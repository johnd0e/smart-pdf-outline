# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pymupdf",
#     "google-genai",
#     "PyYAML",
# ]
# ///

import argparse
import json
import os
import re
import sys
import shutil
import textwrap
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import fitz  # PyMuPDF
import yaml
from google import genai
from google.genai import types

# --- Global State for Interrupt Handling ---
CURRENT_PHASE = "initialization"
MAPPING_YAML_PATH = ""
DISPLAY_YAML_PATH = ""

# --- YAML Setup for Verbatim (Block Scalar) Strings ---
class LiteralStr(str):
    """Custom string class to force YAML block scalar style '|'."""
    pass

def literal_str_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

yaml.add_representer(LiteralStr, literal_str_representer)
yaml.SafeDumper.add_representer(LiteralStr, literal_str_representer)

def clean_and_literalize_prompt(prompt_text):
    """Strips trailing spaces from lines which can cause YAML to fallback to quotes."""
    cleaned = "\n".join(line.rstrip() for line in prompt_text.splitlines()).strip()
    return LiteralStr(cleaned)
# ------------------------------------------------------

DEFAULT_MAPPING_PROMPT = """You are an expert document processing assistant.
Your task is to map a Reference PDF bookmark tree ({ref_lang}) to a Target table of contents data ({target_lang}).

CRITICAL RULES AND LAYOUT INSTRUCTIONS:
1. Attempt to find ALL headings present in the [REF_TOC_TREE]. Do not drop or skip sections unless they genuinely do not exist in the target text.
2. The output titles MUST be written in the {target_lang} language, exactly as they appear in the [TARGET_TOC_DATA].
3. Preserve all chapter numbering (e.g., "1.1", "1.2", "Chapter 1") exactly as it appears in the [REF_TOC_TREE]. Do not reformat the numbering.
4. DO NOT output titles in {ref_lang}. DO NOT translate them yourself.
5. The structural hierarchy of the Target TOC EXACTLY matches the [REF_TOC_TREE]. Preserve the original bookmark order and hierarchy level from the reference.
6. Extract page numbers ONLY from the [TARGET_TOC_DATA].
   IMPORTANT: The target data may be provided as a structured YAML list or raw text. Match the [REF_TOC_TREE] elements to the corresponding items in the target data.
7. If you encounter difficulties, ambiguities, or language mismatches, describe them in the 'processing_notes' field.

[REF_TOC_TREE] ({ref_lang} source structure):
{ref_toc}

[TARGET_TOC_DATA] ({target_lang} translated content):
{target_text}
"""

DEFAULT_EXTRACTION_PROMPT = """You are an expert document processing assistant.
Your task is to find and extract ONLY the complete, detailed Table of Contents (TOC) from the provided document text.

CRITICAL REQUIREMENTS:
1. Scan the ENTIRE provided text FROM START TO FINISH.
2. To prove you have read to the very end, your 'analysis' must explicitly state the LAST heading or text block found at the very bottom of the document text.
3. Many books have a "Brief Contents" (Overview) followed pages later by a "Detailed Contents", often split across multiple blocks.
   Locate the longest, most deeply nested list of chapters.
   WARNING: Use the brief overview as a guide. If the detailed contents ends prematurely, it means you have not extracted it completely.
4. Document your reasoning in the 'analysis' field.
5. Extract the detailed TOC into the 'extracted_toc' JSON array. 
   CRITICAL: You MUST include ALL sub-chapters, sub-sections, and nested items (Level 2, Level 3, etc.). DO NOT output just the high-level chapter titles. DO NOT TRUNCATE.

Document Text:
{raw_text}
"""

DEFAULT_EXTRACTION_REF_HINT = """
[HINT]: The original reference document contains exactly {ref_item_count} bookmark entries across multiple levels. The detailed translated TOC you extract MUST contain roughly the same number of items. If your extraction contains significantly fewer items (e.g. only 10-30 main chapters), YOU ARE EXTRACTING THE WRONG OVERVIEW TOC or summarizing. Keep searching the text for the detailed version and extract EVERY SINGLE nested item!
"""

DEFAULT_APP_CONFIG = {
    "languages": {
        "reference": "auto-detect",
        "target": "auto-detect"
    },
    "prompts": {
        "mapping": DEFAULT_MAPPING_PROMPT,
        "extraction": DEFAULT_EXTRACTION_PROMPT,
        "extraction_ref_hint": DEFAULT_EXTRACTION_REF_HINT
    },
    "generation_config": {
        "model": "gemini-3.1-flash-lite",
        "api_key": "",
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 65536,
    },
    "mapping_schema": {
        "type": "object",
        "properties": {
            "processing_notes": {
                "type": "string",
                "description": "Report any difficulties encountered during mapping, missing pages, or language mismatches here. If none, leave empty."
            },
            "bookmarks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "integer"},
                        "title": {"type": "string"},
                        "page": {"type": "integer"},
                    },
                    "required": ["level", "title", "page"]
                }
            }
        },
        "required": ["bookmarks"]
    },
    "extraction_schema": {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "Chain of thought analysis finding the most detailed TOC, proving scan to the end, and verifying completeness."
            },
            "extracted_toc": {
                "type": "array",
                "description": "The fully structured Table of Contents with resolved hierarchy. MUST NOT BE TRUNCATED.",
                "items": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "integer"},
                        "title": {"type": "string"},
                        "page": {"type": "integer"}
                    },
                    "required": ["level", "title", "page"]
                }
            }
        },
        "required": ["analysis", "extracted_toc"]
    },
    "offset_detection": {
        "scan_limit": 50,
        "header_max_ratio": 0.15,
        "footer_min_ratio": 0.85,
        "min_hits": 2,
    },
    "pdf_save_options": {
        "save_incremental": False,
        "garbage": 1,
        "deflate": True,
        "use_objstms": 1
    }
}

# --- ANSI Coloring Helpers ---
def c_file(text: str) -> str:
    return f"\033[96m{text}\033[0m"

def c_step(step: int | str) -> str:
    return f"\033[94m{step}\033[0m"

def c_err(text: str) -> str:
    return f"\033[91m{text}\033[0m"

def c_warn(text: str) -> str:
    return f"\033[93m{text}\033[0m"

def c_ok(text: str) -> str:
    return f"\033[92m{text}\033[0m"

# -----------------------------

class StepLogger:
    def __init__(self):
        self.step = 1
    def log(self, message: str):
        print(f"\n[{c_step(self.step)}] {message}")
        self.step += 1

logger = StepLogger()

def set_phase(phase_name: str):
    global CURRENT_PHASE
    CURRENT_PHASE = phase_name

def fail(message: str) -> None:
    print(f"\n{c_err('Error:')} {message}", file=sys.stderr)
    raise SystemExit(1)

def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        fail(f"Config file not found: {c_file(path)}")
    except yaml.YAMLError as exc:
        fail(f"Invalid YAML in config file {c_file(path)}: {exc}")


def flat_to_nested(flat_toc):
    root = []
    stack = [(0, root)]
    
    for item in flat_toc:
        level = item['level']
        node = {'title': item['title'], 'page': item['page']}
        
        while stack and stack[-1][0] >= level:
            stack.pop()
            
        if not stack:
            stack = [(0, root)]
            
        parent_children = stack[-1][1]
        parent_children.append(node)
        
        node['children'] = []
        stack.append((level, node['children']))
        
    def remove_empty_children(nodes):
        for n in nodes:
            if not n['children']:
                del n['children']
            else:
                remove_empty_children(n['children'])
                
    remove_empty_children(root)
    return root


def nested_to_flat(nested_toc, current_level=1):
    flat = []
    for node in nested_toc:
        flat.append([current_level, node['title'], node['page']])
        if 'children' in node:
            flat.extend(nested_to_flat(node['children'], current_level + 1))
    return flat


def extract_reference_toc(ref_pdf: str):
    try:
        with fitz.open(ref_pdf) as doc:
            toc = doc.get_toc(simple=True)
    except Exception as exc:
        fail(f"Cannot read reference PDF {c_file(ref_pdf)}: {exc}")
    if not toc:
        fail(f"No bookmarks found in the PDF {c_file(ref_pdf)}")
    return toc


def get_pdf_text_range(pdf_path: str, start_page: int, end_page: int) -> str:
    text_blocks = []
    try:
        with fitz.open(pdf_path) as doc:
            limit = min(end_page, len(doc))
            for i in range(start_page, limit):
                text_blocks.append(doc[i].get_text("text"))
    except Exception as exc:
        fail(f"Failed to read PDF text from {c_file(pdf_path)}: {exc}")
    return "\n".join(text_blocks)


def handle_llm_error(exc: Exception, context: str):
    print(f"\n{c_err('--- LLM Execution Error ---')}")
    print(f"{c_warn('Context:')} {context}")
    
    err_str = str(exc)
    json_match = re.search(r'(\{.*\})', err_str, re.DOTALL)
    if json_match:
        try:
            parsed_err = json.loads(json_match.group(1))
            formatted_err = json.dumps(parsed_err, indent=2, ensure_ascii=False)
            err_str = err_str.replace(json_match.group(1), f"\n{formatted_err}\n")
        except:
            pass

    print(f"{c_err('Details:')}\n{err_str}")
    print(f"{c_err('---------------------------')}\n")
    fail("Failed to process request via LLM.")


def call_llm_with_retry(client, model, contents, config, max_retries=4):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["429", "503", "quota", "overloaded", "busy", "internal server error", "unavailable"]):
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    print(f"    {c_warn(f'LLM API busy/unavailable ({e.__class__.__name__}). Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_retries})')}")
                    time.sleep(sleep_time)
                    continue
            raise e


def extract_toc_via_llm(client, cfg: dict, raw_text: str, ref_toc: list | None) -> str:
    prompt = cfg["prompts"]["extraction"]
    
    if ref_toc:
        ref_count = len(ref_toc)
        hint = cfg["prompts"].get("extraction_ref_hint", "")
        if hint:
            prompt += "\n" + hint.replace("{ref_item_count}", str(ref_count))
    else:
        ref_count = None
            
    prompt = prompt.replace("{raw_text}", raw_text)
    
    kwargs = deepcopy(cfg.get("generation_config", {}))
    model = kwargs.pop("model", "gemini-3.1-flash-lite")
    kwargs.pop("api_key", None)
    kwargs["response_mime_type"] = "application/json"
    kwargs["response_schema"] = cfg.get("extraction_schema")
    gen_cfg = types.GenerateContentConfig(**kwargs)
    
    try:
        response = call_llm_with_retry(client, model, prompt, gen_cfg)
    except Exception as exc:
        handle_llm_error(exc, "Extracting and structuring TOC data")
        
    response_obj = parse_model_response(response)
    
    analysis = response_obj.get("analysis", "")
    if analysis:
        print(f"    {c_warn('Extraction Analysis:')}\n    {analysis}")
        
    extracted = response_obj.get("extracted_toc", [])
    if not extracted:
        print(f"\n{c_err('--- LLM Extraction Error ---')}\n{json.dumps(response_obj, indent=2, ensure_ascii=False)}\n", file=sys.stderr)
        fail("Model returned empty extracted_toc array.")
        
    if ref_count and len(extracted) < (ref_count * 0.4):
        print(f"\n    {c_err('WARNING:')} Extracted TOC has {len(extracted)} items, but reference has {ref_count}. The LLM likely skipped subsections or picked the brief contents!")
        
    return yaml.safe_dump(extracted, allow_unicode=True, sort_keys=False, default_flow_style=False)


def resolve_toc_text(args, cfg: dict, client, target_pdf_abs: str, display_target_pdf: str, ref_toc: list | None) -> str:
    model_name = cfg.get("generation_config", {}).get("model", "gemini-3.1-flash-lite")
    hint_msg = f"(target: ~{len(ref_toc)} items)" if ref_toc else "(no reference)"
    
    if args.toc:
        if args.toc.lower().endswith('.pdf'):
            logger.log(f"Extracting native TOC directly from {c_file(args.toc)}")
            toc_raw = extract_reference_toc(args.toc)
            toc_dicts = [{"level": item[0], "title": item[1], "page": item[2]} for item in toc_raw]
            return yaml.safe_dump(toc_dicts, allow_unicode=True, sort_keys=False, default_flow_style=False)
            
        elif os.path.isfile(args.toc):
            ext = args.toc.lower().split('.')[-1]
            if ext in ['yaml', 'yml', 'json']:
                logger.log(f"Reading structured TOC data from {c_file(args.toc)}")
                return Path(args.toc).read_text(encoding="utf-8")
            else:
                raw_text = Path(args.toc).read_text(encoding="utf-8")
                if not ref_toc:
                    logger.log(f"Using LLM ({c_warn(model_name)}) to structure TOC text from {c_file(args.toc)} {hint_msg}...")
                    return extract_toc_via_llm(client, cfg, raw_text, ref_toc)
                else:
                    logger.log(f"Reading raw TOC text from {c_file(args.toc)}")
                    return raw_text
                    
        elif re.match(r"^\d+(-\d+)?$", args.toc):
            parts = args.toc.split("-")
            start_page = max(0, int(parts[0]) - 1)
            end_page = int(parts[1]) if len(parts) > 1 else start_page + 1
            logger.log(f"Extracting pages {start_page+1} to {end_page} from {c_file(display_target_pdf)}")
            print(f"    -> Using LLM ({c_warn(model_name)}) to locate and structure TOC {hint_msg}...")
            raw_text = get_pdf_text_range(target_pdf_abs, start_page, end_page)
            return extract_toc_via_llm(client, cfg, raw_text, ref_toc)
            
    logger.log(f"Extracting first 50 pages from {c_file(display_target_pdf)}...")
    print(f"    -> Using LLM ({c_warn(model_name)}) to locate and structure TOC {hint_msg}...")
    raw_text = get_pdf_text_range(target_pdf_abs, 0, 50)
    return extract_toc_via_llm(client, cfg, raw_text, ref_toc)


def parse_model_response(response):
    if getattr(response, "parsed", None):
        return response.parsed
    text = (getattr(response, "text", None) or "").strip()
    if text.startswith("```json"):
        text = text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.split("```", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"\n{c_err('--- LLM Output Parsing Error ---')}", file=sys.stderr)
        print(f"{c_warn('Failed to decode JSON from model response.')}", file=sys.stderr)
        print(f"{c_err('Details:')} {exc}", file=sys.stderr)
        print(f"\n{c_warn('Raw Response (Formatted if possible):')}\n", file=sys.stderr)
        print(text, file=sys.stderr)
        print(f"{c_err('--------------------------------')}\n", file=sys.stderr)
        fail("JSON Decode Error")


def validate_entries(response_obj):
    if not isinstance(response_obj, dict):
        formatted_json = json.dumps(response_obj, indent=2, ensure_ascii=False)
        print(f"\n{c_err('--- LLM Schema Validation Error ---')}\n{formatted_json}\n", file=sys.stderr)
        fail("Model returned JSON that does not match the requested schema (expected a root object).")
        
    notes = response_obj.get("processing_notes", "")
    if notes:
        print(f"\n    {c_warn('LLM Processing Report/Notes:')}\n    {notes}\n")

    entries = response_obj.get("bookmarks")
    if not isinstance(entries, list) or not entries:
        formatted_json = json.dumps(response_obj, indent=2, ensure_ascii=False)
        print(f"\n{c_err('--- LLM Schema Validation Error ---')}\n{formatted_json}\n", file=sys.stderr)
        fail("Model returned an empty or missing 'bookmarks' array.")
        
    normalized = []
    for index, item in enumerate(entries, start=1):
        if not isinstance(item, dict):
            fail(f"Entry #{index} is not an object")
        try:
            level = int(item["level"])
            title = str(item["title"]).strip()
            page = int(item["page"])
        except Exception as exc:
            fail(f"Invalid entry #{index}: {exc}")
        normalized.append({"level": level, "title": title, "page": page})
    return normalized


def run_pipeline(args):
    global MAPPING_YAML_PATH, DISPLAY_YAML_PATH
    global logger
    
    display_target_pdf = args.target_pdf
    target_pdf_abs = os.path.abspath(display_target_pdf)
    if not os.path.exists(target_pdf_abs):
        fail(f"Target PDF does not exist: {c_file(display_target_pdf)}")
        
    display_ref_pdf = args.ref_pdf
    ref_pdf_abs = os.path.abspath(display_ref_pdf) if display_ref_pdf else None

    set_phase("setup")
    app_cfg = deepcopy(DEFAULT_APP_CONFIG)
    cfg_file = args.config
    
    if not cfg_file and os.path.exists("app_settings.yaml"):
        cfg_file = "app_settings.yaml"
        print(f"[{c_ok('Info')}] Auto-loading application settings from {c_file(cfg_file)}")
        
    if cfg_file:
        app_cfg = deep_merge(app_cfg, load_yaml_config(cfg_file))

    api_key = args.api_key or app_cfg.get("generation_config", {}).get("api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        fail(f"Provide Gemini API key via {c_warn('--api-key')}, YAML config, or {c_warn('GEMINI_API_KEY')} env var")

    client = genai.Client(api_key=api_key)
    
    if ref_pdf_abs:
        set_phase("extract_reference_toc")
        logger.log(f"Extracting reference bookmarks from: {c_file(display_ref_pdf)}")
        ref_toc = extract_reference_toc(ref_pdf_abs)
        ref_toc_dicts = [{"level": item[0], "title": item[1], "page": item[2]} for item in ref_toc]
        
        if args.save_ref_toc:
            ref_out_path = f"{display_ref_pdf}.ref_toc.yaml"
            nested_ref = flat_to_nested(ref_toc_dicts)
            with open(ref_out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"bookmarks": nested_ref}, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            print(f"    -> Reference TOC saved to: {c_file(ref_out_path)}")
    else:
        ref_toc = None
        ref_toc_dicts = None

    set_phase("extract_target_toc")
    target_toc_data = resolve_toc_text(args, app_cfg, client, target_pdf_abs, display_target_pdf, ref_toc)
    
    is_native_pdf_toc = args.toc and args.toc.lower().endswith('.pdf')
    if is_native_pdf_toc:
        args.save_toc = False
        
    if args.save_toc:
        toc_out_path = f"{display_target_pdf}.target_toc.yaml"
        Path(toc_out_path).write_text(target_toc_data, encoding="utf-8")
        print(f"    -> Extracted Target TOC saved to: {c_file(toc_out_path)}")

    if ref_toc:
        set_phase("llm_mapping")
        model_name = app_cfg.get("generation_config", {}).get("model", "gemini-3.1-flash-lite")
        logger.log(f"Initiating LLM mapping ({c_warn(model_name)}) and parsing response...")
        lang_ref = app_cfg["languages"].get("reference", "auto-detect")
        lang_target = app_cfg["languages"].get("target", "auto-detect")
        
        kwargs = deepcopy(app_cfg.get("generation_config", {}))
        kwargs.pop("model", None)
        kwargs.pop("api_key", None)
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = app_cfg.get("mapping_schema")
        generation_cfg = types.GenerateContentConfig(**kwargs)
        
        ref_toc_yaml = yaml.safe_dump(ref_toc_dicts, allow_unicode=True, sort_keys=False, default_flow_style=False)
        
        prompt = app_cfg["prompts"]["mapping"]
        prompt = prompt.replace("{ref_lang}", lang_ref)
        prompt = prompt.replace("{target_lang}", lang_target)
        prompt = prompt.replace("{ref_toc}", ref_toc_yaml)
        prompt = prompt.replace("{target_text}", target_toc_data)

        try:
            response = call_llm_with_retry(client, model_name, prompt, generation_cfg)
        except Exception as exc:
            handle_llm_error(exc, "Mapping reference TOC to target data")

        response_obj = parse_model_response(response)
        flat_entries = validate_entries(response_obj)
    else:
        set_phase("skip_mapping")
        logger.log("No reference PDF provided. Using extracted/provided TOC directly...")
        extracted_list = yaml.safe_load(target_toc_data)
        flat_entries = validate_entries({"bookmarks": extracted_list})

    nested_entries = flat_to_nested(flat_entries)

    set_phase("offset_detection")
    
    if is_native_pdf_toc:
        logger.log(f"Skipping automatic offset detection. TOC imported directly from a native PDF ({c_file(args.toc)}) contains physical pages.")
        detected_offset = 0
        print(f"    {c_warn('Warning:')} If the source PDF and target PDF differ in pagination (e.g. extra cover pages), you must use {c_file('--offset')} manually.")
    else:
        logger.log(f"Scanning {c_file(display_target_pdf)} to detect page offset...")
        def detect_page_offset_local(doc, detection_cfg):
            scan_limit = int(detection_cfg.get("scan_limit", 50))
            h_ratio = float(detection_cfg.get("header_max_ratio", 0.15))
            f_ratio = float(detection_cfg.get("footer_min_ratio", 0.85))
            m_hits = int(detection_cfg.get("min_hits", 2))
            offsets = []
            for i in range(min(len(doc), scan_limit)):
                page = doc[i]
                rect = page.rect
                blocks = page.get_text("blocks")
                for b in blocks:
                    if len(b) < 7 or b[6] != 0: continue
                    text = b[4].strip()
                    if not text.isdigit(): continue
                    y0, y1 = b[1], b[3]
                    if y1 < rect.height * h_ratio or y0 > rect.height * f_ratio:
                        pnum = int(text)
                        if 0 < pnum <= len(doc):
                            offsets.append((i + 1) - pnum)
            if len(offsets) < m_hits: return 0
            return Counter(offsets).most_common(1)[0][0]

        with fitz.open(target_pdf_abs) as target_doc:
            detected_offset = detect_page_offset_local(target_doc, app_cfg["offset_detection"])

        print(f"    -> Automatically detected page offset: {c_warn(str(detected_offset))}")

    set_phase("save_yaml")
    logger.log("Generating nested YAML mapping file...")
    yaml_out_path = args.out_yaml or f"{display_target_pdf}.bookmarks.yaml"
    
    DISPLAY_YAML_PATH = yaml_out_path
    MAPPING_YAML_PATH = os.path.abspath(yaml_out_path)
    
    pdf_save_options = deepcopy(app_cfg.get("pdf_save_options", {}))
    save_incr = pdf_save_options.pop("save_incremental", False)
    
    mapping_data = {
        "target_pdf": target_pdf_abs,
        "offset": detected_offset,
        "save_incremental": save_incr,
        "save_options": pdf_save_options,
        "bookmarks": nested_entries
    }
    
    with open(MAPPING_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(mapping_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            
    print(f"    -> Unified mapping config saved to: {c_file(yaml_out_path)}")
    
    if args.prepare_only:
        logger.log("Preparation complete. Run 'apply' later manually.")
        return

    logger.log("Proceeding directly to APPLY phase...")
    apply_logic(MAPPING_YAML_PATH, override_offset=None, out_pdf_override=None, display_yaml_path=DISPLAY_YAML_PATH)


def run_apply(args):
    global MAPPING_YAML_PATH, DISPLAY_YAML_PATH
    
    arg = args.command_arg
    if arg.lower().endswith('.pdf'):
        yaml_path = f"{arg}.bookmarks.yaml"
    else:
        yaml_path = arg
        
    DISPLAY_YAML_PATH = yaml_path
    MAPPING_YAML_PATH = os.path.abspath(yaml_path)
    apply_logic(MAPPING_YAML_PATH, override_offset=args.offset, out_pdf_override=args.out_pdf, display_yaml_path=DISPLAY_YAML_PATH)


def apply_logic(mapping_yaml_path, override_offset=None, out_pdf_override=None, display_yaml_path=None):
    set_phase("apply")
    display_path = display_yaml_path or mapping_yaml_path
    mapping = load_yaml_config(mapping_yaml_path)
    
    save_incr = mapping.get("save_incremental", False)
    print(f"\n[{c_step('Apply')}] Loaded mapping from {c_file(display_path)} (Incremental Save: {c_warn('Yes' if save_incr else 'No')})")
    
    target_pdf = mapping.get("target_pdf")
    if not target_pdf or not os.path.exists(target_pdf):
        fail(f"Target PDF '{target_pdf}' specified in YAML does not exist.")

    offset = mapping.get("offset", 0)
    if override_offset is not None:
        offset = override_offset
        print(f"    -> Overriding offset via CLI: {c_warn(str(offset))}")
    else:
        print(f"    -> Using offset from YAML: {c_warn(str(offset))}")

    save_options = mapping.get("save_options", {})
    bookmarks_tree = mapping.get("bookmarks", [])
    
    if not bookmarks_tree:
        fail("No bookmarks found in the YAML file.")

    flat_toc = nested_to_flat(bookmarks_tree)
    final_toc = [[level, title, page + offset] for level, title, page in flat_toc]
    
    base_name = os.path.basename(target_pdf)
    out_pdf_name = out_pdf_override or f"{os.path.splitext(base_name)[0]}_bookmarked.pdf"
    out_pdf_abs = os.path.abspath(out_pdf_name)
    
    print(f"    -> Applying bookmarks to PDF...")
    
    try:
        if save_incr:
            if os.path.abspath(target_pdf) != out_pdf_abs:
                shutil.copy2(target_pdf, out_pdf_abs)
            doc = fitz.open(out_pdf_abs)
            doc.set_toc(final_toc)
            doc.saveIncr()
            doc.close()
        else:
            if save_options.get('garbage', 0) >= 3:
                print(f"    {c_warn('Warning:')} High garbage level (3 or 4) selected. This can cause hangs on complex PDFs.")
            doc = fitz.open(target_pdf)
            doc.set_toc(final_toc)
            doc.save(out_pdf_abs, **save_options)
            doc.close()
    except Exception as exc:
        fail(f"Save failed: {exc}")

    print(f"\n[{c_ok('Success')}] Saved bookmarked PDF as: {c_file(out_pdf_name)}\n")


def build_parser():
    epilog = textwrap.dedent("""
    -------------------------------------------------------------------------------
    USAGE EXAMPLES:
    
    1. Direct Extraction Workflow (No Reference)
       uv run pdf_toc_mapper.py rus.pdf
       
    2. Full Mapping Workflow (With Reference)
       uv run pdf_toc_mapper.py rus.pdf --ref eng.pdf
       
    3. Copy TOC from another PDF
       uv run pdf_toc_mapper.py rus.pdf --toc source.pdf
         
    4. Prepare Only (Stop to edit the YAML)
       uv run pdf_toc_mapper.py rus.pdf --prepare-only
         
    5. Apply (After editing the YAML)
       uv run pdf_toc_mapper.py apply rus.pdf.bookmarks.yaml
         
    6. Generate Settings Config
       uv run pdf_toc_mapper.py savecfg
    -------------------------------------------------------------------------------
    """)
    
    parser = argparse.ArgumentParser(
        description="PDF TOC Mapper. Extracts, maps, and applies bookmarks to a target PDF.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=epilog
    )
    
    parser.add_argument("target_or_cmd", help="Target PDF file OR special command ('apply', 'savecfg')")
    parser.add_argument("command_arg", nargs="?", help="Argument for the special command")
    
    group = parser.add_argument_group("Main Pipeline Arguments")
    group.add_argument("--ref", dest="ref_pdf", help="[Optional] Path to the reference PDF with existing bookmarks", default=None)
    group.add_argument("--toc", help="[Optional] TOC source: a .txt/.yaml/.pdf file, a page num ('15'), or range ('10-15')", default=None)
    
    group.add_argument("--prepare-only", action="store_true", help="[Optional] Stop after creating the mapping YAML. Do not apply bookmarks.")
    
    group.add_argument("--save-toc", action="store_true", help="[Optional] Save the extracted Target TOC to a .target_toc.yaml file")
    group.add_argument("--save-ref-toc", action="store_true", help="[Optional] Save the extracted Reference TOC tree to a .ref_toc.yaml file")
    
    group.add_argument("--config", help="[Optional] Application settings YAML (default: looks for 'app_settings.yaml')", default=None)
    group.add_argument("--out-yaml", help="[Optional] Name for the generated mapping YAML file", default=None)
    group.add_argument("--api-key", help="[Optional] Gemini API key (or set GEMINI_API_KEY env var)", default=None)

    apply_group = parser.add_argument_group("Apply Command Arguments")
    apply_group.add_argument("--offset", type=int, help="[Optional] Override page offset manually during 'apply'")
    apply_group.add_argument("--out-pdf", help="[Optional] Override output PDF name")
    
    return parser


def main(argv=None):
    if len(sys.argv) == 1:
        parser = build_parser()
        parser.print_usage()
        sys.exit(1)

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    
    try:
        target_or_cmd = args.target_or_cmd
        if target_or_cmd in ["apply", "savecfg"]:
            args.command = target_or_cmd
            if args.command == "apply" and not args.command_arg:
                fail("The 'apply' command requires the path to a mapping YAML or target PDF file.")
        else:
            args.command = None
            args.target_pdf = target_or_cmd
            
        if args.command == "apply":
            run_apply(args)
            return
            
        if args.command == "savecfg":
            cfg_path = args.command_arg if args.command_arg else "app_settings.yaml"
            
            cfg_to_dump = deepcopy(DEFAULT_APP_CONFIG)
            for k, v in cfg_to_dump.get('prompts', {}).items():
                cfg_to_dump['prompts'][k] = clean_and_literalize_prompt(v)
                
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_to_dump, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            print(f"[{c_ok('Success')}] Saved default application settings to {c_file(cfg_path)}")
            return

        run_pipeline(args)
        
    except KeyboardInterrupt:
        print(f"\n\n{c_err('Process interrupted by user (KeyboardInterrupt).')}")
        print(f"Interrupted during phase: {c_warn(CURRENT_PHASE)}")
        
        if CURRENT_PHASE == "apply" or CURRENT_PHASE == "save_yaml":
            if DISPLAY_YAML_PATH:
                print(f"\nYou can resume the process later by running:")
                print(f"  uv run pdf_toc_mapper.py apply {c_file(DISPLAY_YAML_PATH)}\n")
        sys.exit(130)


if __name__ == "__main__":
    main()
