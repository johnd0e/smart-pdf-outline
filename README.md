# PDF TOC Mapper

## 🎯 Overview & Goals
This project provides an automated CLI tool designed to solve a specific problem in document management: **extracting, structuring, transferring, and translating PDF Table of Contents (Bookmarks / Outlines).**

Instead of relying on fragile regex parsers to interpret inconsistently formatted PDF TOC pages, this tool uses Large Language Models (Google Gemini) to semantically extract and format bookmarks from plain text pages.

## ✨ Key Features
- **Smart TOC Extraction:** Extracts and structurally formats TOC data from the target PDF automatically (scanning the first 50 pages) or via specific page ranges (e.g., `--toc 10-15`) using a dedicated LLM extraction prompt.
- **Semantic Mapping (Optional):** Uses Gemini to map a well-structured reference bookmark tree to translated text, preserving hierarchy and order.
- **Copy TOC from PDF:** Instantly extract and transfer native bookmarks from one PDF to another (`--toc source.pdf`).
- **Automatic Page Offset Detection:** Scans the target PDF's headers/footers to calculate the exact offset between physical page numbers (1-based index) and printed page numbers.
- **Continuous or Human-in-the-Loop Workflow:** Can run the entire process continuously or stop in a `--prepare-only` phase to output an intermediate YAML mapping, allowing the user to manually verify and correct the layout before altering the PDF.

## 🛠 Prerequisites & Limitations
- **Python 3.10+**
- **[uv](https://github.com/astral-sh/uv):** The script uses PEP 723 inline script metadata, allowing `uv run` to automatically fetch dependencies in an isolated environment.
- **Google Gemini API Key:** Obtainable from [Google AI Studio](https://aistudio.google.com/). Set it via the `GEMINI_API_KEY` env var, pass via `--api-key`, or configure in `app_settings.yaml`.
- **Text Layer Required:** The script relies on PyMuPDF to extract text blocks. It **does not perform OCR**. The target PDF must contain a selectable text layer. Scanned books without a text layer cannot be processed automatically.

## 🚀 Usage Workflow

The target PDF is the primary argument. The tool operates in a continuous pipeline by default, but can be split into `prepare` (extraction & mapping) and `apply` (writing to PDF) phases.

### Step 0: Set API Key
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 1. Direct Extraction (No Reference)
Extract TOC directly from the document's first 50 pages and apply it immediately.
```bash
uv run pdf_toc_mapper.py russian_book.pdf
```

### 2. Using a Reference PDF (`--ref`)
This mode is highly recommended when you have another PDF (e.g., the original English book) that already contains a well-structured bookmark tree. The script uses this reference to guide the LLM, ensuring the target document's TOC hierarchy is perfectly preserved and no nested sections are missed or hallucinated.
```bash
uv run pdf_toc_mapper.py russian_book.pdf --ref english_book.pdf
```

### 3. Copy TOC from another PDF (`--toc`)
If you want to instantly grab native bookmarks from another PDF and inject them into the target (bypassing the LLM).
```bash
uv run pdf_toc_mapper.py target.pdf --toc source.pdf
```
*Note: Because native PDFs provide absolute physical pages, automatic offset detection is disabled in this mode. Use `--offset` manually if the pagination differs between the two files.*

### 4. Prepare Mode (Review before apply)
Stop execution after the `*.bookmarks.yaml` file is generated. You can open the YAML, manually adjust titles/pages, and run `apply` later.

**Option A: Fully Automatic TOC Extraction**
```bash
uv run pdf_toc_mapper.py russian_book.pdf --prepare-only
```

**Option B: Targeted Page Range**
(If you know the TOC is on pages 5 through 10 of the target PDF).
```bash
uv run pdf_toc_mapper.py russian_book.pdf --toc 5-10 --prepare-only
```

### 5. Apply Phase
Apply a manually reviewed YAML mapping to the PDF. The script will read the configuration file automatically.
```bash
uv run pdf_toc_mapper.py apply russian_book.pdf
# or
uv run pdf_toc_mapper.py apply russian_book.bookmarks.yaml
```
*Optional overrides:* `--offset 12` to manually force a page offset, or `--out-pdf custom_name.pdf`.

---

## ⚙️ Configuration (`savecfg`)
You can export the default configuration to modify prompts, models, and PDF save options. The script forces YAML to use block scalars (`|`) so prompts remain unescaped and readable.
```bash
uv run pdf_toc_mapper.py savecfg
```
This generates `app_settings.yaml`. The script will automatically load this file if it exists in the working directory.
Inside the config, you can define:
- `prompts`: Custom system instructions for `mapping` and `extraction`.
- `generation_config`: Model (e.g., `gemini-3.1-flash-lite`), temperature, tokens, and API key.
- `pdf_save_options`: PyMuPDF save arguments (e.g., `save_incremental` and `garbage`).

---

## 🧰 Tools & Approaches Justification

### 1. PyMuPDF (Fitz)
- **Link:** [PyMuPDF](https://pymupdf.readthedocs.io/)
- **Why:** Exceptionally fast C-based (MuPDF) library. Native, robust support for reading and writing `Outlines` (Bookmarks) and extracting text blocks with layout coordinates.
- **Approach:** Used for extracting reference TOCs, reading header/footer coordinates for offset detection, and embedding the final TOC. Supports `saveIncr()` for preventing file bloat.

### 2. Google GenAI SDK (Gemini)
- **Link:** [google-genai](https://googleapis.github.io/python-genai/)
- **Why:** Gemini models offer massive context windows (up to 1M-2M tokens) for cheap/free, making them perfect for stuffing entire book TOCs or 50-page text dumps.
- **Approach:** Utilizes `response_mime_type="application/json"` and `response_schema` to enforce strict JSON array outputs. Two separate prompts are used: one for isolating and structuring TOC data from raw PDF dumps, and one for semantic mapping. Implements an exponential backoff retry strategy for handling API rate limits.

### 3. Page Offset Detection Algorithm
- **Approach:** Scans text blocks of the first N pages. Filters for purely numeric blocks located in the top 15% or bottom 15% of the page bounding box. Uses `collections.Counter` to find the most common (modal) offset, filtering out noise like publication years.

---

## ⚠️ Limitations & Known Issues

1. **Text Layer Required (No OCR):**
   - *Issue:* The script reads text directly from the PDF's digital layer. It does not perform OCR.
   - *Impact:* Scanned PDFs (image-only) will produce empty or garbage text. This tool is designed for digitally created or digitally converted PDFs.
2. **PDF File Size Bloat:**
   - *Issue:* Modifying and saving PDFs with PyMuPDF can sometimes increase file size.
   - *Mitigation:* The script defaults to `garbage=1`. You can toggle `save_incremental: true` in the YAML config to simply append the TOC without rewriting the file.
3. **Offset Detection with Roman Numerals:**
   - *Issue:* Offset detection looks for `text.isdigit()` and ignores Roman numerals (i, ii, iv). Preface pages using Roman numerals cannot be auto-detected.
   - *Mitigation:* Use `--offset` manually or correct page numbers in the YAML before `apply`.
4. **LLM Hallucinations on Complex Layouts:**
   - *Issue:* Multi-column TOC layouts may cause page number mismatches.
   - *Mitigation:* Use `--prepare-only` to review the YAML before writing to the PDF. The `--ref` flag also significantly reduces hallucinations by anchoring the expected structure.

---

## 📚 Development History

- **v1-v3:** Simple CLI, two files, automatic offset detection, TSV output.
- **v4-v5:** Stateful workflow with sidecar JSON config. Strict structured LLM outputs via `response_schema`.
- **v6:** LLM-based TOC text extraction from page ranges.
- **v7-v17:** YAML transition; nested hierarchical output; single continuous pipeline; chain-of-thought extraction; exponential backoff retry.
- **v18-v21 (Current):** Structured JSON extraction (level/title/page schema); `--ref` made optional; `--toc` accepts any format including `.pdf` for native bookmark copy; target PDF becomes positional argument.
