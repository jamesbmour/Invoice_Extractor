"""General document extractor with Docling or multimodal-only LLM modes."""

from __future__ import annotations
import base64
import json
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pdf2image import convert_from_bytes

load_dotenv()

########################### App Configuration & Constants ###########################

APP_TITLE = "General Document Extractor"
DEFAULT_MODEL_NAME = "ministral-3:3b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Fall back to local Ollama instance
SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
SCHEMA_PATH = Path(__file__).with_name("field_schema.json")  # Schema file lives alongside this script

# Define string constants for extraction mode identifiers
MODE_DOCLING = "docling"
MODE_MM = "multimodal"
MODE_OPTIONS = {"Docling + Text LLM": MODE_DOCLING, "Multimodal LLM only": MODE_MM}

# Map MIME types to file extensions for temp file creation
MIME_SUFFIXES = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}

# Provide sensible invoice-focused defaults when no schema file exists
DEFAULT_FIELDS = [
    {
        "name": "vendor_name",
        "description": "Legal or display name of the vendor/seller.",
    },
    {"name": "vendor_address", "description": "Full mailing address for the vendor."},
    {"name": "vendor_email", "description": "Vendor billing or contact email address."},
    {"name": "vendor_phone", "description": "Vendor contact phone number."},
    {"name": "invoice_number", "description": "Unique invoice identifier."},
    {"name": "invoice_date", "description": "Date when the invoice was issued."},
    {"name": "due_date", "description": "Payment due date listed on the invoice."},
    {
        "name": "currency",
        "description": "Invoice currency code or symbol (USD, EUR, $, etc.).",
    },
    {"name": "subtotal", "description": "Amount before tax and fees."},
    {"name": "tax", "description": "Total tax amount charged."},
    {"name": "total", "description": "Final total amount due."},
    {
        "name": "purchase_order_number",
        "description": "Associated PO number if present.",
    },
    {"name": "bill_to", "description": "Billing recipient name and/or address block."},
    {"name": "ship_to", "description": "Shipping recipient name and/or address block."},
    {
        "name": "payment_terms",
        "description": "Payment terms such as Net 30 or due on receipt.",
    },
    {
        "name": "line_items",
        "description": "Array of objects with description, quantity, unit_price, and amount.",
    },
]

########################### State & Type Definitions ###########################

# Define a dict-based extraction state; keys remain optional as each step adds fields incrementally.
ExtractionState = dict[str, Any]


########################### Field Normalization Utilities ###########################


def normalize_name(name: str) -> str:
    """Convert an arbitrary string into a clean, lowercase snake_case key."""
    # Strip non-alphanumeric characters and collapse repeated underscores for consistent key formatting
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())).strip("_")


def normalize_fields(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Deduplicate and clean a list of field definitions, keeping the last occurrence of each key."""
    # Use an ordered dict pattern to deduplicate fields while preserving the last-seen definition
    deduped: dict[str, str] = {}
    for row in rows:
        key = normalize_name(row.get("name", ""))
        if not key:
            continue
        deduped.pop(key, None)  # Move duplicate keys to the end so the latest entry wins
        deduped[key] = str(row.get("description", "")).strip()
    return [{"name": key, "description": description} for key, description in deduped.items()]


########################### Schema Persistence ###########################


def load_fields() -> list[dict[str, str]]:
    """Load field definitions from the JSON schema file, falling back to built-in defaults."""
    # Prefer the saved schema file; fall back to defaults if file is missing, invalid, or empty
    defaults = normalize_fields(DEFAULT_FIELDS)
    if not SCHEMA_PATH.exists():
        return defaults
    try:
        loaded = normalize_fields(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return defaults
    return loaded or defaults


def save_fields(fields: list[dict[str, Any]]) -> None:
    """Persist normalized field definitions to the JSON schema file."""
    # Normalize before saving to ensure the file always contains clean, consistent keys
    SCHEMA_PATH.write_text(json.dumps(normalize_fields(fields), indent=2), encoding="utf-8")


########################### Document Processing ###########################


# Cache the converter across Streamlit reruns to avoid expensive re-initialization
@st.cache_resource
def get_docling_converter() -> Any:
    """Return a singleton Docling DocumentConverter instance."""
    return DocumentConverter()


def extract_text_with_docling(file_name: str, file_type: str, file_bytes: bytes) -> str:
    """Convert a document to markdown text using the Docling library."""
    # Map MIME types to file extensions so Docling can infer the correct parser
    suffix = MIME_SUFFIXES.get(file_type, Path(file_name).suffix or ".bin")  # Fall back to original extension

    # Write bytes to a temp file because Docling requires a file-system path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        result = get_docling_converter().convert(tmp_path)
        return result.document.export_to_markdown().strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # Clean up temp file after conversion


def to_base64_images(file_type: str, file_bytes: bytes) -> list[tuple[str, str]]:
    """Convert a file into a list of (MIME type, base64 string) tuples, one per page/image."""
    # Rasterize every PDF page to PNG so the vision model can consume the full document
    if file_type == "application/pdf":
        encoded_pages: list[tuple[str, str]] = []
        for image in convert_from_bytes(file_bytes, dpi=200):  # 200 dpi balances quality and token size
            buf = BytesIO()
            image.save(buf, format="PNG")
            encoded_pages.append(("image/png", base64.b64encode(buf.getvalue()).decode("utf-8")))
        return encoded_pages
    return [(file_type, base64.b64encode(file_bytes).decode("utf-8"))]


def build_multimodal_content(file_type: str, file_bytes: bytes) -> list[dict[str, Any]]:
    """Construct a multimodal message payload with text prompt and inline base64 images."""
    # Build the multimodal payload from one image (image upload) or many images (multi-page PDF)
    images = to_base64_images(file_type, file_bytes)
    prompt = "Extract information from this document image."
    if file_type == "application/pdf":
        prompt = (
            f"Extract information from this {len(images)}-page PDF document. "
            "Use all pages before returning JSON."
        )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    # Append each page image with an optional page label for multi-page PDFs
    for page_number, (mime, image_b64) in enumerate(images, start=1):
        if file_type == "application/pdf":
            content.append({"type": "text", "text": f"PDF page {page_number}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image_b64}"},
            }
        )
    return content


########################### LLM Prompt Construction ###########################


def build_system_prompt(field_defs: list[dict[str, str]]) -> str:
    """Generate a system prompt that instructs the LLM to extract specific fields as JSON."""
    # Instruct the model to return strict JSON with only the requested fields
    lines = [
        "You extract structured data from documents.",
        "Return valid JSON only.",
        "Extract exactly these keys and no others.",
        "Use null when a field is missing.",
        "Fields:",
    ]
    # Append each field definition as a labelled bullet for the model to reference
    lines.extend(f"- {field['name']}: {field['description'] or 'No description provided.'}" for field in field_defs)
    return "\n".join(lines)


########################### Response Parsing ###########################


def parse_json_content(content: Any) -> tuple[str, dict[str, Any]]:
    """Parse LLM output into a (raw_string, parsed_dict) tuple, handling various response formats."""
    # Handle dict responses directly without re-serializing
    if isinstance(content, dict):
        return json.dumps(content), content

    # Concatenate multi-part message content into a single string
    if isinstance(content, list):
        parts = [
            item
            if isinstance(item, str)
            else item.get("text", "")
            if isinstance(item, dict) and isinstance(item.get("text"), str)
            else str(item)
            for item in content
        ]
        raw = "\n".join(parts).strip()
    else:
        raw = str(content).strip()

    # Strip markdown code fences that some models wrap around JSON output
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return raw, json.loads(cleaned)


########################### Extraction Workflow ###########################


def make_llm(model_name: str) -> ChatOllama:
    """Create an Ollama chat client configured for deterministic JSON output."""
    # Use temperature=0 for deterministic, structured JSON extraction
    return ChatOllama(model=model_name, temperature=0, format="json", base_url=OLLAMA_BASE_URL)


def extract_with_prompt(
    model_name: str,
    field_defs: list[dict[str, str]],
    human_content: str | list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Send field definitions and document content to the LLM, returning parsed JSON."""
    # Send the system prompt and human message to the LLM and parse the structured response
    response = make_llm(model_name).invoke(
        [
            SystemMessage(content=build_system_prompt(field_defs)),
            HumanMessage(content=human_content),
        ]
    )
    return parse_json_content(response.content)


def run_extraction(file_name: str, file_type: str, file_bytes: bytes, model_name: str, mode: str) -> ExtractionState:
    """Orchestrate the full extraction pipeline based on the selected mode."""
    # Bundle all inputs into a shared state dict for downstream steps to reference
    state: ExtractionState = {
        "file_name": file_name,
        "file_type": file_type,
        "file_bytes": file_bytes,
        "model_name": model_name,
        "field_defs": st.session_state["field_defs"],  # Pull live field definitions from session state
        "mode": mode,
    }

    # Build a single-step multimodal flow or a two-step Docling+LLM flow
    if mode == MODE_MM:
        raw_response, extracted_fields = extract_with_prompt(
            model_name,
            state["field_defs"],
            build_multimodal_content(file_type, file_bytes),
        )
        return {
            "file_name": file_name,
            "mode": MODE_MM,
            "raw_response": raw_response,
            "extracted_fields": extracted_fields,
            "document_text": "",  # Return empty document_text since no Docling step was performed
        }

    document_text = extract_text_with_docling(
        file_name, file_type, file_bytes
    )  # Convert to markdown for text-only extraction
    raw_response, extracted_fields = extract_with_prompt(
        model_name,
        state["field_defs"],
        f"Extract information from this document content.\n\n{document_text}",
    )
    return {
        "file_name": file_name,
        "mode": MODE_DOCLING,
        "document_text": document_text,
        "raw_response": raw_response,
        "extracted_fields": extracted_fields,
    }


########################### UI Rendering Helpers ###########################


def render_file_preview(file_type: str, file_name: str, file_bytes: bytes) -> None:
    """Display an uploaded file preview — native image or embedded PDF viewer."""
    # Display raster images natively; embed PDFs using an HTML object tag
    if file_type.startswith("image/"):
        st.image(file_bytes, caption=file_name)
    if file_type == "application/pdf":
        pdf_b64 = base64.b64encode(file_bytes).decode("utf-8")
        st.markdown(
            f'<embed src="data:application/pdf;base64,{pdf_b64}" width="100%" height="700" type="application/pdf">',
            unsafe_allow_html=True,  # Required to inject raw HTML into Streamlit
        )


def render_results(result: ExtractionState) -> None:
    """Display extraction results with JSON preview, CSV download, and debug expanders."""
    st.subheader("Extracted JSON")
    extracted_fields = result.get("extracted_fields", {})
    st.json(extracted_fields)

    # Convert extracted fields into a one-row CSV; serialize nested structures as JSON strings
    normalized_for_csv = (
        {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            for key, value in extracted_fields.items()
        }
        if isinstance(extracted_fields, dict)
        else {"extracted_data": json.dumps(extracted_fields, ensure_ascii=False)}
    )
    csv_bytes = pd.DataFrame([normalized_for_csv]).to_csv(index=False).encode("utf-8")
    file_stem = Path(str(result.get("file_name", "extracted_data"))).stem
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"{file_stem}_extracted.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Raw model output"):
        st.code(result.get("raw_response", ""), language="json")

    # Only show the Docling text expander when text-based extraction was used
    if result.get("mode") == MODE_DOCLING:
        with st.expander("Docling extracted text"):
            st.text(result.get("document_text", ""))


########################### Streamlit Tab Renderers ###########################


def render_extract_tab() -> None:
    """Render the main extraction tab with sidebar controls and result display."""
    with st.sidebar:
        st.subheader("Extraction Settings")
        model_name = st.text_input("Ollama model", value=DEFAULT_MODEL_NAME)
        mode_label = st.radio("Extraction method", list(MODE_OPTIONS))
        uploaded_file = st.file_uploader("Upload invoice", type=SUPPORTED_UPLOAD_TYPES, key="upload_file")

    mode = MODE_OPTIONS[mode_label]

    # Exit early when no file has been uploaded yet to avoid downstream errors
    if not uploaded_file:
        return

    file_bytes = uploaded_file.getvalue()
    st.write(f"File: `{uploaded_file.name}`")
    render_file_preview(uploaded_file.type, uploaded_file.name, file_bytes)

    # Trigger extraction and persist the result so it survives reruns
    if st.button("Extract Information", type="primary"):
        with st.spinner("Running extraction workflow..."):
            st.session_state["extraction_result"] = run_extraction(
                uploaded_file.name, uploaded_file.type, file_bytes, model_name, mode
            )

    if st.session_state.get("extraction_result") is not None:
        render_results(st.session_state["extraction_result"])


def render_fields_tab() -> None:
    """Render the field schema editor tab for customizing extraction fields."""
    st.write("Add or remove field names and descriptions used for extraction.")
    edited = st.data_editor(st.session_state["field_defs"], num_rows="dynamic", width="stretch")

    # Handle both pandas DataFrames and plain list-of-dicts returned by the editor
    rows = edited.to_dict("records") if hasattr(edited, "to_dict") else [dict(row) for row in edited]
    normalized = normalize_fields(rows)

    # Show the user exactly which keys will be sent to the LLM after normalization
    st.caption("Active normalized keys")
    st.code(", ".join(field["name"] for field in normalized) or "(none)")

    left, right = st.columns(2)
    if left.button("Save Fields", type="primary"):
        st.session_state["field_defs"] = normalized
        save_fields(normalized)

    # Reset session state and persisted schema back to the built-in invoice defaults
    if right.button("Reset to Invoice Defaults"):
        defaults = normalize_fields(DEFAULT_FIELDS)
        st.session_state["field_defs"] = defaults
        save_fields(defaults)
        st.rerun()  # Refresh the UI to reflect the reset values


########################### App Entry Point ###########################


def main() -> None:
    """Initialize the Streamlit app and render the tabbed interface."""
    st.set_page_config(page_title="General Extractor", page_icon=":clipboard:")

    # Initialize session state keys on first run to prevent KeyErrors during rendering
    st.session_state.setdefault("field_defs", load_fields())
    st.session_state.setdefault("extraction_result", None)

    st.title(APP_TITLE)

    # Render the two primary tabs for extraction and field management
    extract_tab, fields_tab = st.tabs(["Extract", "Fields"])
    with extract_tab:
        render_extract_tab()
    with fields_tab:
        render_fields_tab()


if __name__ == "__main__":
    main()
