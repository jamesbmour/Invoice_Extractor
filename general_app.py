"""General document extractor using Docling + Ollama + LangGraph."""
from __future__ import annotations
import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, TypedDict
import streamlit as st
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

################################ Configuration & Constants ################################

load_dotenv()

APP_TITLE = "General Document Extractor (Docling + Ollama + LangGraph)"
DEFAULT_MODEL_NAME = "ministral-3:3b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Fall back to localhost if not set
SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
SCHEMA_PATH = Path(__file__).with_name("field_schema.json")  # Store schema alongside this script

# Define default invoice fields used when no saved schema is found
DEFAULT_FIELDS = [
    {"name": "vendor_name", "description": "Legal or display name of the vendor/seller."},
    {"name": "vendor_address", "description": "Full mailing address for the vendor."},
    {"name": "vendor_email", "description": "Vendor billing or contact email address."},
    {"name": "vendor_phone", "description": "Vendor contact phone number."},
    {"name": "invoice_number", "description": "Unique invoice identifier."},
    {"name": "invoice_date", "description": "Date when the invoice was issued."},
    {"name": "due_date", "description": "Payment due date listed on the invoice."},
    {"name": "currency", "description": "Invoice currency code or symbol (USD, EUR, $, etc.)."},
    {"name": "subtotal", "description": "Amount before tax and fees."},
    {"name": "tax", "description": "Total tax amount charged."},
    {"name": "total", "description": "Final total amount due."},
    {"name": "purchase_order_number", "description": "Associated PO number if present."},
    {"name": "bill_to", "description": "Billing recipient name and/or address block."},
    {"name": "ship_to", "description": "Shipping recipient name and/or address block."},
    {"name": "payment_terms", "description": "Payment terms such as Net 30 or due on receipt."},
    {"name": "line_items", "description": "Array of objects with description, quantity, unit_price, and amount."},
]

################################ State Definition ################################

# Define the shared state passed between LangGraph nodes during extraction
class ExtractionState(TypedDict, total=False):
    file_name: str
    file_type: str
    file_bytes: bytes
    model_name: str
    field_defs: list[dict[str, str]]
    document_text: str      # Markdown text produced by Docling
    raw_response: str       # Raw string output from the LLM before parsing
    extracted_fields: dict[str, Any]  # Final parsed JSON from the LLM

################################ Field Normalization & Persistence ################################

def normalize_name(name: str) -> str:
    # Convert any field name to a consistent snake_case key, stripping invalid characters
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())).strip("_")

def normalize_fields(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    # Deduplicate fields by normalized name, keeping the last definition for each key
    deduped: dict[str, str] = {}
    order: list[str] = []
    for row in rows:
        key = normalize_name(row.get("name", ""))
        if not key:
            continue
        # Remove the existing entry so the updated one moves to the end
        if key in deduped:
            order.remove(key)
        order.append(key)
        deduped[key] = str(row.get("description", "")).strip()
    return [{"name": key, "description": deduped[key]} for key in order]

def load_fields() -> list[dict[str, str]]:
    # Prefer the saved schema file; fall back to defaults if the file is missing or empty
    if SCHEMA_PATH.exists():
        loaded = normalize_fields(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
        return loaded or normalize_fields(DEFAULT_FIELDS)
    return normalize_fields(DEFAULT_FIELDS)

def save_fields(fields: list[dict[str, Any]]) -> None:
    # Persist normalized fields to disk so they survive app restarts
    SCHEMA_PATH.write_text(json.dumps(normalize_fields(fields), indent=2), encoding="utf-8")

################################ Document Conversion ################################

# Cache the converter across Streamlit reruns to avoid re-initializing on every interaction
@st.cache_resource
def get_docling_converter() -> Any:
    return DocumentConverter()

def extract_text_with_docling(file_name: str, file_type: str, file_bytes: bytes) -> str:
    # Map MIME types to file extensions so Docling can parse the format correctly
    suffixes = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }
    suffix = suffixes.get(file_type, Path(file_name).suffix or ".bin")  # Preserve original extension as fallback

    # Write file bytes to a temp file, convert it, then clean up immediately
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name
    result = get_docling_converter().convert(temp_path)
    Path(temp_path).unlink(missing_ok=True)  # Remove temp file even if conversion failed

    return result.document.export_to_markdown().strip()

################################ LLM Prompt Building ################################

def build_system_prompt(field_defs: list[dict[str, str]]) -> str:
    # Construct a strict extraction prompt that constrains the LLM to only the defined fields
    lines = [
        "You extract structured data from documents.",
        "Return valid JSON only.",
        "Extract exactly these keys and no others.",
        "Use null when a field is missing.",
        "Fields:",
    ]
    lines.extend(f"- {field['name']}: {field['description'] or 'No description provided.'}" for field in field_defs)
    return "\n".join(lines)

################################ Response Parsing ################################

def parse_json_content(content: Any) -> tuple[str, dict[str, Any]]:
    # Handle dict responses directly, bypassing string parsing
    if isinstance(content, dict):
        return json.dumps(content), content

    # Reassemble list responses (e.g., multi-part tool outputs) into a single string
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
            else:
                pieces.append(str(item))
        raw = "\n".join(pieces).strip()
    else:
        raw = str(content).strip()

    # Strip markdown code fences the LLM may wrap around JSON output
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return raw, json.loads(cleaned)

################################ LangGraph Nodes ################################

def load_document_node(state: ExtractionState) -> ExtractionState:
    # Node 1: Convert the uploaded file to markdown text via Docling
    return {"document_text": extract_text_with_docling(state["file_name"], state["file_type"], state["file_bytes"])}

def extract_fields_node(state: ExtractionState) -> ExtractionState:
    # Node 2: Send the document text to the LLM and parse the structured JSON response
    llm = ChatOllama(model=state["model_name"], temperature=0, format="json", base_url=OLLAMA_BASE_URL)  # temperature=0 for deterministic extraction
    prompt = f"Extract information from this document content.\n\n{state['document_text']}"
    response = llm.invoke([SystemMessage(content=build_system_prompt(state["field_defs"])), HumanMessage(content=prompt)])
    raw_response, extracted_fields = parse_json_content(response.content)
    return {"raw_response": raw_response, "extracted_fields": extracted_fields}

################################ Extraction Workflow ################################

def run_extraction(file_name: str, file_type: str, file_bytes: bytes, model_name: str) -> ExtractionState:
    # Build a two-node LangGraph pipeline: document loading → field extraction
    graph = StateGraph(ExtractionState)
    graph.add_node("load_document", load_document_node)
    graph.add_node("extract_fields", extract_fields_node)
    graph.set_entry_point("load_document")
    graph.add_edge("load_document", "extract_fields")
    graph.add_edge("extract_fields", END)

    # Compile and invoke the graph, injecting the current field schema from session state
    return graph.compile().invoke(
        {"file_name": file_name, "file_type": file_type, "file_bytes": file_bytes, "model_name": model_name, "field_defs": st.session_state["field_defs"]}
    )

################################ UI Rendering ################################

def render_file_preview(file_type: str, file_name: str, file_bytes: bytes) -> None:
    # Render image files inline for quick visual confirmation
    if file_type.startswith("image/"):
        st.image(file_bytes, caption=file_name)

    # Embed PDFs using an HTML embed tag since Streamlit has no native PDF viewer
    if file_type == "application/pdf":
        pdf_b64 = base64.b64encode(file_bytes).decode("utf-8")
        st.markdown(f'<embed src="data:application/pdf;base64,{pdf_b64}" width="100%" height="700" type="application/pdf">', unsafe_allow_html=True)

def render_results(result: ExtractionState) -> None:
    # Display the structured extraction output and collapse verbose debug sections by default
    st.subheader("Extracted JSON")
    st.json(result.get("extracted_fields", {}))
    with st.expander("Raw model output"):
        st.code(result.get("raw_response", ""), language="json")
    with st.expander("Docling extracted text"):
        st.text(result.get("document_text", ""))

def render_extract_tab() -> None:
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL_NAME)
    uploaded_file = st.file_uploader("Upload file", type=SUPPORTED_UPLOAD_TYPES, key="upload_file")

    # Wait for a file before rendering the rest of the tab
    if not uploaded_file:
        return

    file_bytes = uploaded_file.getvalue()
    st.write(f"File: `{uploaded_file.name}`")
    render_file_preview(uploaded_file.type, uploaded_file.name, file_bytes)

    # Trigger the full extraction pipeline and cache the result in session state
    if st.button("Extract Information", type="primary"):
        with st.spinner("Running extraction workflow..."):
            st.session_state["extraction_result"] = run_extraction(uploaded_file.name, uploaded_file.type, file_bytes, model_name)

    # Display persisted results so they survive Streamlit reruns without re-extracting
    if st.session_state.get("extraction_result") is not None:
        render_results(st.session_state["extraction_result"])

def render_fields_tab() -> None:
    st.write("Add or remove field names and descriptions used for extraction.")

    # Allow live editing of the field schema via an interactive table
    edited = st.data_editor(st.session_state["field_defs"], num_rows="dynamic", width="stretch")

    # Normalize editor output regardless of whether it returns a DataFrame or raw dicts
    rows = edited.to_dict("records") if hasattr(edited, "to_dict") else [dict(row) for row in edited]
    normalized = normalize_fields(rows)

    # Show the active normalized key names so users can verify their field definitions
    st.caption("Active normalized keys")
    st.code(", ".join(field["name"] for field in normalized) or "(none)")

    left, right = st.columns(2)
    if left.button("Save Fields", type="primary"):
        st.session_state["field_defs"] = normalized
        save_fields(normalized)

    # Overwrite custom fields with the built-in invoice defaults and refresh the UI
    if right.button("Reset to Invoice Defaults"):
        defaults = normalize_fields(DEFAULT_FIELDS)
        st.session_state["field_defs"] = defaults
        save_fields(defaults)
        st.rerun()

################################ App Entry Point ################################

def main() -> None:
    st.set_page_config(page_title="General Extractor", page_icon=":clipboard:")

    # Initialize session state keys on the first run to avoid KeyError on access
    if "field_defs" not in st.session_state:
        st.session_state["field_defs"] = load_fields()
    if "extraction_result" not in st.session_state:
        st.session_state["extraction_result"] = None

    st.title(APP_TITLE)

    # Separate extraction and field configuration into distinct tabs for clarity
    extract_tab, fields_tab = st.tabs(["Extract", "Fields"])
    with extract_tab:
        render_extract_tab()
    with fields_tab:
        render_fields_tab()

if __name__ == "__main__":
    main()
