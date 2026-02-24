# %%
"""General document extractor with Docling or multimodal-only LLM modes."""

from __future__ import annotations
import base64
import json
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict
import streamlit as st
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pdf2image import convert_from_bytes

load_dotenv()

# %%
########################### App Configuration & Constants ###########################

APP_TITLE = "General Document Extractor"
DEFAULT_MODEL_NAME = "ministral-3:3b"
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)  # Fall back to local Ollama instance
SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
SCHEMA_PATH = Path(__file__).with_name(
    "field_schema.json"
)  # Schema file lives alongside this script

# Define string constants for extraction mode identifiers
MODE_DOCLING = "docling"
MODE_MM = "multimodal"
MODE_OPTIONS = {"Docling + Text LLM": MODE_DOCLING, "Multimodal LLM only": MODE_MM}

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


# Define the LangGraph state schema; all keys are optional to allow partial updates between nodes
class ExtractionState(TypedDict, total=False):
    file_name: str
    file_type: str
    file_bytes: bytes
    model_name: str
    field_defs: list[dict[str, str]]
    mode: str
    document_text: str  # Populated by Docling in text-based mode
    raw_response: str  # Raw LLM output before JSON parsing
    extracted_fields: dict[str, Any]


########################### Field Normalization Utilities ###########################


def normalize_name(name: str) -> str:
    # Strip non-alphanumeric characters and collapse repeated underscores for consistent key formatting
    return re.sub(
        r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    ).strip("_")


def normalize_fields(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    # Use an ordered dict pattern to deduplicate fields while preserving the last-seen definition
    deduped: dict[str, str] = {}
    order: list[str] = []
    for row in rows:
        key = normalize_name(row.get("name", ""))
        if not key:
            continue
        # Move duplicate keys to the end so the latest entry wins
        if key in deduped:
            order.remove(key)
        order.append(key)
        deduped[key] = str(row.get("description", "")).strip()
    return [{"name": key, "description": deduped[key]} for key in order]


########################### Schema Persistence ###########################


def load_fields() -> list[dict[str, str]]:
    # Prefer the saved schema file; fall back to defaults if file is missing or empty
    if SCHEMA_PATH.exists():
        loaded = normalize_fields(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
        return loaded or normalize_fields(DEFAULT_FIELDS)
    return normalize_fields(DEFAULT_FIELDS)


def save_fields(fields: list[dict[str, Any]]) -> None:
    # Normalize before saving to ensure the file always contains clean, consistent keys
    SCHEMA_PATH.write_text(
        json.dumps(normalize_fields(fields), indent=2), encoding="utf-8"
    )


########################### Document Processing ###########################


# Cache the converter across Streamlit reruns to avoid expensive re-initialization
@st.cache_resource
def get_docling_converter() -> Any:
    return DocumentConverter()


def extract_text_with_docling(file_name: str, file_type: str, file_bytes: bytes) -> str:
    # Map MIME types to file extensions so Docling can infer the correct parser
    suffixes = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }
    suffix = suffixes.get(
        file_type, Path(file_name).suffix or ".bin"
    )  # Fall back to original extension

    # Write bytes to a temp file because Docling requires a file-system path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    result = get_docling_converter().convert(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)  # Clean up temp file after conversion
    return result.document.export_to_markdown().strip()


def to_base64_image(file_type: str, file_bytes: bytes) -> tuple[str, str]:
    # Rasterize the first PDF page to PNG so a vision model can consume it
    if file_type == "application/pdf":
        image = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=200)[
            0
        ]  # 200 dpi balances quality and token size
        buf = BytesIO()
        image.save(buf, format="PNG")
        return "image/png", base64.b64encode(buf.getvalue()).decode("utf-8")
    return file_type, base64.b64encode(file_bytes).decode("utf-8")


########################### LLM Prompt Construction ###########################


def build_system_prompt(field_defs: list[dict[str, str]]) -> str:
    # Instruct the model to return strict JSON with only the requested fields
    lines = [
        "You extract structured data from documents.",
        "Return valid JSON only.",
        "Extract exactly these keys and no others.",
        "Use null when a field is missing.",
        "Fields:",
    ]
    lines.extend(
        f"- {field['name']}: {field['description'] or 'No description provided.'}"
        for field in field_defs
    )
    return "\n".join(lines)


########################### Response Parsing ###########################


def parse_json_content(content: Any) -> tuple[str, dict[str, Any]]:
    # Handle dict responses directly without re-serializing
    if isinstance(content, dict):
        return json.dumps(content), content

    # Concatenate multi-part message content into a single string
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(str(item))
        raw = "\n".join(parts).strip()
    else:
        raw = str(content).strip()

    # Strip markdown code fences that some models wrap around JSON output
    cleaned = (
        raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    )
    return raw, json.loads(cleaned)


########################### LangGraph Node Definitions ###########################


def load_document_node(state: ExtractionState) -> ExtractionState:
    # Convert the uploaded file to markdown text for the downstream text-based LLM node
    return {
        "document_text": extract_text_with_docling(
            state["file_name"], state["file_type"], state["file_bytes"]
        )
    }


def extract_docling_node(state: ExtractionState) -> ExtractionState:
    # Use temperature=0 for deterministic, structured JSON extraction
    llm = ChatOllama(
        model=state["model_name"],
        temperature=0,
        format="json",
        base_url=OLLAMA_BASE_URL,
    )
    prompt = (
        f"Extract information from this document content.\n\n{state['document_text']}"
    )
    response = llm.invoke(
        [
            SystemMessage(content=build_system_prompt(state["field_defs"])),
            HumanMessage(content=prompt),
        ]
    )
    raw_response, extracted_fields = parse_json_content(response.content)
    return {"raw_response": raw_response, "extracted_fields": extracted_fields}


def extract_multimodal_node(state: ExtractionState) -> ExtractionState:
    # Use temperature=0 for deterministic, structured JSON extraction
    llm = ChatOllama(
        model=state["model_name"],
        temperature=0,
        format="json",
        base_url=OLLAMA_BASE_URL,
    )
    mime, image_b64 = to_base64_image(state["file_type"], state["file_bytes"])

    # Send the image as a data URL alongside the text instruction
    response = llm.invoke(
        [
            SystemMessage(content=build_system_prompt(state["field_defs"])),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Extract information from this document image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ]
            ),
        ]
    )
    raw_response, extracted_fields = parse_json_content(response.content)
    # Return empty document_text since no Docling step was performed
    return {
        "raw_response": raw_response,
        "extracted_fields": extracted_fields,
        "document_text": "",
    }


########################### Extraction Workflow (LangGraph) ###########################


def run_extraction(
    file_name: str, file_type: str, file_bytes: bytes, model_name: str, mode: str
) -> ExtractionState:
    graph = StateGraph(ExtractionState)

    # Build a single-node graph for multimodal mode or a two-node pipeline for Docling mode
    if mode == MODE_MM:
        graph.add_node("extract_multimodal", extract_multimodal_node)
        graph.set_entry_point("extract_multimodal")
        graph.add_edge("extract_multimodal", END)
    else:
        graph.add_node("load_document", load_document_node)
        graph.add_node("extract_docling", extract_docling_node)
        graph.set_entry_point("load_document")
        graph.add_edge("load_document", "extract_docling")
        graph.add_edge("extract_docling", END)

    return graph.compile().invoke(
        {
            "file_name": file_name,
            "file_type": file_type,
            "file_bytes": file_bytes,
            "model_name": model_name,
            "field_defs": st.session_state[
                "field_defs"
            ],  # Pull live field definitions from session state
            "mode": mode,
        }
    )


########################### UI Rendering Helpers ###########################


def render_file_preview(file_type: str, file_name: str, file_bytes: bytes) -> None:
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
    st.subheader("Extracted JSON")
    st.json(result.get("extracted_fields", {}))

    with st.expander("Raw model output"):
        st.code(result.get("raw_response", ""), language="json")

    # Only show the Docling text expander when text-based extraction was used
    if result.get("mode") == MODE_DOCLING:
        with st.expander("Docling extracted text"):
            st.text(result.get("document_text", ""))


########################### Streamlit Tab Renderers ###########################


def render_extract_tab() -> None:
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL_NAME)
    mode_label = st.radio("Extraction method", list(MODE_OPTIONS), horizontal=True)
    mode = MODE_OPTIONS[mode_label]

    uploaded_file = st.file_uploader(
        "Upload file", type=SUPPORTED_UPLOAD_TYPES, key="upload_file"
    )

    # Exit early when no file has been uploaded yet to avoid downstream errors
    if not uploaded_file:
        return

    file_bytes = uploaded_file.getvalue()
    st.write(f"File: `{uploaded_file.name}`")
    render_file_preview(uploaded_file.type, uploaded_file.name, file_bytes)

    # Warn the user that only the first page will be analysed in multimodal PDF mode
    if mode == MODE_MM and uploaded_file.type == "application/pdf":
        st.caption("Multimodal mode uses the first PDF page as an image input.")

    # Trigger extraction and persist the result so it survives reruns
    if st.button("Extract Information", type="primary"):
        with st.spinner("Running extraction workflow..."):
            st.session_state["extraction_result"] = run_extraction(
                uploaded_file.name, uploaded_file.type, file_bytes, model_name, mode
            )

    if st.session_state.get("extraction_result") is not None:
        render_results(st.session_state["extraction_result"])


def render_fields_tab() -> None:
    st.write("Add or remove field names and descriptions used for extraction.")

    edited = st.data_editor(
        st.session_state["field_defs"], num_rows="dynamic", width="stretch"
    )

    # Handle both pandas DataFrames and plain list-of-dicts returned by the editor
    rows = (
        edited.to_dict("records")
        if hasattr(edited, "to_dict")
        else [dict(row) for row in edited]
    )
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
    st.set_page_config(page_title="General Extractor", page_icon=":clipboard:")

    # Initialize session state keys on first run to prevent KeyErrors during rendering
    if "field_defs" not in st.session_state:
        st.session_state["field_defs"] = load_fields()
    if "extraction_result" not in st.session_state:
        st.session_state["extraction_result"] = None

    st.title(APP_TITLE)

    extract_tab, fields_tab = st.tabs(["Extract", "Fields"])
    with extract_tab:
        render_extract_tab()
    with fields_tab:
        render_fields_tab()


if __name__ == "__main__":
    main()
