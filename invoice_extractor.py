"""
Invoice Extractor app built with Streamlit + LangGraph + Ollama.

This file is organized for teaching:
1) Configuration and constants
2) Pure helper functions (easy to unit-test)
3) LangGraph nodes (business workflow)
4) Streamlit UI (presentation layer)
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pdf2image import convert_from_bytes
from PIL import Image

# Load local environment variables (for OLLAMA_BASE_URL, etc.).
load_dotenv()


# ----------------------------- Configuration ---------------------------------
APP_TITLE = "Invoice Extractor"
DEFAULT_MODEL_NAME = "llava"
DEFAULT_OLLAMA_BASE_URL = "http://eos.local:11434"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
MISSING_VALUE = "N/A"

INVOICE_FIELDS = [
    "INVOICE_NUMBER",
    "DATE",
    "DUE_DATE",
    "VENDOR_NAME",
    "VENDOR_ADDRESS",
    "CUSTOMER_NAME",
    "CUSTOMER_ADDRESS",
    "SUBTOTAL",
    "TAX",
    "TOTAL",
    "CURRENCY",
    "PAYMENT_TERMS",
    "LINE_ITEMS",
    "NOTES",
]
CRITICAL_FIELDS = ["INVOICE_NUMBER", "VENDOR_NAME", "TOTAL", "DATE"]

EXTRACTION_PROMPT = (
    "You are an expert invoice data extractor. Analyze this invoice image and "
    "extract ALL information. Return data in this EXACT format "
    "(use 'N/A' if not found):\n\n"
    "INVOICE_NUMBER: ...\nDATE: ...\nDUE_DATE: ...\nVENDOR_NAME: ...\n"
    "VENDOR_ADDRESS: ...\nCUSTOMER_NAME: ...\nCUSTOMER_ADDRESS: ...\n"
    "SUBTOTAL: ...\nTAX: ...\nTOTAL: ...\nCURRENCY: ...\n"
    "PAYMENT_TERMS: ...\nLINE_ITEMS: item1 (qty x price) | item2 (qty x price) | ...\n"
    "NOTES: ..."
)

CUSTOM_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.result-card {
    background: linear-gradient(135deg, #1a1f2e, #151922);
    border: 1px solid #2a3040;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.result-card h3 {
    color: #60a5fa;
    margin-bottom: .5rem;
    font-size: .85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.result-card p { color: #e2e8f0; font-size: 1.1rem; margin: 0; }
.status-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    padding: .25rem .75rem;
    border-radius: 20px;
    font-size: .8rem;
    font-weight: 500;
}
</style>
"""


# ----------------------------- Workflow State --------------------------------
class InvoiceState(TypedDict):
    image_b64: str
    model_name: str
    raw_text: str
    structured_data: dict[str, str]


# ---------------------------- Pure Helper Logic -------------------------------
def file_bytes_to_base64_png(file_bytes: bytes, mime_type: str) -> str:
    """Convert a PDF (first page) or image upload into base64-encoded PNG."""
    if mime_type == "application/pdf":
        pages = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=200)
        image = pages[0]
    else:
        image = Image.open(io.BytesIO(file_bytes))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_invoice_fields(raw_text: str) -> dict[str, str]:
    """
    Parse key-value lines from model output.

    Expected line format:
    FIELD_NAME: value
    """
    fields: dict[str, str] = {}

    for line in raw_text.strip().splitlines():
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = raw_key.strip().upper()
        if key in INVOICE_FIELDS:
            value = raw_value.strip() or MISSING_VALUE
            fields[key] = value

    for required_key in INVOICE_FIELDS:
        fields.setdefault(required_key, MISSING_VALUE)

    return fields


def add_validation_message(fields: dict[str, str]) -> dict[str, str]:
    """Attach a simple validation status based on critical fields."""
    missing = [key for key in CRITICAL_FIELDS if fields.get(key, MISSING_VALUE) == MISSING_VALUE]
    validation_msg = (
        "All critical fields found"
        if not missing
        else f"Missing critical fields: {', '.join(missing)}"
    )
    validated = dict(fields)
    validated["_VALIDATION"] = validation_msg
    return validated


def normalize_model_output(content: Any) -> str:
    """Normalize LangChain response content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


# ---------------------------- LangGraph Nodes ---------------------------------
def extract_text_node(state: InvoiceState) -> dict[str, str]:
    """Step 1: Send invoice image to Ollama vision model."""
    llm = ChatOllama(
        model=state["model_name"],
        temperature=0,
        base_url=OLLAMA_BASE_URL,
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": EXTRACTION_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{state['image_b64']}"},
            },
        ]
    )

    response = llm.invoke([message])
    return {"raw_text": normalize_model_output(response.content)}


def parse_fields_node(state: InvoiceState) -> dict[str, dict[str, str]]:
    """Step 2: Convert model text output into a structured dictionary."""
    return {"structured_data": parse_invoice_fields(state["raw_text"])}


def validate_output_node(state: InvoiceState) -> dict[str, dict[str, str]]:
    """Step 3: Add a lightweight validation message for teaching/demo use."""
    return {"structured_data": add_validation_message(state["structured_data"])}


@st.cache_resource
def build_graph():
    """Create and cache the LangGraph pipeline once per app session."""
    graph = StateGraph(InvoiceState)
    graph.add_node("extract_text", extract_text_node)
    graph.add_node("parse_fields", parse_fields_node)
    graph.add_node("validate_output", validate_output_node)

    graph.add_edge(START, "extract_text")
    graph.add_edge("extract_text", "parse_fields")
    graph.add_edge("parse_fields", "validate_output")
    graph.add_edge("validate_output", END)

    return graph.compile()


# ------------------------------- UI Helpers ----------------------------------
def render_styles() -> None:
    st.markdown(CUSTOM_STYLES, unsafe_allow_html=True)


def render_field_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <h3>{label.replace("_", " ")}</h3>
            <p>{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_field_pair(data: dict[str, str], left_key: str, right_key: str) -> None:
    left_col, right_col = st.columns(2)
    with left_col:
        render_field_card(left_key, data.get(left_key, MISSING_VALUE))
    with right_col:
        render_field_card(right_key, data.get(right_key, MISSING_VALUE))


def render_invoice_preview(file_bytes: bytes, mime_type: str) -> None:
    if mime_type == "application/pdf":
        pages = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=150)
        st.image(pages[0], caption="Invoice Preview (Page 1)")
        return
    st.image(file_bytes, caption="Invoice Preview")


def process_invoice(file_bytes: bytes, mime_type: str, model_name: str, status: Any) -> InvoiceState:
    """Run the end-to-end extraction flow and return final graph state."""
    status.write("Converting uploaded document to image...")
    image_b64 = file_bytes_to_base64_png(file_bytes, mime_type)

    status.write(f"Sending invoice to Ollama model: {model_name}")
    pipeline = build_graph()

    initial_state: InvoiceState = {
        "image_b64": image_b64,
        "model_name": model_name,
        "raw_text": "",
        "structured_data": {},
    }
    result: InvoiceState = pipeline.invoke(initial_state)
    status.update(label="Extraction complete", state="complete")
    return result


def render_results(result: InvoiceState) -> None:
    """Render extracted fields, validation, raw output, and JSON download."""
    data = result["structured_data"]
    validation = data.get("_VALIDATION", "")
    export_data = {k: v for k, v in data.items() if k != "_VALIDATION"}

    st.markdown("## Extracted Data")
    if validation:
        st.markdown(f'<span class="status-badge">{validation}</span>', unsafe_allow_html=True)

    key_metrics = ["INVOICE_NUMBER", "DATE", "TOTAL", "CURRENCY"]
    metric_cols = st.columns(len(key_metrics))
    for col, key in zip(metric_cols, key_metrics):
        with col:
            render_field_card(key, data.get(key, MISSING_VALUE))

    st.markdown("### Parties")
    render_field_pair(data, "VENDOR_NAME", "CUSTOMER_NAME")
    render_field_pair(data, "VENDOR_ADDRESS", "CUSTOMER_ADDRESS")

    st.markdown("### Financials")
    render_field_pair(data, "SUBTOTAL", "TAX")
    render_field_pair(data, "DUE_DATE", "PAYMENT_TERMS")

    st.markdown("### Line Items")
    render_field_card("LINE_ITEMS", data.get("LINE_ITEMS", MISSING_VALUE))
    render_field_card("NOTES", data.get("NOTES", MISSING_VALUE))

    with st.expander("Raw LLM Output"):
        st.code(result["raw_text"], language="text")

    st.download_button(
        "Download as JSON",
        data=json.dumps(export_data, indent=2),
        file_name="invoice_data.json",
        mime="application/json",
    )


# --------------------------------- App --------------------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":receipt:", layout="wide")
    render_styles()

    st.title(APP_TITLE)
    st.write(
        "Upload a PDF or image invoice to extract structured fields with a local Ollama model."
    )
    st.markdown("---")

    with st.sidebar:
        st.subheader("Configuration")
        model_name = st.text_input(
            "Ollama Model",
            value=DEFAULT_MODEL_NAME,
            help="Vision-capable model name (for example: llava, llava:13b, bakllava).",
        )

    upload_col, preview_col = st.columns([1, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Drop your invoice here",
            type=SUPPORTED_UPLOAD_TYPES,
            help="Supported types: PDF, PNG, JPG, JPEG, WEBP",
        )

    if not uploaded_file:
        return

    file_bytes = uploaded_file.getvalue()

    with preview_col:
        render_invoice_preview(file_bytes, uploaded_file.type)

    st.markdown("---")

    if st.button("Extract Invoice Data", type="primary", width="stretch"):
        with st.status("Processing invoice...", expanded=True) as status:
            st.session_state["result"] = process_invoice(
                file_bytes=file_bytes,
                mime_type=uploaded_file.type,
                model_name=model_name,
                status=status,
            )

    if "result" in st.session_state:
        render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
