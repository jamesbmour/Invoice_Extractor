"""
Teaching-friendly invoice extractor app.

This module is intentionally split into layers:
1. Configuration/constants
2. Pure helper functions
3. LangGraph workflow nodes
4. Streamlit UI
"""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from typing import Any, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pypdf import PdfReader

# Teaching note:
# Read this file top-to-bottom when presenting:
# configuration -> helpers -> graph nodes -> UI entry point.

# Load values from .env (for example: OLLAMA_BASE_URL).
load_dotenv()


# ================================ Configuration =============================== #
APP_TITLE = "Invoice Extractor (LangChain + Ollama + LangGraph)"
DEFAULT_MODEL_NAME = "ministral-3:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg"]


SYSTEM_PROMPT = """You extract structured invoice data.
Return valid JSON only.

Include these keys:
- vendor_name
- vendor_address
- vendor_email
- vendor_phone
- invoice_number
- invoice_date
- due_date
- currency
- subtotal
- tax
- total
- purchase_order_number
- bill_to
- ship_to
- payment_terms
- line_items (array of objects with: description, quantity, unit_price, amount)

Use null when a field is missing."""


# ================================ Workflow State ============================= #
class InvoiceState(TypedDict, total=False):
    """State object shared across LangGraph nodes."""

    # Inputs from the UI upload
    file_name: str
    file_type: str
    file_bytes: bytes
    model_name: str

    # Intermediate + final values produced by graph nodes
    invoice_text: str
    raw_response: str
    extracted_fields: dict[str, Any]


# ================================ Pure Helpers =============================== #
def is_pdf(file_type: str) -> bool:
    """Return True when the upload MIME type is a PDF."""
    return file_type == "application/pdf"


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from each PDF page and combine it into one string."""
    reader = PdfReader(BytesIO(file_bytes))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(page_texts)


def normalize_message_content(content: Any) -> str:
    """Convert LangChain message content into a plain string."""
    # Depending on model/provider settings, content may be string/dict/list.
    # Normalizing early keeps the parser below simple and deterministic.
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content)

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item))
        return "\n".join(parts)

    return str(content)


def remove_markdown_fences(text: str) -> str:
    """Strip markdown code fences around model output, if present."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


def parse_first_json_block(text: str) -> dict[str, Any]:
    """
    Parse the first JSON object/array found in a string.

    Some models prepend or append extra text; this parser is tolerant of that.
    """
    cleaned = remove_markdown_fences(text)
    decoder = json.JSONDecoder()

    # Scan for the first "{" or "[" and try decoding from there.
    # This helps when the model includes extra leading text.
    for index, char in enumerate(cleaned):
        if char not in "{[":
            continue

        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"line_items": parsed}

    return {}


def build_pdf_extraction_prompt(invoice_text: str) -> str:
    """Create the prompt used when a PDF has already been converted to text."""
    return f"Extract all invoice information from this invoice text.\n\n{invoice_text}"


def build_image_message(file_bytes: bytes, file_type: str) -> list[dict[str, str]]:
    """Build the multimodal message content for an image-based extraction call."""
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return [
        {"type": "text", "text": "Extract all invoice information from this image."},
        {"type": "image_url", "image_url": f"data:{file_type};base64,{encoded}"},
    ]


# ================================ Graph Nodes ================================ #
def load_document_node(state: InvoiceState) -> InvoiceState:
    """
    Node 1: Preprocess input document.

    For PDFs we extract text.
    For images we skip this step (vision model reads pixels directly).
    """
    # Graph nodes should be easy to explain as pure transforms:
    # input state -> output delta.
    if is_pdf(state["file_type"]):
        return {"invoice_text": extract_text_from_pdf(state["file_bytes"])}
    return {"invoice_text": ""}


def extract_invoice_node(state: InvoiceState) -> InvoiceState:
    """Node 2: Send document content to Ollama and parse JSON response."""
    # temperature=0 for repeatable demos while teaching.
    llm = ChatOllama(
        model=state["model_name"],
        temperature=0,
        format="json",
        base_url=OLLAMA_BASE_URL,
    )

    if is_pdf(state["file_type"]):
        # PDF path: send extracted text.
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=build_pdf_extraction_prompt(state["invoice_text"])),
            ]
        )
    else:
        # Image path: send a multimodal message with base64 image data.
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=build_image_message(
                        file_bytes=state["file_bytes"],
                        file_type=state["file_type"],
                    )
                ),
            ]
        )

    # With format="json", Ollama should return JSON directly (string or dict).
    if isinstance(response.content, dict):
        extracted_fields = response.content
        raw_response = json.dumps(response.content)
    else:
        raw_response = str(response.content).strip()
        try:
            extracted_fields = json.loads(raw_response)
        except json.JSONDecodeError:
            # Fallback for occasional provider/model deviations.
            raw_response = normalize_message_content(response.content)
            extracted_fields = parse_first_json_block(raw_response)

    return {"raw_response": raw_response, "extracted_fields": extracted_fields}


# ================================ Graph Builder ============================== #
@st.cache_resource
def build_invoice_graph():
    """Create and cache the LangGraph workflow."""
    graph = StateGraph(InvoiceState)
    graph.add_node("load_document", load_document_node)
    graph.add_node("extract_invoice", extract_invoice_node)

    # Linear graph keeps the initial teaching example simple:
    # load_document -> extract_invoice -> END
    graph.set_entry_point("load_document")
    graph.add_edge("load_document", "extract_invoice")
    graph.add_edge("extract_invoice", END)
    return graph.compile()


def run_extraction(file_name: str, file_type: str, file_bytes: bytes, model_name: str) -> InvoiceState:
    """Run the full extraction workflow and return final graph state."""
    # Keep workflow execution in one function so the UI layer stays focused on UX.
    graph = build_invoice_graph()
    return graph.invoke(
        {
            "file_name": file_name,
            "file_type": file_type,
            "file_bytes": file_bytes,
            "model_name": model_name,
        }
    )


# ================================ UI Helpers ================================= #
def render_file_preview(file_type: str, file_name: str, file_bytes: bytes) -> None:
    """Render a preview for image or PDF uploads."""
    if file_type.startswith("image/"):
        st.image(file_bytes, caption=file_name)
        return

    if is_pdf(file_type):
        pdf_b64 = base64.b64encode(file_bytes).decode("utf-8")
        pdf_preview = (
            f'<embed src="data:application/pdf;base64,{pdf_b64}" '
            'width="100%" height="700" type="application/pdf">'
        )
        st.markdown(pdf_preview, unsafe_allow_html=True)


def render_results(result: InvoiceState, file_type: str) -> None:
    """Render structured output plus optional debug sections."""
    # Show parsed JSON first (the main output), then diagnostics.
    st.subheader("Extracted Invoice JSON")
    st.json(result.get("extracted_fields", {}))

    with st.expander("Raw model output"):
        st.code(result.get("raw_response", ""), language="json")

    if is_pdf(file_type):
        with st.expander("Extracted PDF text"):
            st.text(result.get("invoice_text", ""))




# =================================== App ==================================== #
def main() -> None:
    # Step 1: Configure page and explain what the app does.
    st.set_page_config(page_title="Invoice Extractor", page_icon=":receipt:")
    st.title(APP_TITLE)
    st.write("Upload an invoice PDF or image and extract structured fields.")

    # Step 2: Collect user inputs.
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL_NAME)
    uploaded_file = st.file_uploader("Upload invoice", type=SUPPORTED_UPLOAD_TYPES)

    if not uploaded_file:
        return

    # Step 3: Preview uploaded content before processing.
    file_bytes = uploaded_file.getvalue()
    st.write(f"File: `{uploaded_file.name}`")
    render_file_preview(uploaded_file.type, uploaded_file.name, file_bytes)

    # Step 4: Reset previous results when the file changes.
    file_signature = (uploaded_file.name, uploaded_file.type, len(file_bytes))
    if st.session_state.get("active_file_signature") != file_signature:
        st.session_state["active_file_signature"] = file_signature
        st.session_state["extraction_result"] = None

    # Step 5: Run the LangGraph pipeline on button click.
    if st.button("Extract Information", type="primary"):
        with st.spinner("Running extraction workflow..."):
            st.session_state["extraction_result"] = run_extraction(
                file_name=uploaded_file.name,
                file_type=uploaded_file.type,
                file_bytes=file_bytes,
                model_name=model_name,
            )

    # Step 6: Render latest result if available.
    if st.session_state.get("extraction_result") is not None:
        render_results(st.session_state["extraction_result"], uploaded_file.type)


if __name__ == "__main__":
    main()
