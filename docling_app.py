"""Invoice Extractor app using Docling + LangChain + Ollama + LangGraph."""

from __future__ import annotations

import base64
import json
import os
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
APP_TITLE = "Invoice Extractor (Docling + Ollama + LangGraph)"
DEFAULT_MODEL_NAME = "ministral-3:3b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
MODE_DOCLING = "docling"
MODE_MULTIMODAL = "multimodal"
MODE_OPTIONS = {
    "Docling + Text LLM": MODE_DOCLING,
    "Multimodal LLM only": MODE_MULTIMODAL,
}
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


class InvoiceState(TypedDict, total=False):
    file_name: str
    file_type: str
    file_bytes: bytes
    model_name: str
    extraction_mode: str
    invoice_text: str
    raw_response: str
    extracted_fields: dict[str, Any]


@st.cache_resource
def get_docling_converter() -> Any:
    return DocumentConverter()


def extract_text_with_docling(file_name: str, file_type: str, file_bytes: bytes) -> str:
    suffixes = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }
    suffix = suffixes.get(file_type, Path(file_name).suffix or ".bin")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    result = get_docling_converter().convert(tmp_path)
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
    return result.document.export_to_markdown().strip()


def to_base64_image(file_type: str, file_bytes: bytes) -> tuple[str, str]:
    if file_type == "application/pdf":
        image = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=200)[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return "image/png", base64.b64encode(buffer.getvalue()).decode("utf-8")
    return file_type, base64.b64encode(file_bytes).decode("utf-8")


def parse_json_content(content: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(content, dict):
        return json.dumps(content), content
    if isinstance(content, list):
        parts: list[str] = []
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
    cleaned = (
        raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    )
    return raw, json.loads(cleaned)


def load_document_node(state: InvoiceState) -> InvoiceState:
    if state.get("extraction_mode") == MODE_MULTIMODAL:
        return {"invoice_text": ""}
    return {
        "invoice_text": extract_text_with_docling(
            file_name=state["file_name"],
            file_type=state["file_type"],
            file_bytes=state["file_bytes"],
        )
    }


def extract_invoice_node(state: InvoiceState) -> InvoiceState:
    llm = ChatOllama(
        model=state["model_name"],
        temperature=0,
        format="json",
        base_url=OLLAMA_BASE_URL,
    )
    if state.get("extraction_mode") == MODE_MULTIMODAL:
        mime, image_b64 = to_base64_image(state["file_type"], state["file_bytes"])
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Extract all invoice information from this document image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                        },
                    ]
                ),
            ]
        )
    else:
        prompt = f"Extract all invoice information from this document content.\n\n{state['invoice_text']}"
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
    raw_response, extracted_fields = parse_json_content(response.content)
    return {"raw_response": raw_response, "extracted_fields": extracted_fields}


@st.cache_resource
def build_invoice_graph():
    graph = StateGraph(InvoiceState)
    graph.add_node("load_document", load_document_node)
    graph.add_node("extract_invoice", extract_invoice_node)
    graph.set_entry_point("load_document")
    graph.add_edge("load_document", "extract_invoice")
    graph.add_edge("extract_invoice", END)
    return graph.compile()


def run_extraction(
    file_name: str,
    file_type: str,
    file_bytes: bytes,
    model_name: str,
    extraction_mode: str,
) -> InvoiceState:
    return build_invoice_graph().invoke(
        {
            "file_name": file_name,
            "file_type": file_type,
            "file_bytes": file_bytes,
            "model_name": model_name,
            "extraction_mode": extraction_mode,
        }
    )


def render_file_preview(file_type: str, file_name: str, file_bytes: bytes) -> None:
    if file_type.startswith("image/"):
        st.image(file_bytes, caption=file_name)
        return
    if file_type == "application/pdf":
        pdf_b64 = base64.b64encode(file_bytes).decode("utf-8")
        st.markdown(
            f'<embed src="data:application/pdf;base64,{pdf_b64}" width="100%" height="700" type="application/pdf">',
            unsafe_allow_html=True,
        )


def render_results(result: InvoiceState, extraction_mode: str) -> None:
    st.subheader("Extracted Invoice JSON")
    st.json(result.get("extracted_fields", {}))
    with st.expander("Raw model output"):
        st.code(result.get("raw_response", ""), language="json")
    if extraction_mode == MODE_DOCLING:
        with st.expander("Docling extracted text"):
            st.text(result.get("invoice_text", ""))


def main() -> None:
    st.set_page_config(page_title="Invoice Extractor", page_icon=":receipt:")
    st.title(APP_TITLE)
    st.write("Upload an invoice PDF or image and extract structured fields.")
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL_NAME)
    extraction_mode_label = st.radio(
        "Extraction method", list(MODE_OPTIONS), horizontal=True
    )
    extraction_mode = MODE_OPTIONS[extraction_mode_label]
    uploaded_file = st.file_uploader("Upload invoice", type=SUPPORTED_UPLOAD_TYPES)
    if not uploaded_file:
        return
    file_bytes = uploaded_file.getvalue()
    st.write(f"File: `{uploaded_file.name}`")
    render_file_preview(uploaded_file.type, uploaded_file.name, file_bytes)
    if extraction_mode == MODE_MULTIMODAL and uploaded_file.type == "application/pdf":
        st.caption("Multimodal mode uses the first PDF page as image input.")
    if st.button("Extract Information", type="primary"):
        with st.spinner("Running extraction workflow..."):
            st.session_state["extraction_result"] = run_extraction(
                file_name=uploaded_file.name,
                file_type=uploaded_file.type,
                file_bytes=file_bytes,
                model_name=model_name,
                extraction_mode=extraction_mode,
            )
    if st.session_state.get("extraction_result") is not None:
        render_results(st.session_state["extraction_result"], extraction_mode)


if __name__ == "__main__":
    main()
