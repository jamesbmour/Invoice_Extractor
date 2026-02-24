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

############################# Configuration ################################

load_dotenv()
APP_TITLE = "Invoice Extractor (Docling + Ollama + LangGraph)"
DEFAULT_MODEL_NAME = "ministral-3:3b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]

# Define the two extraction pipelines available to the user
MODE_DOCLING = "docling"  # Convert document to text first, then send text to LLM
MODE_MULTIMODAL = "multimodal"  # Send document image directly to a vision-capable LLM
MODE_OPTIONS = {
    "Docling + Text LLM": MODE_DOCLING,
    "Multimodal LLM only": MODE_MULTIMODAL,
}

# Instruct the LLM to return only valid JSON with a fixed schema
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


########################### State Definition ################################


# Define the typed dictionary that flows through each LangGraph node
class InvoiceState(TypedDict, total=False):
    file_name: str
    file_type: str
    file_bytes: bytes
    model_name: str
    extraction_mode: str
    invoice_text: str
    raw_response: str
    extracted_fields: dict[str, Any]


########################## Helper Functions ################################


@st.cache_resource
def get_docling_converter() -> Any:
    """Return a cached Docling DocumentConverter instance."""
    return DocumentConverter()


def extract_text_with_docling(file_name: str, file_type: str, file_bytes: bytes) -> str:
    """Convert a file to markdown text using the Docling library."""
    # Map MIME types to file extensions for the temp file
    suffixes = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }
    suffix = suffixes.get(file_type, Path(file_name).suffix or ".bin")

    # Write bytes to a temp file since Docling requires a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    result = get_docling_converter().convert(tmp_path)

    # Clean up the temporary file after conversion
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    return result.document.export_to_markdown().strip()


def to_base64_image(file_type: str, file_bytes: bytes) -> tuple[str, str]:
    """Encode file bytes as a base64 image string for the multimodal LLM."""
    # Render PDFs to a PNG of the first page since vision models need images
    if file_type == "application/pdf":
        image = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=200)[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return "image/png", base64.b64encode(buffer.getvalue()).decode("utf-8")

    return file_type, base64.b64encode(file_bytes).decode("utf-8")


def parse_json_content(content: Any) -> tuple[str, dict[str, Any]]:
    """Parse the LLM response content into a raw string and a JSON dict."""
    if isinstance(content, dict):
        return json.dumps(content), content

    # Handle list-type responses by concatenating text parts
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

    # Strip markdown code fences the LLM may wrap around the JSON
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return raw, json.loads(cleaned)


########################### LangGraph Nodes ################################


def load_document_node(state: InvoiceState) -> InvoiceState:
    """Convert the uploaded file to text via Docling, or skip for multimodal mode."""
    # Skip text extraction when using the multimodal pipeline
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
    """Send the document to the LLM and parse structured invoice fields."""
    # Initialize the Ollama chat model with JSON output format
    llm = ChatOllama(
        model=state["model_name"],
        temperature=0,  # Use deterministic output for consistent extraction
        format="json",
        base_url=OLLAMA_BASE_URL,
    )

    # Branch based on extraction mode: image-based or text-based
    if state.get("extraction_mode") == MODE_MULTIMODAL:
        mime, image_b64 = to_base64_image(state["file_type"], state["file_bytes"])
        # Send the document as an inline base64 image to the vision model
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
        # Send the Docling-extracted text to the LLM
        prompt = f"Extract all invoice information from this document content.\n\n{state['invoice_text']}"
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])

    raw_response, extracted_fields = parse_json_content(response.content)
    return {"raw_response": raw_response, "extracted_fields": extracted_fields}


######################### Graph Construction ###############################


@st.cache_resource
def build_invoice_graph():
    """Build and compile the two-node LangGraph extraction workflow."""
    graph = StateGraph(InvoiceState)
    graph.add_node("load_document", load_document_node)
    graph.add_node("extract_invoice", extract_invoice_node)
    graph.set_entry_point("load_document")
    graph.add_edge("load_document", "extract_invoice")  # Linear pipeline: load → extract → done
    graph.add_edge("extract_invoice", END)
    return graph.compile()


def run_extraction(
    file_name: str,
    file_type: str,
    file_bytes: bytes,
    model_name: str,
    extraction_mode: str,
) -> InvoiceState:
    """Execute the full extraction graph with the provided inputs."""
    return build_invoice_graph().invoke(
        {
            "file_name": file_name,
            "file_type": file_type,
            "file_bytes": file_bytes,
            "model_name": model_name,
            "extraction_mode": extraction_mode,
        }
    )


########################### UI Rendering ###################################


def render_file_preview(file_type: str, file_name: str, file_bytes: bytes) -> None:
    """Display a preview of the uploaded file in the Streamlit app."""
    if file_type.startswith("image/"):
        st.image(file_bytes, caption=file_name)
        return

    # Embed PDFs directly in the browser using a base64 data URI
    if file_type == "application/pdf":
        pdf_b64 = base64.b64encode(file_bytes).decode("utf-8")
        st.markdown(
            f'<embed src="data:application/pdf;base64,{pdf_b64}" width="100%" height="700" type="application/pdf">',
            unsafe_allow_html=True,
        )


def render_results(result: InvoiceState, extraction_mode: str) -> None:
    """Display the extraction results and optional debug expandables."""
    st.subheader("Extracted Invoice JSON")
    st.json(result.get("extracted_fields", {}))

    with st.expander("Raw model output"):
        st.code(result.get("raw_response", ""), language="json")

    # Show the intermediate Docling text only when that pipeline was used
    if extraction_mode == MODE_DOCLING:
        with st.expander("Docling extracted text"):
            st.text(result.get("invoice_text", ""))


############################# Main App #####################################


def main() -> None:
    """Entry point: configure the Streamlit page and orchestrate the UI."""
    st.set_page_config(page_title="Invoice Extractor", page_icon=":receipt:")
    st.title(APP_TITLE)
    st.write("Upload an invoice PDF or image and extract structured fields.")

    # Render sidebar controls for model selection and file upload
    st.sidebar.header("Settings")
    model_name = st.sidebar.text_input("Ollama model", value=DEFAULT_MODEL_NAME)
    extraction_mode_label = st.sidebar.radio("Extraction method", list(MODE_OPTIONS))
    extraction_mode = MODE_OPTIONS[extraction_mode_label]
    uploaded_file = st.sidebar.file_uploader("Upload invoice", type=SUPPORTED_UPLOAD_TYPES)

    if not uploaded_file:
        return

    file_bytes = uploaded_file.getvalue()
    st.write(f"File: `{uploaded_file.name}`")
    render_file_preview(uploaded_file.type, uploaded_file.name, file_bytes)

    if extraction_mode == MODE_MULTIMODAL and uploaded_file.type == "application/pdf":
        st.caption("Multimodal mode uses the first PDF page as image input.")

    # Trigger extraction and store results in session state for persistence
    if st.button("Extract Information", type="primary"):
        with st.spinner("Running extraction workflow..."):
            st.session_state["extraction_result"] = run_extraction(
                file_name=uploaded_file.name,
                file_type=uploaded_file.type,
                file_bytes=file_bytes,
                model_name=model_name,
                extraction_mode=extraction_mode,
            )

    # Re-render results if they already exist in session state
    if st.session_state.get("extraction_result") is not None:
        render_results(st.session_state["extraction_result"], extraction_mode)


if __name__ == "__main__":
    main()
