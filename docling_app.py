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

# Load environment variables from .env file
load_dotenv()
APP_TITLE = "Invoice Extractor (Docling + Ollama + LangGraph)"
DEFAULT_MODEL_NAME = "ministral-3:3b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]

# Define the two extraction pipelines available to the user
MODE_DOCLING = "docling"  # Convert document to text first, then send text to LLM
MODE_MULTIMODAL = "multimodal"  # Send document image directly to a vision-capable LLM
MODE_OPTIONS = {
    "Docling": MODE_DOCLING,
    "Multimodal LLM": MODE_MULTIMODAL,
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


def extract_text_with_docling(file_name: str, file_type: str, file_bytes: bytes) -> str:
    """Convert a file to markdown text using the Docling library."""


def to_base64_images(file_type: str, file_bytes: bytes) -> list[tuple[str, str]]:
    """Encode file bytes as one or more base64 images for the multimodal LLM."""


def build_multimodal_content(file_type: str, file_bytes: bytes) -> list[dict[str, Any]]:
    """Build the multimodal message content using all pages for PDF inputs."""


def parse_json_content(content: Any) -> tuple[str, dict[str, Any]]:
    """Parse the LLM response content into a raw string and a JSON dict."""


########################### LangGraph Nodes ################################


def load_document_node(state: InvoiceState) -> InvoiceState:
    """Convert the uploaded file to text via Docling, or skip for multimodal mode."""


def extract_invoice_node(state: InvoiceState) -> InvoiceState:
    """Send the document to the LLM and parse structured invoice fields."""


######################### Graph Construction ###############################


@st.cache_resource
def build_invoice_graph():
    """Build and compile the two-node LangGraph extraction workflow."""


def run_extraction(
    file_name: str,
    file_type: str,
    file_bytes: bytes,
    model_name: str,
    extraction_mode: str,
) -> InvoiceState:
    """Execute the full extraction graph with the provided inputs."""


########################### UI Rendering ###################################


def render_file_preview(file_type: str, file_name: str, file_bytes: bytes) -> None:
    """Display a preview of the uploaded file in the Streamlit app."""


def render_results(result: InvoiceState, extraction_mode: str) -> None:
    """Display the extraction results and optional debug expandables."""


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

    # If no file is uploaded, skip the rest of the app
    if not uploaded_file:
        return

    file_bytes = uploaded_file.getvalue()
    st.write(f"File: `{uploaded_file.name}`")
    render_file_preview(uploaded_file.type, uploaded_file.name, file_bytes)

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