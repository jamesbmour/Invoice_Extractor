"""
Invoice Extractor App using LangChain, Ollama, and LangGraph

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


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from each PDF page and combine it into one string."""


def build_pdf_extraction_prompt(invoice_text: str) -> str:
    """Create the prompt used when a PDF has already been converted to text."""


def build_image_message(file_bytes: bytes, file_type: str) -> list[dict[str, str]]:
    """Build the multimodal message content for an image-based extraction call."""


# ================================ Graph Nodes ================================ #
def load_document_node(state: InvoiceState) -> InvoiceState:
    """
    Node 1: Preprocess input document.

    For PDFs we extract text.
    For images we skip this step (vision model reads pixels directly).
    """


def extract_invoice_node(state: InvoiceState) -> InvoiceState:
    """Node 2: Send document content to Ollama and parse JSON response."""


# ================================ Graph Builder ============================== #
@st.cache_resource
def build_invoice_graph():
    """Create and cache the LangGraph workflow."""


def run_extraction(file_name: str, file_type: str, file_bytes: bytes, model_name: str) -> InvoiceState:
    """Run the full extraction workflow and return final graph state."""


# ================================ UI Helpers ================================= #
def render_file_preview(file_type: str, file_name: str, file_bytes: bytes) -> None:
    """Render a preview for image or PDF uploads."""


def render_results(result: InvoiceState, file_type: str) -> None:
    """Render structured output plus optional debug sections."""




# =================================== App ==================================== #
def main() -> None:
    # Step 1: Configure page and explain what the app does.
    st.set_page_config(page_title="Invoice Extractor", page_icon=":receipt:")
    st.title(APP_TITLE)
    st.write("Upload an invoice PDF or image and extract structured fields.")

    # Step 2: Collect user inputs.
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL_NAME)
    uploaded_file = st.file_uploader("Upload invoice", type=SUPPORTED_UPLOAD_TYPES)
    
    # Don't proceed until a file is uploaded.
    if not uploaded_file:
        return

    # Step 3: Preview uploaded content before processing.
    file_bytes = uploaded_file.getvalue()
    st.write(f"File: `{uploaded_file.name}`")
    render_file_preview(uploaded_file.type, uploaded_file.name, file_bytes)

    # Step 4: Reset previous results when the file changes.
    file_signature = (uploaded_file.name, uploaded_file.type, len(file_bytes))
    # We use a simple signature of the file (name, type, size) to detect changes. In a production app, you might want a more robust method (like hashing).
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