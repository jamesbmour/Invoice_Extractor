"""General document extractor with Docling or multimodal-only LLM modes."""
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
from rich import print
load_dotenv()

########################### App Configuration & Constants ###########################

APP_TITLE = "Document Extractor"
# qwen3.5:2b, qwen3.5:4b, qwen3.5:9b, ministral-3:3b, qwen3.5:0.8b
DEFAULT_MODEL_LIST = ['ministral-3:3b','qwen3.5:0.8b', 'qwen3.5:2b', 'qwen3.5:4b', 'qwen3.5:9b']
DEFAULT_MODEL_NAME = "qwen3.5:2b"
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


def normalize_fields(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Deduplicate and clean a list of field definitions, keeping the last occurrence of each key."""


########################### Schema Persistence ###########################


def load_fields() -> list[dict[str, str]]:
    """Load field definitions from the JSON schema file, falling back to built-in defaults."""


def save_fields(fields: list[dict[str, Any]]) -> None:
    """Persist normalized field definitions to the JSON schema file."""


########################### Document Processing ###########################


# Cache the converter across Streamlit reruns to avoid expensive re-initialization
@st.cache_resource
def get_docling_converter() -> Any:
    """Return a singleton Docling DocumentConverter instance. cache_resource ensures single instance per session."""


def extract_text_with_docling(file_name: str, file_type: str, file_bytes: bytes) -> str:
    """Convert a document to markdown text using the Docling library."""  # Clean up temp file after conversion


def to_base64_images(file_type: str, file_bytes: bytes) -> list[tuple[str, str]]:
    """Convert a file into a list of (MIME type, base64 string) tuples, one per page/image."""


def build_multimodal_content(file_type: str, file_bytes: bytes) -> list[dict[str, Any]]:
    """Construct a multimodal message payload with text prompt and inline base64 images."""


########################### LLM Prompt Construction ###########################


def build_system_prompt(field_defs: list[dict[str, str]]) -> str:
    """Generate a system prompt that instructs the LLM to extract specific fields as JSON."""

########################### Response Parsing ###########################


def parse_json_content(content: Any) -> tuple[str, dict[str, Any]]:
    """Parse LLM output into a (raw_string, parsed_dict) tuple, handling various response formats. list is used for multi-part messages from multimodal inputs."""


########################### Extraction Workflow ###########################


def make_llm(model_name: str) -> ChatOllama:
    """Create an Ollama chat client configured for deterministic JSON output."""


def extract_with_prompt(
    model_name: str,
    field_defs: list[dict[str, str]],
    human_content: str | list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Send field definitions and document content to the LLM, returning parsed JSON."""


def run_extraction(file_name: str, file_type: str, file_bytes: bytes, model_name: str, mode: str) -> ExtractionState:
    """Orchestrate the full extraction pipeline based on the selected mode."""


########################### UI Rendering Helpers ###########################


def render_file_preview(file_type: str, file_name: str, file_bytes: bytes) -> None:
    """Display an uploaded file preview — native image or embedded PDF viewer."""


def render_results(result: ExtractionState) -> None:
    """Display extraction results with JSON preview, CSV download, and debug expanders."""


########################### Streamlit Tab Renderers ###########################


def render_extract_tab() -> None:
    """Render the main extraction tab with sidebar controls and result display."""


def render_fields_tab() -> None:
    """Render the field schema editor tab for customizing extraction fields."""  # Refresh the UI to reflect the reset values


########################### App Entry Point ###########################


def main() -> None:
    """Initialize the Streamlit app and render the tabbed interface."""


if __name__ == "__main__":
    main()