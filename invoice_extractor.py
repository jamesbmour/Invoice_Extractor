"""
Invoice Extractor — Streamlit + LangChain + Ollama + LangGraph
Upload a PDF or image invoice and extract structured data via a local LLM.

Requirements:
    pip install streamlit langchain langchain-ollama langgraph \
                pdf2image Pillow pydantic

You also need Ollama running locally with a vision model pulled:
    ollama pull llava
"""

import base64
import io
import tempfile
from typing import TypedDict

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from pdf2image import convert_from_bytes
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://eos.local:11434")

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(page_title="Invoice Extractor", page_icon="🧾", layout="wide")


# ── State schema for LangGraph ───────────────────────────────────────────────
class InvoiceState(TypedDict):
    image_b64: str          # base64-encoded image
    raw_text: str           # raw LLM extraction
    structured_data: dict   # parsed fields


# ── Helper: convert uploaded file to base64 PNG ──────────────────────────────
def file_to_base64_image(uploaded_file) -> str:
    """Convert an uploaded PDF or image to a base64-encoded PNG string."""
    file_bytes = uploaded_file.read()
    mime = uploaded_file.type
    # For PDFs, convert the first page to an image. For images, open directly.
    if mime == "application/pdf":
        images = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=200)
        img = images[0]
    else:
        # For images, we can directly open them with PIL
        img = Image.open(io.BytesIO(file_bytes))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── LangGraph node functions ─────────────────────────────────────────────────
def extract_text(state: InvoiceState) -> dict:
    """Send the invoice image to Ollama vision model and extract raw text."""
    model = ChatOllama(model=st.session_state.get("model_name", "llava"), temperature=0, base_url=OLLAMA_BASE_URL)

    extraction_prompt = (
        "You are an expert invoice data extractor. Analyze this invoice image and "
        "extract ALL information. Return data in this EXACT format (use 'N/A' if not found):\n\n"
        "INVOICE_NUMBER: ...\nDATE: ...\nDUE_DATE: ...\nVENDOR_NAME: ...\n"
        "VENDOR_ADDRESS: ...\nCUSTOMER_NAME: ...\nCUSTOMER_ADDRESS: ...\n"
        "SUBTOTAL: ...\nTAX: ...\nTOTAL: ...\nCURRENCY: ...\n"
        "PAYMENT_TERMS: ...\nLINE_ITEMS: item1 (qty x price) | item2 (qty x price) | ...\n"
        "NOTES: ..."
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": extraction_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{state['image_b64']}"},
            },
        ]
    )

    response = model.invoke([message])
    return {"raw_text": response.content}


def parse_fields(state: InvoiceState) -> dict:
    """Parse the raw extracted text into a structured dictionary."""
    raw = state["raw_text"]
    fields = {}
    field_keys = [
        "INVOICE_NUMBER", "DATE", "DUE_DATE", "VENDOR_NAME", "VENDOR_ADDRESS",
        "CUSTOMER_NAME", "CUSTOMER_ADDRESS", "SUBTOTAL", "TAX", "TOTAL",
        "CURRENCY", "PAYMENT_TERMS", "LINE_ITEMS", "NOTES",
    ]
    # Simple line-by-line parsing based on expected keys
    for line in raw.strip().split("\n"):
        # Check if the line starts with any of the expected field keys
        for key in field_keys:
            # Match lines that start with the key followed by a colon, ignoring case
            if line.strip().upper().startswith(f"{key}:"):
                value = line.split(":", 1)[1].strip()
                # Handle empty values
                fields[key] = value or "N/A"

    # Fill any missing fields
    for key in field_keys:
        # Check if the line starts with any of the expected field keys
        if key not in fields:
            fields[key] = "N/A"

    return {"structured_data": fields}


def validate_output(state: InvoiceState) -> dict:
    """Light validation — flag if critical fields are missing."""
    data = state["structured_data"]
    critical = ["INVOICE_NUMBER", "VENDOR_NAME", "TOTAL", "DATE"]
    missing = [f for f in critical if data.get(f, "N/A") == "N/A"]
    data["_VALIDATION"] = "✅ All critical fields found" if not missing else f"⚠️ Missing: {', '.join(missing)}"
    return {"structured_data": data}


# ── Build the LangGraph pipeline ─────────────────────────────────────────────
def build_graph():
    graph = StateGraph(InvoiceState)
    graph.add_node("extract", extract_text)
    graph.add_node("parse", parse_fields)
    graph.add_node("validate", validate_output)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "parse")
    graph.add_edge("parse", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


# ── Streamlit UI ─────────────────────────────────────────────────────────────
def render_field_card(label: str, value: str):
    """Render a single field as a styled card."""
    st.markdown(f"""
    <div class="result-card">
        <h3>{label.replace('_', ' ')}</h3>
        <p>{value}</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown("# 🧾 Invoice Extractor")
    st.markdown("Upload a PDF or image invoice to extract structured data using a **local LLM** via Ollama.")
    st.markdown("---")

    # ── Custom CSS for a clean, professional look ────────────────────────────────
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .result-card { background: linear-gradient(135deg,#1a1f2e,#151922); border: 1px solid #2a3040;
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    .result-card h3 { color: #60a5fa; margin-bottom: .5rem; font-size: .85rem;
        text-transform: uppercase; letter-spacing: 1.5px; }
    .result-card p { color: #e2e8f0; font-size: 1.1rem; margin: 0; }
    .status-badge { display: inline-block; background: #1e3a5f; color: #60a5fa;
        padding: .25rem .75rem; border-radius: 20px; font-size: .8rem; font-weight: 500; }
    </style>""", unsafe_allow_html=True)

    # Sidebar config
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        model_name = st.text_input("Ollama Model", value="llava",
                                   help="Vision-capable model (e.g. llava, llava:13b, bakllava)")
        st.session_state["model_name"] = model_name

    # File uploader
    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Drop your invoice here",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            help="Supports PDF and common image formats",
        )

    if uploaded_file:
        # Show preview
        with col_preview:
            if uploaded_file.type == "application/pdf":
                uploaded_file.seek(0)
                images = convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1, dpi=150)
                st.image(images[0], caption="Invoice Preview (Page 1)")
            else:
                st.image(uploaded_file, caption="Invoice Preview")

        st.markdown("---")

        # Process button
        if st.button("🚀  Extract Invoice Data", type="primary", use_container_width=True):
            uploaded_file.seek(0)

            with st.status("Processing invoice...", expanded=True) as status:
                result = process_invoice(uploaded_file, model_name, status)
            # Store results
            st.session_state["result"] = result

    # Display results
    if "result" in st.session_state:
        render_results()



def render_results():
    """Render the extracted invoice data in a clean, organized layout."""
    result = st.session_state["result"]
    data = result["structured_data"]
    validation = data.pop("_VALIDATION", "")

    st.markdown("## Extracted Data")
    st.markdown(f'<span class="status-badge">{validation}</span>', unsafe_allow_html=True)
    st.markdown("")

    # Key financial fields in a highlight row
    fin_cols = st.columns(4)
    highlight_fields = ["INVOICE_NUMBER", "DATE", "TOTAL", "CURRENCY"]
    for col, key in zip(fin_cols, highlight_fields):
        with col:
            render_field_card(key, data.get(key, "N/A"))

    # Vendor & Customer
    st.markdown("### Parties")
    party_cols = st.columns(2)
    # Vendor info on the left, customer info on the right
    with party_cols[0]:
        render_field_card("VENDOR_NAME", data.get("VENDOR_NAME", "N/A"))
        render_field_card("VENDOR_ADDRESS", data.get("VENDOR_ADDRESS", "N/A"))
    # Customer info on the right
    with party_cols[1]:
        render_field_card("CUSTOMER_NAME", data.get("CUSTOMER_NAME", "N/A"))
        render_field_card("CUSTOMER_ADDRESS", data.get("CUSTOMER_ADDRESS", "N/A"))

    # Financial details
    st.markdown("### Financials")
    money_cols = render_field_row(3, "SUBTOTAL", data, "TAX")
    with money_cols[2]:
        render_field_card("DUE_DATE", data.get("DUE_DATE", "N/A"))

    # Line items
    st.markdown("### Line Items")
    render_field_card("LINE_ITEMS", data.get("LINE_ITEMS", "N/A"))

    render_field_row(2, "PAYMENT_TERMS", data, "NOTES")
    # Raw output expander
    with st.expander("📄 Raw LLM Output"):
        st.code(result["raw_text"], language="text")

    # Download as JSON
    import json
    json_str = json.dumps(data, indent=2)
    st.download_button(
        "⬇️  Download as JSON",
        data=json_str,
        file_name="invoice_data.json",
        mime="application/json",
    )



def process_invoice(uploaded_file, model_name, status):
    """Process the uploaded file through the LangGraph pipeline."""
    st.write("📸 Converting to image...")
    image_b64 = file_to_base64_image(uploaded_file)

    st.write(f"🤖 Sending to **{model_name}** via Ollama...")
    pipeline = build_graph()
    # Initialize the state with the base64 image and empty fields for text and structured data
    initial_state: InvoiceState = {
        "image_b64": image_b64,
        "raw_text": "",
        "structured_data": {},
    }

    result = pipeline.invoke(initial_state)
    status.update(label="✅ Extraction complete!", state="complete")

    return result


def render_field_row(num_cols, key1, data, key2):
    """Helper to render a row of two fields side by side."""
    result = st.columns(num_cols)
    # Render the first field in the first column and the second field in the second column
    with result[0]:
        render_field_card(key1, data.get(key1, "N/A"))
    # Render the second field in the second column
    with result[1]:
        render_field_card(key2, data.get(key2, "N/A"))
    return result


if __name__ == "__main__":
    main()