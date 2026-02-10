import base64
import json
from io import BytesIO
from typing import TypedDict

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pypdf import PdfReader
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://eos.local:11434")

class InvoiceState(TypedDict, total=False):
    file_name: str
    file_type: str
    file_bytes: bytes
    model: str
    invoice_text: str
    raw_response: str
    extracted_fields: dict


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


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(page_texts)


def load_document_node(state: InvoiceState) -> InvoiceState:
    if state["file_type"] == "application/pdf":
        return {"invoice_text": extract_pdf_text(state["file_bytes"])}
    return {"invoice_text": ""}


def parse_model_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[i:])
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"line_items": parsed}
        except json.JSONDecodeError:
            continue
    return {}


def normalize_response_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item))
        return "\n".join(parts)
    return str(content)


def extract_invoice_node(state: InvoiceState) -> InvoiceState:
    llm = ChatOllama(model=state["model"], temperature=0, format="json", base_url=OLLAMA_BASE_URL)

    if state["file_type"] == "application/pdf":
        user_prompt = (
            "Extract all invoice information from this invoice text.\n\n"
            f"{state['invoice_text']}"
        )
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
    else:
        encoded = base64.b64encode(state["file_bytes"]).decode("utf-8")
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Extract all invoice information from this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:{state['file_type']};base64,{encoded}",
                        },
                    ]
                ),
            ]
        )

    raw_response = normalize_response_content(response.content)
    extracted_fields = parse_model_json(raw_response)
    return {"raw_response": raw_response, "extracted_fields": extracted_fields}


@st.cache_resource
def build_graph():
    graph = StateGraph(InvoiceState)
    graph.add_node("load_document", load_document_node)
    graph.add_node("extract_invoice", extract_invoice_node)
    graph.set_entry_point("load_document")
    graph.add_edge("load_document", "extract_invoice")
    graph.add_edge("extract_invoice", END)
    return graph.compile()


def main():
    st.set_page_config(page_title="Invoice Extractor", page_icon=":receipt:")
    st.title("Invoice Extractor (LangChain + Ollama + LangGraph)")
    st.write("Upload an invoice PDF or image and extract structured fields.")

    model = st.text_input("Ollama model", value="ministral-3:3b")
    uploaded_file = st.file_uploader(
        "Upload invoice",
        type=["pdf", "png", "jpg", "jpeg"],
    )

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        st.write(f"File: `{uploaded_file.name}`")
        if uploaded_file.type.startswith("image/"):
            st.image(
                uploaded_file, caption=uploaded_file.name 
            )
        if uploaded_file.type == "application/pdf":
            pdf_b64 = base64.b64encode(file_bytes).decode("utf-8")
            pdf_preview = (
                f'<embed src="data:application/pdf;base64,{pdf_b64}" '
                'width="100%" height="700" type="application/pdf">'
            )
            st.markdown(pdf_preview, unsafe_allow_html=True)

        if st.button("Extract Information", type="primary"):
            graph = build_graph()
            result = graph.invoke(
                {
                    "file_name": uploaded_file.name,
                    "file_type": uploaded_file.type,
                    "file_bytes": file_bytes,
                    "model": model,
                }
            )

            st.subheader("Extracted Invoice JSON")
            st.json(result["extracted_fields"])

            with st.expander("Raw model output"):
                st.code(result["raw_response"], language="json")

            if uploaded_file.type == "application/pdf":
                with st.expander("Extracted PDF text"):
                    st.text(result["invoice_text"])


if __name__ == "__main__":
    main()
