# Invoice Extractor

A Streamlit web application for extracting structured data from invoice PDFs and images using LangChain, Ollama, and LangGraph.

## 📋 Overview

This application allows users to upload invoice documents (PDF or image formats) and automatically extract key financial information such as vendor details, invoice numbers, dates, amounts, and line items. The extracted data is presented in a structured format and can be downloaded as JSON.

## 🚀 Features

- **Multi-format support**: Upload PDFs or images (PNG, JPG, JPEG, WEBP)
- **Vision-capable AI**: Uses Ollama's vision models (like LLaVA) to analyze invoice images
- **Structured extraction**: Automatically extracts invoice fields including:
  - Vendor information (name, address, email, phone)
  - Invoice details (number, date, due date)
  - Financial information (subtotal, tax, total, currency)
  - Line items with descriptions, quantities, and amounts
  - Payment terms and PO numbers
- **Interactive UI**: Preview uploaded documents before processing
- **Download results**: Export extracted data as JSON
- **Validation**: Automatic validation of critical fields

## 🛠️ Prerequisites

1. **Python 3.12 or later**
2. **Ollama** installed and running locally with a vision-capable model

### Install Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai
```

### Pull a Vision Model

```bash
ollama pull llava
# or for better accuracy
ollama pull llava:13b
```

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/Invoice_Extractor.git
cd Invoice_Extractor
```

### Install dependencies

```bash
pip install streamlit langchain langchain-ollama langgraph pypdf pillow pdf2image python-dotenv
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` to configure:

```ini
# Ollama server URL (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Default model to use (default: llava)
DEFAULT_MODEL=llava
```

## 🎯 Usage

### Run the application

```bash
streamlit run invoice_extractor.py
```

The app will open in your default browser at `http://localhost:8501`.

### Using the App

1. **Select a model**: Choose from available Ollama vision models (e.g., `llava`, `llava:13b`, `bakllava`)
2. **Upload invoice**: Drag and drop or select a PDF or image file
3. **Preview**: View the uploaded document before processing
4. **Extract**: Click the "Extract Invoice Data" button to process
5. **Review results**: View extracted data in a structured format
6. **Download**: Export results as JSON for further processing

## 📂 Project Structure

```
Invoice_Extractor/
├── .env                # Environment configuration
├── .env.example         # Example environment file
├── .gitignore           # Git ignore rules
├── AGENTS.md            # Agent configuration
├── LICENSE              # License file
├── app.py               # Alternative Streamlit app implementation
├── dev-app.py           # Development version of the app
├── invoice_extractor.py # Main Streamlit application
├── README.md            # This file
└── example_docs/        # Sample invoice documents for testing
    ├── Downloadable-PDF-Invoices-Add-On-Samples.pdf
    ├── hourly_invoice_template.png
    ├── image.png
    ├── invoice-0-4.pdf
    ├── invoice-1-3.pdf
    ├── invoice-2-1.pdf
    ├── invoice-3-0.pdf
    ├── invoice-7-0.pdf
    ├── sample-invoice.png
    └── wordpress-pdf-invoice-plugin-sample.pdf
```

## 🔍 Technical Details

### Architecture

The application follows a layered architecture:

1. **Configuration**: Constants and environment settings
2. **Pure Helpers**: Reusable utility functions
3. **LangGraph Nodes**: Workflow processing nodes
4. **UI Components**: Streamlit interface elements

### Workflow

The extraction process follows these steps:

1. **Document Preprocessing**: Convert PDFs to images or prepare image uploads
2. **Vision Analysis**: Send document to Ollama vision model for analysis
3. **Text Extraction**: Parse model output into structured fields
4. **Validation**: Check for critical missing fields
5. **Presentation**: Display results in a user-friendly format

### Supported Models

- `llava` - Lightweight vision model
- `llava:13b` - Larger vision model with better accuracy
- `bakllava` - Alternative vision model
- Any other Ollama vision-capable model

## 📝 Example Output

The extracted data includes these fields:

```json
{
  "INVOICE_NUMBER": "INV-2024-001",
  "DATE": "2024-01-15",
  "DUE_DATE": "2024-02-15",
  "VENDOR_NAME": "Acme Corp",
  "VENDOR_ADDRESS": "123 Business St, Suite 100, New York, NY 10001",
  "CUSTOMER_NAME": "Your Company",
  "CUSTOMER_ADDRESS": "456 Main St, Boston, MA 02101",
  "SUBTOTAL": "1250.00",
  "TAX": "125.00",
  "TOTAL": "1375.00",
  "CURRENCY": "USD",
  "PAYMENT_TERMS": "Net 30",
  "LINE_ITEMS": "Web Development (50 x 25.00) | Hosting (12 x 100.00)",
  "NOTES": "Early payment discount available"
}
```

## 💡 Tips for Best Results

1. **Use high-quality images**: Ensure invoices are clear and legible
2. **Standard formats**: Invoices following common templates extract better
3. **Model selection**: Larger models (like `llava:13b`) provide better accuracy
4. **PDFs**: The first page is used for extraction

## 🐛 Troubleshooting

### Ollama not running

```bash
ollama serve
```

### Model not found

```bash
ollama pull llava
```

### Port already in use

```bash
streamlit run invoice_extractor.py --server.port 8502
```

### Clear cache

```bash
streamlit cache clear
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📞 Support

For questions or issues, please open an issue on GitHub.
