# 🤖 RAG-Based PDF QA Chatbot using Streamlit & Elasticsearch

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot for answering questions over uploaded PDF documents.  
It is designed as a lightweight, local or cloud-deployable system that uses **semantic search** with **Elasticsearch**, and **open-source embeddings** for document retrieval.

Built with **Streamlit**, the application enables intuitive document upload, PDF parsing (with OCR fallback), vector indexing, and natural language interaction powered by an LLM API.

---

## 🔍 Key Features

- 📄 Upload one or multiple PDF files via the UI
- 🧠 Automatic text extraction, chunking, and embedding
- 📦 Storage of semantic vectors into Elasticsearch
- 🔎 Real-time vector search using semantic similarity
- 🧾 OCR fallback using Tesseract for scanned or image PDFs
- 🤖 Retrieval-Augmented Generation (RAG) architecture
- 🧬 Hugging Face sentence embeddings: `all-MiniLM-L6-v2`
- 🗣️ Multilingual input/output support (Greek & English)
- 🎨 Clean chat interface with video background and token-by-token reply streaming
- 📥 Option to download full chat history

---

## ⚙️ Technology Stack

| Component | Technology |
|----------|-------------|
| **Frontend** | Streamlit |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | Elasticsearch |
| **Text Extraction** | PDFPlumber, Tesseract (OCR fallback) |
| **Chunking** | LangChain RecursiveCharacterTextSplitter |
| **LLM API** | External language model (configurable endpoint) |
| **Deployment** | Compatible with local or cloud environments |

---

## 📂 Project Structure



-   `video_background.py`: Main Streamlit app

-   `elastic_injestion.ipynb`: Script to embed and ingest PDFs into Elasticsearch

-   `.env`: Environment variables for credentials (not included)

* * * * *

## 🚀 Getting Started

1.  Install dependencies: Run the last cell of the .ipynb file.

2.  Set up environment: Add your Elasticsearch and LLM API credentials to a `.env` file or use `streamlit secrets`.

3.  Run the app: `streamlit run video_background.py`

* * * * *

**How It Works**

1.  User uploads PDF(s)

2.  Text is extracted and split into overlapping chunks

3.  Each chunk is embedded using a SentenceTransformer model

4.  Embeddings are indexed in Elasticsearch

5.  A user question is also embedded and matched to the top-k relevant chunks

6.  Retrieved chunks and the question are sent to an LLM to generate the answer

7.  The answer is streamed back into the chat interface

* * * * *

**Use Cases**

-   Question answering over technical reports, documents, or manuals

-   Internal documentation search

-   Prototyping intelligent chat interfaces over enterprise documents

-   Research or academic knowledge base exploration

* * * * *

**Security Note**

Keep your credentials secure using a `.env` file or Streamlit secrets. Never expose keys in source code. Ensure proper access control if deployed in production.

* * * * *

**License & Usage**

This project is intended for research, educational, and non-commercial purposes. You may adapt and expand it for your own experimentation, e.g., integrating alternative LLMs or vector store.
