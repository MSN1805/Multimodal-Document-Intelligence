# Multimodal AI-Based Document Intelligence and Retrieval System

**OCR • Semantic Search • BM25 • Hybrid Retrieval • RAG • FAISS • Local LLM**

An AI-powered document intelligence system that extracts information from PDFs and scanned documents, performs semantic and keyword-based retrieval, and generates document-grounded answers using Retrieval-Augmented Generation (RAG).

## Project Overview

The system is designed to work with documents such as:

- Research papers
- Reports
- Resumes
- Invoices
- Scanned PDFs
- Technical documents

It combines OCR, text preprocessing, embeddings, vector search, BM25 keyword retrieval, hybrid ranking, and a local LLM to provide searchable and explainable document analysis.

## Key Features

- PDF text extraction using PyMuPDF
- OCR fallback for scanned pages using Tesseract
- Table extraction using pdfplumber
- Text cleaning and chunking
- Semantic embeddings using Sentence Transformers
- FAISS vector similarity search
- BM25 keyword retrieval
- Hybrid semantic + keyword retrieval
- Persistent document vector databases
- Multi-document retrieval
- Local RAG using Ollama and Llama 3.2
- Source/page-aware citations
- Document summarization
- Structured information extraction
- Evidence and grounding information in the UI
- Streamlit-based interface

## System Architecture

```text
                    ┌──────────────────────┐
                    │     PDF / Document   │
                    └──────────┬───────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Text Extraction + OCR    │
                 │ PyMuPDF + Tesseract      │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Cleaning + Chunking       │
                 └────────────┬─────────────┘
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
        ┌──────────────────┐    ┌──────────────────┐
        │ Sentence        │    │ BM25 Keyword     │
        │ Embeddings      │    │ Retrieval        │
        └────────┬─────────┘    └────────┬─────────┘
                 │                       │
                 ▼                       ▼
        ┌──────────────────┐    ┌──────────────────┐
        │ FAISS Semantic   │    │ Keyword Scores   │
        │ Search           │    │                  │
        └────────┬─────────┘    └────────┬─────────┘
                 └───────────┬───────────┘
                             ▼
                   ┌────────────────────┐
                   │ Hybrid Retrieval   │
                   └─────────┬──────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │ Context / Evidence │
                   └─────────┬──────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │ Local LLM (RAG)    │
                   │ Ollama + Llama 3.2 │
                   └─────────┬──────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │ Answer + Citations │
                   └────────────────────┘
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| UI | Streamlit |
| PDF processing | PyMuPDF |
| Table extraction | pdfplumber |
| OCR | Tesseract + pytesseract |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS |
| Keyword search | BM25 (`rank-bm25`) |
| LLM | Ollama + `llama3.2:3b` |
| Image processing | Pillow, OpenCV |

## Project Structure

```text
Multimodal-Document-Intelligence/
│
├── app.py
├── document_pipeline.py
├── embedding.py
├── vector_store.py
├── rag.py
├── text_processor.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── evaluation/
│   ├── evaluate.py
│   ├── test_questions.json
│   ├── evaluation_report.json
│   └── EVALUATION_REPORT.md
│
├── data/          # local data; ignored by Git
├── models/        # local models; ignored by Git
├── uploads/       # runtime uploads; ignored by Git
├── outputs/       # runtime outputs; ignored by Git
└── vector_db/     # persistent local vector data; ignored by Git
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Multimodal-Document-Intelligence
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Install Tesseract OCR for your operating system and make sure the executable is available to `pytesseract`.

The current Windows configuration expects:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If Tesseract is installed somewhere else, update the path in `document_pipeline.py`.

### 5. Install Ollama and download the local model

Install Ollama, then run:

```powershell
ollama pull llama3.2:3b
```

Make sure Ollama is running before using the RAG features.

## Running the Application

Start Streamlit with:

```powershell
python -m streamlit run app.py
```

The application will open in your browser.

## How It Works

### 1. Document ingestion

A PDF is uploaded through the Streamlit interface. The system attempts normal PDF text extraction first. Pages without useful text can be processed through OCR.

### 2. Preprocessing and chunking

Extracted content is cleaned and divided into smaller overlapping chunks so that relevant evidence can be retrieved efficiently.

### 3. Embeddings

Each text chunk is converted into a numerical embedding using Sentence Transformers.

### 4. Semantic retrieval

FAISS searches for chunks that are semantically similar to the user's query.

### 5. Keyword retrieval

BM25 provides lexical matching, which is useful for exact terminology, names, numbers, and factual queries.

### 6. Hybrid retrieval

Semantic and keyword signals are combined to rank candidate evidence. The V2.2 pipeline also contains targeted query handling for several factual query types.

### 7. RAG generation

The retrieved evidence is supplied to the local LLM. The model generates an answer using the retrieved document context and includes source/page information.

## Evaluation

The V2.2 retrieval pipeline was evaluated on a benchmark containing **18 questions** based on a research paper:

- 16 answerable questions
- 2 explicitly unanswerable questions

### V2.2 Retrieval Results

| Metric | Result |
|---|---:|
| Hit@1 | **87.50%** |
| Recall@3 | **93.75%** |
| Recall@5 | **93.75%** |
| MRR | **88.54%** |
| Average retrieval time | **0.0309 sec/query** |
| Document processing time | **6.395 sec** |
| Chunks | 54 |
| Vectors | 54 |
| Extracted tables | 33 |

These figures measure **retrieval performance**, not overall LLM answer accuracy.

The full local-LLM RAG benchmark was not included because local LLM inference requires substantially longer execution time.

Detailed results are available in `evaluation/EVALUATION_REPORT.md` and `evaluation/evaluation_report.json`.

## Current Limitations

- The current evaluation uses a single source document and a relatively small benchmark.
- Full end-to-end RAG answer accuracy has not been benchmarked.
- Some highly specific factual queries can still retrieve less relevant pages near the top of the ranking.
- Tesseract OCR quality depends on scan quality and document layout.
- Local LLM inference speed depends on available hardware.
- Image/figure understanding using a dedicated vision-language model is outside the current V2 scope.

## Future Scope

- Larger multi-domain evaluation datasets
- Cross-encoder reranking
- Improved table and document-layout understanding
- Vision-language models for figures and images
- Automated hallucination and evidence-consistency detection
- More comprehensive end-to-end RAG evaluation
- Additional document formats
- Authentication and user-specific document libraries
- Cloud deployment

## Research / Project Contribution

The project combines multiple information-retrieval techniques rather than relying only on vector similarity. Hybrid retrieval allows semantic similarity and lexical matching to contribute to evidence selection, while the RAG layer uses the retrieved evidence to generate document-grounded responses.

A further research direction is to combine retrieval evidence, grounding signals, and answer verification into a document-specific confidence mechanism that can flag answers when supporting evidence is insufficient.

## Author

**Mayur Nikam**

Computer Science and Engineering (DATA SCIENCE) Student  
PVPIT, Pune

---

**Note:** Runtime-generated folders such as `vector_db/`, `uploads/`, `outputs/`, and local virtual environments are intentionally excluded from version control.
