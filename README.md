# RAG Telecom Chatbot

A Retrieval-Augmented Generation (RAG) customer care chatbot for telecom support. It answers questions about mobile connectivity, billing, SIM issues, roaming, and related service problems by retrieving relevant context from three knowledge sources and generating responses with Qwen3-32B via Groq.

## Architecture

```
User question
     │
     ▼
Merged Retriever (top-k from each store)
  ├── ChromaDB · faq        (FAQ entries from CSV)
  ├── ChromaDB · tickets    (resolved support tickets from SQLite)
  └── ChromaDB · guides     (PDF guide chunks)
     │
     ▼
ChatPromptTemplate → Qwen3-32B (Groq) → Answer
```

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally via HuggingFace)  
**LLM:** `qwen/qwen3-32b` served by [Groq](https://groq.com)

## Project Structure

```
project_telecom_chatbot/
├── app.py                  # Streamlit web UI
├── main.py                 # CLI entry point
├── rag_chain.py            # Builds the LangChain RAG chain
├── retriever.py            # Merges the three Chroma retrievers
├── ingest_faq.py           # Loads data/faq.csv → Chroma 'faq' collection
├── ingest_tickets.py       # Loads data/tickets.db → Chroma 'tickets' collection
├── ingest_pdf.py           # Loads data/telecom_guide.pdf → Chroma 'guides' collection
├── data/
│   ├── faq.csv             # FAQ question/answer pairs
│   ├── telecom_guide.pdf   # technical guide
│   └── tickets.db          # tickets raised by customers 
└── chroma_store/           # Persisted Chroma vector database (created at ingest)
```

## Prerequisites

- Python 3.11+
- `pip` inside a local virtual environment
- A [Groq API key](https://console.groq.com)
- A Hugging Face token if your environment requires authenticated model downloads

## Setup

**1. Create and activate a virtual environment**

This repository includes a local environment folder named `TelecomRAG-VENV`. You can reuse it if it already works in your machine, or create a fresh one with the standard library.

```bash
python3 -m venv TelecomRAG-VENV
source TelecomRAG-VENV/bin/activate
pip install -r requirements.txt
```

**2. Configure environment variables**

```bash
touch .env
```

Edit `.env` and fill in your keys:

```
groqapikey=your_groq_api_key_here
googleapikey=your_secondary_key_here
```

The current chain reads the environment variables with those names.

**3. Ingest data into Chroma**

Run the three ingestion scripts once to build the vector store:

```bash
python ingest_faq.py
python ingest_tickets.py
python ingest_pdf.py
```

Each script embeds the source data and persists it to `chroma_store/`. Re-run a script only when its source data changes.

## Running the App

**Streamlit web UI**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar has one-click sample questions and a button to clear the conversation history.

## Demo

Here is a quick screenshot of the chatbot interface:

![Telecom RAG chatbot demo](Screenshot.png)

**CLI**

```bash
python main.py
```

Interactive prompt — type a question and press Enter. Type `quit` to exit.

## Data Sources

| Collection | Source file | Granularity |
|---|---|---|
| `faq` | `data/faq.csv` | 1 document per FAQ row |
| `tickets` | `data/tickets.db` | 1 document per resolved ticket |
| `guides` | `data/telecom_guide.pdf` | Chunks of 600 chars with 100-char overlap |

The retriever fetches the top 3 results from each collection, giving up to 9 context documents for each query.

## Ignored Files

These are intentionally ignored in version control and should remain local:

- `.env` for secrets and local configuration
- `TelecomRAG-VENV/` for the Python virtual environment
- `__pycache__/` for Python bytecode caches
- `chroma_store/` for the persisted vector database

If you regenerate embeddings or local data, keep the outputs in those ignored paths unless you intentionally want to version new source files.

## Regenerating Seed Data

```bash
# Seed the SQLite ticket database
python data/seed_tickets.py

# Regenerate the PDF guide
python data/generate_pdf.py
```

After regenerating, re-run the corresponding ingest script.
