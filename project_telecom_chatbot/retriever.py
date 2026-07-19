"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)
"""

import csv
import os
import re

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAQ_PATH = os.path.join(os.path.dirname(__file__), "data", "faq.csv")


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _load_faq_fallback(query: str, limit: int = 3) -> list[Document]:
    if not os.path.exists(FAQ_PATH):
        return []

    query_tokens = _tokenize(query)
    scored_docs: list[tuple[int, Document]] = []

    with open(FAQ_PATH, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = row["question"]
            answer = row["answer"]
            content = f"Q: {question}\nA: {answer}"
            question_lower = question.lower().strip()
            score = 0

            if query.lower().strip() == question_lower:
                score = 10_000
            else:
                question_tokens = _tokenize(question)
                score = len(query_tokens & question_tokens)

            if score > 0:
                scored_docs.append((score, Document(
                    page_content=content,
                    metadata={"source": "faq", "category": row["category"], "faq_id": str(row["id"]), "retrieval": "fallback"},
                )))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs[:limit]]


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    deduped: list[Document] = []
    seen: set[str] = set()

    for doc in documents:
        key = doc.page_content.strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    return deduped


def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    faq_retriever     = faq_store.as_retriever(search_kwargs={"k": k_faq})
    tickets_retriever = tickets_store.as_retriever(search_kwargs={"k": k_tickets})
    guides_retriever  = guides_store.as_retriever(search_kwargs={"k": k_guides})

    def retrieve(query: str) -> list[Document]:
        documents = (
            faq_retriever.invoke(query)
            + tickets_retriever.invoke(query)
            + guides_retriever.invoke(query)
        )
        documents.extend(_load_faq_fallback(query))
        return _dedupe_documents(documents)

    return RunnableLambda(retrieve)
