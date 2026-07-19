"""
Builds the RAG chain:
  merged retriever → prompt → Qwen3-32B on Groq → string output
"""

import os

from dotenv import load_dotenv
load_dotenv()

googleapikey = os.getenv("GOOGLE_API_KEY") 
if not googleapikey:
    googleapikey = os.getenv("googleapikey")

groqapikey = os.getenv("GROQ_API_KEY")
if not groqapikey:
    groqapikey = os.getenv("groqapikey")

model = os.getenv("MODEL", "qwen/qwen3-32b")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retriever import build_retriever

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from two sources:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)

If the context contains a FAQ or ticket that directly answers the question, answer directly from it.
Only say the context is insufficient if the retrieved context is genuinely unrelated or empty.
In that case, say so clearly and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""


def _format_docs(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()
        sections.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        model=model,
        api_key=groqapikey,
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
