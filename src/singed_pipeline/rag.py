from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from singed_pipeline.indexing import vector_store
from dataclasses import dataclass

@dataclass
class RetrievedDocument:
    document : Document
    relevance: float

MIN_RELEVANCE = 0.55

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)


def retrieve_documents(question: str, document_slug: str | None = None ) -> list[RetrievedDocument]:
    metadata_filter = None 

    if document_slug is not None:
        metadata_filter = {
            "document_slug":document_slug
        }

    results = vector_store.similarity_search_with_relevance_scores(
        query=question,
        k=8,
        filter = metadata_filter
    )

    return [
        RetrievedDocument(document = document, relevance = relevance)
        for document, relevance in results
        if relevance >= MIN_RELEVANCE
    ]

def answer_question(question: str, document_slug: str | None = None) -> dict:
    retrieved = retrieve_documents(question = question, document_slug = document_slug)

    if not retrieved:
        return {
            "answer": (
                "The Singed documentation does not currently contain enough information to answer that question"
            ),
            "sources": [],
        }

    context ="\n\n".join(
        item.document.page_content
        for item in retrieved 
    )
   
    prompt = f"""
You are answering questions about the Singed documentation.

Only use the provided context when answering.

If the context does not contain enough information,
say that the documentation does not contain enough information.

Context:

{context}

Question:

{question}
"""

    response = llm.invoke(prompt)

    sources = []

    for item in retrieved:
        doc = item.document

        sources.append({
            "document_slug":doc.metadata["document_slug"],
            "section_slug": doc.metadata["section_slug"],
            "title": doc.metadata.get("section_title"),
        })

    return {
        "answer": response.content,
        "sources": sources,
    }


