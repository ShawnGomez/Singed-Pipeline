from langchain_openai import ChatOpenAI
from singed_pipeline.indexing import vector_store

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)


def retrieve_documents(question: str):
    return vector_store.similarity_search(
        question,
        k=5,
    )

def answer_question(question: str):
    documents = retrieve_documents(question)

    context ="\n\n".join(
        doc.page_content
        for doc in documents
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

    for doc in documents:
        sources.append({
            "document_slug":doc.metadata["document_slug"],
            "section_slug": doc.metadata["section_slug"],
            "title": doc.metadata.get("section_title"),
        })

    return {
        "answer": response.content,
        "sources": sources,
    }


