from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from singed_pipeline.indexing import vector_store
from dataclasses import dataclass
from collections import OrderedDict

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

    groups: "OrderedDict[tuple[str, str], dict]" = OrderedDict()

    for item in retrieved:
        md = item.document.metadata
        key = (md["document_slug"], md["section_slug"])

        if key not in groups:
            groups[key] = {
                "document_slug": md["document_slug"],
                "section_slug": md["section_slug"],
                "document_title": md.get("document_title"),
                "section_title": md.get("section_title"),
                "contents": [],
                "max_relevance": item.relevance,
            }

        groups[key]["contents"].append(item.document.page_content)
        groups[key]["max_relevance"] = max(groups[key]["max_relevance"], item.relevance)

    context_blocks: list[str] = []
    sources: list[dict] = []

    for index, group in enumerate(groups.values(), start = 1):
        label = f"S{index}"
        body = "\n\n".join(group["contents"])

        context_blocks.append(
            f"[{label}]\n"
            f"Document: {group['document_title']}\n"
            f"Section: {group['section_title']}\n"
            f"Relevance: {group['max_relevance']:.3f}\n\n"
            f"{body}"
        )

        sources.append({
            "label": label,
            "document_slug": group["document_slug"],
            "section_slug": group["section_slug"],
            "title": group["section_title"],
        })

    context = "\n\n---\n\n".join(context_blocks)
   
    prompt = f"""You are a technical guide for the project. Explain the documentation clearly.

- Cite every project-specific fact with its source label, e.g. [S1]. Never invent features not in the sources.
- Use general chemical engineering, machine learning, and software engineering knowledge to explain concepts, but do not cite it, and never present it as a documented feature.
- If something is not in the sources, say it is not documented instead of guessing.

<sources>
{context}
</sources>

<question>
{question}
</question>

Answer:"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
    }

