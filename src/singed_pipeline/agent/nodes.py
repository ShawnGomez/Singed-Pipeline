from singed_pipeline.agent.state import AgentState
from singed_pipeline.rag import llm, retrieve_documents
from singed_pipeline.agent.prompts import (
    ANSWER_PROMPT, GRADE_EVIDENCE_PROMPT, REWRITE_QUERY_PROMPT
)
from pydantic import BaseModel

MAX_RETRIEVAL_ATTEMPTS = 2

def retrieve_node(state: AgentState) -> dict:
    search_query = (
        state.get("search_query") 
        or state["question"]
    )

    documents = retrieve_documents(
        question= search_query,
        document_slug=state.get("document_slug"),
    )

    return {
        "retrieved": documents,
        "attempts": state.get("attempts", 0) + 1,
    }



class EvidenceGrade(BaseModel):
    sufficient: bool
    reason: str

grader = llm.with_structured_output(EvidenceGrade)

def grade_evidence_node(state: AgentState)-> dict:
    retrieved = state.get("retrieved", [])

    if not retrieved:
        return{
            "evidence_sufficient": False,
            "evidence_reason": ("The documents didn't meet the relevance threshold"),
        }
    context = "\n\n --- \n\n".join(
        item.document.page_content
        for item in retrieved
    )

    raw_result = grader.invoke(GRADE_EVIDENCE_PROMPT.format(
        question=state["question"],
        search_query=state.get("search_query", state["question"]),
        context=context,
    ))

    result = validate_structured_output(raw_result,EvidenceGrade,)

    return {
        "evidence_sufficient": result.sufficient,
        "evidence_reason": result.reason,
    }

def route_after_grading(state: AgentState) -> str:
    if state.get("evidence_sufficient", False):
        return "answer"

    if state.get("attempts", 0) < MAX_RETRIEVAL_ATTEMPTS:
        return "rewrite"

    return "refuse"

class RewrittenQuery(BaseModel):
    query: str

query_rewriter = llm.with_structured_output(RewrittenQuery)

def rewrite_query_node(state: AgentState)-> dict:
    retrieved = state.get("retrieved", [])

    context_summary = "\n\n".join(
        item.document.page_content[:500]

        for item in retrieved
    )

    raw_result = query_rewriter.invoke(REWRITE_QUERY_PROMPT.format(
        question=state["question"],
        search_query=state.get("search_query", state["question"]),
        evidence_reason=state.get("evidence_reason", "Unknown"),
        context_summary=context_summary,
    ))

    result = validate_structured_output(raw_result, RewrittenQuery)

    return{
        "search_query":result.query,
    }

def answer_node(state: AgentState) -> dict:
    retrieved = state.get("retrieved", [])

    if not retrieved: return refuse_node(state)

    context_blocks: list[str] = []
    sources_by_section: dict[tuple[str, str], dict[str,object]] = {}
    source_labels: dict[tuple[str, str], str] = {}

    for item in retrieved:
        document = item.document
        metadata = document.metadata

        document_slug = str(metadata["document_slug"])
        section_slug = str(metadata["section_slug"])

        source_key = (
            document_slug,
            section_slug,
        )
        source_id = source_labels.setdefault(
            source_key,
            f"S{len(source_labels) + 1}",
        )

        context_blocks.append(
            f"""
[{source_id}]
Document: {metadata.get("document_title")}
Section: {metadata.get("section_title")}

{document.page_content}
""".strip()
        )

        sources_by_section[source_key] = {
            "label": source_id,
            "document_slug": metadata["document_slug"],
            "section_slug": metadata["section_slug"],
            "title": metadata.get("section_title"),
        }

    context = "\n\n --- \n\n".join(context_blocks)

    response = llm.invoke(ANSWER_PROMPT.format(
        context=context,
        question=state["question"],
    ))

    return {
        "answer": str(response.content),
        "sources": list(sources_by_section.values()),
    }


def refuse_node(state: AgentState) -> dict[str,object]:
    return {
        "answer":(
            "The singed documentation does not currently contain enough information to answer that question"
        ),
        "sources": [],
    }

def validate_structured_output[T: BaseModel](
        raw_result: object,
        model_type: type[T],
)->T:
    if isinstance(raw_result, model_type):
        return raw_result

    return model_type.model_validate(raw_result)

