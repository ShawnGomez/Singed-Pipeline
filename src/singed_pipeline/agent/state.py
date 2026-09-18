from typing import TypedDict, NotRequired
from singed_pipeline.models import RetrievedDocument, SourceData 

class AgentState(TypedDict):
    question: str
    document_slug: NotRequired[str | None]

    search_query: NotRequired[str]
    retrieved: NotRequired[list[RetrievedDocument]]

    evidence_sufficient: NotRequired[bool]
    evidence_reason: NotRequired[str]

    attempts: NotRequired[int]
    answer: NotRequired[str]
    sources: NotRequired[list[SourceData]]