from dataclasses import dataclass
from typing import TypedDict

from langchain_core.documents import Document

@dataclass
class RetrievedDocument:
    document: Document
    relevance: float

class SourceData(TypedDict):
    label: str
    document_slug: str
    section_slug:str
    title: str | None