from langchain_core.documents import Document

from singed_pipeline.agent import nodes
from singed_pipeline.rag import RetrievedDocument


def make_retrieved(document_slug: str, section_slug: str) -> RetrievedDocument:
    return RetrievedDocument(
        document=Document(
            page_content="Supported documentation text.",
            metadata={
                "document_slug": document_slug,
                "section_slug": section_slug,
                "document_title": "Guide",
                "section_title": "Setup",
            },
        ),
        relevance=0.9,
    )


def test_retrieve_node_passes_string_query(monkeypatch):
    received = {}

    def fake_retrieve(question, document_slug=None):
        received["question"] = question
        return []

    monkeypatch.setattr(nodes, "retrieve_documents", fake_retrieve)

    result = nodes.retrieve_node({"question": "How do I configure it?"})

    assert received["question"] == "How do I configure it?"
    assert isinstance(received["question"], str)
    assert result["attempts"] == 1


def test_answer_node_labels_deduplicated_sources(monkeypatch):
    class Response:
        content = "Use the documented setup. [S1]"

    class FakeLlm:
        def invoke(self, prompt):
            return Response()

    monkeypatch.setattr(nodes, "llm", FakeLlm())

    result = nodes.answer_node({
        "question": "How?",
        "retrieved": [
            make_retrieved("guide", "setup"),
            make_retrieved("guide", "setup"),
        ],
    })

    assert result["sources"] == [{
        "label": "S1",
        "document_slug": "guide",
        "section_slug": "setup",
        "title": "Setup",
    }]
