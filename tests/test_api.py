from singed_pipeline import api


def test_query_invokes_graph_and_validates_response(monkeypatch):
    captured = {}

    def fake_invoke(state):
        captured.update(state)
        return {
            "answer": "Documented answer. [S1]",
            "sources": [{
                "label": "S1",
                "document_slug": "guide",
                "section_slug": "setup",
                "title": "Setup",
            }],
        }

    monkeypatch.setattr(api.rag_graph, "invoke", fake_invoke)

    response = api.query(api.QueryRequest(message="How?"))

    assert captured["question"] == "How?"
    assert captured["attempts"] == 0
    assert response.sources[0].label == "S1"
