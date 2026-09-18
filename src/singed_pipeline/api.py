from fastapi import FastAPI
from pydantic import BaseModel, Field
from singed_pipeline.agent.graph import rag_graph

app = FastAPI()

class QueryRequest(BaseModel):
    message: str = Field(min_length = 1, max_length = 4000)
    document_slug: str | None = None


class Source(BaseModel):
    label: str
    document_slug: str
    section_slug: str
    title: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

@app.post("/rag/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = rag_graph.invoke({
        "question": request.message,
        "document_slug": request.document_slug,
        "search_query": request.message,
        "attempts": 0,
    })

    answer = result.get("answer")
    if answer is None:
        raise RuntimeError("The RAG graph finished without producing an answer.")


    return QueryResponse.model_validate(result)
