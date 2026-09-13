from fastapi import FastAPI
from pydantic import BaseModel, Field
from singed_pipeline.rag import answer_question

app = FastAPI()

class QueryRequest(BaseModel):
    message: str = Field(min_length = 1, max_length = 4000)
    document_slug: str | None = None


class Source(BaseModel):
    document_slug: str
    section_slug: str
    title: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

@app.post("/rag/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = answer_question(question = request.message, document_slug = request.document_slug,)

    return QueryResponse.model_validate(result)