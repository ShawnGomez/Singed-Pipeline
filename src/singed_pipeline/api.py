from fastapi import FastAPI
from pydantic import BaseModel
from singed_pipeline.rag import answer_question

app = FastAPI()

class QueryRequest(BaseModel):
    message: str
    # conversation_id: str | None = None


class Source(BaseModel):
    document_slug: str
    section_slug: str
    title: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

@app.post("/rag/query", response_model=QueryResponse)
def query(request: QueryRequest):
    return answer_question(request.message)
