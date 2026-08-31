from fastapi import FastAPI
from pydantic import BaseModel

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
def query(request: QueryRequest) -> QueryResponse:
    # make it return void then change it to this for the real build: return answer_question(request.message)
    return QueryResponse(answer="Connected from pipeline $\ket{\psi}$ = $\\alpha \ket{0}$ + $\\beta \ket{1}$", sources=[])
