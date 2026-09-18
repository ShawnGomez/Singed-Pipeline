from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from singed_pipeline.agent.graph import rag_graph
import logging, uuid

app = FastAPI()

logger = logging.getLogger("singed_pipeline.api")

class QueryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
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
    request_id = str(uuid.uuid4())

    try:
        result = rag_graph.invoke(
            {
                "question": request.message,
                "document_slug": request.document_slug,
                "search_query": request.message,
                "attempts": 0,
            })
    except Exception as error:
        logger.error(
            "RAG request failed request_id =%s error_type = %s",
            request_id,
            type(error).__name__
        )

        raise HTTPException(
                status_code = 503,
                detail = {
                    "message": "The assitant is temporarily unavailable.",
                    "request_id": request_id,
                    }
                ) from error

    answer = result.get("answer")
    if answer is None:
        raise RuntimeError("The RAG graph finished without producing an answer.")

    return QueryResponse.model_validate(result)
