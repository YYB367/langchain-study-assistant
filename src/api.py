"""学习资料问答助手的 HTTP 接口。"""

import logging
from time import perf_counter

from fastapi import FastAPI, HTTPException
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict

from src.rag_service import generate_answer, load_saved_vector_store, retrieve_documents


logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    """用户提交的问题。"""

    model_config = ConfigDict(str_strip_whitespace=True, str_min_length=1)
    question: str


class SourceSnippet(BaseModel):
    """答案引用的一条 PDF 原文。"""

    page: int
    text: str


class AskResponse(BaseModel):
    """问答接口返回给用户的内容。"""

    answer: str
    sources: list[SourceSnippet]
    elapsed_ms: float


app = FastAPI(title="学习资料问答助手")


@app.get("/health")
def health() -> dict[str, str]:
    """确认 HTTP 服务可以响应请求。"""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """检索相关段落，生成答案，并附上原文来源。"""
    started_at = perf_counter()
    vector_store = load_saved_vector_store()
    documents = retrieve_documents(vector_store, request.question)
    if not documents:
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        return AskResponse(
            answer="当前 PDF 中没有找到足够相关的资料，无法依据资料回答。",
            sources=[],
            elapsed_ms=elapsed_ms,
        )

    try:
        answer = generate_answer(request.question, documents)
    except OpenAIError as error:
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.warning(
            "DeepSeek 调用失败，耗时 %.2f ms，错误类型：%s",
            elapsed_ms,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="DeepSeek 接口暂时不可用，请稍后重试。",
        ) from None
    sources = [
        SourceSnippet(page=doc.metadata["page"] + 1, text=doc.page_content)
        for doc in documents
    ]
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    return AskResponse(answer=answer, sources=sources, elapsed_ms=elapsed_ms)
