"""
RAG 客服聊天 API：非流式与流式对话接口。
"""
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.engine import ingest_text, query, query_stream
from app.models import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    IngestRequest,
    IngestResponse,
)

chat_router = APIRouter()

_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def _sse_chat_stream(message: str) -> AsyncIterator[str]:
    """将 RAG 流式结果转为 SSE 格式。"""
    try:
        async for chunk, done, sources in query_stream(message):
            data = ChatStreamChunk(
                chunk=chunk,
                done=done,
                sources=sources if done and sources else None,
            )
            yield f"data: {data.model_dump_json(ensure_ascii=False)}\n\n"
    except ValueError as e:
        data = ChatStreamChunk(chunk=str(e), done=True, sources=None)
        yield f"data: {data.model_dump_json(ensure_ascii=False)}\n\n"
    except Exception as e:
        data = ChatStreamChunk(chunk=f"服务异常: {e}", done=True, sources=None)
        yield f"data: {data.model_dump_json(ensure_ascii=False)}\n\n"


@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG 客服对话（非流式）。
    根据知识库与 LLM 返回一条完整回复及引用来源。
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="请使用 POST /api/chat/stream 进行流式对话",
        )
    try:
        answer, sources = query(request.message)
        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            sources=sources,
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {e}")


@chat_router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    RAG 客服对话（SSE 流式）。
    每行一条 JSON（data 字段），含 chunk、done、sources。
    """
    return StreamingResponse(
        _sse_chat_stream(request.message),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )


@chat_router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    将文本写入知识库（Qdrant）。
    用于后续 RAG 检索与回复。
    """
    if not request.text or not request.text.strip():
        return IngestResponse(success=False, message="text 不能为空", doc_count=0)
    try:
        count = ingest_text(request.text.strip())
        return IngestResponse(success=True, message="入库成功", doc_count=count)
    except Exception as e:
        return IngestResponse(success=False, message=str(e), doc_count=0)
