"""
基于 FastAPI 的 RAG 客服 API 入口。

技术栈：FastAPI + LlamaIndex + Qdrant + OpenAI 兼容接口。
运行方式：
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat_router, stream_router
from app.engine import check_qdrant
from app.core.database import init_db, close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化，关闭时清理。"""
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="IntelligentCustomerService API",
    description="基于 LlamaIndex + Qdrant + AI 接口的 RAG 客服系统",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 客服聊天（非流式 + 流式）
app.include_router(chat_router, prefix="/api", tags=["chat"])
# 流式示例接口（SSE/NDJSON/文本）
app.include_router(stream_router, prefix="/api", tags=["stream"])


@app.get("/")
async def root() -> dict:
    """健康检查 / 根路径。"""
    return {"status": "ok", "message": "RAG 客服 API 运行中"}


@app.get("/health")
async def health() -> dict:
    """健康检查，含 Qdrant 连接状态。"""
    qdrant_status = check_qdrant()
    return {"status": "healthy", "qdrant": qdrant_status}
