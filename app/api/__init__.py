"""API 路由：流式示例与 RAG 客服聊天。"""
from app.api.stream import stream_router
from app.api.chat import chat_router

__all__ = ["stream_router", "chat_router"]
