"""
API 请求与响应模型（Pydantic），用于 RAG 客服接口。
"""
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 聊天
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """客服对话请求。"""
    message: str = Field(..., min_length=1, description="用户输入的问题或消息")
    session_id: Optional[str] = Field(None, description="会话 ID，用于多轮上下文（可选）")
    stream: bool = Field(False, description="是否使用流式输出")


class ChatResponse(BaseModel):
    """客服对话响应（非流式）。"""
    answer: str = Field(..., description="回复内容")
    session_id: Optional[str] = None
    sources: List[dict] = Field(default_factory=list, description="引用来源（文档片段等）")


class ChatStreamChunk(BaseModel):
    """流式输出单块数据（SSE/NDJSON 用）。"""
    chunk: str = Field("", description="当前文本片段")
    done: bool = Field(False, description="是否结束")
    sources: Optional[List[dict]] = Field(None, description="仅在 done=True 时可能带来源")


# ---------------------------------------------------------------------------
# 文档入库（Ingest）
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """文档入库请求（可扩展为 URL/文件等）。"""
    text: Optional[str] = Field(None, description="直接传入的文本内容")
    # 后续可扩展：file_url, file_base64 等


class IngestResponse(BaseModel):
    """文档入库响应。"""
    success: bool = True
    message: str = "入库成功"
    doc_count: int = 0


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str = "healthy"
    qdrant: Optional[str] = Field(None, description="Qdrant 连接状态：ok / error")
