"""
应用配置：从环境变量读取，用于 RAG 客服系统。

需配置项示例（.env 或环境变量）：
    QDRANT_HOST=localhost
    QDRANT_PORT=6333
    QDRANT_COLLECTION=rag_customer_service
    OPENAI_API_KEY=sk-xxx
    OPENAI_API_BASE=https://api.openai.com/v1   # 可选，兼容其他 OpenAI 兼容接口
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_customer_service"
    qdrant_api_key: Optional[str] = None
    qdrant_url: Optional[str] = None  # 若设置则优先使用 URL（如 https://xxx.qdrant.io）

    # OpenAI 兼容接口（用于 LLM + Embedding）
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # RAG 行为
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.5
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64

    # Database (MySQL async connection pool)
    database_url: Optional[str] = None  # 若设置则优先使用，否则用下面字段拼接
    database_host: str = "localhost"
    database_port: int = 3306
    database_user: str = "root"
    database_password: str = ""
    database_name: str = "customer_service"
    database_charset: str = "utf8mb4"
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_recycle: int = 3600
    database_echo: bool = False

    def get_database_url(self) -> str:
        """MySQL connection URL for aiomysql."""
        if self.database_url:
            return self.database_url
        pwd = f":{self.database_password}" if self.database_password else ""
        return (
            f"mysql+aiomysql://{self.database_user}{pwd}@"
            f"{self.database_host}:{self.database_port}/{self.database_name}?charset={self.database_charset}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


# 默认导出单例，便于各处直接 from app.config import settings
settings = get_settings()
