"""
RAG 引擎：基于 LlamaIndex + Qdrant + OpenAI 兼容接口。

- 使用 Qdrant 作为向量库，LlamaIndex 构建索引与查询。
- LLM 与 Embedding 使用 OpenAI 兼容 API（可配置 base URL）。
- 支持非流式与流式查询。
"""
import logging
from typing import Any, AsyncIterator, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# 延迟导入，避免未安装依赖时启动报错
_qdrant_client: Any = None
_vector_store: Any = None
_index: Any = None
_query_engine: Any = None
_stream_query_engine: Any = None


def _get_qdrant_client():
    """获取或创建 Qdrant 客户端。"""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    try:
        from qdrant_client import QdrantClient
    except ImportError as e:
        raise ImportError("请安装: pip install qdrant-client") from e

    if settings.qdrant_url:
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    else:
        _qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
    return _qdrant_client


def _get_llm():
    """获取 LLM（OpenAI 兼容）。"""
    from llama_index.core.llms import ChatMessage, MessageRole
    from llama_index.llms.openai import OpenAI

    api_key = settings.openai_api_key or ""
    if not api_key.strip():
        raise ValueError("未配置 OPENAI_API_KEY，请在 .env 或环境变量中设置")

    return OpenAI(
        model=settings.openai_chat_model,
        api_key=api_key,
        api_base=settings.openai_api_base,
    )


def _get_embed_model():
    """获取 Embedding 模型（OpenAI 兼容）。"""
    from llama_index.embeddings.openai import OpenAIEmbedding

    api_key = settings.openai_api_key or ""
    if not api_key.strip():
        raise ValueError("未配置 OPENAI_API_KEY")

    return OpenAIEmbedding(
        model=settings.openai_embed_model,
        api_key=api_key,
        api_base=settings.openai_api_base,
    )


def _ensure_index():
    """确保向量索引与查询引擎已初始化。"""
    global _vector_store, _index, _query_engine, _stream_query_engine
    if _query_engine is not None:
        return

    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    client = _get_qdrant_client()
    _vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
    )
    storage_context = StorageContext.from_defaults(vector_store=_vector_store)
    embed_model = _get_embed_model()
    llm = _get_llm()

    # 从已有 collection 加载索引（不插入新文档）
    _index = VectorStoreIndex.from_vector_store(
        _vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    _query_engine = _index.as_query_engine(
        llm=llm,
        similarity_top_k=settings.rag_top_k,
    )
    _stream_query_engine = _index.as_query_engine(
        llm=llm,
        similarity_top_k=settings.rag_top_k,
        streaming=True,
    )
    logger.info("RAG 引擎已初始化（Qdrant + LlamaIndex）")


def query(message: str) -> tuple[str, List[dict]]:
    """
    非流式 RAG 查询。
    返回 (answer, sources)。
    """
    _ensure_index()
    response = _query_engine.query(message)
    sources = []
    if hasattr(response, "source_nodes") and response.source_nodes:
        for node in response.source_nodes:
            sources.append({
                "text": (node.node.text or "")[:500],
                "score": getattr(node, "score", None),
            })
    return (response.response or "").strip(), sources


async def query_stream(message: str) -> AsyncIterator[tuple[str, bool, Optional[List[dict]]]]:
    """
    流式 RAG 查询。
    产出 (chunk, done, sources)。
    done=True 时 sources 可能非空。
    """
    import asyncio
    from queue import Queue, Empty

    _ensure_index()
    result_queue: Queue = Queue()
    sources_ref: List[dict] = []

    def _run_stream():
        try:
            response = _stream_query_engine.query(message)
            for chunk in response.response_gen:
                result_queue.put(("chunk", chunk, None))
            if hasattr(response, "source_nodes") and response.source_nodes:
                sources_ref.extend([
                    {"text": (n.node.text or "")[:500], "score": getattr(n, "score", None)}
                    for n in response.source_nodes
                ])
        except Exception as e:
            logger.exception("流式查询异常: %s", e)
            result_queue.put(("error", str(e), None))
        result_queue.put(("done", None, None))

    loop = asyncio.get_event_loop()
    fut = loop.run_in_executor(None, _run_stream)
    while True:
        try:
            kind, chunk, _ = result_queue.get_nowait()
        except Empty:
            await asyncio.sleep(0.02)
            continue
        if kind == "chunk":
            yield chunk, False, None
        elif kind == "error":
            yield chunk, True, None
            break
        elif kind == "done":
            yield "", True, sources_ref if sources_ref else None
            break
    await fut


def ingest_text(text: str) -> int:
    """
    将一段文本切分并写入 Qdrant。
    返回写入的节点数（或文档数）。
    """
    from llama_index.core import VectorStoreIndex, Document, StorageContext
    from llama_index.core.node_parser import SentenceSplitter

    _get_qdrant_client()
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    vector_store = QdrantVectorStore(
        client=_get_qdrant_client(),
        collection_name=settings.qdrant_collection,
    )
    embed_model = _get_embed_model()
    parser = SentenceSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    documents = [Document(text=text)]
    nodes = parser.get_nodes_from_documents(documents)
    if not nodes:
        return 0

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    return len(nodes)


def check_qdrant() -> str:
    """检查 Qdrant 连接是否正常。返回 'ok' 或 'error'。"""
    try:
        c = _get_qdrant_client()
        c.get_collections()
        return "ok"
    except Exception as e:
        logger.debug("Qdrant 检查失败: %s", e)
        return "error"


def reset_engine():
    """重置全局引擎（用于测试或重载配置）。"""
    global _qdrant_client, _vector_store, _index, _query_engine, _stream_query_engine
    _qdrant_client = None
    _vector_store = None
    _index = None
    _query_engine = None
    _stream_query_engine = None
