# IntelligentCustomerService

基于 **FastAPI + LlamaIndex + Qdrant + OpenAI 兼容接口** 的 RAG 智能客服系统。

## 技术栈

- **Web**: FastAPI
- **RAG**: LlamaIndex（索引与检索）
- **向量库**: Qdrant
- **LLM / Embedding**: OpenAI 兼容 API（可配置 `OPENAI_API_BASE` 使用第三方接口）

## 快速开始

### 1. 环境

- Python 3.10+
- 已运行的 Qdrant（本地或 Cloud）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

复制环境变量示例并填写密钥：

```bash
cp .env.example .env
# 编辑 .env：至少设置 OPENAI_API_KEY，以及 Qdrant 地址（若不用默认 localhost:6333）
```

### 4. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API 文档: http://localhost:8000/docs  
- 健康检查: http://localhost:8000/health  

### 5. 写入知识库并对话

- **写入文本到知识库**（用于 RAG 检索）:
  - `POST /api/ingest`，Body: `{"text": "你的客服知识库内容..."}`

- **非流式对话**:
  - `POST /api/chat`，Body: `{"message": "用户问题", "stream": false}`

- **流式对话**:
  - `POST /api/chat/stream`，Body: `{"message": "用户问题"}`，响应为 SSE 流

## 项目结构

```
app/
├── main.py           # FastAPI 入口
├── config/           # 配置（环境变量）
├── api/              # 路由：chat（RAG 客服）、stream（流式示例）
├── engine/           # RAG 引擎（LlamaIndex + Qdrant）
├── models/           # 请求/响应模型
├── core/             # 通用逻辑
└── data/             # 数据与文档（可扩展）
```

## 配置说明

| 变量 | 说明 | 默认 |
|------|------|------|
| `QDRANT_HOST` / `QDRANT_PORT` | Qdrant 地址 | localhost / 6333 |
| `QDRANT_COLLECTION` | 向量集合名 | rag_customer_service |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant Cloud 时使用 | - |
| `OPENAI_API_KEY` | LLM 与 Embedding 密钥 | 必填 |
| `OPENAI_API_BASE` | 兼容接口 base URL | 可选 |
| `OPENAI_CHAT_MODEL` / `OPENAI_EMBED_MODEL` | 模型名 | gpt-4o-mini / text-embedding-3-small |
| `RAG_TOP_K` | 检索条数 | 5 |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | 入库分块参数 | 512 / 64 |

## License

MIT
