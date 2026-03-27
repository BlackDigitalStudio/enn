"""
Agentic GraphRAG - Main Application
M2M Backend для автономного LLM-агента

API для навигации по графу кода без загрузки
полной кодовой базы в контекстное окно.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .api.routes import router, set_storage, set_vector_store
from .graph.storage import Neo4jStorage
from .llm.embeddings import VectorStore

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management - подключение к БД при старте"""
    settings = get_settings()

    # Подключение к Neo4j
    neo4j = Neo4jStorage(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password
    )

    connected = neo4j.connect()
    if connected:
        logger.info("Neo4j connected successfully")
        set_storage(neo4j)
    else:
        logger.warning("Neo4j connection failed - some features may not work")

    # Подключение к Qdrant
    vector_store = VectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection
    )
    if vector_store.connect():
        logger.info("Qdrant connected successfully")
        set_vector_store(vector_store)
    else:
        logger.warning("Qdrant connection failed - vector search unavailable")

    yield

    # Cleanup при выключении
    neo4j.close()
    logger.info("Shutting down...")


# Создание FastAPI app
app = FastAPI(
    title="Agentic GraphRAG",
    description="""
## Agentic GraphRAG - M2M API для LLM-агентов

Это backend-система для автономного LLM-агента, работающего с большими кодовыми базами.

### Ключевые концепции:
- **Графовая навигация** вместо загрузки всего кода в контекст
- **Context Pruning** - только метаданные при обходе графа
- **Atomic code fetch** - полный код только когда нужно редактировать
- **Incremental updates** - AST-парсинг только измененных файлов

### Архитектура:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Agent     │────▶│  GraphRAG    │────▶│   Neo4j     │
│   (LLM)     │◀────│   API        │◀────│   Graph     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Qdrant     │
                    │  (vector)   │
                    └─────────────┘
```

### Использование:
1. `POST /api/v1/ingest` - индексация кодовой базы
2. `GET /api/v1/subgraph/{node_id}` - навигация по графу
3. `GET /api/v1/node/{node_id}/code` - получение кода для редактирования
""",
    version="0.1.0",
    lifespan=lifespan
)

# CORS для development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(router)


@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "name": "Agentic GraphRAG",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api-spec")
async def api_spec():
    """
    Возвращает полную спецификацию API в JSON формате.
    Полезно для автоматической генерации клиентского кода.
    """
    return {
        "endpoints": {
            "health": {
                "GET /api/v1/health": {
                    "description": "Проверка статуса системы",
                    "response": "HealthResponse"
                }
            },
            "ingest": {
                "POST /api/v1/ingest": {
                    "description": "Индексация директории",
                    "body": "IngestRequest",
                    "response": "IngestResult"
                },
                "POST /api/v1/ingest/file": {
                    "description": "Индексация одного файла",
                    "query_params": {"file_path": "string"}
                }
            },
            "navigation": {
                "GET /api/v1/subgraph/{node_id}": {
                    "description": "Получение подграфа (Context Pruning)",
                    "params": {
                        "depth": "1-3 (default 2)",
                        "edge_types": "comma-separated filter",
                        "include_code": "boolean (default false)"
                    },
                    "response": "SubgraphResponse"
                },
                "GET /api/v1/node/{node_id}": {
                    "description": "Метаданные узла (без кода)",
                    "response": "NodeResponse"
                },
                "GET /api/v1/node/{node_id}/code": {
                    "description": "Полный исходный код узла",
                    "response": "CodeResponse"
                }
            },
            "search": {
                "POST /api/v1/search": {
                    "description": "Поиск узлов",
                    "body": "SearchRequest"
                }
            },
            "meta": {
                "GET /api/v1/stats": "Статистика графа",
                "GET /api/v1/types": "Допустимые типы узлов",
                "GET /api/v1/edge-types": "Допустимые типы ребер",
                "GET /api/v1/tags": "Допустимые теги"
            }
        },
        "models": {
            "NodeType": ["file", "class", "function", "method", "variable", "namespace", "enum", "struct", "interface", "document", "plan"],
            "EdgeType": ["CALLS", "INCLUDES", "IMPLEMENTS", "INHERITS", "DEPENDS_ON", "DEFINES", "HAS_METHOD", "RELATES_TO", "DESCRIBES"],
            "TagEnum": ["core", "utility", "config", "data", "network", "parser", "storage", "api", "middleware", "handler", "abstract", "interface", "concrete", "deprecated", "entry_point", "test", "mock", "business_logic", "infrastructure", "bounded_context"]
        }
    }


# Обработчик ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
