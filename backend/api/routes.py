"""
Agentic GraphRAG - API Routes
FastAPI эндпоинты для взаимодействия с LLM-агентом
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from ..graph.models import GraphNode, GraphEdge, SubgraphResult, IngestResult, VirtualPatch
from ..graph.storage import Neo4jStorage
from ..parser.cpp_parser import CPPSymbolExtractor, scan_directory
from ..parser.md_parser import MarkdownParser, resolve_mentions_to_edges
from ..llm.embeddings import VectorStore, get_embedding, get_embeddings_batch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["graph"])

# Storage instance — устанавливается из main.py через set_storage()
_storage: Optional[Neo4jStorage] = None


def set_storage(s: Neo4jStorage):
    """Вызывается из main.py при старте приложения"""
    global _storage
    _storage = s


def get_storage() -> Neo4jStorage:
    if _storage is None or _storage._driver is None:
        raise HTTPException(status_code=503, detail="Neo4j not connected")
    return _storage


# Vector store instance
_vector_store: Optional[VectorStore] = None


def set_vector_store(vs: VectorStore):
    global _vector_store
    _vector_store = vs


def get_vector_store() -> VectorStore:
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Qdrant not connected")
    return _vector_store


# ============== Request/Response Models ==============

class IngestRequest(BaseModel):
    """Запрос на индексацию файлов"""
    directory: str = Field(..., description="Путь к директории для индексации")
    extensions: List[str] = Field(default=[".cpp", ".h", ".hpp", ".cc"], description="Расширения файлов")
    recursive: bool = Field(default=True, description="Рекурсивный обход")


class NodeResponse(BaseModel):
    """Ответ с данными узла"""
    node_id: str
    type: str
    name: str
    signature: str
    file_path: str
    line_start: int
    line_end: int
    summary: str
    tags: List[str]
    source_code: Optional[str] = None


class SubgraphRequest(BaseModel):
    """Запрос подграфа"""
    start_node_id: str
    depth: int = Field(default=2, ge=1, le=3)
    edge_types: Optional[List[str]] = None
    virtual_patches: Optional[List[Dict[str, str]]] = None  # Shadow Graph (Фаза B)


class SubgraphResponse(BaseModel):
    """Ответ с подграфом"""
    center_node: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    depth: int
    total_nodes: int
    total_edges: int


class CodeResponse(BaseModel):
    """Ответ с исходным кодом"""
    node_id: str
    source_code: str
    file_path: str
    line_start: int
    line_end: int


class SearchRequest(BaseModel):
    """Запрос поиска"""
    query: Optional[str] = None
    node_type: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = Field(default=50, ge=1, le=100)


class VectorSearchRequest(BaseModel):
    """Запрос векторного поиска (входная точка для агента)"""
    query: str = Field(..., description="Текстовый промпт для поиска")
    node_type: Optional[str] = Field(None, description="Фильтр по типу узла")
    limit: int = Field(default=5, ge=1, le=20)


class ScratchpadEntry(BaseModel):
    """Запись в Scratchpad агента"""
    session_id: str = Field(..., description="ID сессии агента")
    content: str = Field(..., description="Текст записи (план, заметки)")
    node_ids: List[str] = Field(default=[], description="Привязанные node_id")


class VirtualPatchRequest(BaseModel):
    """Запрос на создание виртуального патча"""
    session_id: str
    node_id: str
    file_path: str
    old_code: str
    new_code: str


class StatsResponse(BaseModel):
    """Статистика графа"""
    total_nodes: int
    total_edges: int
    node_types: List[str]
    type_count: int


class HealthResponse(BaseModel):
    """Health check"""
    status: str
    neo4j_connected: bool
    llm_provider: str
    llm_model: str
    llm_configured: bool
    version: str


class LLMTestRequest(BaseModel):
    """Запрос тестирования LLM"""
    prompt: str = Field(..., description="Промпт для LLM")
    system: str = Field(default="", description="Системный промпт")


# ============== API Endpoints ==============

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check - проверка статуса системы"""
    from ..config import get_settings
    settings = get_settings()

    neo4j_ok = False
    if _storage and _storage._driver:
        try:
            _storage._driver.verify_connectivity()
            neo4j_ok = True
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if neo4j_ok else "degraded",
        neo4j_connected=neo4j_ok,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_configured=bool(settings.llm_api_key),
        version="0.1.0"
    )


@router.post("/llm/test")
async def test_llm(request: LLMTestRequest):
    """Тест LLM — отправляет промпт и возвращает ответ модели"""
    from ..llm.client import get_llm_client
    client = get_llm_client()

    if not client.api_key:
        raise HTTPException(status_code=503, detail="LLM API key not configured")

    try:
        response = await client.generate(request.prompt, request.system)
        return {
            "provider": client.provider,
            "model": client.model,
            "response": response,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@router.post("/ingest", response_model=IngestResult)
async def ingest_directory(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Индексация директории с исходным кодом и документацией.

    Шаг 1: Парсит C++ файлы (Tree-sitter), создаёт узлы кода и рёбра.
    Шаг 2: Парсит .md файлы, создаёт DOCUMENT узлы.
    Шаг 3: Автолинковка — привязывает документы к коду через DESCRIBES рёбра.
    """
    s = get_storage()
    cpp_parser = CPPSymbolExtractor(request.directory)
    md_parser = MarkdownParser(request.directory)

    all_nodes = []
    all_edges = []
    errors = []
    files_count = 0

    # --- Шаг 1: C++ файлы ---
    cpp_files = scan_directory(request.directory, request.extensions)
    for file_path in cpp_files:
        try:
            result = cpp_parser.parse_file(file_path)
            if result.success:
                all_nodes.extend(result.nodes)
                all_edges.extend(result.edges)
            else:
                errors.append(f"{file_path}: {result.error}")
        except Exception as e:
            errors.append(f"{file_path}: {str(e)}")
    files_count += len(cpp_files)

    # --- Шаг 2: Markdown файлы ---
    md_files = scan_directory(request.directory, ['.md'])
    md_mentions = []  # (doc_node_id, mentions) для шага 3
    for file_path in md_files:
        try:
            result = md_parser.parse_file(file_path)
            if result.success and result.node:
                all_nodes.append(result.node)
                md_mentions.append((result.node.node_id, result.mentions))
            else:
                errors.append(f"{file_path}: {result.error}")
        except Exception as e:
            errors.append(f"{file_path}: {str(e)}")
    files_count += len(md_files)

    if not all_nodes:
        return IngestResult(
            success=False, files_processed=0,
            nodes_created=0, edges_created=0,
            errors=[f"No files found in {request.directory}"]
        )

    # Bulk insert узлов и рёбер кода
    s.bulk_create_nodes(all_nodes)
    if all_edges:
        s.bulk_create_edges(all_edges)

    # --- Шаг 3: Автолинковка документов к коду ---
    # Берём все узлы кода (не документы) из графа для матчинга
    code_nodes = [n for n in all_nodes if n.type != "document"]
    describe_edges = []
    for doc_node_id, mentions in md_mentions:
        edges = resolve_mentions_to_edges(doc_node_id, mentions, code_nodes)
        describe_edges.extend(edges)

    if describe_edges:
        s.bulk_create_edges(describe_edges)
        all_edges.extend(describe_edges)

    return IngestResult(
        success=len(errors) == 0,
        files_processed=files_count,
        nodes_created=len(all_nodes),
        edges_created=len(all_edges),
        errors=errors
    )


@router.post("/ingest/file", response_model=IngestResult)
async def ingest_file(file_path: str):
    """
    Индексация одного файла.
    """
    s = get_storage()
    parser = CPPSymbolExtractor(file_path)
    
    try:
        result = parser.parse_file(file_path)
        
        if result.success:
            if result.nodes:
                s.bulk_create_nodes(result.nodes)
            if result.edges:
                s.bulk_create_edges(result.edges)
            
            return IngestResult(
                success=True,
                files_processed=1,
                nodes_created=len(result.nodes),
                edges_created=len(result.edges)
            )
        else:
            return IngestResult(
                success=False,
                files_processed=0,
                nodes_created=0,
                edges_created=0,
                errors=[result.error]
            )
    except Exception as e:
        return IngestResult(
            success=False,
            files_processed=0,
            nodes_created=0,
            edges_created=0,
            errors=[str(e)]
        )


@router.get("/subgraph/{node_id}", response_model=SubgraphResponse)
async def get_subgraph(
    node_id: str,
    depth: int = Query(default=2, ge=1, le=3),
    edge_types: Optional[str] = Query(default=None, description="Comma-separated edge types"),
    include_code: bool = Query(default=False, description="Include source code")
):
    """
    Получение подграфа вокруг узла.
    
    ВАЖНО: source_code НЕ включается по умолчанию.
    Это экономит токены LLM при навигации.
    Для получения кода используйте /node/{node_id}/code
    """
    s = get_storage()
    
    # Парсинг типов ребер
    edge_filter = None
    if edge_types:
        edge_filter = [t.strip() for t in edge_types.split(",")]
    
    subgraph = s.get_subgraph(node_id, depth, edge_filter, include_code)
    
    if not subgraph:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    return subgraph.to_dict(include_code)


@router.get("/node/{node_id}", response_model=NodeResponse)
async def get_node(node_id: str):
    """Получение метаданных узла (без source_code)"""
    s = get_storage()
    node = s.get_node(node_id)
    
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    return NodeResponse(**node.to_api_dict(include_code=False))


@router.get("/node/{node_id}/code", response_model=CodeResponse)
async def get_node_code(node_id: str):
    """
    Получение полного исходного кода узла.
    
    Агент вызывает этот эндпоинт ТОЛЬКО когда принял решение
    редактировать конкретный узел.
    """
    s = get_storage()
    node = s.get_node(node_id)
    
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    return CodeResponse(
        node_id=node.node_id,
        source_code=node.source_code,
        file_path=node.file_path,
        line_start=node.line_start,
        line_end=node.line_end
    )


@router.post("/search", response_model=List[NodeResponse])
async def search_nodes(request: SearchRequest):
    """
    Поиск узлов по тексту или фильтрам.
    
    Используется как fallback если векторный поиск (Milvus/Qdrant) недоступен.
    """
    s = get_storage()
    nodes = s.search_nodes(
        query=request.query,
        node_type=request.node_type,
        tags=request.tags,
        limit=request.limit
    )
    
    return [NodeResponse(**n.to_api_dict(include_code=False)) for n in nodes]


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Получение статистики графа"""
    s = get_storage()
    stats = s.get_stats()
    return StatsResponse(**stats)


@router.delete("/clear")
async def clear_graph():
    """Очистка всего графа и векторного хранилища (для тестирования)"""
    s = get_storage()
    s.clear_all()
    if _vector_store:
        _vector_store.delete_all()
    _virtual_patches.clear()
    _scratchpads.clear()
    return {"message": "Graph, vectors, patches and scratchpads cleared"}


@router.get("/types")
async def get_node_types():
    """Получение списка допустимых типов узлов"""
    from ..graph.models import NodeType
    return [t.value for t in NodeType]


@router.get("/edge-types")
async def get_edge_types():
    """Получение списка допустимых типов ребер"""
    from ..graph.models import EdgeType
    return [t.value for t in EdgeType]


@router.get("/tags")
async def get_allowed_tags():
    """Получение списка допустимых тегов"""
    from ..graph.models import TagEnum
    return [t.value for t in TagEnum]


# ============== Phase C: Vector Search ==============

@router.post("/search/vector")
async def vector_search(request: VectorSearchRequest):
    """
    Векторный поиск — ВХОДНАЯ ТОЧКА для агента.

    Агент отправляет текстовый промпт → получает стартовые node_id
    для дальнейшей навигации по графу через get_subgraph.
    """
    vs = get_vector_store()

    # Генерируем эмбеддинг запроса
    query_embedding = await get_embedding(request.query)
    if not query_embedding:
        raise HTTPException(status_code=503, detail="Failed to generate embedding")

    results = vs.search(
        query_vector=query_embedding,
        limit=request.limit,
        node_type=request.node_type
    )

    return {
        "query": request.query,
        "results": results,
        "count": len(results)
    }


@router.post("/index/embeddings")
async def index_embeddings():
    """
    Генерирует эмбеддинги для всех узлов графа и сохраняет в Qdrant.
    Вызывается после ingest, чтобы активировать векторный поиск.
    """
    s = get_storage()
    vs = get_vector_store()

    # Забираем все узлы из Neo4j
    all_nodes = s.search_nodes(limit=1000)
    if not all_nodes:
        return {"indexed": 0, "message": "No nodes in graph"}

    # Формируем текст для эмбеддинга: name + summary + type
    texts = []
    for node in all_nodes:
        text = f"{node.type}: {node.name}"
        if node.summary:
            text += f" - {node.summary}"
        if node.signature:
            text += f" ({node.signature})"
        texts.append(text)

    # Батч-генерация эмбеддингов через Gemini
    embeddings = await get_embeddings_batch(texts)

    # Сохраняем в Qdrant
    items = []
    for node, emb in zip(all_nodes, embeddings):
        if emb is None:
            continue
        items.append({
            "node_id": node.node_id,
            "embedding": emb,
            "payload": {
                "type": node.type,
                "name": node.name,
                "summary": node.summary,
                "file_path": node.file_path,
            }
        })

    count = vs.upsert_batch(items)
    return {"indexed": count, "total_nodes": len(all_nodes)}


# ============== Phase C: LLM Summary Generation ==============

@router.post("/generate/summaries")
async def generate_summaries():
    """
    Генерирует summary для всех узлов без summary через LLM.
    Выполняется при индексации, НЕ при навигации.
    """
    from ..llm.client import get_llm_client
    s = get_storage()
    client = get_llm_client()

    if not client.api_key:
        raise HTTPException(status_code=503, detail="LLM not configured")

    all_nodes = s.search_nodes(limit=1000)
    updated = 0

    for node in all_nodes:
        # Пропускаем если summary уже осмысленный (не автогенерированный шаблон)
        if node.summary and not node.summary.startswith(("Class ", "Function ", "Document")):
            continue
        if not node.source_code:
            continue

        code_snippet = node.source_code[:500]  # Первые 500 символов
        prompt = (
            f"Describe this {node.type} in one sentence (max 100 chars). "
            f"Name: {node.name}\n\nCode:\n{code_snippet}"
        )

        summary = await client.generate(prompt, system="Be concise. Respond with only the description.")
        if summary:
            summary = summary.strip()[:200]
            node.summary = summary
            s.update_node(node)
            updated += 1

    return {"updated": updated, "total_nodes": len(all_nodes)}


# ============== Phase C: Shadow Graph (Virtual Patches) ==============

# In-memory хранилище патчей (per-session)
_virtual_patches: Dict[str, List[Dict[str, Any]]] = {}


@router.post("/patches")
async def create_patch(request: VirtualPatchRequest):
    """Сохранение виртуального патча (незакоммиченное изменение)"""
    from ..graph.models import VirtualPatch

    patch = VirtualPatch(
        node_id=request.node_id,
        file_path=request.file_path,
        old_code=request.old_code,
        new_code=request.new_code,
        session_id=request.session_id,
    )

    if request.session_id not in _virtual_patches:
        _virtual_patches[request.session_id] = []
    _virtual_patches[request.session_id].append(patch.to_dict())

    return {"status": "stored", "session_id": request.session_id,
            "patches_count": len(_virtual_patches[request.session_id])}


@router.get("/patches/{session_id}")
async def get_patches(session_id: str):
    """Получение всех патчей сессии"""
    patches = _virtual_patches.get(session_id, [])
    return {"session_id": session_id, "patches": patches}


@router.delete("/patches/{session_id}")
async def clear_patches(session_id: str):
    """Удаление патчей сессии (после коммита)"""
    removed = len(_virtual_patches.pop(session_id, []))
    return {"session_id": session_id, "removed": removed}


@router.get("/node/{node_id}/code/shadow")
async def get_node_code_with_patches(node_id: str, session_id: str = Query(...)):
    """
    Получение кода узла С НАЛОЖЕННЫМИ виртуальными патчами.

    Агент видит актуальную версию кода, даже если изменения
    ещё не записаны в основную БД.
    """
    s = get_storage()
    node = s.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    code = node.source_code
    patches = _virtual_patches.get(session_id, [])

    # Применяем патчи для этого узла последовательно
    for patch in patches:
        if patch["node_id"] == node_id:
            code = code.replace(patch["old_code"], patch["new_code"])

    return {
        "node_id": node.node_id,
        "source_code": code,
        "file_path": node.file_path,
        "patched": code != node.source_code,
        "session_id": session_id,
    }


# ============== Phase C: Scratchpad ==============

# In-memory scratchpad (per-session)
_scratchpads: Dict[str, List[Dict[str, Any]]] = {}


@router.post("/scratchpad")
async def write_scratchpad(entry: ScratchpadEntry):
    """
    Запись в Scratchpad агента.

    Агент использует внешний буфер вместо хранения
    архитектуры в контекстном окне.
    """
    from datetime import datetime

    record = {
        "content": entry.content,
        "node_ids": entry.node_ids,
        "created_at": datetime.utcnow().isoformat(),
    }

    if entry.session_id not in _scratchpads:
        _scratchpads[entry.session_id] = []
    _scratchpads[entry.session_id].append(record)

    return {
        "session_id": entry.session_id,
        "entry_index": len(_scratchpads[entry.session_id]) - 1,
        "status": "stored",
    }


@router.get("/scratchpad/{session_id}")
async def read_scratchpad(session_id: str):
    """Чтение всего Scratchpad сессии"""
    entries = _scratchpads.get(session_id, [])
    return {
        "session_id": session_id,
        "entries": entries,
        "count": len(entries),
    }


@router.delete("/scratchpad/{session_id}")
async def clear_scratchpad(session_id: str):
    """Очистка Scratchpad сессии"""
    removed = len(_scratchpads.pop(session_id, []))
    return {"session_id": session_id, "removed": removed}
