"""
Agentic GraphRAG - API Routes
FastAPI эндпоинты для взаимодействия с LLM-агентом
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import os
import zipfile
import tempfile
import shutil
import asyncio
import hashlib

from ..graph.models import GraphNode, GraphEdge, SubgraphResult, IngestResult, VirtualPatch
from ..graph.storage import Neo4jStorage
from ..parser.txt_converter import scan_and_filter
from ..llm.embeddings import VectorStore, get_embedding, get_embeddings_batch
from ..llm.entity_extractor import extract_entities_batch, resolve_entities

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


@router.post("/ingest")
async def ingest_directory(request: IngestRequest):
    """Legacy endpoint — redirects to /pipeline."""
    from fastapi.responses import RedirectResponse
    return {"message": "Use POST /api/v1/pipeline instead", "redirect": "/api/v1/pipeline"}


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


# ============== Upload & Auto-Pipeline ==============

UPLOAD_DIR = "/app/uploads"


@router.post("/upload")
async def upload_project(file: UploadFile = File(...)):
    """
    Загрузка проекта через браузер (ZIP-архив или отдельные файлы).
    Распаковывает в /app/uploads/{project_name}/ внутри контейнера.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Определяем имя проекта
    original = file.filename or "project"
    project_name = os.path.splitext(original)[0]
    project_dir = os.path.join(UPLOAD_DIR, project_name)

    # Читаем файл
    content = await file.read()

    if original.endswith('.zip'):
        # ZIP — распаковываем
        os.makedirs(project_dir, exist_ok=True)
        tmp = os.path.join(UPLOAD_DIR, original)
        with open(tmp, 'wb') as f:
            f.write(content)
        with zipfile.ZipFile(tmp, 'r') as z:
            z.extractall(project_dir)
        os.remove(tmp)
    else:
        # Одиночный файл
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(project_dir, original), 'wb') as f:
            f.write(content)

    # Сканируем и фильтруем
    scan = scan_and_filter(project_dir)

    return {
        "project_name": project_name,
        "project_dir": project_dir,
        "scan": scan["stats"],
        "cpp_files": len(scan["cpp_files"]),
        "doc_files": len(scan["doc_files"]),
        "txt_files": len(scan["txt_files"]),
        "other_text": len(scan.get("other_text", [])),
    }


def _split_into_chunks(text: str, max_size: int = 2000) -> List[str]:
    """Split text into chunks ≤ max_size, breaking on paragraph boundaries."""
    if len(text) <= max_size:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current = ""

    for para in paragraphs:
        # If single paragraph exceeds max_size, split by newlines then by chars
        if len(para) > max_size:
            if current:
                chunks.append(current)
                current = ""
            lines = para.split('\n')
            for line in lines:
                if len(current) + len(line) + 1 > max_size:
                    if current:
                        chunks.append(current)
                    # If single line exceeds max_size, hard split
                    while len(line) > max_size:
                        chunks.append(line[:max_size])
                        line = line[max_size:]
                    current = line
                else:
                    current = current + '\n' + line if current else line
        elif len(current) + len(para) + 2 > max_size:
            if current:
                chunks.append(current)
            current = para
        else:
            current = current + '\n\n' + para if current else para

    if current.strip():
        chunks.append(current)

    return chunks if chunks else [text[:max_size]]


class PipelineRequest(BaseModel):
    directory: Optional[str] = None
    project_name: Optional[str] = None


@router.post("/pipeline")
async def auto_pipeline(request: PipelineRequest):
    """
    Universal pipeline: scan → document nodes → entity extraction → embeddings.

    4 clean steps. No AST parsers, no MD parsers.
    Entity extraction is THE CORE — DeepSeek reads every document
    and extracts entities, facts, topics, skills, preferences + edges.
    """
    from ..llm.client import get_llm_client
    import time as _time
    import math

    pipeline_start = _time.time()

    directory = request.directory
    project_name = request.project_name
    if project_name:
        directory = os.path.join(UPLOAD_DIR, project_name)
    if not directory or not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    s = get_storage()
    vs = get_vector_store()
    client = get_llm_client()
    steps = []
    errors = []

    # ================================================================
    # STEP 1: Scan & Filter — find all text files
    # ================================================================
    step_start = _time.time()
    scan = scan_and_filter(directory)
    all_files = scan["cpp_files"] + scan["doc_files"] + scan["txt_files"] + scan.get("other_text", [])
    steps.append({
        "step": "scan",
        "time_s": round(_time.time() - step_start, 2),
        "total_files": len(all_files),
        "stats": scan["stats"],
    })
    logger.info(f"Step 1 (scan): {len(all_files)} files found")

    # ================================================================
    # STEP 2: Create document chunks — each file split into ≤2000 char chunks
    # Full content preserved, no truncation.
    # ================================================================
    step_start = _time.time()
    doc_nodes = []
    chunk_edges = []
    CHUNK_SIZE = 2000

    for fp in all_files:
        try:
            with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            if len(content.strip()) < 10:
                continue

            rel_path = os.path.relpath(fp, directory) if directory else fp
            file_hash = hashlib.sha256(rel_path.encode()).hexdigest()[:16]
            name = os.path.basename(fp)

            # Split into chunks on paragraph boundaries
            chunks = _split_into_chunks(content, CHUNK_SIZE)

            prev_node_id = None
            for ci, chunk_text in enumerate(chunks):
                node_id = f"doc::{file_hash}::{ci}"
                doc_nodes.append(GraphNode(
                    node_id=node_id,
                    type="document",
                    name=f"{name}::chunk_{ci}",
                    signature=rel_path,
                    file_path=rel_path,
                    line_start=0,
                    line_end=0,
                    source_code=chunk_text,
                    summary=f"{name} [{ci+1}/{len(chunks)}]",
                    tags=[],
                ))
                # NEXT_CHUNK edge for ordering
                if prev_node_id:
                    chunk_edges.append(GraphEdge(
                        source_id=prev_node_id,
                        target_id=node_id,
                        edge_type="NEXT_CHUNK",
                    ))
                prev_node_id = node_id
        except Exception as e:
            errors.append(f"{fp}: {e}")

    if doc_nodes:
        s.bulk_create_nodes(doc_nodes)
    if chunk_edges:
        s.bulk_create_edges(chunk_edges)

    steps.append({
        "step": "create_chunks",
        "time_s": round(_time.time() - step_start, 2),
        "files": len(all_files),
        "chunks_created": len(doc_nodes),
        "chunk_edges": len(chunk_edges),
        "errors": len(errors),
    })
    logger.info(f"Step 2 (chunks): {len(all_files)} files → {len(doc_nodes)} chunks")

    # ================================================================
    # STEP 3: Entity Extraction — THE CORE
    # DeepSeek reads each document → extracts entities + edges
    # ================================================================
    step_start = _time.time()
    entity_nodes = []
    entity_edges = []
    extraction_api_calls = 0
    extraction_tokens = {"input": 0, "output": 0}

    if client.api_key and doc_nodes:
        # Prepare items for batch extraction
        extraction_items = []
        for node in doc_nodes:
            content = node.source_code or ""
            if len(content.strip()) < 20:
                continue
            extraction_items.append({
                "node_id": node.node_id,
                "name": node.name,
                "type": node.type,
                "content": content,
            })

        if extraction_items:
            batch_size = 10
            concurrency = 20
            logger.info(f"Step 3 (extraction): {len(extraction_items)} items, batch_size={batch_size}, concurrency={concurrency}")

            raw_entities, raw_edges = await extract_entities_batch(
                client, extraction_items, batch_size=batch_size, concurrency=concurrency
            )
            extraction_api_calls = math.ceil(len(extraction_items) / batch_size)
            logger.info(
                f"Extraction done: {len(raw_entities)} entities, {len(raw_edges)} edges"
            )

            # Entity resolution — deduplicate by normalized name
            name_map = resolve_entities(raw_entities)

            # Convert ExtractedEntity → GraphNode
            seen_entity_ids = set()
            for ent in raw_entities:
                canonical = name_map.get(ent.name, ent.name)
                ent_node_id = f"{ent.type}::{canonical}"

                if ent_node_id in seen_entity_ids:
                    continue
                seen_entity_ids.add(ent_node_id)

                entity_nodes.append(GraphNode(
                    node_id=ent_node_id,
                    type=ent.type,
                    name=canonical,
                    signature="",
                    file_path="",
                    line_start=0,
                    line_end=0,
                    source_code="",
                    summary=ent.summary,
                    tags=[],
                ))

            if entity_nodes:
                s.bulk_create_nodes(entity_nodes)

            # Create edges: document → entity (MENTIONS)
            for ent in raw_entities:
                canonical = name_map.get(ent.name, ent.name)
                ent_node_id = f"{ent.type}::{canonical}"
                if ent.source_node_id:
                    entity_edges.append(GraphEdge(
                        source_id=ent.source_node_id,
                        target_id=ent_node_id,
                        edge_type="MENTIONS",
                        metadata={"confidence": ent.confidence},
                    ))

            # Create edges: entity → entity (semantic)
            entity_by_name = {n.name: n.node_id for n in entity_nodes}
            for ed in raw_edges:
                src_canonical = name_map.get(ed.source_name, ed.source_name)
                tgt_canonical = name_map.get(ed.target_name, ed.target_name)
                src_id = entity_by_name.get(src_canonical)
                tgt_id = entity_by_name.get(tgt_canonical)
                if src_id and tgt_id:
                    entity_edges.append(GraphEdge(
                        source_id=src_id,
                        target_id=tgt_id,
                        edge_type=ed.edge_type,
                        metadata={"weight": ed.weight},
                    ))

            if entity_edges:
                s.bulk_create_edges(entity_edges)

    steps.append({
        "step": "entity_extraction",
        "time_s": round(_time.time() - step_start, 2),
        "entity_nodes": len(entity_nodes),
        "entity_edges": len(entity_edges),
        "api_calls": extraction_api_calls,
    })
    logger.info(f"Step 3 (extraction): {len(entity_nodes)} entities, {len(entity_edges)} edges")

    # ================================================================
    # STEP 4: Embeddings — vectorize ALL nodes (documents + entities)
    # ================================================================
    step_start = _time.time()
    all_nodes = doc_nodes + entity_nodes
    emb_count = 0

    if all_nodes:
        texts = [
            n.source_code if n.type == "document" and n.source_code
            else f"{n.type}: {n.name} - {n.summary}"
            for n in all_nodes
        ]
        embeddings = await get_embeddings_batch(texts)
        items = []
        for node, emb in zip(all_nodes, embeddings):
            if emb:
                items.append({
                    "node_id": node.node_id,
                    "embedding": emb,
                    "payload": {
                        "type": node.type, "name": node.name,
                        "summary": node.summary, "file_path": node.file_path,
                    }
                })
        emb_count = vs.upsert_batch(items)

    steps.append({
        "step": "embeddings",
        "time_s": round(_time.time() - step_start, 2),
        "indexed": emb_count,
    })
    logger.info(f"Step 4 (embeddings): {emb_count} vectors indexed")

    total_time = round(_time.time() - pipeline_start, 2)
    return {
        "success": True,
        "project_dir": directory,
        "total_time_s": total_time,
        "total_files": len(all_files),
        "total_nodes": len(all_nodes),
        "total_edges": len(entity_edges),
        "steps": steps,
        "errors": errors[:10],
    }


# ============== Bulk Operations (N+1 fix) ==============

@router.post("/nodes/bulk")
async def get_nodes_bulk(node_ids: List[str]):
    """
    Получение метаданных нескольких узлов за 1 запрос.

    Решение N+1 для навигации: агент запрашивает 50 узлов
    одним вызовом вместо 50 отдельных GET /node/{id}.
    """
    s = get_storage()
    nodes = s.get_nodes_bulk(node_ids)
    return [n.to_api_dict(include_code=False) for n in nodes]


@router.post("/nodes/bulk/code")
async def get_nodes_code_bulk(node_ids: List[str]):
    """
    Получение кода нескольких узлов за 1 запрос.
    Для случаев когда агент уже решил что ему нужны конкретные файлы.
    """
    s = get_storage()
    nodes = s.get_nodes_bulk(node_ids)
    return [
        {
            "node_id": n.node_id,
            "source_code": n.source_code,
            "file_path": n.file_path,
            "line_start": n.line_start,
            "line_end": n.line_end,
        }
        for n in nodes
    ]


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
    Генерирует эмбеддинги для ВСЕХ узлов графа и сохраняет в Qdrant.
    Обрабатывает порциями по 500 узлов чтобы не переполнить RAM.
    """
    s = get_storage()
    vs = get_vector_store()

    # Считаем общее кол-во узлов
    total_in_graph = s.count_nodes()
    if total_in_graph == 0:
        return {"indexed": 0, "message": "No nodes in graph"}

    total_indexed = 0
    chunk_size = 500
    offset = 0

    while offset < total_in_graph:
        # Забираем порцию узлов
        nodes_chunk = s.search_nodes(limit=chunk_size, skip=offset)
        if not nodes_chunk:
            break

        # Формируем текст для эмбеддинга — документы по содержимому, остальное по метаданным
        texts = []
        for node in nodes_chunk:
            if node.type == "document" and node.source_code:
                texts.append(node.source_code[:500])
            else:
                text = f"{node.type}: {node.name}"
                if node.summary:
                    text += f" - {node.summary}"
                texts.append(text)

        # Батч-генерация эмбеддингов через Gemini
        embeddings = await get_embeddings_batch(texts)

        # Сохраняем в Qdrant
        items = []
        for node, emb in zip(nodes_chunk, embeddings):
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
        total_indexed += count
        offset += len(nodes_chunk)
        logger.info(f"Embeddings: {total_indexed}/{total_in_graph} indexed")

    return {"indexed": total_indexed, "total_nodes": total_in_graph}


# ============== Phase C: LLM Summary Generation ==============

@router.post("/generate/summaries")
async def generate_summaries():
    """
    Батч-генерация summary для всех узлов через LLM.

    Решение N+1: вместо 1 API-вызова на узел отправляем
    пачки по 30 узлов за 1 вызов. 100K узлов = ~3.3K вызовов
    вместо 100K (30x экономия токенов и времени).
    """
    from ..llm.client import get_llm_client
    s = get_storage()
    client = get_llm_client()

    if not client.api_key:
        raise HTTPException(status_code=503, detail="LLM not configured")

    all_nodes = s.search_nodes(limit=100000)

    # Фильтруем: только узлы без осмысленного summary и с кодом
    needs_summary = [
        n for n in all_nodes
        if n.source_code
        and (not n.summary or n.summary.startswith(("Class ", "Function ", "Document")))
    ]

    if not needs_summary:
        return {"updated": 0, "total_nodes": len(all_nodes), "api_calls": 0}

    # Батч-генерация: 30 узлов за 1 LLM-вызов
    summaries = await client.batch_summarize(needs_summary, batch_size=30)

    # Запись в Neo4j
    updated = 0
    for node in needs_summary:
        if node.node_id in summaries:
            node.summary = summaries[node.node_id]
            s.update_node(node)
            updated += 1

    import math
    api_calls = math.ceil(len(needs_summary) / 30)

    return {
        "updated": updated,
        "total_nodes": len(all_nodes),
        "api_calls": api_calls,
        "savings": f"{len(needs_summary)} nodes in {api_calls} calls instead of {len(needs_summary)}"
    }


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


# ============== Agent Query (RAG) ==============

class AgentQueryRequest(BaseModel):
    question: str = Field(..., description="Вопрос к агенту")
    top_k: int = Field(default=10, description="Кол-во узлов из vector search")
    depth: int = Field(default=1, description="Глубина subgraph вокруг найденных узлов")


@router.post("/agent/query")
async def agent_query(request: AgentQueryRequest):
    """
    RAG-запрос: vector search → subgraph context → LLM answer.

    Полный цикл GraphRAG с метриками:
    1. Эмбеддинг вопроса → Qdrant vector search → top_k стартовых узлов
    2. Для каждого узла — get_subgraph(depth) → контекст
    3. Собираем промпт с контекстом → Gemini → ответ
    4. Возвращаем ответ + метрики (токены, время, узлы)
    """
    import time
    start = time.time()

    s = get_storage()
    vs = get_vector_store()
    from ..llm.client import get_llm_client
    client = get_llm_client()

    if not client.api_key:
        raise HTTPException(status_code=503, detail="LLM API key not configured")

    # Step 1: Vector search
    query_embedding = await get_embedding(request.question)
    if not query_embedding:
        raise HTTPException(status_code=503, detail="Failed to generate query embedding")

    vector_results = vs.search(
        query_vector=query_embedding,
        limit=request.top_k,
    )

    if not vector_results:
        raise HTTPException(status_code=404, detail="No relevant nodes found")

    # Step 2: Expand context via subgraph
    context_nodes = {}  # node_id -> node_info
    context_edges = []

    for vr in vector_results:
        node_id = vr.get("node_id", "")
        if not node_id:
            continue
        # Get the node itself
        node = s.get_node(node_id)
        if node:
            context_nodes[node_id] = node
        # Expand subgraph
        subgraph = s.get_subgraph(node_id, depth=request.depth)
        if subgraph:
            for n in subgraph.nodes:
                if n.node_id not in context_nodes:
                    context_nodes[n.node_id] = n
            context_edges.extend(subgraph.edges)

    # Step 3: Build prompt with context
    context_parts = []
    for nid, node in context_nodes.items():
        info = f"[{node.type}] {node.name}"
        if node.signature:
            info += f" | signature: {node.signature}"
        if node.summary:
            info += f" | summary: {node.summary}"
        if node.file_path:
            info += f" | file: {node.file_path}"
        # Include source code for small nodes (< 2000 chars)
        if node.source_code and len(node.source_code) < 2000:
            info += f"\n```\n{node.source_code}\n```"
        context_parts.append(info)

    edge_parts = []
    seen_edges = set()
    for e in context_edges:
        key = f"{e.source_id}->{e.target_id}:{e.edge_type}"
        if key not in seen_edges:
            seen_edges.add(key)
            edge_parts.append(f"  {e.source_id} --[{e.edge_type}]--> {e.target_id}")

    context_text = "\n\n".join(context_parts)
    edges_text = "\n".join(edge_parts[:50])  # Limit edges

    system_prompt = (
        "You are a knowledge assistant with access to a universal knowledge graph. "
        "The graph contains entities, facts, topics, skills, preferences, and documents from diverse sources "
        "(code, literature, conversations, recipes, finance, etc). "
        "Answer questions based ONLY on the provided context. "
        "If context contains contradictions, explain both sides with their sources. "
        "If the context doesn't contain enough information, say so explicitly. "
        "Answer in the same language as the question."
    )

    user_prompt = f"""QUESTION: {request.question}

GRAPH CONTEXT ({len(context_nodes)} nodes, {len(seen_edges)} edges):

{context_text}

RELATIONSHIPS:
{edges_text}

Based on this context, provide a detailed answer to the question."""

    context_chars = len(user_prompt)

    # Step 4: Call LLM with metrics
    result = await client.generate_with_metrics(user_prompt, system_prompt)

    total_time = round(time.time() - start, 2)

    return {
        "question": request.question,
        "answer": result["text"],
        "metrics": {
            "total_time_s": total_time,
            "llm_time_s": result["time_s"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"],
            "context_nodes": len(context_nodes),
            "context_edges": len(seen_edges),
            "context_chars": context_chars,
            "vector_results": len(vector_results),
        },
        "sources": [
            {"node_id": nid, "type": n.type, "name": n.name, "file": n.file_path}
            for nid, n in list(context_nodes.items())[:20]
        ],
    }


# ============== Entity Extraction ==============

@router.post("/extract/entities")
async def extract_entities_from_graph():
    """
    Извлекает сущности (ENTITY, FACT, TOPIC, SKILL) из всех узлов графа.
    Создаёт новые узлы-сущности и семантические рёбра.
    Обрабатывает порциями по 10 узлов за один LLM-вызов.
    """
    import hashlib
    import math

    s = get_storage()
    from ..llm.client import get_llm_client
    client = get_llm_client()

    if not client.api_key:
        raise HTTPException(status_code=503, detail="LLM API key not configured")

    # Забираем все узлы порциями
    total_in_graph = s.count_nodes()
    if total_in_graph == 0:
        return {"message": "No nodes in graph"}

    all_items = []
    offset = 0
    chunk_size = 500

    while offset < total_in_graph:
        nodes = s.search_nodes(limit=chunk_size, skip=offset)
        if not nodes:
            break
        for node in nodes:
            content = node.source_code or node.summary or node.name
            if len(content.strip()) < 20:
                continue
            all_items.append({
                "node_id": node.node_id,
                "name": node.name,
                "type": node.type,
                "content": content,
            })
        offset += len(nodes)

    logger.info(f"Entity extraction: {len(all_items)} nodes to process")

    # Батч-экстракция
    entities, edges = await extract_entities_batch(client, all_items, batch_size=10)
    logger.info(f"Extracted: {len(entities)} entities, {len(edges)} edges")

    # Entity resolution (дедупликация)
    name_map = resolve_entities(entities)

    # Создаём узлы-сущности в Neo4j
    entity_nodes = {}  # canonical_name → GraphNode
    for ent in entities:
        canonical = name_map.get(ent.name, ent.name)
        if canonical in entity_nodes:
            continue  # Уже создан

        node_id = f"{ent.type}::{canonical}"
        entity_nodes[canonical] = GraphNode(
            node_id=node_id,
            type=ent.type,
            name=canonical,
            signature="",
            file_path="",
            line_start=0,
            line_end=0,
            source_code="",
            summary=ent.summary,
            tags=[],
        )

    # Bulk insert entity nodes
    new_nodes = list(entity_nodes.values())
    if new_nodes:
        created = s.bulk_create_nodes(new_nodes)
        logger.info(f"Created {created} entity nodes")

    # Создаём семантические рёбра
    # 1) source_node → entity (MENTIONS)
    semantic_edges = []
    for ent in entities:
        canonical = name_map.get(ent.name, ent.name)
        if canonical not in entity_nodes:
            continue
        entity_node = entity_nodes[canonical]

        semantic_edges.append(GraphEdge(
            source_id=ent.source_node_id,
            target_id=entity_node.node_id,
            edge_type="MENTIONS",
            metadata={"confidence": ent.confidence, "weight": ent.confidence},
        ))

    # 2) entity → entity edges from extractor
    for ed in edges:
        src_canonical = name_map.get(ed.source_name, ed.source_name)
        tgt_canonical = name_map.get(ed.target_name, ed.target_name)
        if src_canonical not in entity_nodes or tgt_canonical not in entity_nodes:
            continue

        semantic_edges.append(GraphEdge(
            source_id=entity_nodes[src_canonical].node_id,
            target_id=entity_nodes[tgt_canonical].node_id,
            edge_type=ed.edge_type,
            metadata={"weight": ed.weight},
        ))

    # Bulk insert edges
    if semantic_edges:
        edge_count = s.bulk_create_edges(semantic_edges)
        logger.info(f"Created {edge_count} semantic edges")

    api_calls = math.ceil(len(all_items) / 10)

    return {
        "total_source_nodes": len(all_items),
        "entities_extracted": len(entities),
        "unique_entities": len(entity_nodes),
        "semantic_edges": len(semantic_edges),
        "entity_resolution_merges": sum(1 for k, v in name_map.items() if k != v),
        "api_calls": api_calls,
    }
