"""
ENN - External Neural Network - API Routes
Entities = neurons. Edges = synapses. LLM thinks freely.
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import os
import zipfile
import hashlib
import json

from ..graph.models import GraphNode, GraphEdge
from ..graph.storage import Storage
from ..parser.txt_converter import scan_and_filter
from ..llm.entity_extractor import extract_all_files, merge_file_states

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["graph"])

_storage: Optional[Storage] = None


def set_storage(s: Storage):
    global _storage
    _storage = s


def get_storage() -> Storage:
    if _storage is None or _storage._conn is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    return _storage


# ============== Models ==============

class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    llm_provider: str
    llm_model: str
    llm_configured: bool
    version: str


class StatsResponse(BaseModel):
    total_nodes: int
    total_edges: int
    node_types: List[str]
    type_count: int


class SearchRequest(BaseModel):
    query: Optional[str] = None
    node_type: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=200)


class PipelineRequest(BaseModel):
    directory: Optional[str] = None
    project_name: Optional[str] = None


class AgentQueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask the ENN")
    max_iterations: int = Field(default=10, ge=1, le=25, description="Max thinking iterations")


# ============== Health & Stats ==============

@router.get("/health", response_model=HealthResponse)
async def health_check():
    from ..config import get_settings
    settings = get_settings()
    db_ok = False
    if _storage and _storage._conn:
        try:
            _storage._conn.execute("SELECT 1")
            db_ok = True
        except Exception:
            pass
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        db_connected=db_ok,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_configured=bool(settings.llm_api_key),
        version="3.0.0"
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    s = get_storage()
    return StatsResponse(**s.get_stats())


@router.delete("/clear")
async def clear_graph():
    s = get_storage()
    s.clear_all()
    return {"message": "Graph cleared"}


# ============== Graph Access ==============

@router.get("/node/{node_id}")
async def get_node(node_id: str):
    s = get_storage()
    node = s.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return node.to_api_dict(include_code=True)


@router.get("/subgraph/{node_id}")
async def get_subgraph(
    node_id: str,
    depth: int = Query(default=2, ge=1, le=3),
    edge_types: Optional[str] = Query(default=None),
    include_code: bool = Query(default=False),
):
    s = get_storage()
    edge_filter = [t.strip() for t in edge_types.split(",")] if edge_types else None
    subgraph = s.get_subgraph(node_id, depth, edge_filter, include_code)
    if not subgraph:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return subgraph.to_dict(include_code)


@router.post("/search")
async def search_nodes(request: SearchRequest):
    s = get_storage()
    nodes = s.search_nodes(query=request.query, node_type=request.node_type, limit=request.limit)
    return [n.to_api_dict(include_code=False) for n in nodes]


# ============== Upload & Pipeline ==============

UPLOAD_DIR = "/app/uploads"


def _split_into_chunks(text: str, max_size: int = 12000) -> List[str]:
    if len(text) <= max_size:
        return [text]
    chunks = []
    paragraphs = text.split('\n\n')
    current = ""
    for para in paragraphs:
        if len(para) > max_size:
            if current:
                chunks.append(current)
                current = ""
            lines = para.split('\n')
            for line in lines:
                if len(current) + len(line) + 1 > max_size:
                    if current:
                        chunks.append(current)
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
    return chunks if chunks else [text]


@router.post("/upload")
async def upload_project(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    original = file.filename or "project"
    project_name = os.path.splitext(original)[0]
    project_dir = os.path.join(UPLOAD_DIR, project_name)
    content = await file.read()
    if original.endswith('.zip'):
        os.makedirs(project_dir, exist_ok=True)
        tmp = os.path.join(UPLOAD_DIR, original)
        with open(tmp, 'wb') as f:
            f.write(content)
        with zipfile.ZipFile(tmp, 'r') as z:
            for member in z.namelist():
                member_path = os.path.realpath(os.path.join(project_dir, member))
                if not member_path.startswith(os.path.realpath(project_dir) + os.sep):
                    raise HTTPException(400, f"Zip path traversal: {member}")
            z.extractall(project_dir)
        os.remove(tmp)
    else:
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(project_dir, original), 'wb') as f:
            f.write(content)
    return await auto_pipeline(PipelineRequest(directory=project_dir))


@router.post("/pipeline")
async def auto_pipeline(request: PipelineRequest):
    """ENN pipeline: scan → chunks → neural extraction (entities + synapses)."""
    from ..llm.client import get_llm_client
    import time as _time

    pipeline_start = _time.time()
    directory = request.directory
    if request.project_name:
        directory = os.path.join(UPLOAD_DIR, request.project_name)
    if not directory or not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    s = get_storage()
    client = get_llm_client()
    steps = []
    errors = []

    # STEP 1: Scan
    step_start = _time.time()
    scan = scan_and_filter(directory)
    all_files = scan["files"]
    steps.append({"step": "scan", "time_s": round(_time.time() - step_start, 2), "files": len(all_files)})
    logger.info(f"Step 1 (scan): {len(all_files)} files")

    # STEP 2: Create chunks
    step_start = _time.time()
    doc_nodes = []
    chunk_edges = []
    CHUNK_SIZE = 12000

    for fp in all_files:
        try:
            with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            if len(content.strip()) < 10:
                continue
            rel_path = os.path.relpath(fp, directory)
            file_hash = hashlib.sha256(rel_path.encode()).hexdigest()[:16]
            name = os.path.basename(fp)
            chunks = _split_into_chunks(content, CHUNK_SIZE)
            prev_id = None
            for ci, chunk_text in enumerate(chunks):
                node_id = f"doc::{file_hash}::{ci}"
                doc_nodes.append(GraphNode(
                    node_id=node_id, type="document", name=f"{name}::chunk_{ci}",
                    signature=rel_path, file_path=rel_path, line_start=0, line_end=0,
                    source_code=chunk_text, summary=f"{name} [{ci+1}/{len(chunks)}]", tags=[],
                ))
                if prev_id:
                    chunk_edges.append(GraphEdge(source_id=prev_id, target_id=node_id, edge_type="NEXT_CHUNK"))
                prev_id = node_id
        except Exception as e:
            errors.append(f"{fp}: {e}")

    if doc_nodes:
        s.bulk_create_nodes(doc_nodes)
    if chunk_edges:
        s.bulk_create_edges(chunk_edges)
    steps.append({"step": "chunks", "time_s": round(_time.time() - step_start, 2),
                   "files": len(all_files), "chunks": len(doc_nodes)})
    logger.info(f"Step 2 (chunks): {len(all_files)} files → {len(doc_nodes)} chunks")

    # STEP 3: Neural Extraction (neurons + synapses)
    step_start = _time.time()
    entity_nodes = []
    entity_edges = []

    if client.api_key and doc_nodes:
        files_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for node in doc_nodes:
            content = node.source_code or ""
            if len(content.strip()) < 20:
                continue
            fp = node.file_path or "unknown"
            if fp not in files_chunks:
                files_chunks[fp] = []
            files_chunks[fp].append({"node_id": node.node_id, "content": content})

        logger.info(f"Step 3 (extraction): {sum(len(c) for c in files_chunks.values())} chunks across {len(files_chunks)} files")
        file_states = await extract_all_files(client, files_chunks, concurrency=10)
        global_state = merge_file_states(file_states)
        logger.info(f"Extraction done: {len(global_state.entities)} neurons, {len(global_state.edges)} synapses")

        entity_node_ids = {}
        source_files = list(files_chunks.keys())
        source_hash = hashlib.sha256(",".join(source_files).encode()).hexdigest()[:8]
        for name, ent in global_state.entities.items():
            node_id = f"entity::{name}::{source_hash}"
            entity_node_ids[name] = node_id
            entity_nodes.append(GraphNode(
                node_id=node_id, type=ent.type, name=name,
                signature=",".join(source_files)[:200], file_path="", line_start=0, line_end=0,
                source_code="", summary=ent.summary, tags=[],
            ))

        if entity_nodes:
            s.bulk_create_nodes(entity_nodes)

        for name, ent in global_state.entities.items():
            ent_id = entity_node_ids.get(name)
            if not ent_id:
                continue
            for chunk_id in set(ent.source_chunks):
                entity_edges.append(GraphEdge(source_id=chunk_id, target_id=ent_id,
                                              edge_type="MENTIONS", metadata={"confidence": ent.confidence}))

        for ed in global_state.edges:
            src_id = entity_node_ids.get(ed.source)
            tgt_id = entity_node_ids.get(ed.target)
            if src_id and tgt_id:
                meta = {"weight": ed.weight}
                if ed.evidence_starts:
                    meta["evidence_starts"] = ed.evidence_starts
                if ed.evidence_ends:
                    meta["evidence_ends"] = ed.evidence_ends
                if ed.source_chunk:
                    meta["source_chunk"] = ed.source_chunk
                entity_edges.append(GraphEdge(source_id=src_id, target_id=tgt_id,
                                              edge_type=ed.edge_type, metadata=meta))

        if entity_edges:
            s.bulk_create_edges(entity_edges)

    steps.append({"step": "extraction", "time_s": round(_time.time() - step_start, 2),
                   "neurons": len(entity_nodes), "synapses": len(entity_edges)})
    logger.info(f"Step 3: {len(entity_nodes)} neurons, {len(entity_edges)} synapses")

    total_time = round(_time.time() - pipeline_start, 2)
    return {
        "success": True, "project_dir": directory, "total_time_s": total_time,
        "total_files": len(all_files), "total_nodes": len(doc_nodes) + len(entity_nodes),
        "total_edges": len(entity_edges) + len(chunk_edges), "steps": steps,
        "errors": errors[:10],
    }


# ============== ENN Query — Free Thinking ==============

THINK_PROMPT = """Ты находишься в Графе Знаний. В нём есть сущности, обрывки информации и связи между сущностями и информацией.

Твоя задача — дать точный ответ на вопрос пользователя используя Граф Знаний.

Вопрос: {question}

Обзор графа (самые связанные сущности):
{graph_overview}

Инструменты:
- search:слово — найти сущности по слову
- explore:имя — показать все связи этой сущности
- follow:имя — перейти по связи к этой сущности и увидеть её связи
- read_evidence:имя — прочитать только выделенный текст привязанный к связям этой сущности (быстро, экономно)
- read_chunk:id — прочитать полный текст чанка (медленно, но полная информация)
- read_sequential:имя — прочитать все чанки с этой сущностью по порядку
- answer:текст — дать ответ

Блокнот (найденные факты):
{scratchpad}

Инструкция:
1. Выдели из вопроса главные сущности и используй search. Если ничего не найдено — пробуй синонимы или explore пока не найдёшь подходящее слово.
2. Используй explore чтобы увидеть все связи найденной сущности.
3. Используй follow чтобы перейти по релевантной связи к другой сущности.
4. Используй read_evidence чтобы прочитать выделенный текст (экономнее чем read_chunk). Если нужно больше контекста — используй read_chunk.
5. Повторяй шаги 2-4 пока не соберёшь достаточно фактов для ответа.

Записывай найденные факты в поле notes — они сохранятся в блокноте. Если блокнот превысит 80% контекста, он будет сжат до ключевых фактов.

Как только данных достаточно — используй answer:текст.

Верни JSON:
{{
  "thinking": "ход мысли",
  "notes": "релевантные факты для ответа (добавляются в блокнот)",
  "action": "инструмент:параметр"
}}"""


@router.post("/agent/query")
async def agent_query(request: AgentQueryRequest):
    """ENN query: LLM thinks freely with access to the neural network."""
    import time as _time
    start = _time.time()

    s = get_storage()
    from ..llm.client import get_llm_client
    client = get_llm_client()

    if not client.api_key:
        raise HTTPException(status_code=503, detail="LLM not configured")

    # Graph overview: top neurons by connectivity
    top_neurons = s.get_root_categories(limit=30)
    graph_overview = "\n".join([
        f"- {n['name']} [{n['type']}]: {n.get('summary', '')[:100]}"
        for n in top_neurons
    ]) if top_neurons else "(empty graph)"

    scratchpad = ""
    steps = []
    total_tokens = 0
    all_sources = []
    answer_text = "I could not find enough information to answer."

    for iteration in range(request.max_iterations):
        prompt = THINK_PROMPT.format(
            question=request.question,
            graph_overview=graph_overview,
            scratchpad=scratchpad or "(empty — start thinking)",
        )

        result = await client.generate_with_metrics(prompt)
        total_tokens += result.get("total_tokens", 0)

        # Strip <think> blocks from reasoning models
        text = result["text"]
        if "<think>" in text and "</think>" in text:
            text = text[text.find("</think>") + 8:].strip()

        data = _parse_json(text)
        thinking = data.get("thinking", "")
        notes = data.get("notes", "")
        action = data.get("action", "answer:I don't have enough information")

        if notes:
            scratchpad += f"\n[{iteration+1}] {notes}"

        steps.append({"step": iteration + 1, "thinking": thinking[:200], "action": action})

        # Execute action
        if action.startswith("answer:"):
            answer_text = action[7:]
            break

        elif action.startswith("search:"):
            keyword = action[7:]
            matches = s.search_entities_by_name(keyword, limit=10)
            all_sources.extend(matches)
            if matches:
                info = "\n".join([f"- {m['name']} [{m['type']}]: {m.get('summary', '')}" for m in matches])
                scratchpad += f"\n[search:{keyword}] {info}"
            else:
                scratchpad += f"\n[search:{keyword}] nothing found"

        elif action.startswith("explore:"):
            entity_name = action[8:]
            matches = s.search_entities_by_name(entity_name, limit=1)
            if matches:
                all_sources.extend(matches)
                related = s.get_related(matches[0]["node_id"], limit=20)
                edges_info = "\n".join([
                    f"  {'→' if r.get('outgoing') else '←'}[{r.get('edge_type','?')}] {r['name']}: {r.get('summary','')[:80]}"
                    for r in related
                ])
                scratchpad += f"\n[explore:{entity_name}]\n{edges_info}"
            else:
                scratchpad += f"\n[explore:{entity_name}] not found"

        elif action.startswith("follow:"):
            # Переход по связи — explore целевой сущности
            entity_name = action[7:]
            matches = s.search_entities_by_name(entity_name, limit=1)
            if matches:
                all_sources.extend(matches)
                node = s.get_node(matches[0]["node_id"])
                if node:
                    scratchpad += f"\n[follow:{entity_name}] [{node.type}] {node.name}: {node.summary}"
                related = s.get_related(matches[0]["node_id"], limit=20)
                edges_info = "\n".join([
                    f"  {'→' if r.get('outgoing') else '←'}[{r.get('edge_type','?')}] {r['name']}: {r.get('summary','')[:80]}"
                    for r in related
                ])
                if edges_info:
                    scratchpad += f"\n{edges_info}"
            else:
                scratchpad += f"\n[follow:{entity_name}] not found"

        elif action.startswith("read_evidence:"):
            # Прочитать только выделенный текст привязанный к связям (быстро)
            entity_name = action[14:]
            evidences = s.get_evidence_for_entity(entity_name, limit=10)
            if evidences:
                for ev in evidences:
                    chunk_id = ev.get("source_chunk", "")
                    if chunk_id and ev.get("evidence_starts"):
                        chunk_node = s.get_node(chunk_id)
                        if chunk_node and chunk_node.source_code:
                            src = chunk_node.source_code
                            si = src.find(ev["evidence_starts"])
                            if si >= 0:
                                ei = src.find(ev.get("evidence_ends", ""), max(0, si))
                                if ei > si:
                                    fragment = src[si:ei + len(ev.get('evidence_ends', ''))]
                                else:
                                    fragment = src[si:si + 500]
                                scratchpad += f"\n[evidence:{ev['source']}→{ev['target']}] {fragment}"
            else:
                scratchpad += f"\n[read_evidence:{entity_name}] no evidence found"

        elif action.startswith("read_chunk:"):
            chunk_id = action[11:]
            node = s.get_node(chunk_id)
            if node and node.source_code:
                scratchpad += f"\n[chunk] {node.source_code[:2000]}"

        elif action.startswith("read_sequential:"):
            entity_name = action[16:]
            chunks = s.get_chunks_for_entity(entity_name, limit=20)
            if chunks:
                def _chunk_sort_key(c):
                    nid = c.get("node_id", "")
                    try:
                        return int(nid.rsplit("::", 1)[-1])
                    except ValueError:
                        return 0
                sorted_chunks = sorted(chunks, key=_chunk_sort_key)
                seq = "\n".join([ch.get("content", "")[:500] for ch in sorted_chunks if ch.get("content")])
                scratchpad += f"\n[sequential:{entity_name}] {seq[:3000]}"

        # Compress if needed (80% of 205K tokens ≈ 480K chars)
        if len(scratchpad) > 480000:
            cr = await client.generate_with_metrics(f"Compress to key facts, keep all names and relationships:\n\n{scratchpad}")
            total_tokens += cr.get("total_tokens", 0)
            scratchpad = f"[compressed]\n{cr['text']}"

    # Cache answer
    try:
        ah = hashlib.sha256(request.question.encode()).hexdigest()[:12]
        s.create_node(GraphNode(
            node_id=f"answer::{ah}", type="cached_answer",
            name=request.question[:100].lower(), signature=request.question,
            file_path="", line_start=0, line_end=0,
            source_code=answer_text[:2000], summary=answer_text[:200], tags=[],
        ))
        for ent in all_sources[:5]:
            s.create_edge(GraphEdge(source_id=f"answer::{ah}", target_id=ent["node_id"], edge_type="ANSWERED_FROM"))
    except Exception:
        pass

    return {
        "question": request.question,
        "answer": answer_text,
        "metrics": {
            "total_time_s": round(_time.time() - start, 2),
            "total_tokens": total_tokens,
            "iterations": len(steps),
        },
        "thinking": steps,
        "sources": [{"node_id": e["node_id"], "type": e["type"], "name": e["name"]} for e in all_sources[:10]],
    }


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Strip <think> blocks from reasoning models
    if "<think>" in text and "</think>" in text:
        text = text[text.find("</think>") + 8:].strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return {"thinking": "parse error", "notes": "", "action": "answer:Failed to parse response"}
