"""
Tree Base - API Routes
Universal knowledge graph with graph navigation.
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import os
import zipfile
import hashlib
import json

from ..graph.models import GraphNode, GraphEdge, IngestResult
from ..graph.storage import Neo4jStorage
from ..parser.txt_converter import scan_and_filter
from ..llm.entity_extractor import extract_all_files, merge_file_states

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["graph"])

_storage: Optional[Neo4jStorage] = None


def set_storage(s: Neo4jStorage):
    global _storage
    _storage = s


def get_storage() -> Neo4jStorage:
    if _storage is None or _storage._driver is None:
        raise HTTPException(status_code=503, detail="Neo4j not connected")
    return _storage


# ============== Models ==============

class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
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
    question: str = Field(..., description="Question to ask the knowledge graph")
    max_depth: int = Field(default=3, description="Max navigation depth")


# ============== Health & Stats ==============

@router.get("/health", response_model=HealthResponse)
async def health_check():
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
        version="2.0.0"
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


# ============== Graph Navigation ==============

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


def _split_into_chunks(text: str, max_size: int = 8000) -> List[str]:
    """Split text into chunks <= max_size, breaking on paragraph boundaries."""
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

    return chunks if chunks else [text[:max_size]]


@router.post("/upload")
async def upload_project(file: UploadFile = File(...)):
    """Upload file → auto-run pipeline."""
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
            z.extractall(project_dir)
        os.remove(tmp)
    else:
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(project_dir, original), 'wb') as f:
            f.write(content)

    request = PipelineRequest(directory=project_dir)
    return await auto_pipeline(request)


@router.post("/pipeline")
async def auto_pipeline(request: PipelineRequest):
    """
    Tree Base pipeline: scan → chunks → entity extraction. No embeddings needed.
    """
    from ..llm.client import get_llm_client
    import time as _time
    import math

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
    CHUNK_SIZE = 8000

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
                    node_id=node_id, type="document",
                    name=f"{name}::chunk_{ci}", signature=rel_path,
                    file_path=rel_path, line_start=0, line_end=0,
                    source_code=chunk_text,
                    summary=f"{name} [{ci+1}/{len(chunks)}]", tags=[],
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

    # STEP 3: Entity Extraction (context-aware)
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
        logger.info(f"Extraction done: {len(global_state.entities)} entities, {len(global_state.edges)} edges")

        entity_node_ids = {}
        for name, ent in global_state.entities.items():
            node_id = f"entity::{name}"
            entity_node_ids[name] = node_id
            entity_nodes.append(GraphNode(
                node_id=node_id, type=ent.type, name=name,
                signature="", file_path="", line_start=0, line_end=0,
                source_code="", summary=ent.summary, tags=[],
            ))

        if entity_nodes:
            s.bulk_create_nodes(entity_nodes)

        for name, ent in global_state.entities.items():
            ent_id = entity_node_ids.get(name)
            if not ent_id:
                continue
            for chunk_id in ent.source_chunks:
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
                   "entities": len(entity_nodes), "edges": len(entity_edges)})
    logger.info(f"Step 3 (extraction): {len(entity_nodes)} entities, {len(entity_edges)} edges")

    total_time = round(_time.time() - pipeline_start, 2)
    return {
        "success": True, "project_dir": directory, "total_time_s": total_time,
        "total_files": len(all_files), "total_nodes": len(all_nodes),
        "total_edges": len(entity_edges) + len(chunk_edges), "steps": steps,
        "errors": errors[:10],
    }


# ============== Agent Query (Graph Navigation) ==============

NAVIGATE_PROMPT = """You are navigating a knowledge graph to find information.

QUESTION: {question}

AVAILABLE NODES:
{nodes_list}

Pick the most relevant nodes. Return ONLY valid JSON (no markdown):
{{"selected": ["name1", "name2"], "reasoning": "why"}}

Select up to 5. If none relevant: {{"selected": [], "reasoning": "none relevant"}}"""


SCRATCHPAD_PROMPT = """You are searching a knowledge graph to answer a question.

QUESTION: {question}

SCRATCHPAD (what you've found so far):
{scratchpad}

NEW INFORMATION:
{new_info}

Decide what to do next. Return ONLY valid JSON (no markdown):
{{
  "add_to_scratchpad": "summarize the useful new information in 1-3 sentences",
  "sufficient": true/false,
  "next_action": "explore:entity_name" or "read_chunk:chunk_id" or "answer"
}}

- "sufficient": true if you have enough to answer the question fully
- "explore:name": follow edges of this entity to learn more
- "read_chunk:id": load full chunk text for details
- "answer": you have enough, generate the final answer"""


ANSWER_PROMPT = """You are a knowledge assistant with access to a universal knowledge graph.
The graph contains entities from diverse sources (code, literature, recipes, finance, psychology, etc).

Answer based ONLY on the provided context. If contradictions exist, explain both sides.
If insufficient context, say so. Answer in the same language as the question.

QUESTION: {question}

COLLECTED CONTEXT:
{context}

RELATIONSHIPS:
{edges}"""


@router.post("/agent/query")
async def agent_query(request: AgentQueryRequest):
    """Graph navigation query: LLM navigates graph to find answers."""
    import time as _time
    start = _time.time()

    s = get_storage()
    from ..llm.client import get_llm_client
    client = get_llm_client()

    if not client.api_key:
        raise HTTPException(status_code=503, detail="LLM not configured")

    navigation_steps = []
    found_entities = []
    total_nav_tokens = 0

    # Step 1: LLM navigates root categories
    roots = s.get_root_categories(limit=50)
    if not roots:
        raise HTTPException(status_code=404, detail="Knowledge graph is empty")

    nodes_list = "\n".join([f"- {r['name']} [{r['type']}]: {r.get('summary', '')}" for r in roots])
    nav_prompt = NAVIGATE_PROMPT.format(question=request.question, nodes_list=nodes_list)
    nav_result = await client.generate_with_metrics(nav_prompt)
    total_nav_tokens += nav_result.get("total_tokens", 0)

    nav_data = _parse_nav(nav_result["text"])
    navigation_steps.append({"step": "root_nav", "options": len(roots), "selected": nav_data.get("selected", [])})

    # Step 2: Drill down into selected categories
    for name in nav_data.get("selected", [])[:3]:
        matches = [r for r in roots if r["name"].lower() == name.lower()]
        if not matches:
            matches = s.search_entities_by_name(name, limit=1)
        if not matches:
            continue

        children = s.get_children(matches[0]["node_id"], limit=30)
        if children:
            cl = "\n".join([f"- {c['name']} [{c['type']}]: {c.get('summary', '')}" for c in children])
            nav2 = await client.generate_with_metrics(NAVIGATE_PROMPT.format(question=request.question, nodes_list=cl))
            total_nav_tokens += nav2.get("total_tokens", 0)
            nav2_data = _parse_nav(nav2["text"])
            navigation_steps.append({"step": f"drill:{name}", "options": len(children), "selected": nav2_data.get("selected", [])})
            for cn in nav2_data.get("selected", [])[:5]:
                cm = [c for c in children if c["name"].lower() == cn.lower()]
                if cm:
                    found_entities.append(cm[0])
        else:
            found_entities.append(matches[0])

    if not found_entities:
        raise HTTPException(status_code=404, detail="No relevant information found")

    # Step 3: Scratchpad loop — iteratively gather context
    scratchpad = ""
    edge_parts = []
    all_sources = list(found_entities[:10])
    max_iterations = 8

    for iteration in range(max_iterations):
        # Gather new info from current entities
        new_info_parts = []
        for ent in found_entities[:5]:
            node = s.get_node(ent["node_id"])
            if not node:
                continue
            info = f"[{node.type}] {node.name}"
            if node.summary:
                info += f": {node.summary}"
            new_info_parts.append(info)

            for rel in s.get_related(ent["node_id"], limit=15):
                d = "→" if rel.get("outgoing") else "←"
                edge_parts.append(f"  {node.name} {d}[{rel.get('edge_type', '?')}]{d} {rel['name']}")
                if rel.get("summary"):
                    new_info_parts.append(f"  [{rel['type']}] {rel['name']}: {rel['summary']}")

            for chunk in s.get_chunks_for_entity(ent.get("name", ""), limit=2):
                if chunk.get("content"):
                    new_info_parts.append(f"[source: {chunk.get('file_path', '?')}]\n{chunk['content']}")

        new_info = "\n".join(new_info_parts)

        # Ask LLM: enough or explore more?
        sp_prompt = SCRATCHPAD_PROMPT.format(
            question=request.question,
            scratchpad=scratchpad or "(empty — first iteration)",
            new_info=new_info[:4000],
        )
        sp_result = await client.generate_with_metrics(sp_prompt)
        total_nav_tokens += sp_result.get("total_tokens", 0)
        sp_data = _parse_nav(sp_result["text"])

        # Update scratchpad
        addition = sp_data.get("add_to_scratchpad", "")
        if addition:
            scratchpad += f"\n[iter {iteration+1}] {addition}"

        navigation_steps.append({
            "step": f"scratchpad_iter_{iteration+1}",
            "sufficient": sp_data.get("sufficient", False),
            "next_action": sp_data.get("next_action", "answer"),
        })

        # Check if sufficient
        if sp_data.get("sufficient", False) or sp_data.get("next_action") == "answer":
            break

        # Follow next action
        next_action = sp_data.get("next_action", "")
        if next_action.startswith("explore:"):
            entity_name = next_action[8:]
            matches = s.search_entities_by_name(entity_name, limit=3)
            found_entities = matches
            all_sources.extend(matches)
        elif next_action.startswith("read_chunk:"):
            chunk_id = next_action[11:]
            node = s.get_node(chunk_id)
            if node and node.source_code:
                scratchpad += f"\n[chunk {chunk_id}] {node.source_code[:2000]}"
        else:
            break

    # Step 4: Final answer from scratchpad
    answer_prompt = ANSWER_PROMPT.format(
        question=request.question,
        context=scratchpad,
        edges="\n".join(list(set(edge_parts))[:30]),
    )
    result = await client.generate_with_metrics(answer_prompt)

    return {
        "question": request.question,
        "answer": result["text"],
        "metrics": {
            "total_time_s": round(_time.time() - start, 2),
            "llm_time_s": result["time_s"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"] + total_nav_tokens,
            "nav_tokens": total_nav_tokens,
            "scratchpad_iterations": len([s for s in navigation_steps if "scratchpad" in s.get("step", "")]),
            "context_entities": len(all_sources),
        },
        "navigation": navigation_steps,
        "sources": [{"node_id": e["node_id"], "type": e["type"], "name": e["name"]} for e in all_sources[:10]],
    }


def _parse_nav(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
    return {"selected": [], "reasoning": "parse error"}
