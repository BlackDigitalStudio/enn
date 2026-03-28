"""
Entity Extractor — context-aware extraction with write-access to the knowledge graph.

Core principle: extraction is NOT "create entities from chunk".
It IS "update the knowledge graph with new information from this chunk".

Each chunk sees existing entities and can:
- Reuse existing entity (same name, same type)
- Update summary of existing entity with new information
- Add new edges between existing entities
- Update edge types when relationships evolve (enemies → friends)
- Create new entities only when genuinely new

Processing order:
- Sequential within one file (chunks of a book go in order)
- Parallel between files (16 books processed simultaneously)
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Context-aware extraction prompt — sees existing entities
CONTEXT_EXTRACTION_PROMPT = """You are a knowledge graph updater. Read the new content and update the knowledge graph.

EXISTING ENTITIES IN GRAPH:
{existing_entities}

NEW CONTENT (from {source_name}):
{content}

---

Return ONLY valid JSON (no markdown):

{{
  "entities": [
    {{
      "name": "canonical name (lowercase, reuse existing names when possible)",
      "type": "most specific type (character, location, ability, ingredient, class, concept...)",
      "summary": "updated description incorporating new info (same language as content)",
      "action": "create|update",
      "confidence": 0.0-1.0
    }}
  ],
  "edges": [
    {{
      "source": "entity name",
      "target": "entity name",
      "type": "descriptive relationship (KNOWS, LOCATED_IN, PART_OF, FIGHTS, LOVES, CREATES...)",
      "weight": 0.0-1.0,
      "action": "create|update"
    }}
  ]
}}

RULES:
- REUSE existing entity names — do NOT create "меркурий" if "меркурий" already exists
- REUSE existing types — if "меркурий" is already "character", keep it "character"
- UPDATE summaries — if you learn something new about an existing entity, set action="update"
- UPDATE edges — if a relationship changed (enemies became friends), set action="update" with new type
- CREATE only genuinely new entities not in the existing list
- Summaries in the same language as source content
- Maximum 15 entities + 20 edges per chunk
- Skip confidence < 0.3
"""

# First chunk prompt (no existing entities yet)
INITIAL_EXTRACTION_PROMPT = """You are a knowledge graph builder. Extract entities and relationships from this content.

CONTENT (from {source_name}):
{content}

---

Return ONLY valid JSON (no markdown):

{{
  "entities": [
    {{
      "name": "canonical name (lowercase)",
      "type": "most specific type (character, location, ability, ingredient, class, concept...)",
      "summary": "one-line description (same language as content)",
      "confidence": 0.0-1.0
    }}
  ],
  "edges": [
    {{
      "source": "entity name",
      "target": "entity name",
      "type": "descriptive relationship (KNOWS, LOCATED_IN, PART_OF, FIGHTS, LOVES, CREATES...)",
      "weight": 0.0-1.0
    }}
  ]
}}

RULES:
- Use most specific type: "character" not "entity", "ingredient" not "thing"
- Entity names: lowercase, canonical
- Summaries in the same language as source content
- Maximum 15 entities + 20 edges
- Skip confidence < 0.3
"""


@dataclass
class EntityState:
    """Живое состояние сущности в процессе extraction."""
    name: str
    type: str
    summary: str
    confidence: float
    source_chunks: List[str] = field(default_factory=list)  # chunk IDs where mentioned


@dataclass
class EdgeState:
    """Живое состояние ребра."""
    source: str
    target: str
    edge_type: str
    weight: float
    source_chunk: str = ""


class KnowledgeGraphState:
    """
    In-memory state of the knowledge graph during extraction.
    Updated after each chunk. Passed to LLM as context.
    """

    def __init__(self):
        self.entities: Dict[str, EntityState] = {}  # name → EntityState
        self.edges: List[EdgeState] = []

    def format_for_prompt(self, max_entities: int = 50) -> str:
        """Format existing entities for inclusion in extraction prompt."""
        if not self.entities:
            return "(empty — this is the first chunk)"

        # Sort by confidence descending, take top N
        sorted_ents = sorted(self.entities.values(), key=lambda e: -e.confidence)[:max_entities]
        lines = []
        for ent in sorted_ents:
            lines.append(f"- {ent.name} [{ent.type}]: {ent.summary}")
        return "\n".join(lines)

    def update_from_extraction(self, entities: List[Dict], edges: List[Dict], chunk_id: str):
        """Update state with extraction results from one chunk."""
        for ent in entities:
            name = ent["name"].lower().strip()
            if not name:
                continue
            conf = float(ent.get("confidence", 0.5))
            if conf < 0.3:
                continue

            if name in self.entities:
                # Update existing
                existing = self.entities[name]
                if ent.get("summary"):
                    existing.summary = ent["summary"]
                if conf > existing.confidence:
                    existing.confidence = conf
                existing.source_chunks.append(chunk_id)
            else:
                # Create new
                self.entities[name] = EntityState(
                    name=name,
                    type=ent.get("type", "entity"),
                    summary=ent.get("summary", ""),
                    confidence=conf,
                    source_chunks=[chunk_id],
                )

        for ed in edges:
            src = ed.get("source", "").lower().strip()
            tgt = ed.get("target", "").lower().strip()
            if not src or not tgt:
                continue
            w = float(ed.get("weight", 0.5))
            if w < 0.2:
                continue

            edge_type = ed.get("type", "RELATES_TO")

            # Check if edge already exists — update type if action=update
            updated = False
            if ed.get("action") == "update":
                for existing_edge in self.edges:
                    if existing_edge.source == src and existing_edge.target == tgt:
                        existing_edge.edge_type = edge_type
                        existing_edge.weight = w
                        existing_edge.source_chunk = chunk_id
                        updated = True
                        break

            if not updated:
                self.edges.append(EdgeState(
                    source=src, target=tgt,
                    edge_type=edge_type, weight=w,
                    source_chunk=chunk_id,
                ))


async def _extract_chunk_with_context(
    llm_client,
    content: str,
    source_name: str,
    chunk_id: str,
    graph_state: KnowledgeGraphState,
    max_retries: int = 2,
) -> Optional[Dict]:
    """Extract entities from one chunk, with context of existing graph state."""

    existing = graph_state.format_for_prompt()

    if graph_state.entities:
        prompt = CONTEXT_EXTRACTION_PROMPT.format(
            existing_entities=existing,
            source_name=source_name,
            content=content,
        )
    else:
        prompt = INITIAL_EXTRACTION_PROMPT.format(
            source_name=source_name,
            content=content,
        )

    for attempt in range(max_retries + 1):
        try:
            response = await llm_client.generate(prompt)
            if not response:
                continue
            data = _parse_json_response(response)
            if data:
                return data
        except Exception as e:
            logger.warning(f"Extraction attempt {attempt+1} failed for {chunk_id}: {e}")
            if attempt < max_retries:
                await asyncio.sleep(1)

    logger.error(f"Extraction failed after {max_retries+1} attempts: {chunk_id}")
    return None


async def extract_file_sequential(
    llm_client,
    chunks: List[Dict[str, Any]],
    file_name: str,
) -> KnowledgeGraphState:
    """
    Process all chunks of ONE file sequentially.
    Each chunk sees entities extracted from previous chunks.
    Returns the final graph state for this file.
    """
    state = KnowledgeGraphState()

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        if len(content.strip()) < 20:
            continue

        chunk_id = chunk["node_id"]

        data = await _extract_chunk_with_context(
            llm_client, content, file_name, chunk_id, state
        )

        if data:
            state.update_from_extraction(
                data.get("entities", []),
                data.get("edges", []),
                chunk_id,
            )

        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            logger.info(
                f"  {file_name}: {i+1}/{len(chunks)} chunks, "
                f"{len(state.entities)} entities, {len(state.edges)} edges"
            )

    return state


async def extract_all_files(
    llm_client,
    files_chunks: Dict[str, List[Dict[str, Any]]],
    concurrency: int = 10,
) -> Dict[str, KnowledgeGraphState]:
    """
    Process multiple files in parallel.
    Each file is processed sequentially (chunks in order).
    Different files run in parallel (no dependency between them).

    Args:
        files_chunks: {file_path: [{"node_id": ..., "content": ...}, ...]}
        concurrency: max parallel files

    Returns:
        {file_path: KnowledgeGraphState}
    """
    results: Dict[str, KnowledgeGraphState] = {}
    file_list = list(files_chunks.items())

    logger.info(f"Entity extraction: {len(file_list)} files, concurrency={concurrency}")

    for batch_start in range(0, len(file_list), concurrency):
        batch = file_list[batch_start:batch_start + concurrency]

        tasks = [
            extract_file_sequential(llm_client, chunks, file_name)
            for file_name, chunks in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for (file_name, _), result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logger.error(f"File extraction failed: {file_name}: {result}")
                continue
            results[file_name] = result

        processed = min(batch_start + len(batch), len(file_list))
        total_entities = sum(len(s.entities) for s in results.values())
        total_edges = sum(len(s.edges) for s in results.values())
        logger.info(
            f"Entity extraction: {processed}/{len(file_list)} files done, "
            f"{total_entities} entities, {total_edges} edges"
        )

    return results


def merge_file_states(
    file_states: Dict[str, KnowledgeGraphState],
) -> KnowledgeGraphState:
    """
    Merge entity states from multiple files into one global state.
    Entities with the same name are merged (summaries combined, confidence maxed).
    """
    global_state = KnowledgeGraphState()

    for file_name, state in file_states.items():
        for name, ent in state.entities.items():
            if name in global_state.entities:
                existing = global_state.entities[name]
                # Keep higher confidence
                if ent.confidence > existing.confidence:
                    existing.confidence = ent.confidence
                # Append summary if different
                if ent.summary and ent.summary != existing.summary:
                    existing.summary = ent.summary  # Latest wins
                existing.source_chunks.extend(ent.source_chunks)
            else:
                global_state.entities[name] = EntityState(
                    name=ent.name,
                    type=ent.type,
                    summary=ent.summary,
                    confidence=ent.confidence,
                    source_chunks=list(ent.source_chunks),
                )

        global_state.edges.extend(state.edges)

    return global_state


def _parse_json_response(text: str) -> Any:
    """Parse JSON from LLM response (may contain markdown wrapper)."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == start_char:
                depth += 1
            elif text[idx] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:idx + 1])
                    except json.JSONDecodeError:
                        break

    logger.warning(f"Failed to parse JSON: {text[:200]}")
    return None
