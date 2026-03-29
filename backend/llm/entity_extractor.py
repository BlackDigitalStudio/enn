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
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# File logger for full extraction trace
EXTRACTION_LOG_DIR = "/app/logs"


def _log_to_file(chunk_id: str, chunk_text: str, prompt: str, response: str, parsed: Any, file_name: str = ""):
    """Write full extraction trace to log file."""
    try:
        os.makedirs(EXTRACTION_LOG_DIR, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in file_name)
        log_file = os.path.join(EXTRACTION_LOG_DIR, f"extraction_{safe_name}.log")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"CHUNK: {chunk_id} | TIME: {datetime.utcnow().isoformat()}\n")
            f.write(f"{'='*80}\n")
            f.write(f"\n--- CHUNK TEXT ({len(chunk_text)} chars) ---\n")
            f.write(chunk_text[:2000])
            if len(chunk_text) > 2000:
                f.write(f"\n... [{len(chunk_text) - 2000} more chars]\n")
            f.write(f"\n\n--- PROMPT ({len(prompt)} chars) ---\n")
            f.write(prompt[:3000])
            if len(prompt) > 3000:
                f.write(f"\n... [{len(prompt) - 3000} more chars]\n")
            f.write(f"\n\n--- LLM RESPONSE ({len(response)} chars) ---\n")
            f.write(response)
            f.write(f"\n\n--- PARSED RESULT ---\n")
            if parsed:
                f.write(json.dumps(parsed, ensure_ascii=False, indent=2))
            else:
                f.write("None (parse failed)")
            f.write(f"\n")
    except Exception as e:
        logger.warning(f"Failed to write extraction log: {e}")

CONTEXT_EXTRACTION_PROMPT = """Твоя задача — извлекать из любого текста (код, законы, художественная литература, статьи) строгие факты, зависимости и логику в виде троек: [Сущность 1] | [Связь] | [Сущность 2].

1. СУЩНОСТИ (Объекты / Концепции / Акторы)
Это любые самостоятельные элементы текста: люди, термины, переменные в коде, ингредиенты, законы, эмоции или абстрактные понятия.

Правило: Пиши максимально коротко (обычно 1-3 слова), в именительном падеже. Убирай лишние прилагательные, если они не меняют суть. Заменяй местоимения (он, она, это) на конкретные названия сущностей из текста.

Примеры: Unreal Engine 5, Статья 12, Биткоин, Пользователь, Функция Render(), Свекла.

Уже известные сущности (если сущность уже есть — используй то же имя):
{existing_entities}

2. СВЯЗИ (Взаимодействия / Зависимости / Свойства)
Это то, как Сущность 1 влияет на Сущность 2, относится к ней или зависит от нее.

Правило: Связь должна отражать конкретное действие, иерархию, свойство или конфликт. Запрещено использовать пустые слова вроде "связан_с", "упоминается_вместе_с".

Примеры связей: наследует_класс, гарантирует_право, требует_ингредиент, противоречит, является_частью, испытывает_эмоцию. Можно и нужно создавать свои.

3. ВЫДЕЛЕНИЕ ТЕКСТА
Для каждой связи укажи фрагмент исходного текста который её подтверждает: первые несколько слов фрагмента и последние несколько слов. По этим маркерам программа вырежет точную цитату.

ПРИМЕРЫ ИЗ РАЗНЫХ ДОМЕНОВ:

Текст (Программирование): "Класс Player Character наследует базовую логику от Actor и использует функцию Move()."
[Класс Player Character] | наследует_от | [Класс Actor]
[Класс Player Character] | вызывает_функцию | [Функция Move]

Текст (Кулинария): "Для классического борща обязательно нужна свекла и немного уксуса для цвета."
[Классический борщ] | требует_ингредиент | [Свекла]
[Классический борщ] | требует_ингредиент | [Уксус]
[Уксус] | сохраняет_свойство | [Цвет]

Текст (Юриспруденция): "Статья 15 Конституции запрещает любую цензуру в СМИ."
[Статья 15 Конституции] | запрещает | [Цензура]
[Цензура] | применяется_к | [СМИ]

---

Текст ({source_name}):
{content}

---

Верни JSON:

{{
  "entities": [
    {{
      "name": "имя (1-3 слова, именительный падеж)",
      "type": "кто/что это",
      "summary": "краткое описание на языке текста",
      "action": "create если новая, update если уже есть"
    }}
  ],
  "edges": [
    {{
      "source": "имя сущности 1",
      "target": "имя сущности 2",
      "type": "конкретная_связь",
      "evidence_starts": "первые слова фрагмента текста",
      "evidence_ends": "последние слова фрагмента текста"
    }}
  ],
  "contradictions": [
    {{
      "entity": "имя сущности",
      "old_fact": "что было известно раньше",
      "new_fact": "что стало известно сейчас",
      "evidence_starts": "первые слова подтверждения"
    }}
  ]
}}"""

INITIAL_EXTRACTION_PROMPT = """Твоя задача — извлекать из любого текста строгие факты, зависимости и логику в виде троек: [Сущность 1] | [Связь] | [Сущность 2].

1. СУЩНОСТИ — любые самостоятельные элементы: люди, термины, места, предметы, события. Пиши коротко (1-3 слова), в именительном падеже. Заменяй местоимения на конкретные имена.

2. СВЯЗИ — конкретное действие, иерархия, свойство или конфликт. Запрещено: "связан_с", "упоминается_вместе_с". Создавай свои связи.

3. ВЫДЕЛЕНИЕ ТЕКСТА — для каждой связи укажи первые и последние слова фрагмента текста который её подтверждает.

Текст ({source_name}):
{content}

---

Верни JSON:

{{
  "entities": [
    {{
      "name": "имя (1-3 слова, именительный падеж)",
      "type": "кто/что это",
      "summary": "краткое описание на языке текста"
    }}
  ],
  "edges": [
    {{
      "source": "имя сущности 1",
      "target": "имя сущности 2",
      "type": "конкретная_связь",
      "evidence_starts": "первые слова фрагмента текста",
      "evidence_ends": "последние слова фрагмента текста"
    }}
  ]
}}"""


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
    evidence_starts: str = ""
    evidence_ends: str = ""


class KnowledgeGraphState:
    """
    In-memory state of the knowledge graph during extraction.
    Updated after each chunk. Passed to LLM as context.
    """

    def __init__(self):
        self.entities: Dict[str, EntityState] = {}  # name → EntityState
        self.edges: List[EdgeState] = []

    def format_for_prompt(self, chunk_text: str = "", max_entities: int = 50) -> str:
        """Format existing entities for inclusion in extraction prompt.
        Only includes entities whose names appear in the chunk text (keyword pre-filter).
        Scales to 100K+ entities — prompt stays small regardless of graph size.
        """
        if not self.entities:
            return "(empty — this is the first chunk)"

        chunk_lower = chunk_text.lower() if chunk_text else ""

        if chunk_lower:
            # Only include entities mentioned in this chunk
            relevant = [ent for ent in self.entities.values()
                        if ent.name in chunk_lower]
            # Always include entities with many source chunks (key characters/concepts)
            if len(relevant) < 5:
                top = sorted(self.entities.values(), key=lambda e: -len(e.source_chunks))[:10]
                seen = {e.name for e in relevant}
                for ent in top:
                    if ent.name not in seen:
                        relevant.append(ent)
        else:
            relevant = sorted(self.entities.values(), key=lambda e: -len(e.source_chunks))[:max_entities]

        relevant = relevant[:max_entities]

        if not relevant:
            return "(no matching entities for this chunk)"

        lines = []
        for ent in relevant:
            lines.append(f"- {ent.name} [{ent.type}]: {ent.summary}")
        return "\n".join(lines)

    def update_from_extraction(self, entities: List[Dict], edges: List[Dict], chunk_id: str) -> Dict[str, int]:
        """Update state with extraction results from one chunk. Returns counts."""
        new_count = 0
        updated_count = 0

        for ent in entities:
            name = ent.get("name", "").lower().strip()
            # Remove [type] suffix from name: "кён [персонаж]" → "кён"
            if "[" in name and name.endswith("]"):
                name = name[:name.index("[")].strip()
            if not name:
                continue

            if name in self.entities:
                # Update existing
                updated_count += 1
                existing = self.entities[name]
                if ent.get("summary"):
                    existing.summary = ent["summary"]
                # Update type if new type is more specific
                if ent.get("type") and ent["type"] != existing.type:
                    if len(ent["type"]) > len(existing.type) or existing.type in ("entity", "concept", "thing"):
                        existing.type = ent["type"]
                existing.source_chunks.append(chunk_id)
            else:
                # Create new
                new_count += 1
                self.entities[name] = EntityState(
                    name=name,
                    type=ent.get("type", "entity"),
                    summary=ent.get("summary", ""),
                    confidence=1.0,
                    source_chunks=[chunk_id],
                )

        for ed in edges:
            src = ed.get("source", "").lower().strip()
            tgt = ed.get("target", "").lower().strip()
            if not src or not tgt:
                continue
            w = float(ed.get("weight", 0.8))

            edge_type = ed.get("type", "RELATES_TO")

            evidence_starts = ed.get("evidence_starts", "")
            evidence_ends = ed.get("evidence_ends", "")

            # Check if edge already exists — update type if action=update
            updated = False
            if ed.get("action") == "update":
                for existing_edge in self.edges:
                    if existing_edge.source == src and existing_edge.target == tgt:
                        existing_edge.edge_type = edge_type
                        existing_edge.weight = w
                        existing_edge.source_chunk = chunk_id
                        existing_edge.evidence_starts = evidence_starts
                        existing_edge.evidence_ends = evidence_ends
                        updated = True
                        break

            if not updated:
                self.edges.append(EdgeState(
                    source=src, target=tgt,
                    edge_type=edge_type, weight=w,
                    source_chunk=chunk_id,
                    evidence_starts=evidence_starts,
                    evidence_ends=evidence_ends,
                ))

        return {"new": new_count, "updated": updated_count}


async def _extract_chunk_with_context(
    llm_client,
    content: str,
    source_name: str,
    chunk_id: str,
    graph_state: KnowledgeGraphState,
    max_retries: int = 4,
) -> Optional[Dict]:
    """Extract entities from one chunk, with context of existing graph state."""

    existing = graph_state.format_for_prompt(chunk_text=content)

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
                logger.warning(f"Empty response for {chunk_id} (attempt {attempt+1}), prompt length={len(prompt)}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1, 2, 4 sec
                continue
            data = _parse_response(response)
            _log_to_file(chunk_id, content, prompt, response, data, source_name)
            if data and (data.get("entities") or data.get("edges")):
                return data
            if not data:
                logger.warning(f"Unparseable response for {chunk_id} (attempt {attempt+1}): {response[:100]}")
        except Exception as e:
            logger.warning(f"Extraction error for {chunk_id} (attempt {attempt+1}): {e}")
        if attempt < max_retries:
            await asyncio.sleep(2 ** attempt)

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
    failed_chunks = []

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        if len(content.strip()) < 20:
            continue

        chunk_id = chunk["node_id"]

        data = await _extract_chunk_with_context(
            llm_client, content, file_name, chunk_id, state
        )

        if data:
            new_ents = data.get("entities", [])
            new_edges = data.get("edges", [])
            counts = state.update_from_extraction(new_ents, new_edges, chunk_id)
            ent_names = [e.get("name", "?") for e in new_ents[:5]]
            edge_types = [e.get("type", "?") for e in new_edges[:3]]
            logger.info(
                f"  [{i+1}/{len(chunks)}] {counts['new']} new + {counts['updated']} updated entities, +{len(new_edges)} edges | "
                f"entities: {ent_names} | edges: {edge_types} | "
                f"total: {len(state.entities)} entities, {len(state.edges)} edges"
            )
        else:
            failed_chunks.append(i)
            logger.warning(f"  [{i+1}/{len(chunks)}] no data returned (will retry)")

        await asyncio.sleep(0.5)

    # Retry failed chunks once more
    if failed_chunks:
        logger.info(f"  {file_name}: retrying {len(failed_chunks)} failed chunks")
        for i in failed_chunks:
            chunk = chunks[i]
            chunk_id = chunk["node_id"]
            content = chunk.get("content", "")
            await asyncio.sleep(2)
            data = await _extract_chunk_with_context(llm_client, content, file_name, chunk_id, state)
            if data:
                new_ents = data.get("entities", [])
                new_edges = data.get("edges", [])
                state.update_from_extraction(new_ents, new_edges, chunk_id)
                logger.info(f"  [{i+1}/{len(chunks)}] RETRY OK: +{len(new_ents)} entities, +{len(new_edges)} edges")
            else:
                logger.error(f"  [{i+1}/{len(chunks)}] RETRY FAILED — chunk lost")

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
                # Update type when merging if new type is more specific
                if ent.type and ent.type != existing.type:
                    if len(ent.type) > len(existing.type) or existing.type in ("entity", "concept", "thing"):
                        existing.type = ent.type
                # Append summary if different
                if ent.summary and ent.summary != existing.summary:
                    existing.summary = ent.summary  # Latest wins
                existing.source_chunks.extend(ent.source_chunks)
            else:
                global_state.entities[name] = EntityState(
                    name=ent.name,
                    type=ent.type,
                    summary=ent.summary,
                    confidence=1.0,
                    source_chunks=list(ent.source_chunks),
                )

        global_state.edges.extend(state.edges)

    return global_state


def _parse_response(text: str) -> Optional[Dict]:
    """Parse extraction response — JSON primary, text fallback.
    With 131K max output from MiniMax, JSON won't truncate.
    """
    text = text.strip()

    # Strip <think> blocks from reasoning models
    if "<think>" in text and "</think>" in text:
        text = text[text.find("</think>") + 8:].strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try JSON first
    try:
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start >= 0 and json_end > json_start:
            data = json.loads(text[json_start:json_end + 1])
            if isinstance(data, dict):
                # Clean entity names: remove [brackets]
                for ent in data.get("entities", []):
                    name = ent.get("name", "")
                    if "[" in name and name.endswith("]"):
                        bracket = name.index("[")
                        ent["name"] = name[:bracket].strip()
                    ent["name"] = ent.get("name", "").lower().strip()
                    # Validate type
                    etype = ent.get("type", "")
                    if len(etype) > 30:
                        ent["summary"] = etype + " " + ent.get("summary", "")
                        ent["type"] = "концепция"
                return data
    except json.JSONDecodeError:
        pass

    # Try JSON array
    try:
        json_start = text.find("[")
        json_end = text.rfind("]")
        if json_start >= 0 and json_end > json_start:
            data = json.loads(text[json_start:json_end + 1])
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data[0]
    except json.JSONDecodeError:
        pass

    # Text fallback: parse NEW:/UPD:/EDGE: lines
    entities = []
    edges = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("NEW:", "UPD:")):
            action = "create" if line.startswith("NEW:") else "update"
            parts = line[4:].split("|")
            if len(parts) >= 2:
                name = parts[0].strip().lower()
                if "[" in name and name.endswith("]"):
                    name = name[:name.index("[")].strip()
                etype = parts[1].strip().lower() if len(parts[1].strip()) <= 30 else "концепция"
                summary = parts[2].strip() if len(parts) >= 3 else ""
                entities.append({"name": name, "type": etype, "summary": summary, "action": action})
        elif line.startswith("EDGE:"):
            rest = line[5:].strip()
            arrow = rest.find("->")
            if arrow >= 0:
                source = rest[:arrow].strip().lower()
                after = rest[arrow+2:].split("|")
                if len(after) >= 2:
                    edges.append({"source": source, "target": after[0].strip().lower(),
                                  "type": after[1].strip(), "weight": 0.8,
                                  "evidence_starts": "", "evidence_ends": ""})

    if entities or edges:
        return {"entities": entities, "edges": edges}

    logger.warning(f"Failed to parse response: {text[:200]}")
    return None
