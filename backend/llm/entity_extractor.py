"""
Entity Extractor — извлечение сущностей, фактов и топиков из любого контента.

Работает с любым типом данных:
- Код (C++, Python) → технические сущности, зависимости
- Документация (.md) → описания, архитектурные решения
- Диалоги с нейросетью → факты о пользователе, предпочтения
- Произвольный текст → топики, именованные сущности, ключевые факты

Выход: список сущностей (ENTITY, FACT, TOPIC, SKILL, PREFERENCE)
       + список рёбер между ними и исходным узлом
"""

import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Промпт для LLM — универсальная экстракция
EXTRACTION_PROMPT = """You are a knowledge graph entity extractor. Analyze the content below and extract structured entities.

CONTENT TYPE: {content_type}
SOURCE: {source_name}

CONTENT:
{content}

---

Extract ALL meaningful entities. Return ONLY valid JSON (no markdown, no commentary):

{{
  "entities": [
    {{
      "name": "unique canonical name (lowercase, no duplicates)",
      "type": "choose the most specific type (e.g. person, character, location, item, ability, ingredient, library, class, function, concept, event, organization, technology, recipe, emotion, relationship...)",
      "summary": "one-line description in the same language as the content",
      "confidence": 0.0-1.0
    }}
  ],
  "edges": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "type": "choose the most descriptive relationship (e.g. KNOWS, USES, LOCATED_IN, PART_OF, FIGHTS, LOVES, CREATES, DEPENDS_ON, TEACHES, CONTRADICTS, EVOLVES_INTO...)",
      "weight": 0.0-1.0
    }}
  ]
}}

RULES:
- Use the most specific type possible — "character" not "entity", "ingredient" not "thing"
- Entity names must be canonical (lowercase, singular)
- Summary should be in the same language as the source content
- Confidence < 0.3 = skip
- For code: class names, libraries, patterns — NOT every variable
- For text: people, places, events, claims, relationships — NOT filler words
- Maximum 20 entities + 30 edges per chunk (focus on most important)
"""

# Батч-промпт для обработки нескольких узлов за раз
BATCH_EXTRACTION_PROMPT = """You are a knowledge graph entity extractor. Process these {count} items and extract entities from each.

{items_text}

---

Return ONLY valid JSON array (no markdown):
[
  {{
    "source_id": "item ID from above",
    "entities": [
      {{"name": "canonical name (lowercase)", "type": "most specific type (person, location, item, ability, library, class, concept, event, ingredient, technology...)", "summary": "one-line description in the same language as content", "confidence": 0.0-1.0}}
    ],
    "edges": [
      {{"source": "entity name", "target": "entity name", "type": "descriptive relationship (KNOWS, USES, LOCATED_IN, PART_OF, CREATES, DEPENDS_ON, CONTRADICTS...)", "weight": 0.0-1.0}}
    ]
  }}
]

RULES:
- Max 10 entities + 15 edges per item
- Use specific types — "character" not "entity", "ingredient" not "thing"
- Entity names: lowercase, canonical, deduplicated across items
- Summaries in the same language as source content
- Skip confidence < 0.3
"""


@dataclass
class ExtractedEntity:
    """Извлечённая сущность"""
    name: str
    type: str  # entity, topic, fact, skill, preference
    summary: str
    confidence: float
    source_node_id: str  # откуда извлечено


@dataclass
class ExtractedEdge:
    """Извлечённое ребро"""
    source_name: str
    target_name: str
    edge_type: str
    weight: float
    source_node_id: str  # контекст откуда извлечено


async def extract_entities_single(
    llm_client,
    content: str,
    content_type: str,
    source_name: str,
    source_node_id: str,
) -> Tuple[List[ExtractedEntity], List[ExtractedEdge]]:
    """
    Извлекает сущности из одного куска контента.
    """
    if not content or len(content.strip()) < 20:
        return [], []

    # Обрезаем контент до 4000 символов (экономия токенов)
    truncated = content[:4000]

    prompt = EXTRACTION_PROMPT.format(
        content_type=content_type,
        source_name=source_name,
        content=truncated,
    )

    try:
        response = await llm_client.generate(prompt)
        if not response:
            return [], []

        data = _parse_json_response(response)
        if not data:
            return [], []

        entities = []
        for e in data.get("entities", []):
            conf = float(e.get("confidence", 0))
            if conf < 0.3:
                continue
            entities.append(ExtractedEntity(
                name=e["name"].lower().strip(),
                type=e.get("type", "entity"),
                summary=e.get("summary", ""),
                confidence=conf,
                source_node_id=source_node_id,
            ))

        edges = []
        for ed in data.get("edges", []):
            w = float(ed.get("weight", 0.5))
            if w < 0.2:
                continue
            edges.append(ExtractedEdge(
                source_name=ed["source"].lower().strip(),
                target_name=ed["target"].lower().strip(),
                edge_type=ed.get("type", "RELATES_TO"),
                weight=w,
                source_node_id=source_node_id,
            ))

        return entities, edges

    except Exception as e:
        logger.error(f"Entity extraction failed for {source_name}: {e}")
        return [], []


async def _process_one_batch(
    llm_client,
    batch: List[Dict[str, Any]],
    batch_idx: int,
    total: int,
) -> Tuple[List[ExtractedEntity], List[ExtractedEdge]]:
    """Process a single batch of items through LLM. Called in parallel."""
    entities = []
    edges = []

    items_text = ""
    for idx, item in enumerate(batch):
        content = (item.get("content") or "")[:2000]
        items_text += f"\n--- ITEM {idx+1} (id: {item['node_id']}, type: {item['type']}) ---\n"
        items_text += f"Name: {item['name']}\n"
        items_text += f"Content:\n{content}\n"

    prompt = BATCH_EXTRACTION_PROMPT.format(
        count=len(batch),
        items_text=items_text,
    )

    try:
        response = await llm_client.generate(prompt)
        if not response:
            return entities, edges

        results = _parse_json_response(response)
        if not results:
            return entities, edges

        if isinstance(results, dict):
            results = [results]

        for result in results:
            src_id = result.get("source_id", "")

            for e in result.get("entities", []):
                conf = float(e.get("confidence", 0))
                if conf < 0.3:
                    continue
                entities.append(ExtractedEntity(
                    name=e["name"].lower().strip(),
                    type=e.get("type", "entity"),
                    summary=e.get("summary", ""),
                    confidence=conf,
                    source_node_id=src_id,
                ))

            for ed in result.get("edges", []):
                w = float(ed.get("weight", 0.5))
                if w < 0.2:
                    continue
                edges.append(ExtractedEdge(
                    source_name=ed["source"].lower().strip(),
                    target_name=ed["target"].lower().strip(),
                    edge_type=ed.get("type", "RELATES_TO"),
                    weight=w,
                    source_node_id=src_id,
                ))

    except Exception as e:
        logger.error(f"Batch extraction failed (batch {batch_idx}): {e}")

    return entities, edges


async def extract_entities_batch(
    llm_client,
    items: List[Dict[str, Any]],
    batch_size: int = 50,
    concurrency: int = 10,
) -> Tuple[List[ExtractedEntity], List[ExtractedEdge]]:
    """
    Батч-экстракция сущностей с параллельными вызовами.

    - batch_size: сколько файлов в одном LLM-вызове (50 ≈ 30K input tokens)
    - concurrency: сколько вызовов отправляем одновременно (10 параллельных)

    19K файлов: 384 батча → 39 раундов по 10 → ~30 мин вместо 53 часов.
    """
    import asyncio

    all_entities = []
    all_edges = []

    # Split items into batches
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])

    total_items = len(items)
    processed = 0

    # Process batches in parallel groups
    for round_idx in range(0, len(batches), concurrency):
        group = batches[round_idx:round_idx + concurrency]

        tasks = [
            _process_one_batch(llm_client, batch, round_idx + j, total_items)
            for j, batch in enumerate(group)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Parallel batch failed: {result}")
                continue
            ents, eds = result
            all_entities.extend(ents)
            all_edges.extend(eds)

        processed += sum(len(b) for b in group)
        logger.info(f"Entity extraction: {processed}/{total_items} items processed ({len(all_entities)} entities, {len(all_edges)} edges)")

    return all_entities, all_edges


def resolve_entities(
    entities: List[ExtractedEntity],
    similarity_threshold: float = 0.85,
) -> Dict[str, str]:
    """
    Entity resolution — дедупликация сущностей.

    Двухфазная стратегия:
    1. Строковая: нормализация, точное совпадение, подстроки (быстро)
    2. Семантическая: cosine similarity эмбеддингов имён (для кросс-язычных матчей)

    "кулинария" ↔ "cooking" → merge (семантически одно и то же)
    "react" ↔ "react hooks" → НЕ merge (подстрока, но разные типы)

    Returns: {original_name: canonical_name}
    """
    name_map: Dict[str, str] = {}
    canonical: Dict[str, ExtractedEntity] = {}

    # Phase 1: String-based resolution
    for ent in entities:
        normalized = _normalize_name(ent.name)

        if normalized in canonical:
            name_map[ent.name] = canonical[normalized].name
            if ent.confidence > canonical[normalized].confidence:
                canonical[normalized] = ent
            continue

        # Substring match — only if names are very close (avoids "react" ↔ "reactivity")
        merged = False
        for existing_norm, existing_ent in list(canonical.items()):
            if normalized == existing_norm:
                continue  # Already handled above
            # Only substring-merge if one is ≥80% of the other (close names)
            shorter, longer = sorted([normalized, existing_norm], key=len)
            if shorter in longer and len(shorter) / len(longer) >= 0.8:
                if len(normalized) >= len(existing_norm):
                    canonical[normalized] = ent
                    name_map[existing_ent.name] = ent.name
                    name_map[ent.name] = ent.name
                else:
                    name_map[ent.name] = existing_ent.name
                merged = True
                break

        if not merged:
            canonical[normalized] = ent
            name_map[ent.name] = ent.name

    # Phase 2: Semantic resolution via embeddings
    # Merge entities with different names but same meaning (cross-language)
    try:
        from ..llm.embeddings import _get_model
        model = _get_model()

        canon_list = list(canonical.items())
        if len(canon_list) > 1:
            # Compare "name: summary" pairs, not just names
            # This prevents merging "python (language)" with "python (snake)"
            texts = [f"{ent.name}: {ent.summary}" for _, ent in canon_list]
            embeddings = model.encode(texts, normalize_embeddings=True)

            for i in range(len(canon_list)):
                for j in range(i + 1, len(canon_list)):
                    sim = float(embeddings[i] @ embeddings[j])
                    if sim >= similarity_threshold:
                        # Merge j into i (i is canonical)
                        name_j = canon_list[j][1].name
                        name_i = canon_list[i][1].name
                        name_map[name_j] = name_i
                        logger.info(f"Semantic merge: '{name_j}' → '{name_i}' (sim={sim:.3f})")
    except Exception as e:
        logger.warning(f"Semantic entity resolution skipped: {e}")

    return name_map


def _normalize_name(name: str) -> str:
    """Нормализация имени для дедупликации"""
    return name.lower().strip().replace("-", "_").replace(" ", "_").replace("::", "_")


def _parse_json_response(text: str) -> Any:
    """Парсим JSON из ответа LLM (может содержать markdown-обёртку)"""
    text = text.strip()

    # Убираем markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)

    # Пробуем парсить как есть
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Ищем первый { или [
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Ищем соответствующую закрывающую скобку
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

    logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
    return None
