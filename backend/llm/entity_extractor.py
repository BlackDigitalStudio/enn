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
      "type": "entity|topic|fact|skill|preference",
      "summary": "one-line description",
      "confidence": 0.0-1.0
    }}
  ],
  "edges": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "type": "MENTIONS|REVEALS|BELONGS_TO|SKILLED_IN|DESCRIBES|CONTRADICTS|RELATES_TO",
      "weight": 0.0-1.0
    }}
  ]
}}

RULES:
- "entity" = named thing (person, library, class, product, place)
- "topic" = thematic cluster (programming, cooking, finance, health)
- "fact" = specific verifiable claim ("uses CMake 3.31", "Bitcoin ATH = $108K")
- "skill" = competency ("C++ templates", "React hooks")
- "preference" = user preference ("prefers dark mode", "likes minimal UI")
- Confidence < 0.3 = skip it
- Entity names must be canonical (lowercase, singular, no file paths)
- For code: extract class names, libraries, patterns — NOT every function call
- For text: extract key concepts, people, claims — NOT filler words
- Maximum 20 entities per chunk (focus on most important)
- Maximum 30 edges per chunk
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
      {{"name": "...", "type": "entity|topic|fact|skill|preference", "summary": "...", "confidence": 0.0-1.0}}
    ],
    "edges": [
      {{"source": "...", "target": "...", "type": "MENTIONS|REVEALS|BELONGS_TO|SKILLED_IN|DESCRIBES|RELATES_TO", "weight": 0.0-1.0}}
    ]
  }}
]

RULES:
- Max 10 entities + 15 edges per item
- Entity names: lowercase, canonical, deduplicated across items
- Skip confidence < 0.3
- For code: class names, libraries, patterns — not every variable
- For docs: key concepts, claims, references — not filler
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
    similarity_threshold: float = 0.9,
) -> Dict[str, str]:
    """
    Entity resolution — дедупликация сущностей.

    Простая стратегия (без эмбеддингов):
    - Точное совпадение имени → merge
    - Имя является подстрокой другого → merge к более длинному
    - Совпадение с нормализацией (пробелы, дефисы, подчёркивания) → merge

    Returns: {original_name: canonical_name}
    """
    name_map: Dict[str, str] = {}
    canonical: Dict[str, ExtractedEntity] = {}

    for ent in entities:
        normalized = _normalize_name(ent.name)

        # Точное совпадение после нормализации
        if normalized in canonical:
            name_map[ent.name] = canonical[normalized].name
            # Обновляем confidence если выше
            if ent.confidence > canonical[normalized].confidence:
                canonical[normalized] = ent
            continue

        # Проверяем подстроки
        merged = False
        for existing_norm, existing_ent in list(canonical.items()):
            if normalized in existing_norm or existing_norm in normalized:
                # Берём более длинное имя как каноническое
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
