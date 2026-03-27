"""
Agentic GraphRAG - Markdown Parser
Парсер документации (.md файлов) для создания DOCUMENT узлов
и автоматической привязки к узлам кода через DESCRIBES рёбра.
"""

import os
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Set

from ..graph.models import GraphNode, GraphEdge, NodeType, EdgeType


@dataclass
class ParsedMarkdown:
    """Результат парсинга .md файла"""
    file_path: str
    node: GraphNode = None
    mentions: Set[str] = field(default_factory=set)  # Имена символов кода найденные в тексте
    success: bool = True
    error: str = ""


class MarkdownParser:
    """
    Детерминированный парсер Markdown-документов.

    Создаёт DOCUMENT узел из .md файла и извлекает
    упоминания символов кода (имена классов, функций)
    для последующей автолинковки через DESCRIBES рёбра.

    LLM здесь НЕ участвует — только regex/string matching.
    """

    # Паттерны для извлечения упоминаний кода из текста
    # Backtick-обёрнутые идентификаторы: `PhysicsWorld`, `RigidBody::update`
    RE_BACKTICK = re.compile(r'`([A-Z][A-Za-z0-9_:]+)`')
    # CamelCase слова (вероятные имена классов): PhysicsBody, RigidBody
    RE_CAMELCASE = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b')
    # Scope-qualified имена: PhysicsWorld::step, RigidBody::checkCollision
    RE_SCOPED = re.compile(r'\b([A-Z][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*)\b')

    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)

    def _generate_node_id(self, name: str, file_path: str) -> str:
        """Генерация уникального ID для document узла"""
        content = f"document:{name}:{file_path}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"document::{name}#{hash_suffix}"

    def _extract_title(self, content: str) -> str:
        """Извлечение заголовка из первого # заголовка или имени файла"""
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return ""

    def _extract_summary(self, content: str) -> str:
        """Первый абзац после заголовка как summary"""
        lines = content.split('\n')
        summary_lines = []
        past_title = False
        for line in lines:
            stripped = line.strip()
            if not past_title:
                if stripped.startswith('# '):
                    past_title = True
                continue
            if stripped == '':
                if summary_lines:
                    break
                continue
            summary_lines.append(stripped)
        return ' '.join(summary_lines)[:300]  # Max 300 chars

    def _extract_mentions(self, content: str) -> Set[str]:
        """
        Извлечение упоминаний символов кода из текста.
        Ищет:
        1. `BacktickIdentifiers` — явные ссылки на код
        2. CamelCase слова — вероятные имена классов
        3. Scoped names (Class::method) — явные ссылки на методы
        """
        mentions = set()

        for match in self.RE_BACKTICK.finditer(content):
            mentions.add(match.group(1))

        for match in self.RE_SCOPED.finditer(content):
            mentions.add(match.group(1))

        for match in self.RE_CAMELCASE.finditer(content):
            mentions.add(match.group(1))

        return mentions

    def parse_file(self, file_path: str) -> ParsedMarkdown:
        """Парсинг одного .md файла"""
        result = ParsedMarkdown(file_path=file_path)

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            result.success = False
            result.error = f"Failed to read: {e}"
            return result

        if not content.strip():
            result.success = False
            result.error = "Empty file"
            return result

        # Имя файла без расширения как fallback имя
        basename = os.path.splitext(os.path.basename(file_path))[0]
        title = self._extract_title(content) or basename
        summary = self._extract_summary(content) or f"Document: {title}"

        line_count = content.count('\n') + 1

        node = GraphNode(
            node_id=self._generate_node_id(basename, file_path),
            type=NodeType.DOCUMENT.value,
            name=title,
            signature=f"doc::{basename}",
            file_path=file_path,
            line_start=1,
            line_end=line_count,
            source_code=content,
            summary=summary,
            tags=["business_logic"],  # Документация = бизнес-логика по ТЗ
        )

        result.node = node
        result.mentions = self._extract_mentions(content)
        return result


def resolve_mentions_to_edges(
    doc_node_id: str,
    mentions: Set[str],
    existing_nodes: List[GraphNode]
) -> List[GraphEdge]:
    """
    Резолвит упоминания символов кода в тексте документа
    к конкретным узлам графа и создаёт DESCRIBES рёбра.

    Матчинг: имя из текста совпадает с name узла кода.
    Например, "PhysicsWorld" в .md матчится на class::PhysicsWorld.

    Args:
        doc_node_id: ID документного узла (source)
        mentions: Множество имён, найденных в тексте
        existing_nodes: Список существующих узлов кода из графа

    Returns:
        Список DESCRIBES рёбер
    """
    edges = []
    matched = set()

    for node in existing_nodes:
        if node.type == NodeType.DOCUMENT.value:
            continue  # Не линкуем документы к документам

        # Точное совпадение имени
        if node.name in mentions and node.node_id not in matched:
            edges.append(GraphEdge(
                source_id=doc_node_id,
                target_id=node.node_id,
                edge_type=EdgeType.DESCRIBES.value,
                metadata={"match_type": "exact_name", "matched_text": node.name}
            ))
            matched.add(node.node_id)

    return edges
