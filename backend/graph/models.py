"""
Agentic GraphRAG - Graph Models
Модели данных для графа знаний
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import json


class NodeType(Enum):
    """Типы узлов графа — универсальная схема для кода, текста и знаний"""
    # === Код (AST) ===
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    NAMESPACE = "namespace"
    ENUM = "enum"
    STRUCT = "struct"
    INTERFACE = "interface"

    # === Документы ===
    DOCUMENT = "document"      # Любой текстовый документ / чанк диалога
    PLAN = "plan"              # Планы агента

    # === Универсальные сущности (Knowledge Graph) ===
    ENTITY = "entity"          # Именованная сущность: персона, продукт, технология, место
    TOPIC = "topic"            # Тематический кластер: "cooking", "finance", "physics_engine"
    FACT = "fact"              # Конкретный факт: "user knows Python", "Bitcoin ATH = $108K"
    SKILL = "skill"            # Навык/компетенция: "C++ expert", "knows React"
    PREFERENCE = "preference"  # Предпочтение пользователя: "prefers dark mode"


class EdgeType(Enum):
    """Типы связей — структурные (AST) + семантические (Knowledge Graph)"""
    # === Структурные (детерминированные, из AST) ===
    CALLS = "CALLS"
    INCLUDES = "INCLUDES"
    IMPLEMENTS = "IMPLEMENTS"
    INHERITS = "INHERITS"
    DEPENDS_ON = "DEPENDS_ON"
    DEFINES = "DEFINES"
    HAS_METHOD = "HAS_METHOD"

    # === Семантические (из entity extractor) ===
    RELATES_TO = "RELATES_TO"        # Общая связь между сущностями
    DESCRIBES = "DESCRIBES"          # Документ описывает код/сущность
    MENTIONS = "MENTIONS"            # Контент упоминает сущность
    REVEALS = "REVEALS"              # Контент раскрывает факт ("я python-разработчик")
    BELONGS_TO = "BELONGS_TO"        # Сущность принадлежит топику
    CONTRADICTS = "CONTRADICTS"      # Факт противоречит другому факту
    SAME_AS = "SAME_AS"              # Entity resolution: два узла = одна сущность
    SKILLED_IN = "SKILLED_IN"        # Персона владеет навыком
    SUPERSEDES = "SUPERSEDES"        # Новый факт заменяет старый (temporal)


class TagEnum(Enum):
    """
    Жесткий Enum для тегов.
    LLM НЕ может генерировать новые теги.
    """
    # Архитектурные теги (код)
    CORE = "core"
    UTILITY = "utility"
    CONFIG = "config"
    DATA = "data"
    NETWORK = "network"
    PARSER = "parser"
    STORAGE = "storage"
    API = "api"
    MIDDLEWARE = "middleware"
    HANDLER = "handler"

    # Статусные теги
    ABSTRACT = "abstract"
    INTERFACE = "interface"
    CONCRETE = "concrete"
    DEPRECATED = "deprecated"

    # Теги компонентов
    ENTRY_POINT = "entry_point"
    TEST = "test"
    MOCK = "mock"

    # Семантические теги (код)
    BUSINESS_LOGIC = "business_logic"
    INFRASTRUCTURE = "infrastructure"
    BOUNDED_CONTEXT = "bounded_context"

    # Универсальные теги (знания)
    PERSONAL = "personal"           # Факт о пользователе
    TECHNICAL = "technical"         # Техническая информация
    CREATIVE = "creative"           # Творческий контент
    FINANCIAL = "financial"         # Финансы / крипто / инвестиции
    HEALTH = "health"               # Здоровье
    SOCIAL = "social"               # Общение / отношения
    EDUCATION = "education"         # Обучение / книги / курсы
    PHILOSOPHY = "philosophy"       # Философия / религия / мировоззрение
    ENTERTAINMENT = "entertainment" # Развлечения / ранобе / игры
    TOOLCHAIN = "toolchain"         # Инструменты сборки / CI/CD


@dataclass
class GraphNode:
    """
    Узел графа знаний.
    
    Properties:
    - node_id: Уникальный ID (формат: type::name#hash)
    - type: Тип узла из NodeType
    - name: Имя (сигнатура без полного пути)
    - signature: Полная сигнатура
    - file_path: Путь к файлу
    - line_start, line_end: Позиция в файле
    - source_code: Полный исходный код
    - summary: Краткое описание (LLM, Фаза B)
    - tags: Теги из TagEnum
    """
    node_id: str
    type: str
    name: str
    signature: str
    file_path: str
    line_start: int
    line_end: int
    source_code: str = ""
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    embedding_id: Optional[str] = None  # ID в векторной БД (Фаза B)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для JSON"""
        return {
            "node_id": self.node_id,
            "type": self.type,
            "name": self.name,
            "signature": self.signature,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "source_code": self.source_code,
            "summary": self.summary,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "embedding_id": self.embedding_id
        }
    
    def to_api_dict(self, include_code: bool = False) -> Dict[str, Any]:
        """
        Конвертация для API.
        По умолчанию source_code НЕ включается (экономия токенов).
        """
        d = self.to_dict()
        if not include_code:
            d.pop("source_code", None)
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        """Создание из словаря"""
        def _to_str(val) -> str:
            """Convert Neo4j DateTime or any value to ISO string"""
            if val is None:
                return datetime.utcnow().isoformat()
            if isinstance(val, str):
                return val
            if hasattr(val, "isoformat"):
                return val.isoformat()
            return str(val)

        return cls(
            node_id=data["node_id"],
            type=data["type"],
            name=data["name"],
            signature=data.get("signature", ""),
            file_path=data.get("file_path", ""),
            line_start=data.get("line_start", 0),
            line_end=data.get("line_end", 0),
            source_code=data.get("source_code", ""),
            summary=data.get("summary", ""),
            tags=data.get("tags", []),
            created_at=_to_str(data.get("created_at")),
            updated_at=_to_str(data.get("updated_at")),
            embedding_id=data.get("embedding_id")
        )


@dataclass
class GraphEdge:
    """
    Ребро графа (связь между узлами).
    
    Properties:
    - source_id: ID исходного узла
    - target_id: ID целевого узла
    - edge_type: Тип связи из EdgeType
    - metadata: Дополнительные данные
    """
    source_id: str
    target_id: str
    edge_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.edge_type,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=data["type"],
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat())
        )


@dataclass 
class SubgraphResult:
    """
    Результат запроса подграфа.
    Используется API эндпоинтом get_subgraph.
    """
    center_node: GraphNode
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    depth: int
    total_nodes: int
    total_edges: int
    
    def to_dict(self, include_code: bool = False) -> Dict[str, Any]:
        return {
            "center_node": self.center_node.to_api_dict(include_code),
            "nodes": [n.to_api_dict(include_code) for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "depth": self.depth,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges
        }


@dataclass
class IngestResult:
    """Результат индексации файла/директории"""
    success: bool
    files_processed: int
    nodes_created: int
    edges_created: int
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "files_processed": self.files_processed,
            "nodes_created": self.nodes_created,
            "edges_created": self.edges_created,
            "errors": self.errors
        }


@dataclass
class VirtualPatch:
    """
    Виртуальный патч для Shadow Graph (Фаза B).
    Незакоммиченные изменения из текущей сессии агента.
    """
    node_id: str
    file_path: str
    old_code: str
    new_code: str
    session_id: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "file_path": self.file_path,
            "old_code": self.old_code,
            "new_code": self.new_code,
            "session_id": self.session_id,
            "created_at": self.created_at
        }


# Валидация тегов
ALLOWED_TAGS = {tag.value for tag in TagEnum}


def validate_tags(tags: List[str]) -> List[str]:
    """
    Валидация тегов.
    Принимает любые теги — система не ограничивает категории.
    """
    return [t.lower().strip() for t in tags if t and t.strip()]
