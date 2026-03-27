"""
Agentic GraphRAG - Neo4j Storage
Хранилище графа на базе Neo4j
"""

import os
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, ConstraintError

from .models import GraphNode, GraphEdge, SubgraphResult, NodeType, EdgeType, validate_tags

logger = logging.getLogger(__name__)


class Neo4jStorage:
    """
    Хранилище графа на Neo4j.
    
    Операции:
    - create_node / create_edge
    - get_node / get_neighbors
    - get_subgraph (с Context Pruning)
    - bulk operations (UPSERT)
    - search by type/tag
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Optional[Driver] = None
    
    def connect(self) -> bool:
        """Установка соединения с Neo4j"""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Проверка соединения
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j: {self.uri}")
            
            # Инициализация индексов
            self._create_constraints()
            return True
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
        except Exception as e:
            logger.error(f"Neo4j connection error: {e}")
            return False
    
    def close(self):
        """Закрытие соединения"""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")
    
    def _create_constraints(self):
        """Создание индексов и constraints"""
        constraints = [
            "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.node_id IS UNIQUE",
            "CREATE INDEX node_type IF NOT EXISTS FOR (n:Node) ON (n.type)",
            "CREATE INDEX node_name IF NOT EXISTS FOR (n:Node) ON (n.name)",
            "CREATE INDEX edge_type IF NOT EXISTS FOR ()-[e:EDGE]-() ON (e.type)"
        ]
        
        with self._driver.session(database=self.database) as session:
            for cql in constraints:
                try:
                    session.run(cql)
                except ConstraintError:
                    pass  # Constraint уже существует
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ============== Node Operations ==============
    
    def create_node(self, node: GraphNode) -> bool:
        """
        Создание узла в графе.
        Использует UPSERT (MERGE) для инкрементальных обновлений.
        """
        # Валидация тегов
        node.tags = validate_tags(node.tags)
        
        cypher = """
        MERGE (n:Node {node_id: $node_id})
        SET n.type = $type,
            n.name = $name,
            n.signature = $signature,
            n.file_path = $file_path,
            n.line_start = $line_start,
            n.line_end = $line_end,
            n.source_code = $source_code,
            n.summary = $summary,
            n.tags = $tags,
            n.updated_at = datetime()
        RETURN n
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, **node.to_dict())
            return result.single() is not None
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Получение узла по ID"""
        cypher = "MATCH (n:Node {node_id: $node_id}) RETURN n"
        
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, node_id=node_id)
            record = result.single()
            
            if record:
                return GraphNode.from_dict(dict(record["n"]))
            return None
    
    def get_node_code(self, node_id: str) -> Optional[str]:
        """Получение только исходного кода узла"""
        node = self.get_node(node_id)
        return node.source_code if node else None
    
    def delete_node(self, node_id: str) -> bool:
        """Удаление узла и связанных ребер"""
        cypher = """
        MATCH (n:Node {node_id: $node_id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, node_id=node_id)
            record = result.single()
            return record and record["deleted"] > 0
    
    def get_nodes_bulk(self, node_ids: List[str]) -> List[GraphNode]:
        """Получение нескольких узлов за 1 Cypher-запрос (решение N+1)"""
        if not node_ids:
            return []
        cypher = """
        MATCH (n:Node)
        WHERE n.node_id IN $node_ids
        RETURN n
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, node_ids=node_ids)
            return [GraphNode.from_dict(dict(record["n"])) for record in result]

    def update_node(self, node: GraphNode) -> bool:
        """Обновление свойств существующего узла"""
        cypher = """
        MATCH (n:Node {node_id: $node_id})
        SET n.summary = $summary,
            n.updated_at = datetime()
        RETURN n
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, node_id=node.node_id, summary=node.summary)
            return result.single() is not None

    def bulk_create_nodes(self, nodes: List[GraphNode]) -> int:
        """Массовое создание узлов в транзакции"""
        count = 0
        with self._driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                for node in nodes:
                    node.tags = validate_tags(node.tags)
                    cypher = """
                    MERGE (n:Node {node_id: $node_id})
                    SET n.type = $type,
                        n.name = $name,
                        n.signature = $signature,
                        n.file_path = $file_path,
                        n.line_start = $line_start,
                        n.line_end = $line_end,
                        n.source_code = $source_code,
                        n.summary = $summary,
                        n.tags = $tags,
                        n.updated_at = datetime()
                    """
                    tx.run(cypher, **node.to_dict())
                    count += 1
                tx.commit()
        return count
    
    # ============== Edge Operations ==============
    
    def create_edge(self, edge: GraphEdge) -> bool:
        """Создание ребра между узлами"""
        cypher = """
        MATCH (source:Node {node_id: $source_id})
        MATCH (target:Node {node_id: $target_id})
        MERGE (source)-[e:EDGE]->(target)
        SET e.type = $type,
            e.metadata = $metadata,
            e.created_at = datetime()
        RETURN e
        """
        
        with self._driver.session(database=self.database) as session:
            params = edge.to_dict()
            params["metadata"] = json.dumps(params.get("metadata", {}))
            result = session.run(cypher, **params)
            return result.single() is not None

    def delete_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        """Удаление ребра"""
        cypher = """
        MATCH (source:Node {node_id: $source_id})-[e:EDGE {type: $type}]->(target:Node {node_id: $target_id})
        DELETE e
        RETURN count(e) as deleted
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, source_id=source_id, target_id=target_id, type=edge_type)
            record = result.single()
            return record and record["deleted"] > 0
    
    def bulk_create_edges(self, edges: List[GraphEdge]) -> int:
        """Массовое создание ребер"""
        count = 0
        with self._driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                for edge in edges:
                    cypher = """
                    MATCH (source:Node {node_id: $source_id})
                    MATCH (target:Node {node_id: $target_id})
                    MERGE (source)-[e:EDGE]->(target)
                    SET e.type = $type, e.metadata = $metadata, e.created_at = datetime()
                    """
                    params = edge.to_dict()
                    import json as _json
                    params["metadata"] = _json.dumps(params.get("metadata", {}))
                    tx.run(cypher, **params)
                    count += 1
                tx.commit()
        return count
    
    # ============== Navigation ==============
    
    def get_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        edge_types: List[str] = None,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Получение соседей узла с заданной глубиной.

        Args:
            node_id: ID стартового узла
            depth: Глубина обхода (1-3)
            edge_types: Фильтр по типам ребер
            direction: Направление ("outgoing", "incoming", "both")
        """
        max_depth = min(depth, 3)

        # Формирование паттерна направления
        if direction == "outgoing":
            left, right = "-", "->"
        elif direction == "incoming":
            left, right = "<-", "-"
        else:
            left, right = "-", "-"

        # Фильтр по типу ребра
        type_filter = ""
        params: Dict[str, Any] = {"node_id": node_id}
        if edge_types:
            type_filter = "WHERE ALL(r IN relationships(path) WHERE r.type IN $edge_types)"
            params["edge_types"] = edge_types

        cypher = f"""
        MATCH path = (start:Node {{node_id: $node_id}}){left}[*1..{max_depth}]{right}(end:Node)
        {type_filter}
        WITH nodes(path) AS path_nodes, relationships(path) AS path_edges
        UNWIND path_nodes AS n
        WITH collect(DISTINCT n) AS unique_nodes, path_edges
        UNWIND path_edges AS e
        WITH unique_nodes, collect(DISTINCT e) AS unique_edges
        RETURN unique_nodes, unique_edges
        """

        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            record = result.single()

            if not record:
                return [], []

            nodes = [GraphNode.from_dict(dict(n)) for n in record["unique_nodes"]]

            # Ребра из Neo4j relationship objects
            edges = []
            for rel in record["unique_edges"]:
                edge = GraphEdge(
                    source_id=dict(rel.start_node)["node_id"],
                    target_id=dict(rel.end_node)["node_id"],
                    edge_type=rel.get("type", "RELATES_TO"),
                    metadata=json.loads(rel.get("metadata", "{}")) if rel.get("metadata") else {}
                )
                edges.append(edge)

            return nodes, edges
    
    def get_subgraph(
        self,
        node_id: str,
        depth: int = 2,
        edge_types: List[str] = None,
        include_code: bool = False
    ) -> Optional[SubgraphResult]:
        """
        Получение подграфа с Context Pruning.
        
        Это основной API эндпоинт для навигации агента.
        Возвращает метаданные без source_code по умолчанию.
        """
        center_node = self.get_node(node_id)
        if not center_node:
            return None
        
        nodes, edges = self.get_neighbors(node_id, depth, edge_types)
        
        # Добавляем центральный узел если его нет в соседях
        node_ids = {n.node_id for n in nodes}
        if center_node.node_id not in node_ids:
            nodes.insert(0, center_node)
        
        return SubgraphResult(
            center_node=center_node,
            nodes=nodes,
            edges=edges,
            depth=depth,
            total_nodes=len(nodes),
            total_edges=len(edges)
        )
    
    # ============== Search ==============
    
    def search_nodes(
        self,
        query: str = None,
        node_type: str = None,
        tags: List[str] = None,
        limit: int = 50
    ) -> List[GraphNode]:
        """
        Поиск узлов по различным критериям.
        Используется как fallback если векторный поиск недоступен.
        """
        conditions = []
        params: Dict[str, Any] = {"lim": limit}

        if query:
            conditions.append("(n.name CONTAINS $q OR n.signature CONTAINS $q)")
            params["q"] = query

        if node_type:
            conditions.append("n.type = $ntype")
            params["ntype"] = node_type

        if tags:
            conditions.append("ANY(tag IN $tags WHERE tag IN n.tags)")
            params["tags"] = tags

        where_clause = " AND ".join(conditions) if conditions else "true"

        cypher = f"""
        MATCH (n:Node)
        WHERE {where_clause}
        RETURN n
        ORDER BY n.updated_at DESC
        LIMIT $lim
        """

        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            return [GraphNode.from_dict(dict(record["n"])) for record in result]
    
    # ============== Statistics ==============
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики графа"""
        cypher = """
        MATCH (n:Node)
        OPTIONAL MATCH ()-[e]->()
        RETURN count(DISTINCT n) as node_count,
               count(DISTINCT labels(n)) as type_count,
               count(e) as edge_count,
               collect(DISTINCT n.type) as types
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher)
            record = result.single()
            
            if record:
                return {
                    "total_nodes": record["node_count"],
                    "total_edges": record["edge_count"],
                    "node_types": record["types"],
                    "type_count": record["type_count"]
                }
            return {"total_nodes": 0, "total_edges": 0, "node_types": [], "type_count": 0}
    
    def clear_all(self):
        """Очистка всего графа (для тестирования)"""
        cypher = "MATCH (n) DETACH DELETE n"
        with self._driver.session(database=self.database) as session:
            session.run(cypher)
        logger.info("Graph cleared")
