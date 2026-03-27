"""
Agentic GraphRAG - Embeddings & Vector Search
Генерация эмбеддингов через Gemini API + хранение в Qdrant.

Используется ТОЛЬКО как входная точка для агента:
агент отправляет текстовый промпт → получает стартовый node_id.
"""

import logging
import aiohttp
import hashlib
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)

from ..config import get_settings

logger = logging.getLogger(__name__)

# Gemini embedding model output dimension
EMBEDDING_DIM = 3072


class VectorStore:
    """
    Qdrant-обёртка для хранения и поиска эмбеддингов узлов графа.

    Каждый узел хранится как точка в Qdrant:
    - id: числовой хеш от node_id
    - vector: эмбеддинг от Gemini
    - payload: {node_id, type, name, summary}
    """

    def __init__(self, url: str = None, collection: str = None):
        settings = get_settings()
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self.client: Optional[QdrantClient] = None

    def connect(self) -> bool:
        try:
            self.client = QdrantClient(url=self.url)
            # Создаём коллекцию если нет
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection not in collections:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection}")
            else:
                logger.info(f"Qdrant collection exists: {self.collection}")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            return False

    def _node_id_to_int(self, node_id: str) -> int:
        """Конвертация строкового node_id в числовой ID для Qdrant"""
        return int(hashlib.md5(node_id.encode()).hexdigest()[:15], 16)

    def upsert_node(
        self,
        node_id: str,
        embedding: List[float],
        payload: Dict[str, Any]
    ) -> bool:
        """Сохранение эмбеддинга узла в Qdrant"""
        if not self.client:
            return False
        try:
            point = PointStruct(
                id=self._node_id_to_int(node_id),
                vector=embedding,
                payload={"node_id": node_id, **payload}
            )
            self.client.upsert(
                collection_name=self.collection,
                points=[point]
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant upsert failed for {node_id}: {e}")
            return False

    def upsert_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> int:
        """
        Батч-вставка. Каждый item: {node_id, embedding, payload}
        """
        if not self.client:
            return 0
        points = []
        for item in items:
            points.append(PointStruct(
                id=self._node_id_to_int(item["node_id"]),
                vector=item["embedding"],
                payload={"node_id": item["node_id"], **item["payload"]}
            ))
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=points
            )
            return len(points)
        except Exception as e:
            logger.error(f"Qdrant batch upsert failed: {e}")
            return 0

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        node_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск ближайших узлов по вектору.

        Returns:
            [{node_id, type, name, summary, score}, ...]
        """
        if not self.client:
            return []

        query_filter = None
        if node_type:
            query_filter = Filter(
                must=[FieldCondition(key="type", match=MatchValue(value=node_type))]
            )

        try:
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
            )
            return [
                {
                    **hit.payload,
                    "score": hit.score
                }
                for hit in results.points
            ]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def delete_all(self):
        """Очистка коллекции"""
        if self.client:
            try:
                self.client.delete_collection(self.collection)
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                )
            except Exception as e:
                logger.error(f"Qdrant clear failed: {e}")


async def get_embedding(text: str, api_key: str = None) -> Optional[List[float]]:
    """
    Генерация эмбеддинга через Gemini Embedding API.

    Модель: text-embedding-004 (768 dim, бесплатная)
    """
    settings = get_settings()
    key = api_key or settings.llm_api_key
    if not key:
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-embedding-001:embedContent?key={key}"
    )
    payload = {
        "content": {"parts": [{"text": text}]}
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"Embedding API error {resp.status}: {error}")
                return None
            data = await resp.json()
            try:
                return data["embedding"]["values"]
            except (KeyError, IndexError):
                logger.error(f"Unexpected embedding response: {data}")
                return None


async def get_embeddings_batch(
    texts: List[str],
    api_key: str = None,
    batch_size: int = 100,
) -> List[Optional[List[float]]]:
    """
    Батч-генерация эмбеддингов через Gemini batchEmbedContents API.

    Решение N+1: вместо 1 HTTP-вызова на текст отправляем
    пачки по batch_size (до 100) за 1 вызов.
    100K узлов = ~1K вызовов вместо 100K (100x экономия).
    """
    settings = get_settings()
    key = api_key or settings.llm_api_key
    if not key:
        return [None] * len(texts)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-embedding-001:batchEmbedContents?key={key}"
    )
    model = "models/gemini-embedding-001"

    results: List[Optional[List[float]]] = []

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            payload = {
                "requests": [
                    {"content": {"parts": [{"text": t}]}, "model": model}
                    for t in chunk
                ]
            }
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"Batch embedding error {resp.status}: {error}")
                        results.extend([None] * len(chunk))
                        continue
                    data = await resp.json()
                    for emb in data.get("embeddings", []):
                        results.append(emb.get("values"))
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                results.extend([None] * len(chunk))

    return results
