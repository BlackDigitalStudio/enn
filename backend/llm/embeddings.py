"""
Agentic GraphRAG - Embeddings & Vector Search
Генерация эмбеддингов через Gemini Embedding API + хранение в Qdrant.

Один разум: Gemini для extraction + Gemini для embeddings = единое семантическое пространство.
"""

import logging
import hashlib
import aiohttp
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)

from ..config import get_settings

logger = logging.getLogger(__name__)

# Gemini embedding-001 output dimension
EMBEDDING_DIM = 768


class VectorStore:
    """
    Qdrant-обёртка для хранения и поиска эмбеддингов узлов графа.
    """

    def __init__(self, url: str = None, collection: str = None):
        settings = get_settings()
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self.client: Optional[QdrantClient] = None

    def connect(self) -> bool:
        try:
            self.client = QdrantClient(url=self.url)
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection not in collections:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection} (dim={EMBEDDING_DIM})")
            else:
                info = self.client.get_collection(self.collection)
                existing_dim = info.config.params.vectors.size
                if existing_dim != EMBEDDING_DIM:
                    logger.warning(f"Collection dim mismatch: {existing_dim} vs {EMBEDDING_DIM}, recreating")
                    self.client.delete_collection(self.collection)
                    self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(
                            size=EMBEDDING_DIM,
                            distance=Distance.COSINE
                        )
                    )
                else:
                    logger.info(f"Qdrant collection exists: {self.collection}")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            return False

    def _node_id_to_int(self, node_id: str) -> int:
        return int(hashlib.sha256(node_id.encode()).hexdigest()[:16], 16)

    def upsert_node(self, node_id: str, embedding: List[float], payload: Dict[str, Any]) -> bool:
        if not self.client:
            return False
        try:
            point = PointStruct(
                id=self._node_id_to_int(node_id),
                vector=embedding,
                payload={"node_id": node_id, **payload}
            )
            self.client.upsert(collection_name=self.collection, points=[point])
            return True
        except Exception as e:
            logger.error(f"Qdrant upsert failed for {node_id}: {e}")
            return False

    def upsert_batch(self, items: List[Dict[str, Any]], chunk_size: int = 200) -> int:
        if not self.client:
            return 0
        total = 0
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            points = [
                PointStruct(
                    id=self._node_id_to_int(item["node_id"]),
                    vector=item["embedding"],
                    payload={"node_id": item["node_id"], **item["payload"]}
                )
                for item in chunk
            ]
            try:
                self.client.upsert(collection_name=self.collection, points=points)
                total += len(points)
            except Exception as e:
                logger.error(f"Qdrant chunk upsert failed ({i}-{i+len(chunk)}): {e}")
        return total

    def search(self, query_vector: List[float], limit: int = 5, node_type: str = None) -> List[Dict[str, Any]]:
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
            return [{**hit.payload, "score": hit.score} for hit in results.points]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def delete_all(self):
        if self.client:
            try:
                self.client.delete_collection(self.collection)
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
                )
            except Exception as e:
                logger.error(f"Qdrant clear failed: {e}")


async def get_embedding(text: str, api_key: str = None) -> Optional[List[float]]:
    """Gemini Embedding API — single text."""
    settings = get_settings()
    key = api_key or settings.llm_api_key
    if not key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={key}"
    payload = {"content": {"parts": [{"text": text[:2000]}]}}

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
    Gemini batchEmbedContents API — up to 100 texts per call.
    Same mind as Gemini 2.5 Flash extraction = aligned semantic space.
    """
    settings = get_settings()
    key = api_key or settings.llm_api_key
    if not key:
        return [None] * len(texts)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={key}"
    model = "models/gemini-embedding-001"

    results: List[Optional[List[float]]] = []

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            payload = {
                "requests": [
                    {"content": {"parts": [{"text": t[:2000]}]}, "model": model}
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


# For entity resolution — local model (fast, no API calls for internal matching)
_local_model = None


def _get_model():
    """Lazy-load local sentence-transformers for entity resolution only."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        model_name = settings.default_embedding_model
        logger.info(f"Loading local embedding model for entity resolution: {model_name}")
        _local_model = SentenceTransformer(model_name)
    return _local_model
