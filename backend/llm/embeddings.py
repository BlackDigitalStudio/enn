"""
Agentic GraphRAG - Embeddings & Vector Search
Генерация эмбеддингов через локальный sentence-transformers + хранение в Qdrant.

Провайдер-агностик: не зависит от LLM-провайдера (DeepSeek/Gemini/OpenAI).
Используется ТОЛЬКО как входная точка для агента:
агент отправляет текстовый промпт → получает стартовый node_id.
"""

import logging
import hashlib
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)

from ..config import get_settings

logger = logging.getLogger(__name__)

# sentence-transformers/all-MiniLM-L6-v2 → 384 dim
EMBEDDING_DIM = 384

# Lazy-loaded model (загружается один раз при первом вызове)
_model = None


def _get_model():
    """Lazy-load sentence-transformers model"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        model_name = settings.default_embedding_model
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded, dim={_model.get_sentence_embedding_dimension()}")
    return _model


class VectorStore:
    """
    Qdrant-обёртка для хранения и поиска эмбеддингов узлов графа.

    Каждый узел хранится как точка в Qdrant:
    - id: числовой хеш от node_id
    - vector: эмбеддинг от sentence-transformers
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
                # Проверяем размерность — если не совпадает, пересоздаём
                info = self.client.get_collection(self.collection)
                existing_dim = info.config.params.vectors.size
                if existing_dim != EMBEDDING_DIM:
                    logger.warning(
                        f"Collection dim mismatch: {existing_dim} vs {EMBEDDING_DIM}, recreating"
                    )
                    self.client.delete_collection(self.collection)
                    self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(
                            size=EMBEDDING_DIM,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Recreated Qdrant collection: {self.collection} (dim={EMBEDDING_DIM})")
                else:
                    logger.info(f"Qdrant collection exists: {self.collection}")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            return False

    def _node_id_to_int(self, node_id: str) -> int:
        return int(hashlib.sha256(node_id.encode()).hexdigest()[:16], 16)

    def upsert_node(
        self,
        node_id: str,
        embedding: List[float],
        payload: Dict[str, Any]
    ) -> bool:
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
        items: List[Dict[str, Any]],
        chunk_size: int = 200
    ) -> int:
        if not self.client:
            return 0
        total = 0
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            points = []
            for item in chunk:
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
                total += len(points)
            except Exception as e:
                logger.error(f"Qdrant chunk upsert failed ({i}-{i+len(chunk)}): {e}")
        return total

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        node_type: str = None
    ) -> List[Dict[str, Any]]:
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
    Генерация эмбеддинга через локальный sentence-transformers.
    Не требует API ключа — работает на CPU.
    """
    try:
        model = _get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


async def get_embeddings_batch(
    texts: List[str],
    api_key: str = None,
    batch_size: int = 256,
) -> List[Optional[List[float]]]:
    """
    Батч-генерация эмбеддингов через локальный sentence-transformers.
    Работает на CPU, не требует API.
    batch_size=256 — оптимально для RAM/CPU.
    """
    if not texts:
        return []

    try:
        model = _get_model()
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            embeddings = model.encode(chunk, normalize_embeddings=True, show_progress_bar=False)
            results.extend([e.tolist() for e in embeddings])
        return results
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        return [None] * len(texts)
