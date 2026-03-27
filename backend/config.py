# Agentic GraphRAG - Configuration
# Все настройки через environment variables

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "graphrag_nodes"

    # LLM Configuration (провайдер-агностик)
    # Поддерживаемые провайдеры: "gemini", "openai", "anthropic"
    llm_provider: str = "gemini"
    llm_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    llm_base_url: Optional[str] = None  # Для кастомных endpoint'ов

    # Application Settings
    app_name: str = "Agentic GraphRAG"
    debug: bool = True

    # Graph Settings
    max_subgraph_depth: int = 3
    default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
