# Tree Base - Configuration

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # LLM Configuration
    # Поддерживаемые провайдеры: "gemini", "openai", "anthropic"
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_model: str = "MiniMax-M2.7-highspeed"
    llm_base_url: Optional[str] = "https://api.minimax.io/v1/chat/completions"

    # Application Settings
    app_name: str = "Tree Base"
    debug: bool = True

    # Graph Settings
    max_subgraph_depth: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
