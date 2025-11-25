"""
Configuration settings for the application.
Loads settings from environment variables and .env file.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Ollama Configuration
    OLLAMA_URL: str = "http://ollama:11434"
    LLM_MODEL: str = "llama3.2:latest"
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    
    # Qdrant Configuration
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "internal_docs"
    VECTOR_SIZE: int = 768  # nomic-embed-text dimension
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Embedding Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # LLM Settings
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.7
    TOP_K: int = 5  # Number of documents to retrieve
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings