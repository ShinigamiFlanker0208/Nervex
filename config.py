"""
SoulGraph Central Configuration

Manages environment variables, model selection, database connection strings,
and execution engine security boundaries using Pydantic Settings.
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Core settings for SoulGraph infrastructure.
    Loads from environment variables or .env file.
    """
    # Application Config
    APP_NAME: str = "SoulGraph: Autonomous Research Translation & Execution Engine"
    DEBUG: bool = True
    ENVIRONMENT: str = "local_development"

    # API Keys for LLM Extraction & LangGraph Loop
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq API Key for fast Pydantic extraction")
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Gemini API Key for advanced multimodal reasoning")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="Fallback / secondary LLM provider API Key")
    
    # Embedding & OCR Local Configurations
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CUDA_DEVICE_ID: int = Field(default=0, description="GPU ID for Marker/Surya local execution")
    
    # Neo4j Knowledge Graph Configuration
    NEO4J_URI: str = Field(default="bolt://localhost:7687", description="Neo4j bolt connection string")
    NEO4J_USERNAME: str = Field(default="neo4j", description="Neo4j auth username")
    NEO4J_PASSWORD: str = Field(default="password", description="Neo4j auth password")
    NEO4J_DATABASE: str = Field(default="neo4j", description="Neo4j target database")

    # Vector DB (Chroma / Milvus) Configuration
    VECTOR_DB_PATH: Path = Field(default=Path("./data/chromadb"), description="Local storage path for vector embeddings")
    COLLECTION_NAME: str = "soulgraph_math_chunks"
    
    # Sandbox & Docker Execution Constraints
    DOCKER_SANDBOX_IMAGE: str = "soulgraph-runtime:latest"
    MAX_EXECUTION_ATTEMPTS: int = 5
    CONTAINER_TIMEOUT_SECONDS: int = 300
    TRANSLATION_TIMEOUT_SECONDS: float = 1800.0  # whole LangGraph loop (LLM thinking + up to 5 sandbox runs)
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:7860"]
    CONTAINER_MEMORY_LIMIT_MB: int = 4096
    CONTAINER_CPU_QUOTA: int = 100000  # Default cgroups CPU cap
    ENABLE_GPU_PASSTHRU: bool = True
    MAX_UPLOAD_MB: int = 50
    API_HOST: str = "127.0.0.1"  # loopback by default: this API can run code on your GPU
    API_PORT: int = 8000
    
    # Directories
    BASE_DIR: Path = Path(__file__).parent.resolve()
    UPLOAD_DIR: Path = Path(__file__).parent.resolve() / "uploads"
    ARTIFACTS_DIR: Path = Path(__file__).parent.resolve() / "execution_artifacts"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def create_directories(self) -> None:
        """
        # TODO: Implement checking and creating required local storage folders (UPLOAD_DIR, ARTIFACTS_DIR, VECTOR_DB_PATH).
        """
        directories=[self.UPLOAD_DIR, self.ARTIFACTS_DIR, self.VECTOR_DB_PATH]
        for directory in directories:
            try:
                os.makedirs(directory)
            except FileExistsError:
                print(f"Directory {directory} already exists")
        # TODO: Use pathlib or os.makedirs to ensure local workspace directories exist.



# TODO: Instantiate a singleton instance of Settings to be imported across the project.
settings = Settings()
