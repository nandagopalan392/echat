from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "ChatPDF RAG API"
    VERSION: str = "1.0.0"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # DeepSeek Configuration
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"
    
    # File Upload Settings
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["pdf"]
    
    # Vector Store Settings
    VECTOR_STORE_DIR: Path = Path("vector_store")
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Chat Settings
    MAX_HISTORY_LENGTH: int = 10
    DEFAULT_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000
    
    # Database Settings
    CHROMA_DB_DIR: Path = Path("chroma_db")
    
    # Cache Settings
    CACHE_DIR: Path = Path("cache")
    CACHE_TTL: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# API Tags Metadata
API_TAGS_METADATA = [
    {
        "name": "chat",
        "description": "Chat operations with PDF context"
    },
    {
        "name": "documents",
        "description": "PDF document management"
    },
    {
        "name": "sessions",
        "description": "Chat session management"
    }
]

# Create required directories
def create_directories():
    for directory in [
        settings.UPLOAD_DIR,
        settings.VECTOR_STORE_DIR,
        settings.CHROMA_DB_DIR,
        settings.CACHE_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories
create_directories()
