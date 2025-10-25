"""
Application Configuration
Centralized configuration for the entire application
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Enhanced Chat API"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # 1 hour (reduced for better security)
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # 7 days
    
    # Cookie Settings
    COOKIE_SECURE: bool = False  # Set to True in production with HTTPS
    COOKIE_HTTPONLY: bool = True
    COOKIE_SAMESITE: str = "lax"  # "strict", "lax", or "none"
    COOKIE_DOMAIN: str = None  # Set to your domain in production
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "*"  # Remove in production
    ]
    
    # Database Settings
    SQLITE_DB_PATH: str = "/app/data/db"
    CHROMA_DB_PATH: str = "/app/data/chroma_db"
    
    # AI Models Configuration
    OLLAMA_HOST: str = "http://ollama:11434"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"
    
    # RAG Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    TOP_K_RESULTS: int = 5
    
    # File Upload Settings
    UPLOAD_DIR: Path = Path("/app/data/uploads")
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "pptx", "txt", "xlsx"]
    
    # Vector Store Settings
    VECTOR_STORE_DIR: Path = Path("/app/data/vector_store")
    
    # Celery/Redis Configuration
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Chat Settings
    MAX_HISTORY_LENGTH: int = 10
    DEFAULT_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000
    
    # Cache Settings
    CACHE_DIR: Path = Path("/app/data/cache")
    CACHE_TTL: int = 3600  # 1 hour
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = 'utf-8'

# Create global settings instance
settings = Settings()

# API Tags Metadata for FastAPI documentation
API_TAGS_METADATA = [
    {
        "name": "auth",
        "description": "Authentication and authorization operations"
    },
    {
        "name": "chat",
        "description": "Chat operations with RAG context"
    },
    {
        "name": "documents",
        "description": "Document management and processing"
    },
    {
        "name": "users",
        "description": "User management operations"
    },
    {
        "name": "admin",
        "description": "Administrative operations"
    },
    {
        "name": "models",
        "description": "AI model management"
    },
    {
        "name": "evaluation",
        "description": "Model evaluation and testing"
    },
    {
        "name": "finetuning",
        "description": "Model fine-tuning operations"
    }
]

# Create required directories
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        settings.UPLOAD_DIR,
        settings.VECTOR_STORE_DIR,
        Path(settings.CHROMA_DB_PATH),
        settings.CACHE_DIR,
        Path(settings.SQLITE_DB_PATH)
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
create_directories()
