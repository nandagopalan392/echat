"""
Database connection and session management
"""
import sqlite3
import logging
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages SQLite database connections"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_dir = Path(settings.SQLITE_DB_PATH)
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_dir / 'main.db')  # Main application database
        else:
            self.db_path = db_path
        
        logger.info(f"Database path: {self.db_path}")
        self._ensure_db_directory()
    
    def _ensure_db_directory(self):
        """Ensure database directory exists with proper permissions"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        try:
            db_dir.chmod(0o777)
        except Exception as e:
            logger.warning(f"Could not set directory permissions: {e}")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections
        
        Usage:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(...)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = None):
        """
        Execute a query and return results
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """
        Execute an update/insert/delete query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.rowcount


# Global database instance
_db_instance = None


def get_db() -> DatabaseConnection:
    """
    Get or create database instance
    This is a dependency that can be injected into FastAPI routes
    
    Returns:
        DatabaseConnection instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance


def get_db_session() -> Generator:
    """
    Dependency for getting database sessions in FastAPI
    
    Usage:
        @app.get("/endpoint")
        def endpoint(db = Depends(get_db_session)):
            # use db
    """
    db = get_db()
    try:
        yield db
    finally:
        pass  # SQLite doesn't need explicit session cleanup
