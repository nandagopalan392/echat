"""
File Repository
Database operations for file metadata
"""
import sqlite3
import logging
from typing import List, Dict, Optional
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class FileRepository:
    """Repository for file metadata operations"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize file repository
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_dir = os.getenv('SQLITE_DB_PATH', '/app/data/db')
            db_dir = Path(db_dir).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = os.path.join(db_dir, 'chat.db')
        else:
            self.db_path = db_path
        
        # Ensure directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        db_dir.chmod(0o777)
        
        logger.info(f"Using database path for files: {self.db_path}")
        self._init_tables()
    
    def _init_tables(self):
        """Initialize file-related tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create files table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        format TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        uploaded_by TEXT NOT NULL,
                        is_folder BOOLEAN DEFAULT FALSE,
                        folder_path TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("File tables initialized")
        except Exception as e:
            logger.error(f"Error initializing file tables: {e}")
            raise
    
    def save_file_info(
        self,
        filename: str,
        format: str,
        size: int,
        uploaded_by: str,
        is_folder: bool = False,
        folder_path: Optional[str] = None
    ) -> int:
        """
        Save file metadata
        
        Args:
            filename: Name of the file
            format: File format/extension
            size: File size in bytes
            uploaded_by: Username who uploaded the file
            is_folder: Whether this is part of a folder upload
            folder_path: Path within folder structure
            
        Returns:
            File ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO files (filename, format, size, uploaded_by, is_folder, folder_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (filename, format, size, uploaded_by, is_folder, folder_path))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving file info: {e}")
            raise
    
    def get_file_by_id(self, file_id: int) -> Optional[Dict]:
        """
        Get file metadata by ID
        
        Args:
            file_id: File ID
            
        Returns:
            File metadata dict or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM files WHERE id = ?
                ''', (file_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting file {file_id}: {e}")
            return None
    
    def get_files_by_user(self, username: str) -> List[Dict]:
        """
        Get all files uploaded by a user
        
        Args:
            username: Username
            
        Returns:
            List of file metadata dicts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM files WHERE uploaded_by = ?
                    ORDER BY upload_date DESC
                ''', (username,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting files for user {username}: {e}")
            return []
    
    def get_all_files(self) -> List[Dict]:
        """
        Get all files
        
        Returns:
            List of file metadata dicts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM files ORDER BY upload_date DESC
                ''')
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting all files: {e}")
            return []
    
    def delete_file(self, file_id: int) -> bool:
        """
        Delete file metadata
        
        Args:
            file_id: File ID
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM files WHERE id = ?', (file_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {e}")
            return False
    
    def delete_files_by_user(self, username: str) -> int:
        """
        Delete all files for a user
        
        Args:
            username: Username
            
        Returns:
            Number of files deleted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM files WHERE uploaded_by = ?', (username,))
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Error deleting files for user {username}: {e}")
            return 0


# Singleton instance
_file_repository = None


def get_file_repository() -> FileRepository:
    """Get file repository singleton"""
    global _file_repository
    if _file_repository is None:
        _file_repository = FileRepository()
    return _file_repository
