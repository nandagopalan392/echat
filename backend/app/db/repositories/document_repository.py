"""
Document repository for database operations
"""
import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.db.base import DatabaseConnection
from app.db.models.document import FileInfo, ChunkingConfig

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for document-related database operations"""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    # File operations
    def save_file_info(self, filename: str, format: str, size: int, uploaded_by: str,
                      is_folder: bool = False, folder_path: Optional[str] = None) -> Optional[int]:
        """Save file information to database"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''INSERT INTO files (filename, format, size, uploaded_by, is_folder, folder_path)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (filename, format, size, uploaded_by, is_folder, folder_path)
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving file info: {e}")
            return None
    
    def get_all_files(self) -> List[FileInfo]:
        """Get all files from database"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT id, filename, format, size, upload_date, uploaded_by, is_folder, folder_path
                       FROM files
                       ORDER BY upload_date DESC'''
                )
                rows = cursor.fetchall()
                return [FileInfo.from_db_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting all files: {e}")
            return []
    
    def get_file_by_id(self, file_id: int) -> Optional[FileInfo]:
        """Get file by ID"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT id, filename, format, size, upload_date, uploaded_by, is_folder, folder_path
                       FROM files
                       WHERE id = ?''',
                    (file_id,)
                )
                row = cursor.fetchone()
                return FileInfo.from_db_row(row) if row else None
        except Exception as e:
            logger.error(f"Error getting file by ID: {e}")
            return None
    
    def get_file_by_name(self, filename: str) -> Optional[FileInfo]:
        """Get file by filename"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT id, filename, format, size, upload_date, uploaded_by, is_folder, folder_path
                       FROM files
                       WHERE filename = ?''',
                    (filename,)
                )
                row = cursor.fetchone()
                return FileInfo.from_db_row(row) if row else None
        except Exception as e:
            logger.error(f"Error getting file by name: {e}")
            return None
    
    def delete_file(self, file_id: int) -> bool:
        """Delete file from database"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM files WHERE id = ?', (file_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def delete_file_by_name(self, filename: str) -> bool:
        """Delete file by filename"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM files WHERE filename = ?', (filename,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting file by name: {e}")
            return False
    
    def get_file_stats(self) -> Dict[str, Any]:
        """Get file statistics"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total files count
                cursor.execute('SELECT COUNT(*) FROM files WHERE is_folder = 0')
                total_files = cursor.fetchone()[0]
                
                # Total size
                cursor.execute('SELECT SUM(size) FROM files WHERE is_folder = 0')
                total_size = cursor.fetchone()[0] or 0
                
                # Files by format
                cursor.execute(
                    '''SELECT format, COUNT(*) as count
                       FROM files
                       WHERE is_folder = 0
                       GROUP BY format'''
                )
                by_format = dict(cursor.fetchall())
                
                return {
                    'total_files': total_files,
                    'total_size': total_size,
                    'by_format': by_format
                }
        except Exception as e:
            logger.error(f"Error getting file stats: {e}")
            return {'total_files': 0, 'total_size': 0, 'by_format': {}}
    
    # Chunking config operations
    def save_chunking_config(self, user_id: str, method: str, config_data: Dict) -> bool:
        """Save chunking configuration for a user"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute(
                    '''INSERT OR REPLACE INTO chunking_configs 
                       (user_id, method, config_data, created_at, updated_at)
                       VALUES (?, ?, ?, 
                               COALESCE((SELECT created_at FROM chunking_configs 
                                        WHERE user_id = ? AND method = ?), ?),
                               ?)''',
                    (user_id, method, json.dumps(config_data), user_id, method, now, now)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving chunking config: {e}")
            return False
    
    def get_chunking_config(self, user_id: str, method: str) -> Optional[Dict]:
        """Get chunking configuration for a user and method"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT config_data FROM chunking_configs
                       WHERE user_id = ? AND method = ?''',
                    (user_id, method)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
        except Exception as e:
            logger.error(f"Error getting chunking config: {e}")
            return None
    
    def get_all_chunking_configs(self, user_id: str) -> Dict[str, Dict]:
        """Get all chunking configurations for a user"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT method, config_data FROM chunking_configs
                       WHERE user_id = ?''',
                    (user_id,)
                )
                rows = cursor.fetchall()
                return {row[0]: json.loads(row[1]) for row in rows}
        except Exception as e:
            logger.error(f"Error getting all chunking configs: {e}")
            return {}
