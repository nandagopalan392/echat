"""
Configuration repository for database operations
"""
import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime

from app.db.base import DatabaseConnection
from app.db.models.config import ModelSettings, RetrievalConfig

logger = logging.getLogger(__name__)


class ConfigRepository:
    """Repository for configuration-related database operations"""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    # Model settings operations
    def get_model_settings(self) -> Optional[ModelSettings]:
        """Get model settings"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT id, llm, embedding, parameters, updated_at, provider, embedding_provider
                       FROM model_settings
                       ORDER BY id DESC
                       LIMIT 1'''
                )
                row = cursor.fetchone()
                return ModelSettings.from_db_row(row) if row else None
        except Exception as e:
            logger.error(f"Error getting model settings: {e}")
            return None
    
    def save_model_settings(self, llm: str, embedding: str, parameters: Dict,
                           provider: str = 'ollama', embedding_provider: str = 'ollama') -> bool:
        """Save model settings"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute(
                    '''INSERT INTO model_settings 
                       (llm, embedding, parameters, provider, embedding_provider, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (llm, embedding, json.dumps(parameters), provider, embedding_provider, now)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving model settings: {e}")
            return False
    
    def update_model_settings(self, llm: str, embedding: str, parameters: Dict,
                             provider: str = 'ollama', embedding_provider: str = 'ollama') -> bool:
        """Update existing model settings or create new"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Check if settings exist
                cursor.execute('SELECT id FROM model_settings LIMIT 1')
                if cursor.fetchone():
                    # Update existing
                    cursor.execute(
                        '''UPDATE model_settings
                           SET llm = ?, embedding = ?, parameters = ?, 
                               provider = ?, embedding_provider = ?, updated_at = ?
                           WHERE id = (SELECT id FROM model_settings ORDER BY id DESC LIMIT 1)''',
                        (llm, embedding, json.dumps(parameters), provider, embedding_provider, now)
                    )
                else:
                    # Insert new
                    cursor.execute(
                        '''INSERT INTO model_settings 
                           (llm, embedding, parameters, provider, embedding_provider, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)''',
                        (llm, embedding, json.dumps(parameters), provider, embedding_provider, now)
                    )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating model settings: {e}")
            return False
    
    # Retrieval config operations
    def get_retrieval_config(self, user_id: Optional[str] = None) -> Optional[Dict]:
        """Get retrieval configuration"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT config FROM retrieval_configs
                       WHERE user_id IS ? OR user_id = ?
                       ORDER BY updated_at DESC
                       LIMIT 1''',
                    (user_id, user_id)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
        except Exception as e:
            logger.error(f"Error getting retrieval config: {e}")
            return None
    
    def save_retrieval_config(self, config: Dict, user_id: Optional[str] = None) -> bool:
        """Save retrieval configuration"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute(
                    '''INSERT OR REPLACE INTO retrieval_configs 
                       (user_id, config, created_at, updated_at)
                       VALUES (?, ?, 
                               COALESCE((SELECT created_at FROM retrieval_configs WHERE user_id IS ?), ?),
                               ?)''',
                    (user_id, json.dumps(config), user_id, now, now)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving retrieval config: {e}")
            return False
    
    def update_retrieval_config(self, config: Dict, user_id: Optional[str] = None) -> bool:
        """Update retrieval configuration"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Check if config exists
                cursor.execute(
                    'SELECT id FROM retrieval_configs WHERE user_id IS ? OR user_id = ?',
                    (user_id, user_id)
                )
                if cursor.fetchone():
                    # Update existing
                    cursor.execute(
                        '''UPDATE retrieval_configs
                           SET config = ?, updated_at = ?
                           WHERE user_id IS ? OR user_id = ?''',
                        (json.dumps(config), now, user_id, user_id)
                    )
                else:
                    # Insert new
                    cursor.execute(
                        '''INSERT INTO retrieval_configs (user_id, config, created_at, updated_at)
                           VALUES (?, ?, ?, ?)''',
                        (user_id, json.dumps(config), now, now)
                    )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating retrieval config: {e}")
            return False
