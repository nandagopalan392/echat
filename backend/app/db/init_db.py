"""
Database Initialization Module

Handles database schema creation and initial data seeding.
Migrated from chat_db.py init_db() method to clean architecture.
"""
import os
import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization and schema creation"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database initializer
        
        Args:
            db_path: Optional path to database file. If None, uses settings.
        """
        if db_path is None:
            self.db_path = settings.SQLITE_DB_PATH
        else:
            self.db_path = db_path
        
        # Ensure directory exists with proper permissions
        db_file = Path(self.db_path)
        db_dir = db_file.parent
        
        logger.info(f"Database file path: {db_file}")
        logger.info(f"Database directory path: {db_dir}")
        logger.info(f"Database directory absolute: {db_dir.absolute()}")
        
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Database directory created successfully: {db_dir}")
            logger.info(f"Database directory exists: {db_dir.exists()}")
            logger.info(f"Database directory is writable: {os.access(db_dir, os.W_OK)}")
        except Exception as e:
            logger.error(f"Failed to create database directory: {e}", exc_info=True)
            raise
        
        try:
            db_dir.chmod(0o777)
        except Exception as e:
            logger.warning(f"Could not set directory permissions: {e}")
        
        logger.info(f"Database initializer using path: {self.db_path}")
    
    def init_database(self) -> None:
        """
        Initialize database with all required tables and default data.
        This is idempotent - can be called multiple times safely.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Initialize all tables
                self._create_users_table(cursor)
                self._create_files_table(cursor)
                self._create_chat_tables(cursor)
                self._create_cache_table(cursor)
                self._create_config_tables(cursor)
                self._create_evaluation_tables(cursor)
                self._create_experiment_tables(cursor)
                self._create_rlhf_tables(cursor)
                
                # Add any missing columns (migrations)
                self._apply_migrations(cursor)
                
                conn.commit()
                logger.info("✅ Database initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Database initialization error: {str(e)}")
            raise
    
    def _create_users_table(self, cursor: sqlite3.Cursor) -> None:
        """Create users table and default admin user"""
        # Check if users table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='users'
        """)
        
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_admin BOOLEAN DEFAULT FALSE,
                    is_active BOOLEAN DEFAULT TRUE,
                    role TEXT DEFAULT 'Engineer',
                    email TEXT,
                    last_login TIMESTAMP
                )
            ''')
            
            # Create default admin user
            admin_password = 'admin123'
            hashed_password = hashlib.sha256(admin_password.encode()).hexdigest()
            cursor.execute(
                'INSERT INTO users (username, password_hash, is_admin, role) VALUES (?, ?, ?, ?)',
                ('admin', hashed_password, True, 'Admin')
            )
            logger.info("Created users table and admin user")
    
    def _create_files_table(self, cursor: sqlite3.Cursor) -> None:
        """Create files table for document management"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                format TEXT NOT NULL,
                size INTEGER NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                uploaded_by TEXT NOT NULL,
                is_folder BOOLEAN DEFAULT FALSE,
                folder_path TEXT,
                FOREIGN KEY (uploaded_by) REFERENCES users (username)
            )
        ''')
        logger.debug("Files table ready")
    
    def _create_chat_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create chat-related tables"""
        # Chat sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                topic TEXT,
                created_at TIMESTAMP,
                last_updated TIMESTAMP
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                content TEXT,
                is_user BOOLEAN,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
            )
        ''')
        logger.debug("Chat tables ready")
    
    def _create_cache_table(self, cursor: sqlite3.Cursor) -> None:
        """Create response cache table"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY,
                query_hash TEXT UNIQUE,
                query TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        logger.debug("Cache table ready")
    
    def _create_config_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create configuration tables"""
        # Chunking configurations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunking_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                method TEXT NOT NULL,
                config_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, method),
                FOREIGN KEY (user_id) REFERENCES users (username)
            )
        ''')
        
        # Model settings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                llm TEXT NOT NULL,
                embedding TEXT NOT NULL,
                parameters TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Retrieval configurations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retrieval_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                config TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id)
            )
        ''')
        logger.debug("Configuration tables ready")
    
    def _create_evaluation_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create evaluation-related tables"""
        # Evaluation metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                message_id INTEGER,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                context TEXT,
                groundedness_score REAL,
                context_relevance_score REAL,
                answer_quality_score REAL,
                latency_ms INTEGER,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id),
                FOREIGN KEY (message_id) REFERENCES messages (id)
            )
        ''')
        
        # Evaluation datasets
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                document_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'Processing',
                created_by TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT,
                FOREIGN KEY (created_by) REFERENCES users (username)
            )
        ''')
        
        # Evaluation dataset documents mapping
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_dataset_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                document_id INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES evaluation_datasets (id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES files (id) ON DELETE CASCADE,
                UNIQUE(dataset_id, document_id)
            )
        ''')
        
        # Evaluation results - drop and recreate to ensure schema is correct
        cursor.execute('DROP TABLE IF EXISTS evaluation_results')
        cursor.execute('''
            CREATE TABLE evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                dataset_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                expected_answer TEXT,
                actual_answer TEXT,
                context TEXT,
                groundedness_score REAL,
                relevance_score REAL,
                quality_score REAL,
                latency_ms INTEGER,
                model_used TEXT,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES evaluation_datasets (id) ON DELETE CASCADE,
                FOREIGN KEY (task_id) REFERENCES evaluation_tasks (task_id) ON DELETE CASCADE
            )
        ''')
        
        # Evaluation tasks (for background processing)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                dataset_id INTEGER,
                task_type TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                total_queries INTEGER DEFAULT 0,
                query TEXT,
                response TEXT,
                context_chunks INTEGER,
                conversation_id TEXT,
                user_id TEXT,
                metadata TEXT,
                groundedness_score REAL,
                answer_relevance_score REAL,
                context_relevance_score REAL,
                overall_score REAL,
                evaluation_time REAL,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES evaluation_datasets(id)
            )
        ''')
        logger.debug("Evaluation tables ready")
    
    def _create_experiment_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create experiment/training/dataset tables"""
        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                user_id TEXT NOT NULL,
                base_model TEXT NOT NULL,
                model_provider TEXT DEFAULT 'huggingface',
                status TEXT NOT NULL,
                config TEXT NOT NULL,
                dataset_path TEXT,
                model_path TEXT,
                metrics TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Training logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                epoch INTEGER,
                step INTEGER,
                loss REAL,
                eval_loss REAL,
                learning_rate REAL,
                accuracy REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        # Datasets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                num_samples INTEGER,
                format TEXT,
                description TEXT,
                status TEXT DEFAULT 'Processing',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Dataset samples table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                sample_index INTEGER NOT NULL,
                input_text TEXT NOT NULL,
                output_text TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE,
                UNIQUE(dataset_id, sample_index)
            )
        ''')
        logger.debug("Experiment tables ready")
    
    def _create_rlhf_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create RLHF (Reinforcement Learning from Human Feedback) tables"""
        # RLHF feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rlhf_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                chosen_index INTEGER,
                username TEXT,
                comment TEXT,
                created_at TEXT
            )
        ''')
        
        # RLHF response options table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rlhf_response_options (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                question TEXT,
                response_option_0 TEXT,
                response_option_1 TEXT,
                chosen_response TEXT,
                chosen_index INTEGER,
                username TEXT,
                created_at TEXT,
                message_id INTEGER
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_rlhf_feedback_session_id 
            ON rlhf_feedback (session_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_rlhf_feedback_username 
            ON rlhf_feedback (username)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_rlhf_response_options_session_id 
            ON rlhf_response_options (session_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_rlhf_response_options_username 
            ON rlhf_response_options (username)
        ''')
        logger.debug("RLHF tables ready")
    
    def _apply_migrations(self, cursor: sqlite3.Cursor) -> None:
        """
        Apply schema migrations - add columns that might be missing in older databases.
        This ensures backward compatibility with existing databases.
        """
        migrations = [
            # Model settings migrations
            ("model_settings", "provider", "TEXT DEFAULT 'ollama'"),
            ("model_settings", "embedding_provider", "TEXT DEFAULT 'ollama'"),
            
            # Evaluation datasets migrations
            ("evaluation_datasets", "file_path", "TEXT"),
            ("evaluation_datasets", "question_count", "INTEGER DEFAULT 0"),
            
            # Finetuning datasets migrations
            ("datasets", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ]
        
        for table, column, definition in migrations:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                logger.info(f"Added column '{column}' to table '{table}'")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.debug(f"Column '{column}' already exists in '{table}'")
                else:
                    logger.warning(f"Could not add column '{column}' to '{table}': {e}")
    

    def verify_schema(self) -> bool:
        """
        Verify that all required tables exist.
        
        Returns:
            True if schema is valid, False otherwise
        """
        required_tables = [
            'users', 'files', 'chat_sessions', 'messages', 'response_cache',
            'chunking_configs', 'model_settings', 'retrieval_configs',
            'evaluation_metrics', 'evaluation_datasets', 'evaluation_dataset_documents',
            'evaluation_results', 'evaluation_tasks',
            'experiments', 'training_logs', 'datasets', 'dataset_samples',
            'rlhf_feedback', 'rlhf_response_options'
        ]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}
                
                missing_tables = set(required_tables) - existing_tables
                if missing_tables:
                    logger.error(f"Missing tables: {missing_tables}")
                    return False
                
                logger.info(f"✅ Schema verified - all {len(required_tables)} tables present")
                return True
                
        except Exception as e:
            logger.error(f"Schema verification failed: {e}")
            return False


# Singleton instance
_db_initializer: Optional[DatabaseInitializer] = None


def get_database_initializer(db_path: Optional[str] = None) -> DatabaseInitializer:
    """
    Get singleton database initializer instance.
    
    Args:
        db_path: Optional path to database file
        
    Returns:
        DatabaseInitializer instance
    """
    global _db_initializer
    if _db_initializer is None:
        _db_initializer = DatabaseInitializer(db_path)
    return _db_initializer


def initialize_database(db_path: Optional[str] = None) -> None:
    """
    Initialize database with all required tables and default data.
    Convenience function for quick initialization.
    
    Args:
        db_path: Optional path to database file
    """
    initializer = get_database_initializer(db_path)
    initializer.init_database()
    initializer.verify_schema()
