"""
Document Storage Service
Handles raw document storage in MinIO and tracks ingestion metadata in SQLite
"""

import os
import sqlite3
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from minio import Minio
from minio.error import S3Error
import tempfile

logger = logging.getLogger(__name__)

class DocumentStorageService:
    """Service for managing document storage and ingestion tracking"""
    
    def __init__(self):
        # MinIO configuration
        self.minio_endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        self.minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'minio_user')
        self.minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'minio_password')
        self.minio_secure = os.getenv('MINIO_SECURE', 'false').lower() == 'true'
        self.bucket_name = os.getenv('MINIO_BUCKET', 'documents')
        
        # SQLite configuration
        self.db_path = os.getenv('DOCUMENT_DB_PATH', '/app/data/documents.db')
        
        # Initialize services
        self._init_minio()
        self._init_database()
    
    def _init_minio(self):
        """Initialize MinIO client and ensure bucket exists"""
        try:
            self.minio_client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=self.minio_secure
            )
            
            # Create bucket if it doesn't exist
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")
            else:
                logger.info(f"MinIO bucket exists: {self.bucket_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize MinIO: {e}")
            raise
    
    def _init_database(self):
        """Initialize SQLite database for tracking document ingestion"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    file_size INTEGER NOT NULL,
                    content_type TEXT,
                    minio_object_name TEXT NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create ingestion_metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ingestion_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    vector_store_collection TEXT NOT NULL,
                    chunk_count INTEGER,
                    ingestion_status TEXT DEFAULT 'pending',
                    ingested_at TIMESTAMP,
                    error_message TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (id),
                    UNIQUE(document_id, embedding_model)
                )
            ''')
            
            # Create index for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_file_hash ON documents(file_hash)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_embedding_model ON ingestion_metadata(embedding_model)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_metadata(ingestion_status)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Document tracking database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize document database: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for deduplication"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def store_document(self, file_path: str, original_filename: str, content_type: str = None) -> Dict:
        """
        Store document in MinIO and record metadata in SQLite
        Returns document info or existing document if duplicate
        """
        try:
            # Calculate file hash for deduplication
            file_hash = self._calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)
            
            # Check if document already exists
            existing_doc = self._get_document_by_hash(file_hash)
            if existing_doc:
                logger.info(f"Document already exists: {original_filename} (hash: {file_hash[:16]}...)")
                return existing_doc
            
            # Generate unique object name in MinIO
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_extension = Path(original_filename).suffix
            minio_object_name = f"{timestamp}_{file_hash[:16]}{file_extension}"
            
            # Upload to MinIO
            self.minio_client.fput_object(
                self.bucket_name,
                minio_object_name,
                file_path,
                content_type=content_type
            )
            
            # Store metadata in SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO documents (filename, file_hash, file_size, content_type, minio_object_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (original_filename, file_hash, file_size, content_type, minio_object_name))
            
            document_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            document_info = {
                'id': document_id,
                'filename': original_filename,
                'file_hash': file_hash,
                'file_size': file_size,
                'content_type': content_type,
                'minio_object_name': minio_object_name,
                'uploaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"Document stored successfully: {original_filename}")
            return document_info
            
        except Exception as e:
            logger.error(f"Failed to store document {original_filename}: {e}")
            raise
    
    def get_document_file(self, document_id: int) -> str:
        """
        Retrieve document from MinIO and return temporary file path
        Returns path to temporary file that should be cleaned up by caller
        """
        try:
            # Get document metadata
            doc_info = self._get_document_by_id(document_id)
            if not doc_info:
                raise ValueError(f"Document not found: {document_id}")
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(doc_info['filename']).suffix
            )
            temp_file.close()
            
            # Download from MinIO
            self.minio_client.fget_object(
                self.bucket_name,
                doc_info['minio_object_name'],
                temp_file.name
            )
            
            logger.debug(f"Retrieved document {document_id} to {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to retrieve document {document_id}: {e}")
            raise
    
    def track_ingestion(self, document_id: int, embedding_model: str, 
                       vector_store_collection: str, chunk_count: int = None,
                       metadata: Dict = None) -> bool:
        """Track successful ingestion of document with specific embedding model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update or insert ingestion metadata
            cursor.execute('''
                INSERT OR REPLACE INTO ingestion_metadata 
                (document_id, embedding_model, vector_store_collection, chunk_count, 
                 ingestion_status, ingested_at, metadata_json)
                VALUES (?, ?, ?, ?, 'completed', ?, ?)
            ''', (
                document_id, 
                embedding_model, 
                vector_store_collection, 
                chunk_count,
                datetime.now().isoformat(),
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Tracked ingestion: doc_id={document_id}, model={embedding_model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track ingestion: {e}")
            return False
    
    def mark_ingestion_failed(self, document_id: int, embedding_model: str, error_message: str):
        """Mark ingestion as failed for document and model combination"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO ingestion_metadata 
                (document_id, embedding_model, vector_store_collection, 
                 ingestion_status, error_message, ingested_at)
                VALUES (?, ?, ?, 'failed', ?, ?)
            ''', (document_id, embedding_model, '', error_message, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.warning(f"Marked ingestion failed: doc_id={document_id}, model={embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to mark ingestion failed: {e}")
    
    def mark_ingestion_pending(self, document_id: int, embedding_model: str):
        """Mark ingestion as pending for document and model combination"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO ingestion_metadata 
                (document_id, embedding_model, vector_store_collection, 
                 ingestion_status, error_message, ingested_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
            ''', (document_id, embedding_model, '', '', ''))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Marked ingestion pending: doc_id={document_id}, model={embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to mark ingestion pending: {e}")
    
    def get_documents_for_model(self, embedding_model: str) -> List[Dict]:
        """Get all documents that have been successfully ingested for a specific model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT d.*, im.chunk_count, im.ingested_at, im.vector_store_collection
                FROM documents d
                INNER JOIN ingestion_metadata im ON d.id = im.document_id
                WHERE im.embedding_model = ? AND im.ingestion_status = 'completed'
                ORDER BY d.uploaded_at DESC
            ''', (embedding_model,))
            
            rows = cursor.fetchall()
            conn.close()
            
            documents = []
            for row in rows:
                documents.append({
                    'id': row[0],
                    'filename': row[1],
                    'file_hash': row[2],
                    'file_size': row[3],
                    'content_type': row[4],
                    'minio_object_name': row[5],
                    'uploaded_at': row[6],
                    'chunk_count': row[8],
                    'ingested_at': row[9],
                    'vector_store_collection': row[10]
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get documents for model {embedding_model}: {e}")
            return []
    
    def get_pending_ingestions(self, embedding_model: str) -> List[Dict]:
        """Get documents that need to be ingested for a specific model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all documents that haven't been ingested for this model
            cursor.execute('''
                SELECT d.*
                FROM documents d
                LEFT JOIN ingestion_metadata im ON d.id = im.document_id 
                    AND im.embedding_model = ?
                WHERE im.id IS NULL OR im.ingestion_status IN ('pending', 'failed')
                ORDER BY d.uploaded_at ASC
            ''', (embedding_model,))
            
            rows = cursor.fetchall()
            conn.close()
            
            documents = []
            for row in rows:
                documents.append({
                    'id': row[0],
                    'filename': row[1],
                    'file_hash': row[2],
                    'file_size': row[3],
                    'content_type': row[4],
                    'minio_object_name': row[5],
                    'uploaded_at': row[6]
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get pending ingestions for model {embedding_model}: {e}")
            return []
    
    def _get_document_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Get document by file hash"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM documents WHERE file_hash = ?', (file_hash,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'filename': row[1],
                    'file_hash': row[2],
                    'file_size': row[3],
                    'content_type': row[4],
                    'minio_object_name': row[5],
                    'uploaded_at': row[6]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document by hash: {e}")
            return None
    
    def _get_document_by_id(self, document_id: int) -> Optional[Dict]:
        """Get document by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM documents WHERE id = ?', (document_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'filename': row[1],
                    'file_hash': row[2],
                    'file_size': row[3],
                    'content_type': row[4],
                    'minio_object_name': row[5],
                    'uploaded_at': row[6],
                    'last_modified': row[7]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document by ID: {e}")
            return None
    
    def list_all_documents(self) -> List[Dict]:
        """List all stored documents with their ingestion status across models"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT d.*, 
                       GROUP_CONCAT(im.embedding_model || ':' || im.ingestion_status) as model_status
                FROM documents d
                LEFT JOIN ingestion_metadata im ON d.id = im.document_id
                GROUP BY d.id
                ORDER BY d.uploaded_at DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            documents = []
            for row in rows:
                model_status = {}
                if row[8]:  # model_status
                    for status_pair in row[8].split(','):
                        model, status = status_pair.split(':')
                        model_status[model] = status
                
                documents.append({
                    'id': row[0],
                    'filename': row[1],
                    'file_hash': row[2],
                    'file_size': row[3],
                    'content_type': row[4],
                    'minio_object_name': row[5],
                    'uploaded_at': row[6],
                    'last_modified': row[7],
                    'model_status': model_status
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def delete_document(self, document_id: int) -> bool:
        """Delete document from both MinIO and SQLite"""
        try:
            # Get document info
            doc_info = self._get_document_by_id(document_id)
            if not doc_info:
                logger.warning(f"Document not found for deletion: {document_id}")
                return False
            
            # Delete from MinIO
            try:
                self.minio_client.remove_object(self.bucket_name, doc_info['minio_object_name'])
            except S3Error as e:
                if e.code != 'NoSuchKey':
                    raise
                logger.warning(f"Object not found in MinIO: {doc_info['minio_object_name']}")
            
            # Delete from SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete ingestion metadata first (foreign key constraint)
            cursor.execute('DELETE FROM ingestion_metadata WHERE document_id = ?', (document_id,))
            cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Document deleted successfully: {doc_info['filename']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def document_exists_in_storage(self, document_id: int) -> bool:
        """Check if a document exists in MinIO storage"""
        try:
            # Get document metadata
            doc_info = self._get_document_by_id(document_id)
            if not doc_info:
                return False
            
            # Check if object exists in MinIO
            try:
                self.minio_client.stat_object(self.bucket_name, doc_info['minio_object_name'])
                return True
            except S3Error as e:
                if e.code == 'NoSuchKey':
                    return False
                raise e
                
        except Exception as e:
            logger.error(f"Failed to check document existence: {e}")
            return False
    
    def cleanup_orphaned_documents(self) -> int:
        """Remove documents from database that no longer exist in MinIO storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all documents
            cursor.execute('SELECT id, filename, minio_object_name FROM documents')
            all_docs = cursor.fetchall()
            
            orphaned_count = 0
            for doc_id, filename, minio_object_name in all_docs:
                try:
                    # Check if object exists in MinIO
                    self.minio_client.stat_object(self.bucket_name, minio_object_name)
                except S3Error as e:
                    if e.code == 'NoSuchKey':
                        # Document doesn't exist in MinIO, remove from database
                        logger.warning(f"Removing orphaned document from database: {filename} (ID: {doc_id})")
                        
                        # Remove from ingestion_metadata table first (foreign key constraint)
                        cursor.execute('DELETE FROM ingestion_metadata WHERE document_id = ?', (doc_id,))
                        
                        # Remove from documents table
                        cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
                        
                        orphaned_count += 1
                    else:
                        logger.error(f"Error checking document {filename}: {e}")
            
            conn.commit()
            conn.close()
            
            if orphaned_count > 0:
                logger.info(f"Cleaned up {orphaned_count} orphaned documents from database")
            
            return orphaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned documents: {e}")
            return 0


# Global instance
_document_storage = None

def get_document_storage() -> DocumentStorageService:
    """Get global document storage service instance"""
    global _document_storage
    if _document_storage is None:
        _document_storage = DocumentStorageService()
    return _document_storage
