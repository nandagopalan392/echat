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

    def get_document(self, document_id: int) -> Optional[Dict]:
        """Get document by ID with metadata from ingestion_metadata"""
        try:
            # Get basic document info
            doc_info = self._get_document_by_id(document_id)
            if not doc_info:
                return None
            
            # Get additional info from ingestion metadata (without user_id since it doesn't exist)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT ingestion_status, error_message, embedding_model 
                FROM ingestion_metadata 
                WHERE document_id = ? 
                ORDER BY ingested_at DESC 
                LIMIT 1
            ''', (document_id,))
            
            ingestion_row = cursor.fetchone()
            conn.close()
            
            if ingestion_row:
                doc_info['status'] = ingestion_row[0]
                doc_info['error_message'] = ingestion_row[1]
                doc_info['embedding_model'] = ingestion_row[2]
            else:
                # No ingestion metadata found, set default status
                doc_info['status'] = 'not_ingested'
                doc_info['error_message'] = None
                doc_info['embedding_model'] = None
            
            return doc_info
            
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            return None

    def get_file_content(self, document_id: int) -> Optional[bytes]:
        """Get file content as bytes from MinIO"""
        try:
            # Get document metadata
            doc_info = self._get_document_by_id(document_id)
            if not doc_info:
                return None
            
            # Get object from MinIO
            response = self.minio_client.get_object(
                self.bucket_name,
                doc_info['minio_object_name']
            )
            
            content = response.read()
            response.close()
            response.release_conn()
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to get file content for document {document_id}: {e}")
            return None

    def get_document_info(self, document_id: int) -> Optional[Dict]:
        """Get document information - alias for get_document method"""
        return self.get_document(document_id)
    
    def get_document_content(self, document_id: int) -> Optional[str]:
        """Get document text content for dataset generation using enhanced document processor"""
        try:
            # Use enhanced document processor for better text extraction
            temp_file_path = self.get_temp_file_path(document_id)
            if not temp_file_path:
                return None
            
            try:
                from app.core.rag.chunking.enhanced_document_processor import get_document_processor
                processor = get_document_processor()
                
                # Extract text using enhanced document processor
                result = processor.process_document(temp_file_path)
                
                # Handle ChunkingResult or direct documents list
                if hasattr(result, 'chunks'):
                    documents = result.chunks
                elif hasattr(result, '__iter__'):
                    documents = result
                else:
                    documents = [result]  # Single document case
                
                # Combine all text content
                combined_text = []
                for doc in documents:
                    if hasattr(doc, 'page_content') and doc.page_content:
                        combined_text.append(doc.page_content.strip())
                
                if combined_text:
                    return '\n\n'.join(combined_text)
                
            except Exception as e:
                logger.warning(f"Enhanced document processor failed for document {document_id}: {e}")
                # Fallback to basic content extraction
                return self._get_basic_document_content(document_id)
            
            finally:
                # Clean up temporary file
                try:
                    if temp_file_path and Path(temp_file_path).exists():
                        Path(temp_file_path).unlink()
                except Exception as e:
                    logger.warning(f"Could not clean up temporary file {temp_file_path}: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document content for {document_id}: {e}")
            return None

    def get_document_content_for_dataset(self, document_id: int) -> Optional[str]:
        """Get raw document text content for dataset generation WITHOUT chunking"""
        try:
            # Use enhanced document processor but extract raw text without chunking
            temp_file_path = self.get_temp_file_path(document_id)
            if not temp_file_path:
                return None
            
            try:
                from app.core.rag.chunking.enhanced_document_processor import get_document_processor
                processor = get_document_processor()
                
                # Try to extract using enhanced document processor first
                try:
                    # Process document to get raw text content
                    result = processor.process_document(temp_file_path)
                    
                    # Extract documents from ChunkingResult
                    if hasattr(result, 'chunks'):
                        documents = result.chunks
                    else:
                        documents = result  # fallback if it's already a list
                    
                    # Combine all text content without chunking
                    combined_text = []
                    for doc in documents:
                        if hasattr(doc, 'page_content') and doc.page_content:
                            combined_text.append(doc.page_content.strip())
                    
                    if combined_text:
                        return '\n\n'.join(combined_text)
                        
                except Exception as e:
                    logger.warning(f"Enhanced document processor failed for document {document_id}: {e}")
                
                # Fallback to basic loader methods
                file_ext = Path(temp_file_path).suffix.lower()
                
                if file_ext == '.pdf':
                    try:
                        from langchain_community.document_loaders import PyPDFLoader
                        loader = PyPDFLoader(temp_file_path)
                        documents = loader.load()
                        combined_text = []
                        for doc in documents:
                            if hasattr(doc, 'page_content') and doc.page_content:
                                combined_text.append(doc.page_content.strip())
                        if combined_text:
                            return '\n\n'.join(combined_text)
                    except Exception as e:
                        logger.warning(f"PyPDF extraction failed: {e}")
                
                elif file_ext in ['.docx', '.doc']:
                    try:
                        from langchain_community.document_loaders import Docx2txtLoader
                        loader = Docx2txtLoader(temp_file_path)
                        documents = loader.load()
                        if documents and hasattr(documents[0], 'page_content'):
                            return documents[0].page_content.strip()
                    except Exception as e:
                        logger.warning(f"DOCX extraction failed: {e}")
                
                elif file_ext in ['.pptx', '.ppt']:
                    try:
                        from langchain_community.document_loaders import UnstructuredPowerPointLoader
                        loader = UnstructuredPowerPointLoader(temp_file_path)
                        documents = loader.load()
                        combined_text = []
                        for doc in documents:
                            if hasattr(doc, 'page_content') and doc.page_content:
                                combined_text.append(doc.page_content.strip())
                        if combined_text:
                            return '\n\n'.join(combined_text)
                    except Exception as e:
                        logger.warning(f"PowerPoint extraction failed: {e}")
                
                elif file_ext in ['.xlsx', '.xls', '.csv']:
                    try:
                        import pandas as pd
                        if file_ext == '.csv':
                            df = pd.read_csv(temp_file_path)
                        else:
                            df = pd.read_excel(temp_file_path)
                        
                        # Convert dataframe to readable text
                        text_content = f"Spreadsheet Content:\n\n{df.to_string(index=False)}"
                        return text_content
                    except Exception as e:
                        logger.warning(f"Spreadsheet extraction failed: {e}")
                
                elif file_ext in ['.txt', '.md']:
                    try:
                        with open(temp_file_path, 'r', encoding='utf-8') as f:
                            return f.read().strip()
                    except Exception as e:
                        logger.warning(f"Text file extraction failed: {e}")
                
                # Final fallback to basic content extraction
                return self._get_basic_document_content(document_id)
                
            finally:
                # Clean up temporary file
                try:
                    if temp_file_path and Path(temp_file_path).exists():
                        Path(temp_file_path).unlink()
                except Exception as e:
                    logger.warning(f"Could not clean up temporary file {temp_file_path}: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get raw document content for dataset generation {document_id}: {e}")
            return None

    def get_temp_file_path(self, document_id: int) -> Optional[str]:
        """Download file to temporary location and return path"""
        try:
            import tempfile
            import os
            
            # Get document metadata
            doc_info = self._get_document_by_id(document_id)
            if not doc_info:
                return None
            
            # Get file content
            file_content = self.get_file_content(document_id)
            if not file_content:
                return None
            
            # Create temporary file with proper extension
            filename = doc_info['filename']
            file_ext = Path(filename).suffix
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Failed to create temporary file for document {document_id}: {e}")
            return None

    def _get_basic_document_content(self, document_id: int) -> Optional[str]:
        """Basic document content extraction (fallback method)"""
        try:
            # Get file content as bytes
            file_content = self.get_file_content(document_id)
            if not file_content:
                return None
            
            # Get document info to check content type
            doc_info = self.get_document(document_id)
            if not doc_info:
                return None
            
            content_type = doc_info.get('content_type', '').lower()
            
            # For text files, decode directly
            if 'text' in content_type or 'json' in content_type:
                try:
                    return file_content.decode('utf-8')
                except UnicodeDecodeError:
                    return file_content.decode('latin-1')
            
            # For PDFs and other document types, we'd need to extract text
            # For now, try to decode as text anyway (fallback)
            try:
                text_content = file_content.decode('utf-8')
                # Basic check if it looks like readable text
                if len(text_content.strip()) > 0:
                    return text_content
            except UnicodeDecodeError:
                pass
            
            # If we can't decode as text, return a placeholder
            logger.warning(f"Could not extract text content from document {document_id} with content type {content_type}")
            return f"[Content from {doc_info.get('filename', 'document')} - {content_type}]"
            
        except Exception as e:
            logger.error(f"Failed to get basic document content for {document_id}: {e}")
            return None

    def update_document_status(self, document_id: int, status: str, error_message: str = None):
        """Update document status in ingestion metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current embedding model for this document
            cursor.execute('''
                SELECT embedding_model 
                FROM ingestion_metadata 
                WHERE document_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            ''', (document_id,))
            
            result = cursor.fetchone()
            embedding_model = result[0] if result else 'unknown'
            
            # Update or insert status
            if status == 'pending':
                self.mark_ingestion_pending(document_id, embedding_model)
            elif status == 'failed':
                self.mark_ingestion_failed(document_id, embedding_model, error_message or 'Unknown error')
            elif status == 'completed':
                # Mark as completed
                cursor.execute('''
                    UPDATE ingestion_metadata 
                    SET status = 'completed', error_message = NULL, processed_at = CURRENT_TIMESTAMP
                    WHERE document_id = ? AND embedding_model = ?
                ''', (document_id, embedding_model))
                
                if cursor.rowcount == 0:
                    # Insert new record if none exists
                    cursor.execute('''
                        INSERT INTO ingestion_metadata 
                        (document_id, embedding_model, status, processed_at) 
                        VALUES (?, ?, 'completed', CURRENT_TIMESTAMP)
                    ''', (document_id, embedding_model))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating document status: {e}")

    def _migrate_database(self):
        """Migrate database schema to add new columns if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if chunking_method column exists in documents table
            cursor.execute("PRAGMA table_info(documents)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'chunking_method' not in columns:
                logger.info("Adding chunking_method column to documents table")
                cursor.execute('ALTER TABLE documents ADD COLUMN chunking_method TEXT DEFAULT "general"')
                
            if 'chunking_config' not in columns:
                logger.info("Adding chunking_config column to documents table")
                cursor.execute('ALTER TABLE documents ADD COLUMN chunking_config TEXT')
            
            # Check ingestion_metadata table
            cursor.execute("PRAGMA table_info(ingestion_metadata)")
            ing_columns = [column[1] for column in cursor.fetchall()]
            
            if 'chunking_method' not in ing_columns:
                logger.info("Adding chunking_method column to ingestion_metadata table")
                cursor.execute('ALTER TABLE ingestion_metadata ADD COLUMN chunking_method TEXT DEFAULT "general"')
                
            if 'chunking_config' not in ing_columns:
                logger.info("Adding chunking_config column to ingestion_metadata table")
                cursor.execute('ALTER TABLE ingestion_metadata ADD COLUMN chunking_config TEXT')
            
            conn.commit()
            conn.close()
            logger.info("Database migration completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to migrate database: {e}")
            raise

    def _init_database(self):
        """Initialize SQLite database for tracking document ingestion"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Run migration first to add any missing columns
            if os.path.exists(self.db_path):
                self._migrate_database()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables with chunking configuration support
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_size INTEGER NOT NULL,
                    content_type TEXT,
                    minio_object_name TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    chunking_method TEXT DEFAULT 'general',
                    chunking_config TEXT  -- JSON string of chunking configuration
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ingestion_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    vector_store_collection TEXT NOT NULL,
                    chunk_count INTEGER,
                    ingestion_status TEXT DEFAULT 'pending', -- pending, completed, failed
                    error_message TEXT,
                    ingested_at TEXT,
                    metadata_json TEXT,  -- Additional metadata as JSON
                    chunking_method TEXT DEFAULT 'general',
                    chunking_config TEXT,  -- JSON string of chunking configuration used
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
                    UNIQUE (document_id, embedding_model)
                )
            ''')
            
            # Add indexes for better performance
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents (file_hash)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_ingestion_model ON ingestion_metadata (embedding_model)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_metadata (ingestion_status)''')
            
            conn.commit()
            conn.close()
            
            logger.info("Document storage database initialized")
            
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
    
    def store_document(self, file_path: str, original_filename: str, content_type: str = None, 
                      chunking_method: str = "general", chunking_config: Dict = None) -> Dict:
        """
        Store document in MinIO and record metadata in SQLite
        Returns document info or existing document if duplicate
        """
        try:
            # Calculate file hash for deduplication
            file_hash = self._calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)
            
            # Check if document already exists
            existing_doc = self.get_document_by_hash(file_hash)
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
            
            # Store only the filename (not the full path) for consistency
            filename_only = Path(original_filename).name
            
            cursor.execute('''
                INSERT INTO documents (filename, file_hash, file_size, content_type, 
                                     minio_object_name, uploaded_at, chunking_method, chunking_config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                filename_only,  # Store only filename, not full path 
                file_hash, 
                file_size, 
                content_type, 
                minio_object_name,
                datetime.now().isoformat(),
                chunking_method,
                json.dumps(chunking_config) if chunking_config else None
            ))
            
            document_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            document_info = {
                'id': document_id,
                'filename': filename_only,  # Use consistent filename
                'file_hash': file_hash,
                'file_size': file_size,
                'content_type': content_type,
                'minio_object_name': minio_object_name,
                'uploaded_at': datetime.now().isoformat(),
                'chunking_method': chunking_method,
                'chunking_config': chunking_config
            }
            
            logger.info(f"Document stored successfully: {filename_only}")
            return document_info
            
        except Exception as e:
            logger.error(f"Failed to store document {original_filename}: {e}")
            raise
    
    def get_document_with_config(self, document_id: int) -> Optional[Dict]:
        """Get document by ID with chunking configuration"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT d.id, d.filename, d.file_hash, d.file_size, d.content_type, 
                       d.minio_object_name, d.uploaded_at, d.chunking_method, d.chunking_config
                FROM documents d
                WHERE d.id = ?
            ''', (document_id,))
            
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
                    'chunking_method': row[7] if row[7] else 'general',
                    'chunking_config': json.loads(row[8]) if row[8] else None
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting document with config {document_id}: {e}")
            return None

    def update_document_metadata(self, document_id: int, chunking_method: str = None, 
                                chunk_count: int = None, metadata: Dict = None) -> bool:
        """Update document metadata after reingestion"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update document record if chunking method provided
            if chunking_method:
                cursor.execute('''
                    UPDATE documents 
                    SET chunking_method = ? 
                    WHERE id = ?
                ''', (chunking_method, document_id))
            
            # Update ingestion metadata if provided
            if chunk_count is not None or metadata is not None:
                # Get current embedding model
                from rag import get_chatpdf_instance
                rag_instance = get_chatpdf_instance()
                embedding_model = rag_instance.embedding_model if rag_instance else 'unknown'
                
                # Update existing ingestion record
                update_fields = []
                update_values = []
                
                if chunk_count is not None:
                    update_fields.append('chunk_count = ?')
                    update_values.append(chunk_count)
                
                if metadata is not None:
                    update_fields.append('metadata_json = ?')
                    update_values.append(json.dumps(metadata))
                
                if chunking_method:
                    update_fields.append('chunking_method = ?')
                    update_values.append(chunking_method)
                
                # Add update timestamp
                update_fields.append('ingested_at = ?')
                update_values.append(datetime.now().isoformat())
                
                # Add WHERE conditions
                update_values.extend([document_id, embedding_model])
                
                query = f'''
                    UPDATE ingestion_metadata 
                    SET {', '.join(update_fields)}
                    WHERE document_id = ? AND embedding_model = ?
                '''
                
                cursor.execute(query, update_values)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated metadata for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document metadata: {e}")
            return False

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
                       metadata: Dict = None, chunking_method: str = "general",
                       chunking_config: Dict = None) -> bool:
        """Track successful ingestion of document with specific embedding model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update or insert ingestion metadata
            cursor.execute('''
                INSERT OR REPLACE INTO ingestion_metadata 
                (document_id, embedding_model, vector_store_collection, chunk_count, 
                 ingestion_status, ingested_at, metadata_json, chunking_method, chunking_config)
                VALUES (?, ?, ?, ?, 'completed', ?, ?, ?, ?)
            ''', (
                document_id, 
                embedding_model, 
                vector_store_collection, 
                chunk_count,
                datetime.now().isoformat(),
                json.dumps(metadata) if metadata else None,
                chunking_method,
                json.dumps(chunking_config) if chunking_config else None
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
                SELECT d.*, im.chunk_count, im.ingested_at, im.vector_store_collection, 
                       COALESCE(im.chunking_method, d.chunking_method, 'general') as chunking_method
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
                    'vector_store_collection': row[10],
                    'chunking_method': row[11] if row[11] else 'general'
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
    
    def get_document_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Get document by file hash"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Debug: Check total documents in DB
            cursor.execute('SELECT COUNT(*) FROM documents')
            total_docs = cursor.fetchone()[0]
            logger.info(f"Total documents in database: {total_docs}")
            
            cursor.execute('SELECT * FROM documents WHERE file_hash = ?', (file_hash,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                logger.info(f"Found duplicate document: {row[1]} (hash: {file_hash[:16]}...)")
                return {
                    'id': row[0],
                    'filename': row[1],
                    'file_hash': row[2],
                    'file_size': row[3],
                    'content_type': row[4],
                    'minio_object_name': row[5],
                    'upload_date': row[6],  # Changed from uploaded_at to match API response
                    'chunking_method': row[7] if len(row) > 7 else 'general',
                    'chunking_config': row[8] if len(row) > 8 else None
                }
            else:
                logger.info(f"No duplicate found for hash: {file_hash[:16]}...")
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
    
    def _get_document_by_filename(self, filename: str) -> Optional[Dict]:
        """Get document by filename"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, filename, file_hash, file_size, content_type, 
                       minio_object_name, uploaded_at, last_modified
                FROM documents 
                WHERE filename = ?
            ''', (filename,))
            
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
            logger.error(f"Failed to get document by filename {filename}: {e}")
            return None
    
    def list_all_documents(self) -> List[Dict]:
        """List all stored documents with their ingestion status across models"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current RAG instance to determine current embedding model
            try:
                from rag import get_chatpdf_instance
                current_rag = get_chatpdf_instance()
                current_embedding_model = current_rag.embedding_model if current_rag else None
            except ImportError:
                # Handle potential circular import
                current_embedding_model = None
            
            cursor.execute('''
                SELECT d.id, d.filename, d.file_hash, d.file_size, d.content_type, 
                       d.minio_object_name, d.uploaded_at,
                       GROUP_CONCAT(im.embedding_model || ':' || im.ingestion_status) as model_status,
                       im_current.ingestion_status as current_model_status,
                       im_current.embedding_model as current_embedding_model,
                       COALESCE(im_current.chunking_method, d.chunking_method, 'general') as chunking_method
                FROM documents d
                LEFT JOIN ingestion_metadata im ON d.id = im.document_id
                LEFT JOIN ingestion_metadata im_current ON d.id = im_current.document_id 
                    AND im_current.embedding_model = ?
                GROUP BY d.id
                ORDER BY d.uploaded_at DESC
            ''', (current_embedding_model,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Debug logging
            logger.debug(f"Found {len(rows)} documents for current embedding model: {current_embedding_model}")
            if rows and len(rows) > 0:
                logger.debug(f"Sample row structure: {len(rows[0])} columns")
                logger.debug(f"Sample row: {rows[0]}")
            
            documents = []
            for row in rows:
                model_status = {}
                # model_status is now at index 7
                if row[7]:  # model_status
                    for status_pair in row[7].split(','):
                        parts = status_pair.split(':')
                        if len(parts) == 2:
                            model, status = parts
                            model_status[model] = status
                
                # Determine if document is indexed for current embedding model
                current_status = row[8]  # current_model_status
                indexed = current_status == 'completed' if current_status else False
                
                # Get chunking method
                chunking_method = row[10] if row[10] else 'general'  # chunking_method column
                
                # Get embedding model for current document (prioritize current model, fallback to any available)
                if indexed and current_embedding_model:
                    embedding_model = current_embedding_model
                elif model_status:
                    # Find the most recent successfully completed ingestion
                    completed_models = [model for model, status in model_status.items() if status == 'completed']
                    if completed_models:
                        embedding_model = completed_models[0]  # Take the first completed one
                    else:
                        embedding_model = list(model_status.keys())[0]  # Take any available model
                else:
                    embedding_model = 'Unknown'
                
                documents.append({
                    'id': row[0],
                    'filename': row[1],
                    'file_hash': row[2],
                    'size': row[3] if row[3] is not None else 0,  # Ensure size is not None
                    'content_type': row[4],
                    'minio_object_name': row[5],
                    'upload_date': row[6],  # Frontend expects 'upload_date', not 'uploaded_at'
                    'last_modified': None,  # Not included in this query
                    'model_status': model_status,
                    'indexed': indexed,  # Frontend expects this field
                    'embedding_model': embedding_model,  # Frontend expects this field
                    'chunking_method': chunking_method  # Add chunking method
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def delete_document(self, document_id: int) -> bool:
        """
        Comprehensive document deletion:
        1. Delete chunks from vector store (all embedding models)
        2. Delete file from MinIO
        3. Delete metadata from SQLite
        """
        try:
            # Get document info first
            doc_info = self._get_document_by_id(document_id)
            if not doc_info:
                logger.warning(f"Document not found for deletion: {document_id}")
                return False
            
            deletion_success = True
            filename = doc_info['filename']
            
            # Step 1: Delete from vector store across all embedding models
            try:
                from rag import get_chatpdf_instance
                rag_instance = get_chatpdf_instance()
                
                if rag_instance:
                    vector_deletion_success = rag_instance.remove_document_by_id(document_id)
                    if not vector_deletion_success:
                        logger.warning(f"Failed to completely remove document {document_id} from vector stores")
                        deletion_success = False
                else:
                    logger.warning("RAG instance not available for vector store cleanup")
                    deletion_success = False
                    
            except Exception as e:
                logger.error(f"Error removing document {document_id} from vector stores: {e}")
                deletion_success = False
            
            # Step 2: Delete from MinIO
            try:
                self.minio_client.remove_object(self.bucket_name, doc_info['minio_object_name'])
                logger.info(f"Removed file from MinIO: {doc_info['minio_object_name']}")
            except S3Error as e:
                if e.code != 'NoSuchKey':
                    logger.error(f"Error removing from MinIO: {e}")
                    deletion_success = False
                else:
                    logger.warning(f"Object not found in MinIO (already deleted?): {doc_info['minio_object_name']}")
            except Exception as e:
                logger.error(f"Error removing from MinIO: {e}")
                deletion_success = False
            
            # Step 3: Delete from SQLite (metadata and ingestion records)
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Delete ingestion metadata first (foreign key constraint)
                cursor.execute('DELETE FROM ingestion_metadata WHERE document_id = ?', (document_id,))
                deleted_ingestion_records = cursor.rowcount
                
                # Delete document record
                cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
                deleted_document_records = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                logger.info(f"Deleted {deleted_ingestion_records} ingestion records and {deleted_document_records} document record for {filename}")
                
                if deleted_document_records == 0:
                    logger.warning(f"No document record found to delete for ID {document_id}")
                    deletion_success = False
                    
            except Exception as e:
                logger.error(f"Error deleting from database: {e}")
                deletion_success = False
            
            # Summary
            if deletion_success:
                logger.info(f"Document deleted successfully: {filename} (ID: {document_id})")
            else:
                logger.warning(f"Document deletion completed with some failures: {filename} (ID: {document_id})")
            
            return deletion_success
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def delete_document_by_filename(self, filename: str) -> bool:
        """Delete document by filename from both MinIO and SQLite"""
        try:
            # Get document info
            doc_info = self._get_document_by_filename(filename)
            if not doc_info:
                logger.warning(f"Document not found for deletion: {filename}")
                return False
            
            return self.delete_document(doc_info['id'])
            
        except Exception as e:
            logger.error(f"Failed to delete document {filename}: {e}")
            return False

    def delete_multiple_documents(self, document_ids: List[int]) -> Dict[str, int]:
        """
        Bulk delete multiple documents
        Returns a summary of deletion results
        """
        try:
            results = {
                'total': len(document_ids),
                'successful': 0,
                'failed': 0,
                'failed_ids': []
            }
            
            logger.info(f"Starting bulk deletion of {len(document_ids)} documents")
            
            for doc_id in document_ids:
                try:
                    if self.delete_document(doc_id):
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        results['failed_ids'].append(doc_id)
                except Exception as e:
                    logger.error(f"Error deleting document {doc_id}: {e}")
                    results['failed'] += 1
                    results['failed_ids'].append(doc_id)
            
            logger.info(f"Bulk deletion completed: {results['successful']} successful, {results['failed']} failed")
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk deletion: {e}")
            return {
                'total': len(document_ids),
                'successful': 0,
                'failed': len(document_ids),
                'failed_ids': document_ids
            }
    
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

    def clear_all_documents(self) -> bool:
        """Clear all documents from both database and MinIO storage"""
        try:
            logger.info("Clearing all documents from storage...")
            
            # Clear MinIO bucket
            try:
                objects = self.minio_client.list_objects(self.bucket_name, recursive=True)
                for obj in objects:
                    self.minio_client.remove_object(self.bucket_name, obj.object_name)
                    logger.info(f"Removed object from MinIO: {obj.object_name}")
            except Exception as e:
                logger.warning(f"Error clearing MinIO bucket: {e}")
            
            # Clear database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM ingestion_metadata')
            cursor.execute('DELETE FROM documents')
            
            conn.commit()
            conn.close()
            
            logger.info("All documents cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear all documents: {e}")
            return False


# Global instance
_document_storage = None

def get_document_storage() -> DocumentStorageService:
    """Get global document storage service instance"""
    global _document_storage
    if _document_storage is None:
        _document_storage = DocumentStorageService()
    return _document_storage
