"""
Document Service
Business logic for document management operations
"""
import logging
import mimetypes
from typing import List, Dict, Any, Optional
from pathlib import Path
import datetime

from app.config.chunking import ChunkingMethod, ChunkingConfig, FileFormatSupport
from app.db.repositories.file_repository import get_file_repository
from app.services.rag_service import get_rag_service
from app.services.storage_service import get_storage_service

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document management operations"""
    
    def __init__(self):
        """Initialize document service"""
        self.upload_progress = {}
    
    def process_upload(
        self,
        file_contents: bytes,
        filename: str,
        user_id: str,
        is_folder: bool = False,
        folder_path: str = "",
        chunking_method: str = "auto",
        chunking_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process uploaded file
        
        Args:
            file_contents: File content bytes
            filename: Original filename
            user_id: User ID uploading the file
            is_folder: Whether file is part of folder upload
            folder_path: Path within folder structure
            chunking_method: Chunking method to use
            chunking_config: Chunking configuration dict
            
        Returns:
            Upload result dictionary
        """
        file_id = f"upload_{datetime.datetime.now().timestamp()}"
        self.upload_progress[file_id] = 0
        
        failed_files = []
        processed_files = []
        
        try:
            # Create temp directory
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True, mode=0o755)
            
            # Determine file path
            if is_folder and folder_path:
                folder_structure = Path(folder_path).parent
                full_path = temp_dir / folder_structure
                full_path.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / folder_path
            else:
                temp_path = temp_dir / f"temp_{file_id}_{filename}"
            
            # Save file
            temp_path.write_bytes(file_contents)
            self.upload_progress[file_id] = 30
            
            # Determine file extension and chunking method
            file_ext = filename.split('.')[-1].lower()
            
            # Set up chunking configuration
            if chunking_method == "auto":
                selected_method = FileFormatSupport.get_optimal_method(file_ext)
            else:
                try:
                    selected_method = ChunkingMethod(chunking_method)
                except ValueError:
                    logger.warning(f"Invalid chunking method '{chunking_method}', using general")
                    selected_method = ChunkingMethod.GENERAL
            
            # Create chunking configuration
            config = ChunkingConfig(
                method=selected_method,
                **chunking_config
            ) if chunking_config else ChunkingConfig(method=selected_method)
            
            logger.info(f"Processing {filename} with method {selected_method.value}")
            
            # Process file based on type
            success = self._process_file(
                temp_path, filename, selected_method, config, user_id, file_ext
            )
            
            if success:
                processed_files.append(filename)
            else:
                failed_files.append(filename)
            
            self.upload_progress[file_id] = 90
            
            # Save file info to database
            file_repo = get_file_repository()
            file_repo.save_file_info(
                filename=filename,
                format=file_ext,
                size=len(file_contents),
                uploaded_by=user_id,
                is_folder=is_folder,
                folder_path=folder_path if is_folder else None
            )
            
            self.upload_progress[file_id] = 100
            
            return {
                "message": f"Upload complete. Processed: {len(processed_files)} files, Failed: {len(failed_files)} files",
                "file_id": file_id,
                "processed_files": processed_files,
                "failed_files": failed_files,
                "folder_path": folder_path if is_folder else None
            }
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            failed_files.append(filename)
            self.upload_progress[file_id] = -1
            return {
                "message": "Upload completed with errors",
                "file_id": file_id,
                "processed_files": processed_files,
                "failed_files": failed_files,
                "error": str(e)
            }
            
        finally:
            # Clean up temp files
            self._cleanup_temp_files(temp_dir, temp_path if 'temp_path' in locals() else None)
    
    def _process_file(
        self,
        temp_path: Path,
        filename: str,
        chunking_method,
        chunking_config,
        user_id: str,
        file_ext: str
    ) -> bool:
        """Process a single file"""
        # Document file extensions
        doc_extensions = (
            '.pdf', '.docx', '.doc', '.txt', '.md', '.csv', 
            '.xlsx', '.xls', '.ppt', '.pptx', '.html', 
            '.json', '.eml', '.jpg', '.jpeg', '.png', 
            '.gif', '.tif', '.tiff'
        )
        
        if filename.lower().endswith(doc_extensions):
            logger.info(f"Processing document file: {filename}")
            rag = get_rag_service()
            return rag.ingest_with_storage_and_chunking(
                str(temp_path),
                filename,
                chunking_method,
                chunking_config,
                user_id
            )
        else:
            # For other files, store in MinIO only
            logger.info(f"Storing non-document file: {filename}")
            try:
                content_type, _ = mimetypes.guess_type(filename)
                if not content_type:
                    content_type = 'application/octet-stream'
                
                storage = get_storage_service()
                doc_info = storage.store_document(
                    str(temp_path),
                    filename,
                    content_type,
                    chunking_method.value,
                    chunking_config.to_dict()
                )
                
                return doc_info is not None
                
            except Exception as e:
                logger.error(f"Error storing file {filename}: {str(e)}")
                return False
    
    def _cleanup_temp_files(self, temp_dir: Path, temp_path: Optional[Path] = None):
        """Clean up temporary files"""
        try:
            if temp_path and temp_path.exists():
                temp_path.unlink()
            for path in sorted([p for p in temp_dir.rglob('*') if p.is_dir()], reverse=True):
                try:
                    path.rmdir()
                except OSError:
                    pass
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with their ingestion status"""
        rag = get_rag_service()
        return rag.get_all_documents()
    
    def get_ingested_documents(self) -> Dict[str, Any]:
        """Get documents ingested for current embedding model"""
        rag = get_rag_service()
        return {
            "documents": rag.get_ingested_documents(),
            "embedding_model": rag.embedding_model
        }
    
    def reingest_documents(self, embedding_model: str) -> bool:
        """Re-ingest all documents for a new embedding model"""
        rag = get_rag_service()
        return rag.reingest_for_model_switch(embedding_model)
    
    def reingest_specific_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Re-ingest specific documents with custom config"""
        rag = get_rag_service()
        return rag.reingest_specific_documents_with_config(documents)
    
    def delete_document(self, document_id: int) -> bool:
        """Delete a document by ID"""
        storage = get_storage_service()
        return storage.delete_document(document_id)
    
    def bulk_delete_documents(self, document_ids: List[int]) -> Dict[str, Any]:
        """Bulk delete multiple documents"""
        storage = get_storage_service()
        return storage.delete_multiple_documents(document_ids)
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from storage"""
        storage = get_storage_service()
        return storage.clear_all_documents()
    
    def cleanup_orphaned_documents(self) -> int:
        """Cleanup orphaned documents"""
        storage = get_storage_service()
        return storage.cleanup_orphaned_documents()
    
    def check_duplicate(self, filename: str, file_hash: str) -> Dict[str, Any]:
        """Check if a file is a duplicate"""
        storage = get_storage_service()
        existing_file = storage.get_document_by_hash(file_hash)
        
        if existing_file:
            return {
                "exists": True,
                "existing_file": {
                    "filename": existing_file.get('filename'),
                    "upload_date": existing_file.get('upload_date'),
                    "size": existing_file.get('file_size'),
                    "content_type": existing_file.get('content_type'),
                    "hash": existing_file.get('file_hash')
                }
            }
        return {"exists": False}
    
    def delete_by_filename(self, filename: str) -> bool:
        """Delete a document by filename"""
        storage = get_storage_service()
        return storage.delete_document_by_filename(filename)
    
    def get_document_chunks(self, filename: str) -> Dict[str, Any]:
        """Get all chunks for a document"""
        rag = get_rag_service()
        storage = get_storage_service()
        
        if not rag or not rag.vector_store:
            raise ValueError("Vector store not available")
        
        # Get document info
        doc_info = None
        all_docs = storage.list_all_documents()
        for doc in all_docs:
            if doc['filename'] == filename:
                doc_info = doc
                break
        
        # Get chunks from ChromaDB
        chroma_client = rag.vector_store._client
        collection_name = rag._get_collection_name()
        collection = chroma_client.get_collection(name=collection_name)
        
        # Get all chunks for this document
        results = collection.get(
            where={"source": filename},
            include=["embeddings", "documents", "metadatas"]
        )
        
        chunks = []
        if results['ids']:
            for i, chunk_id in enumerate(results['ids']):
                chunks.append({
                    "id": chunk_id,
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i] if results['metadatas'] else {}
                })
        
        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "chunks": chunks,
            "embedding_model": rag.embedding_model
        }
    
    def get_upload_progress(self, file_id: str) -> int:
        """Get upload progress for a file"""
        return self.upload_progress.get(file_id, 0)


# Singleton instance
_document_service = None


def get_document_service() -> DocumentService:
    """Get document service singleton"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service
