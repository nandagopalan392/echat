"""
Document Service
Business logic for document management operations
"""
import logging
import mimetypes
from typing import List, Dict, Any, Optional
from pathlib import Path
import datetime
from datetime import timedelta

from app.config.chunking import ChunkingMethod, ChunkingConfig, FileFormatSupport
from app.db.repositories.file_repository import get_file_repository
from app.services.rag_service import get_rag_service
from app.services.storage_service import get_storage_service

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document management operations"""
    
    def __init__(self):
        """Initialize document service"""
        self._progress: Dict[str, Dict[str, any]] = {}
        self._max_age_minutes = 60  # Clean up entries older than 1 hour
    
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
        
        # Initialize progress tracking
        self.create_upload_progress(file_id, filename, len(file_contents))
        self.update_upload_progress(file_id, 10, "uploading", "Reading file...")
        
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
            self.update_upload_progress(file_id, 30, "uploading", "File uploaded, preparing for processing...")
            
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
            
            # Update progress before processing
            self.update_upload_progress(file_id, 50, "processing", "Processing document...")
            
            # Process file based on type
            success = self._process_file(
                temp_path, filename, selected_method, config, user_id, file_ext
            )
            
            if success:
                processed_files.append(filename)
            else:
                failed_files.append(filename)
            
            self.update_upload_progress(file_id, 90, "processing", "Finalizing...")
            
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
            
            # Mark upload as completed successfully
            self.complete_upload_progress(file_id, success=True, message="Upload completed successfully")
            
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
            
            # Mark upload as failed
            self.complete_upload_progress(file_id, success=False, message=f"Upload failed: {str(e)}")
            
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
    
    # Upload Progress Tracking Methods
    
    def create_upload_progress(self, file_id: str, filename: str, total_size: int = 0) -> None:
        """
        Initialize upload progress tracking
        
        Args:
            file_id: Unique identifier for the upload
            filename: Name of the file being uploaded
            total_size: Total size in bytes (0 if unknown)
        """
        self._progress[file_id] = {
            "filename": filename,
            "total_size": total_size,
            "progress": 0,
            "status": "initializing",
            "message": "Upload initialized",
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        logger.info(f"Upload tracking initialized for {file_id}: {filename}")
    
    def update_upload_progress(
        self, 
        file_id: str, 
        progress: int, 
        status: str = "uploading",
        message: str = ""
    ) -> None:
        """
        Update upload progress
        
        Args:
            file_id: Upload identifier
            progress: Progress percentage (0-100)
            status: Current status (uploading, processing, completed, failed)
            message: Optional status message
        """
        if file_id not in self._progress:
            logger.warning(f"Attempted to update non-existent upload: {file_id}")
            return
        
        self._progress[file_id].update({
            "progress": max(0, min(100, progress)),  # Clamp to 0-100
            "status": status,
            "message": message or f"{status.capitalize()}...",
            "updated_at": datetime.datetime.utcnow()
        })
        logger.debug(f"Upload progress updated for {file_id}: {progress}% - {status}")
    
    def get_upload_progress(self, file_id: str) -> Optional[Dict[str, any]]:
        """
        Get current upload progress
        
        Args:
            file_id: Upload identifier
            
        Returns:
            Progress data dictionary or None if not found
        """
        if file_id not in self._progress:
            return None
        
        progress_data = self._progress[file_id].copy()
        # Convert datetime objects to ISO format
        progress_data["created_at"] = progress_data["created_at"].isoformat()
        progress_data["updated_at"] = progress_data["updated_at"].isoformat()
        
        return progress_data
    
    def complete_upload_progress(self, file_id: str, success: bool = True, message: str = "") -> None:
        """
        Mark upload as completed
        
        Args:
            file_id: Upload identifier
            success: Whether upload succeeded
            message: Optional completion message
        """
        if file_id not in self._progress:
            logger.warning(f"Attempted to complete non-existent upload: {file_id}")
            return
        
        status = "completed" if success else "failed"
        default_message = "Upload completed successfully" if success else "Upload failed"
        
        self._progress[file_id].update({
            "progress": 100 if success else self._progress[file_id]["progress"],
            "status": status,
            "message": message or default_message,
            "updated_at": datetime.datetime.utcnow()
        })
        logger.info(f"Upload {status} for {file_id}")
    
    def delete_upload_progress(self, file_id: str) -> bool:
        """
        Remove upload tracking data
        
        Args:
            file_id: Upload identifier
            
        Returns:
            True if deleted, False if not found
        """
        if file_id in self._progress:
            del self._progress[file_id]
            logger.info(f"Upload tracking deleted for {file_id}")
            return True
        return False
    
    def cleanup_stale_uploads(self) -> int:
        """
        Remove old upload tracking entries
        
        Returns:
            Number of entries cleaned up
        """
        cutoff_time = datetime.datetime.utcnow() - timedelta(minutes=self._max_age_minutes)
        stale_ids = [
            file_id for file_id, data in self._progress.items()
            if data["updated_at"] < cutoff_time
        ]
        
        for file_id in stale_ids:
            del self._progress[file_id]
        
        if stale_ids:
            logger.info(f"Cleaned up {len(stale_ids)} stale upload entries")
        
        return len(stale_ids)
    
    def get_all_upload_progress(self) -> Dict[str, Dict[str, any]]:
        """
        Get all current upload progress entries
        
        Returns:
            Dictionary of all uploads
        """
        result = {}
        for file_id, data in self._progress.items():
            progress_data = data.copy()
            progress_data["created_at"] = progress_data["created_at"].isoformat()
            progress_data["updated_at"] = progress_data["updated_at"].isoformat()
            result[file_id] = progress_data
        
        return result
    
    def get_upload_count(self) -> int:
        """Get number of tracked uploads"""
        return len(self._progress)


# Singleton instance
_document_service = None


def get_document_service() -> DocumentService:
    """Get document service singleton"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service
