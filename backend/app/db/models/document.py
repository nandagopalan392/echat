"""
Document-related database models
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class FileInfo:
    """File information model"""
    id: Optional[int] = None
    filename: str = ""
    format: str = ""
    size: int = 0
    upload_date: Optional[datetime] = None
    uploaded_by: str = ""
    is_folder: bool = False
    folder_path: Optional[str] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create FileInfo from database row"""
        if row is None:
            return None
        return cls(
            id=row['id'] if 'id' in row.keys() else row[0],
            filename=row['filename'] if 'filename' in row.keys() else row[1],
            format=row['format'] if 'format' in row.keys() else row[2],
            size=row['size'] if 'size' in row.keys() else row[3],
            upload_date=row['upload_date'] if 'upload_date' in row.keys() else row[4],
            uploaded_by=row['uploaded_by'] if 'uploaded_by' in row.keys() else row[5],
            is_folder=bool(row['is_folder'] if 'is_folder' in row.keys() else row[6]),
            folder_path=row['folder_path'] if 'folder_path' in row.keys() else (row[7] if len(row) > 7 else None)
        )


@dataclass
class ChunkingConfig:
    """Chunking configuration model"""
    id: Optional[int] = None
    user_id: str = ""
    method: str = ""
    config_data: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create ChunkingConfig from database row"""
        if row is None:
            return None
        return cls(
            id=row['id'] if 'id' in row.keys() else row[0],
            user_id=row['user_id'] if 'user_id' in row.keys() else row[1],
            method=row['method'] if 'method' in row.keys() else row[2],
            config_data=row['config_data'] if 'config_data' in row.keys() else row[3],
            created_at=row['created_at'] if 'created_at' in row.keys() else row[4],
            updated_at=row['updated_at'] if 'updated_at' in row.keys() else row[5]
        )
