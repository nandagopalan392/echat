"""
Retrieval Configuration System

Handles retrieval settings like similarity threshold, keyword weight, and reranker options.
Migrated from retrieval_config.py to app structure.
"""

import json
import os
import logging
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval"""
    similarity_threshold: float = 0.2  # Minimum similarity score for retrieving chunks
    keyword_similarity_weight: float = 0.7  # Weight of keyword similarity (0.0-1.0)
    reranker_enabled: bool = False  # Whether to use reranker model
    reranker_model: str = ""  # Name of the reranker model
    reranker_provider: str = "ollama"  # Provider for the reranker model (ollama, huggingface)
    max_chunks: int = 5  # Maximum number of chunks to retrieve
    search_type: str = "similarity"  # Type of search (similarity, mmr, etc.)
    auto_merging_enabled: bool = False  # Whether to enable auto merging retrieval
    auto_merging_similarity_threshold: float = 0.8  # Similarity threshold for merging chunks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'similarity_threshold': self.similarity_threshold,
            'keyword_similarity_weight': self.keyword_similarity_weight,
            'reranker_enabled': self.reranker_enabled,
            'reranker_model': self.reranker_model,
            'reranker_provider': self.reranker_provider,
            'max_chunks': self.max_chunks,
            'search_type': self.search_type,
            'auto_merging_enabled': self.auto_merging_enabled,
            'auto_merging_similarity_threshold': self.auto_merging_similarity_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalConfig':
        """Create config from dictionary with validation"""
        import math
        
        # Create a copy to avoid modifying the original
        validated_data = data.copy() if data else {}
        
        # Validate and fix numeric fields
        numeric_fields = {
            'similarity_threshold': 0.2,
            'keyword_similarity_weight': 0.7,
            'max_chunks': 5,
            'auto_merging_similarity_threshold': 0.8
        }
        
        for field, default_value in numeric_fields.items():
            if field in validated_data:
                value = validated_data[field]
                # Handle NaN, None, or invalid values
                if value is None or (isinstance(value, (int, float)) and math.isnan(value)):
                    validated_data[field] = default_value
                else:
                    try:
                        validated_data[field] = float(value) if field != 'max_chunks' else int(value)
                        # Clamp values to valid ranges
                        if field == 'similarity_threshold':
                            validated_data[field] = max(0.0, min(1.0, validated_data[field]))
                        elif field == 'keyword_similarity_weight':
                            validated_data[field] = max(0.0, min(1.0, validated_data[field]))
                        elif field == 'max_chunks':
                            validated_data[field] = max(1, min(20, validated_data[field]))
                        elif field == 'auto_merging_similarity_threshold':
                            validated_data[field] = max(0.0, min(1.0, validated_data[field]))
                    except (ValueError, TypeError):
                        validated_data[field] = default_value
        
        # Validate boolean fields
        boolean_fields = ['reranker_enabled', 'auto_merging_enabled']
        for field in boolean_fields:
            if field in validated_data:
                value = validated_data[field]
                if not isinstance(value, bool):
                    # Convert string representations to boolean
                    if isinstance(value, str):
                        validated_data[field] = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        validated_data[field] = bool(value)
        
        # Validate string fields
        string_fields = ['reranker_model', 'reranker_provider', 'search_type']
        for field in string_fields:
            if field in validated_data and not isinstance(validated_data[field], str):
                validated_data[field] = str(validated_data[field]) if validated_data[field] is not None else ""
        
        # Set default for reranker_provider if not provided
        if 'reranker_provider' not in validated_data or not validated_data['reranker_provider']:
            validated_data['reranker_provider'] = 'ollama'
        
        # Validate reranker_provider against allowed values
        valid_providers = ['ollama', 'huggingface']
        if 'reranker_provider' in validated_data:
            if validated_data['reranker_provider'] not in valid_providers:
                logger.warning(f"Invalid reranker_provider: {validated_data['reranker_provider']}, defaulting to 'ollama'")
                validated_data['reranker_provider'] = 'ollama'
        
        # Validate search_type against allowed values
        valid_search_types = ['similarity', 'mmr', 'similarity_score_threshold', 'hybrid']
        if 'search_type' in validated_data:
            if validated_data['search_type'] not in valid_search_types:
                logger.warning(f"Invalid search_type: {validated_data['search_type']}, defaulting to 'similarity'")
                validated_data['search_type'] = 'similarity'
        
        return cls(**validated_data)


class RetrievalConfigManager:
    """Manager for retrieval configurations"""
    
    def __init__(self, config_dir: str = "/app/data/retrieval_configs", document_repository=None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.default_config = self._load_default_config()
        
        # Use injected repository or import from app structure
        if document_repository:
            self.db = document_repository
        else:
            try:
                from app.db.repositories import DocumentRepository
                from app.db import DatabaseConnection
                self.db = DocumentRepository(DatabaseConnection())
            except ImportError:
                logger.warning("DocumentRepository not available, using file-based config only")
                self.db = None
    
    def _load_default_config(self) -> RetrievalConfig:
        """Load default retrieval configuration"""
        return RetrievalConfig(
            similarity_threshold=0.2,
            keyword_similarity_weight=0.7,
            reranker_enabled=False,
            reranker_model="",
            reranker_provider="ollama",
            max_chunks=5,
            search_type="similarity",
            auto_merging_enabled=False,
            auto_merging_similarity_threshold=0.8
        )
    
    def get_config(self, user_id: Optional[str] = None) -> RetrievalConfig:
        """Get retrieval configuration for user or default"""
        # Try to load user-specific config from database first
        if user_id and self.db:
            try:
                user_config_data = self.db.get_user_retrieval_config(user_id)
                if user_config_data:
                    return RetrievalConfig.from_dict(user_config_data)
            except Exception as e:
                logger.warning(f"Could not load user retrieval config: {e}")
        
        # Try to load from file
        config_file = self.config_dir / f"retrieval_{user_id or 'default'}.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                return RetrievalConfig.from_dict(config_data)
            except Exception as e:
                logger.warning(f"Could not load retrieval config from file: {e}")
        
        return self.default_config
    
    def save_config(self, config: RetrievalConfig, user_id: Optional[str] = None) -> bool:
        """Save retrieval configuration"""
        try:
            # Save to database if available
            if user_id and self.db:
                try:
                    self.db.save_user_retrieval_config(user_id, config.to_dict())
                    logger.info(f"Saved retrieval config to database for user {user_id}")
                except Exception as e:
                    logger.warning(f"Could not save to database: {e}")
            
            # Save to file as backup
            config_file = self.config_dir / f"retrieval_{user_id or 'default'}.json"
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            logger.info(f"Saved retrieval config to file: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save retrieval config: {e}")
            return False
    
    def validate_config(self, config: RetrievalConfig) -> List[str]:
        """Validate configuration and return list of warnings"""
        warnings = []
        
        if config.similarity_threshold < 0.0 or config.similarity_threshold > 1.0:
            warnings.append(f"Similarity threshold {config.similarity_threshold} should be between 0.0 and 1.0")
        
        if config.keyword_similarity_weight < 0.0 or config.keyword_similarity_weight > 1.0:
            warnings.append(f"Keyword similarity weight {config.keyword_similarity_weight} should be between 0.0 and 1.0")
        
        if config.max_chunks < 1 or config.max_chunks > 20:
            warnings.append(f"Max chunks {config.max_chunks} should be between 1 and 20")
        
        if config.reranker_enabled and not config.reranker_model:
            warnings.append("Reranker is enabled but no model is specified")
        
        if config.search_type not in ["similarity", "mmr", "similarity_score_threshold", "hybrid"]:
            warnings.append(f"Unknown search type: {config.search_type}")
        
        if config.auto_merging_similarity_threshold < 0.0 or config.auto_merging_similarity_threshold > 1.0:
            warnings.append(f"Auto merging similarity threshold {config.auto_merging_similarity_threshold} should be between 0.0 and 1.0")
        
        return warnings


# Global instance
_retrieval_config_manager = None


def get_retrieval_config_manager() -> RetrievalConfigManager:
    """Get the global retrieval config manager instance"""
    global _retrieval_config_manager
    if _retrieval_config_manager is None:
        _retrieval_config_manager = RetrievalConfigManager()
    return _retrieval_config_manager
