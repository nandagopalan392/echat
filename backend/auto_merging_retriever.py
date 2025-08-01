"""
Auto Merging Retrieval System
Combines similar document chunks to provide more comprehensive context.
"""

import logging
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class DocumentWithScore:
    """Document wrapper with similarity score for merging operations."""
    
    def __init__(self, document: Document, score: float = 0.0):
        self.document = document
        self.score = score
        
    def get_score(self) -> float:
        return self.score

class AutoMergingRetriever:
    """
    Auto Merging Retrieval that combines document chunks using hierarchical relationships.
    
    This implementation combines approaches:
    1. Hierarchical parent-child merging 
    2. Sequential chunk filling for continuous context
    3. Similarity-based clustering for related content
    4. Cross-source merging for comprehensive coverage
    """
    
    def __init__(
        self, 
        similarity_threshold: float = 0.8,
        simple_ratio_thresh: float = 0.5,
        max_iterations: int = 3,
        verbose: bool = False
    ):
        """
        Initialize the auto merging retriever.
        
        Args:
            similarity_threshold: Minimum similarity score for merging chunks (0.0-1.0)
            simple_ratio_thresh: Ratio threshold for parent-child merging (0.0-1.0)
            max_iterations: Maximum iterations for iterative merging
            verbose: Enable verbose logging
        """
        self.similarity_threshold = similarity_threshold
        self.simple_ratio_thresh = simple_ratio_thresh
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
    def merge_documents(self, documents: List[Document]) -> List[Document]:
        """
        Apply hierarchical auto merging to combine document chunks.
        
        Args:
            documents: List of retrieved documents (preferably reranked)
            
        Returns:
            List of merged documents with enhanced context
        """
        if not documents or len(documents) < 2:
            logger.info(f"Auto merging: Not enough documents to merge ({len(documents)})")
            return documents
            
        print(f"[AUTO_MERGE] Starting hierarchical auto merging with {len(documents)} documents")
        
        try:
            # Convert documents to scored format
            scored_docs = [DocumentWithScore(doc, 1.0 - i * 0.1) for i, doc in enumerate(documents)]
            
            # Iterative merging process
            current_docs = scored_docs
            iteration = 0
            
            while iteration < self.max_iterations:
                print(f"[AUTO_MERGE] Iteration {iteration + 1}: Processing {len(current_docs)} documents")
                
                # Try different merging strategies
                new_docs, is_changed = self._try_merging(current_docs)
                
                if not is_changed:
                    print(f"[AUTO_MERGE] No changes in iteration {iteration + 1}, stopping")
                    break
                    
                current_docs = new_docs
                iteration += 1
            
            # Extract final documents and sort by score
            final_docs = [doc_score.document for doc_score in sorted(current_docs, key=lambda x: x.get_score(), reverse=True)]
            
            print(f"[AUTO_MERGE] Completed: {len(documents)} → {len(final_docs)} documents after {iteration} iterations")
            return final_docs
            
        except Exception as e:
            logger.error(f"Error in auto merging: {str(e)}", exc_info=True)
            # Return original documents on error
            return documents
    
    def _try_merging(self, docs: List[DocumentWithScore]) -> Tuple[List[DocumentWithScore], bool]:
        """
        Try different merging strategies in order of priority.
        
        Args:
            docs: List of documents with scores
            
        Returns:
            Tuple of (merged_docs, is_changed)
        """
        # Strategy 1: Fill in sequential gaps
        docs, is_changed_1 = self._fill_sequential_gaps(docs)
        
        # Strategy 2: Hierarchical parent-child merging
        docs, is_changed_2 = self._merge_hierarchical_chunks(docs)
        
        # Strategy 3: Similarity-based clustering
        docs, is_changed_3 = self._merge_similar_chunks(docs)
        
        return docs, (is_changed_1 or is_changed_2 or is_changed_3)
    
    def _fill_sequential_gaps(self, docs: List[DocumentWithScore]) -> Tuple[List[DocumentWithScore], bool]:
        """
        Fill gaps between sequential document chunks from the same source.        
        
        """
        if len(docs) < 2:
            return docs, False
            
        print(f"[AUTO_MERGE] Checking sequential gaps between {len(docs)} documents")
        
        # Group by source and sort by chunk index
        source_groups = self._group_docs_by_source_with_scores(docs)
        new_docs = []
        is_changed = False
        
        for source, source_docs in source_groups.items():
            # Sort by chunk index if available
            sorted_docs = self._sort_docs_by_sequence(source_docs)
            
            # Look for gaps and fill them
            filled_docs = []
            for i, doc_score in enumerate(sorted_docs):
                filled_docs.append(doc_score)
                
                # Check if there's a gap to the next document
                if i < len(sorted_docs) - 1:
                    current_chunk_idx = self._extract_chunk_index(doc_score.document)
                    next_chunk_idx = self._extract_chunk_index(sorted_docs[i + 1].document)
                    
                    # If there's exactly one missing chunk, create a merged document
                    if (current_chunk_idx is not None and next_chunk_idx is not None and 
                        next_chunk_idx == current_chunk_idx + 2):
                        
                        # Create a merged document that spans the gap
                        merged_doc = self._create_spanning_document(
                            doc_score.document, 
                            sorted_docs[i + 1].document
                        )
                        avg_score = (doc_score.get_score() + sorted_docs[i + 1].get_score()) / 2
                        filled_docs.append(DocumentWithScore(merged_doc, avg_score))
                        is_changed = True
                        
                        if self.verbose:
                            print(f"[AUTO_MERGE] Filled sequential gap: chunks {current_chunk_idx} to {next_chunk_idx}")
            
            new_docs.extend(filled_docs)
        
        return new_docs, is_changed
    
    def _merge_hierarchical_chunks(self, docs: List[DocumentWithScore]) -> Tuple[List[DocumentWithScore], bool]:
        """
        Merge child chunks into parent chunks when ratio threshold is met.        
        
        """
        print(f"[AUTO_MERGE] Checking hierarchical relationships for {len(docs)} documents")
        
        # Group documents by potential parent relationships
        parent_child_map = self._build_parent_child_relationships(docs)
        
        if not parent_child_map:
            return docs, False
        
        docs_to_remove = set()
        docs_to_add = []
        is_changed = False
        
        for parent_info, child_docs in parent_child_map.items():
            if len(child_docs) < 2:
                continue
                
            # Calculate ratio of retrieved children vs total children
            total_children = self._estimate_total_children(parent_info, child_docs)
            ratio = len(child_docs) / max(total_children, 1)
            
            if ratio >= self.simple_ratio_thresh:
                # Merge children into parent
                merged_doc = self._create_parent_document(parent_info, child_docs)
                avg_score = sum(doc.get_score() for doc in child_docs) / len(child_docs)
                docs_to_add.append(DocumentWithScore(merged_doc, avg_score))
                
                # Mark children for removal
                for doc in child_docs:
                    docs_to_remove.add(id(doc))
                    
                is_changed = True
                
                if self.verbose:
                    print(f"[AUTO_MERGE] Merged {len(child_docs)} children into parent (ratio: {ratio:.2f})")
        
        # Create new document list
        new_docs = [doc for doc in docs if id(doc) not in docs_to_remove]
        new_docs.extend(docs_to_add)
        
        return new_docs, is_changed
    
    def _merge_similar_chunks(self, docs: List[DocumentWithScore]) -> Tuple[List[DocumentWithScore], bool]:
        """
        Merge documents based on content similarity.
        
        This is the original similarity-based approach.
        """
        if len(docs) < 2:
            return docs, False
            
        print(f"[AUTO_MERGE] Checking similarity-based merging for {len(docs)} documents")
        
        # Calculate similarity matrix
        documents = [doc.document for doc in docs]
        similarity_matrix = self._calculate_similarity_matrix(documents)
        
        # Find clusters of similar documents
        clusters = self._find_similar_clusters_with_scores(similarity_matrix, docs)
        
        new_docs = []
        is_changed = False
        
        for cluster in clusters:
            if len(cluster) > 1:
                # Merge cluster
                merged_doc = self._merge_cluster_documents([doc.document for doc in cluster])
                avg_score = sum(doc.get_score() for doc in cluster) / len(cluster)
                new_docs.append(DocumentWithScore(merged_doc, avg_score))
                is_changed = True
                
                if self.verbose:
                    print(f"[AUTO_MERGE] Similarity-merged {len(cluster)} chunks (threshold: {self.similarity_threshold})")
            else:
                new_docs.extend(cluster)
        
        return new_docs, is_changed
    
    # Helper methods for enhanced merging
    def _group_docs_by_source_with_scores(self, docs: List[DocumentWithScore]) -> Dict[str, List[DocumentWithScore]]:
        """Group documents with scores by their source."""
        groups = defaultdict(list)
        
        for doc_score in docs:
            source = self._extract_source_identifier(doc_score.document)
            groups[source].append(doc_score)
            
        return dict(groups)
    
    def _sort_docs_by_sequence(self, docs: List[DocumentWithScore]) -> List[DocumentWithScore]:
        """Sort documents by their sequential order (chunk index, page number, etc.)."""
        def get_sort_key(doc_score):
            doc = doc_score.document
            metadata = doc.metadata or {}
            
            # Try to extract sequence indicators
            for key in ['chunk_index', 'chunk_id', 'page', 'page_number', 'section']:
                if key in metadata:
                    try:
                        if isinstance(metadata[key], str) and metadata[key].isdigit():
                            return int(metadata[key])
                        elif isinstance(metadata[key], (int, float)):
                            return int(metadata[key])
                    except (ValueError, TypeError):
                        pass
            
            # Fallback: use position in current list
            return 0
        
        try:
            return sorted(docs, key=get_sort_key)
        except:
            return docs
    
    def _extract_chunk_index(self, document: Document) -> Optional[int]:
        """Extract chunk index from document metadata."""
        metadata = document.metadata or {}
        
        for key in ['chunk_index', 'chunk_id', 'index']:
            if key in metadata:
                try:
                    value = metadata[key]
                    if isinstance(value, str) and value.isdigit():
                        return int(value)
                    elif isinstance(value, (int, float)):
                        return int(value)
                except (ValueError, TypeError):
                    pass
        
        return None
    
    def _create_spanning_document(self, doc1: Document, doc2: Document) -> Document:
        """Create a document that spans between two sequential documents."""
        # Combine content with clear separation
        combined_content = f"{doc1.page_content.strip()}\n\n[SEQUENTIAL_MERGE]\n\n{doc2.page_content.strip()}"
        
        # Merge metadata
        merged_metadata = self._merge_metadata_simple([doc1, doc2])
        merged_metadata['sequential_merge'] = True
        merged_metadata['spans_chunks'] = True
        
        return Document(page_content=combined_content, metadata=merged_metadata)
    
    def _build_parent_child_relationships(self, docs: List[DocumentWithScore]) -> Dict[str, List[DocumentWithScore]]:
        """
        Build parent-child relationships based on document metadata.
        
        Returns a mapping of parent_info -> list of child documents.
        """
        parent_child_map = defaultdict(list)
        
        for doc_score in docs:
            metadata = doc_score.document.metadata or {}
            
            # Look for parent indicators in metadata
            parent_info = self._extract_parent_info(metadata)
            if parent_info:
                parent_child_map[parent_info].append(doc_score)
        
        # Only return groups with multiple children
        return {k: v for k, v in parent_child_map.items() if len(v) > 1}
    
    def _extract_parent_info(self, metadata: Dict) -> Optional[str]:
        """Extract parent information from metadata."""
        # Look for parent indicators
        source = metadata.get('source', '')
        page = metadata.get('page', metadata.get('page_number', ''))
        section = metadata.get('section', metadata.get('chapter', ''))
        
        # Create parent identifier
        if source and page:
            return f"{source}_page_{page}"
        elif source and section:
            return f"{source}_section_{section}"
        elif source:
            return f"{source}_doc"
        
        return None
    
    def _estimate_total_children(self, parent_info: str, child_docs: List[DocumentWithScore]) -> int:
        """Estimate total number of children for a parent."""
        # Simple heuristic: assume we have at least as many children as we've found
        # In a real implementation, this could query the document store
        return max(len(child_docs), 3)  # Assume at least 3 children per parent
    
    def _create_parent_document(self, parent_info: str, child_docs: List[DocumentWithScore]) -> Document:
        """Create a parent document by merging child documents."""
        # Sort children by sequence
        sorted_children = self._sort_docs_by_sequence(child_docs)
        
        # Combine content
        contents = []
        for i, doc_score in enumerate(sorted_children):
            content = doc_score.document.page_content.strip()
            contents.append(f"[CHILD {i+1}]\n{content}")
        
        combined_content = f"[PARENT MERGED FROM {len(child_docs)} CHILDREN]\n\n" + "\n\n".join(contents)
        
        # Merge metadata
        child_documents = [doc_score.document for doc_score in child_docs]
        merged_metadata = self._merge_metadata_simple(child_documents)
        merged_metadata['parent_merge'] = True
        merged_metadata['parent_info'] = parent_info
        merged_metadata['merged_children'] = len(child_docs)
        
        return Document(page_content=combined_content, metadata=merged_metadata)
    
    def _find_similar_clusters_with_scores(self, similarity_matrix: np.ndarray, docs: List[DocumentWithScore]) -> List[List[DocumentWithScore]]:
        """Find clusters of similar documents using the similarity matrix."""
        n = len(docs)
        visited = set()
        clusters = []
        
        for i in range(n):
            if i in visited:
                continue
                
            # Start a new cluster
            cluster = [docs[i]]
            visited.add(i)
            
            # Find similar documents to add to this cluster
            for j in range(i + 1, n):
                if j in visited:
                    continue
                    
                # Check if documents i and j are similar enough to merge
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    cluster.append(docs[j])
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _merge_cluster_documents(self, documents: List[Document]) -> Document:
        """Merge a cluster of similar documents into a single comprehensive document."""
        if len(documents) == 1:
            return documents[0]
            
        # Sort documents by potential order
        sorted_docs = self._sort_documents_by_order(documents)
        
        # Combine content intelligently
        merged_content = self._combine_content(sorted_docs)
        
        # Merge metadata
        merged_metadata = self._merge_metadata_simple(sorted_docs)
        
        return Document(page_content=merged_content, metadata=merged_metadata)
    
    def _merge_metadata_simple(self, documents: List[Document]) -> Dict:
        """Simple metadata merging for various document types."""
        merged_metadata = {}
        
        # Collect all unique metadata keys
        all_keys = set()
        for doc in documents:
            if doc.metadata:
                all_keys.update(doc.metadata.keys())
        
        # Merge metadata intelligently
        for key in all_keys:
            values = []
            for doc in documents:
                if doc.metadata and key in doc.metadata:
                    value = doc.metadata[key]
                    if value not in values:
                        values.append(value)
            
            # Store merged values
            if len(values) == 1:
                merged_metadata[key] = values[0]
            elif len(values) > 1:
                if all(isinstance(v, (int, float)) for v in values):
                    merged_metadata[key] = f"{min(values)}-{max(values)}"
                else:
                    merged_metadata[key] = " | ".join(str(v) for v in values)
        
        # Add merging metadata
        merged_metadata['merged_chunks'] = len(documents)
        merged_metadata['auto_merged'] = True
        
        return merged_metadata
    
    # Legacy methods (keeping for compatibility)
    def _group_documents_by_source(self, documents: List[Document]) -> Dict[str, List[Document]]:
        
        for doc in documents:
            # Extract source identifier from metadata
            source = self._extract_source_identifier(doc)
            groups[source].append(doc)
            
        return dict(groups)
    
    def _extract_source_identifier(self, document: Document) -> str:
        """Extract a source identifier from document metadata."""
        metadata = document.metadata or {}
        
        # Try different metadata keys for source identification
        for key in ['source', 'file_path', 'filename', 'page', 'document_id']:
            if key in metadata:
                source_value = metadata[key]
                if isinstance(source_value, str):
                    return source_value
                elif isinstance(source_value, (int, float)):
                    return str(source_value)
        
        # Fallback: use first 50 characters of content as identifier
        content_preview = document.page_content[:50].strip()
        return f"unknown_source_{hash(content_preview) % 10000}"
    
    def _merge_document_group(self, documents: List[Document]) -> List[Document]:
        """Merge documents from the same source based on similarity and context."""
        if len(documents) <= 1:
            return documents
            
        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(documents)
        
        # Find clusters of similar documents
        clusters = self._find_similar_clusters(similarity_matrix, documents)
        
        # Merge each cluster
        merged_docs = []
        for cluster in clusters:
            if len(cluster) > 1:
                merged_doc = self._merge_cluster(cluster)
                merged_docs.append(merged_doc)
                print(f"[AUTO_MERGE] Merged {len(cluster)} chunks into 1 comprehensive chunk")
            else:
                merged_docs.extend(cluster)
                
        return merged_docs
    
    def _calculate_similarity_matrix(self, documents: List[Document]) -> np.ndarray:
        """Calculate pairwise similarity matrix for documents."""
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
            
        except Exception as e:
            logger.warning(f"Error calculating similarity matrix: {e}")
            # Return identity matrix as fallback
            n = len(documents)
            return np.eye(n)
    
    def _find_similar_clusters(self, similarity_matrix: np.ndarray, documents: List[Document]) -> List[List[Document]]:
        """Find clusters of similar documents using the similarity matrix."""
        n = len(documents)
        visited = set()
        clusters = []
        
        for i in range(n):
            if i in visited:
                continue
                
            # Start a new cluster
            cluster = [documents[i]]
            visited.add(i)
            
            # Find similar documents to add to this cluster
            for j in range(i + 1, n):
                if j in visited:
                    continue
                    
                # Check if documents i and j are similar enough to merge
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    cluster.append(documents[j])
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _merge_cluster(self, documents: List[Document]) -> Document:
        """Merge a cluster of similar documents into a single comprehensive document."""
        if len(documents) == 1:
            return documents[0]
            
        # Sort documents by potential order (if available in metadata)
        sorted_docs = self._sort_documents_by_order(documents)
        
        # Combine content intelligently
        merged_content = self._combine_content(sorted_docs)
        
        # Merge metadata
        merged_metadata = self._merge_metadata(sorted_docs)
        
        # Create merged document
        merged_doc = Document(
            page_content=merged_content,
            metadata=merged_metadata
        )
        
        return merged_doc
    
    def _sort_documents_by_order(self, documents: List[Document]) -> List[Document]:
        """Sort documents by logical order (page numbers, sections, etc.)."""
        def get_sort_key(doc):
            metadata = doc.metadata or {}
            
            # Try to extract page number or section number
            for key in ['page', 'page_number', 'section', 'chunk_index']:
                if key in metadata:
                    try:
                        return int(metadata[key])
                    except (ValueError, TypeError):
                        pass
            
            # Fallback: use document position in original list
            return 0
        
        try:
            return sorted(documents, key=get_sort_key)
        except:
            # If sorting fails, return original order
            return documents
    
    def _combine_content(self, documents: List[Document]) -> str:
        """Intelligently combine content from multiple documents."""
        contents = []
        
        for i, doc in enumerate(documents):
            content = doc.page_content.strip()
            
            # Add source information for merged content
            metadata = doc.metadata or {}
            source_info = ""
            
            if 'page' in metadata:
                source_info = f"(Page {metadata['page']})"
            elif 'section' in metadata:
                source_info = f"(Section {metadata['section']})"
            elif len(documents) > 1:
                source_info = f"(Part {i + 1})"
            
            if source_info:
                content = f"{source_info} {content}"
            
            contents.append(content)
        
        # Join with double newlines for clear separation
        merged_content = "\n\n".join(contents)
        
        # Add a header indicating this is merged content
        if len(documents) > 1:
            header = f"[MERGED CONTENT FROM {len(documents)} RELATED CHUNKS]\n\n"
            merged_content = header + merged_content
        
        return merged_content
    
    def _merge_metadata(self, documents: List[Document]) -> Dict:
        """Merge metadata from multiple documents."""
        merged_metadata = {}
        
        # Collect all unique metadata keys
        all_keys = set()
        for doc in documents:
            if doc.metadata:
                all_keys.update(doc.metadata.keys())
        
        # Merge metadata intelligently
        for key in all_keys:
            values = []
            for doc in documents:
                if doc.metadata and key in doc.metadata:
                    value = doc.metadata[key]
                    if value not in values:
                        values.append(value)
            
            # Store merged values
            if len(values) == 1:
                merged_metadata[key] = values[0]
            elif len(values) > 1:
                # For multiple values, create a combined representation
                if all(isinstance(v, (int, float)) for v in values):
                    # For numeric values, store range
                    merged_metadata[key] = f"{min(values)}-{max(values)}"
                else:
                    # For text values, join with separator
                    merged_metadata[key] = " | ".join(str(v) for v in values)
        
        # Add merging metadata
        merged_metadata['merged_chunks'] = len(documents)
        merged_metadata['auto_merged'] = True
        
        return merged_metadata
    
    def _cross_source_merge(self, documents: List[Document]) -> List[Document]:
        """Perform cross-source merging for documents from different sources."""
        if len(documents) <= 1:
            return documents
            
        print(f"[AUTO_MERGE] Checking cross-source merging for {len(documents)} documents")
        
        # Calculate similarity between documents from different sources
        similarity_matrix = self._calculate_similarity_matrix(documents)
        
        # Find high-similarity pairs across sources
        merged_docs = []
        used_indices = set()
        
        for i in range(len(documents)):
            if i in used_indices:
                continue
                
            # Find the most similar document to merge with
            similar_docs = [documents[i]]
            used_indices.add(i)
            
            for j in range(i + 1, len(documents)):
                if j in used_indices:
                    continue
                    
                # Check if they're from different sources but highly similar
                source_i = self._extract_source_identifier(documents[i])
                source_j = self._extract_source_identifier(documents[j])
                
                if (source_i != source_j and 
                    similarity_matrix[i][j] >= self.similarity_threshold):
                    similar_docs.append(documents[j])
                    used_indices.add(j)
            
            # Merge similar documents or keep individual
            if len(similar_docs) > 1:
                merged_doc = self._merge_cluster(similar_docs)
                merged_docs.append(merged_doc)
                print(f"[AUTO_MERGE] Cross-source merged {len(similar_docs)} chunks from different sources")
            else:
                merged_docs.extend(similar_docs)
        
        return merged_docs

def create_auto_merging_retriever(
    similarity_threshold: float = 0.8,
    simple_ratio_thresh: float = 0.5,
    max_iterations: int = 3,
    verbose: bool = False
) -> AutoMergingRetriever:
    """
    Factory function to create an auto merging retriever.
    
    Args:
        similarity_threshold: Minimum similarity score for merging chunks (0.0-1.0)
        simple_ratio_thresh: Ratio threshold for parent-child merging (0.0-1.0) 
        max_iterations: Maximum iterations for iterative merging
        verbose: Enable verbose logging
        
    Returns:
        AutoMergingRetriever instance
    """
    return AutoMergingRetriever(
        similarity_threshold=similarity_threshold,
        simple_ratio_thresh=simple_ratio_thresh,
        max_iterations=max_iterations,
        verbose=verbose
    )
