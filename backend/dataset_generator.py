"""
Dataset Generator for TruLens Evaluation
Generates evaluation datasets from documents using Ollama LLMs
"""

import asyncio
import logging
import json
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DatasetItem:
    """Single item in evaluation dataset"""
    query: str
    expected_response: str
    expected_chunks: List[Dict[str, str]]
    metadata: Dict[str, Any]

@dataclass
class GeneratedDataset:
    """Complete generated dataset"""
    name: str
    description: str
    items: List[DatasetItem]
    generation_metadata: Dict[str, Any]

class DatasetGenerator:
    """Generate evaluation datasets from documents using Ollama LLMs"""
    
    def __init__(self, ollama_base_url: str = "http://ollama:11434"):
        self.ollama_base_url = ollama_base_url
        self.client = httpx.AsyncClient(timeout=120.0)
        
    async def generate_dataset_from_documents(
        self,
        documents: List[Dict[str, Any]],
        dataset_name: str,
        dataset_description: str,
        num_questions_per_doc: int = 3,
        model_name: str = "llama3",
        difficulty_levels: List[str] = ["easy", "medium", "hard"]
    ) -> GeneratedDataset:
        """
        Generate evaluation dataset from a list of documents
        
        Args:
            documents: List of document objects with content and metadata
            dataset_name: Name for the generated dataset
            dataset_description: Description of the dataset
            num_questions_per_doc: Number of questions to generate per document
            model_name: Ollama model to use for generation
            difficulty_levels: List of difficulty levels to generate
        """
        logger.info(f"Starting dataset generation for {len(documents)} documents")
        
        all_items = []
        generation_stats = {
            "total_documents": len(documents),
            "questions_per_document": num_questions_per_doc,
            "model_used": model_name,
            "generation_start": datetime.now().isoformat(),
            "success_count": 0,
            "error_count": 0
        }
        
        for doc_idx, document in enumerate(documents):
            try:
                logger.info(f"Processing document {doc_idx + 1}/{len(documents)}: {document.get('filename', 'Unknown')}")
                
                # Extract content from document
                content = await self._extract_document_content(document)
                if not content:
                    logger.warning(f"No content extracted from document {document.get('filename')}")
                    continue
                
                # Generate questions for this document
                doc_items = await self._generate_questions_for_document(
                    content=content,
                    document_metadata=document,
                    num_questions=num_questions_per_doc,
                    model_name=model_name,
                    difficulty_levels=difficulty_levels
                )
                
                all_items.extend(doc_items)
                generation_stats["success_count"] += len(doc_items)
                
                # Add small delay to avoid overwhelming the LLM
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing document {document.get('filename', 'Unknown')}: {e}")
                generation_stats["error_count"] += 1
                continue
        
        generation_stats["generation_end"] = datetime.now().isoformat()
        generation_stats["total_questions_generated"] = len(all_items)
        
        logger.info(f"Dataset generation completed. Generated {len(all_items)} questions from {len(documents)} documents")
        
        return GeneratedDataset(
            name=dataset_name,
            description=dataset_description,
            items=all_items,
            generation_metadata=generation_stats
        )
    
    async def _extract_document_content(self, document: Dict[str, Any]) -> str:
        """Extract text content from document object using raw content extraction (no chunking)"""
        try:
            # Import here to avoid circular imports
            from document_storage import get_document_storage
            
            # If document has direct content
            if 'content' in document:
                return document['content']
            
            # If document has text field
            if 'text' in document:
                return document['text']
            
            # If document has extracted_text
            if 'extracted_text' in document:
                return document['extracted_text']
            
            # Try to get content from file using raw document extraction (no chunking)
            doc_storage = get_document_storage()
            
            # Get the document ID and try to extract content
            doc_id = document.get('id') or document.get('document_id')
            if doc_id:
                try:
                    # Use the raw document content extraction method (no chunking)
                    content = doc_storage.get_document_content_for_dataset(doc_id)
                    if content:
                        return content
                            
                except Exception as e:
                    logger.warning(f"Could not extract raw content for document {doc_id}: {e}")
            
            # Final fallback: return filename as placeholder
            return f"Content from {document.get('filename', 'document')}"
            
        except Exception as e:
            logger.error(f"Error extracting content from document: {e}")
            return ""
    
    async def _generate_questions_for_document(
        self,
        content: str,
        document_metadata: Dict[str, Any],
        num_questions: int,
        model_name: str,
        difficulty_levels: List[str]
    ) -> List[DatasetItem]:
        """Generate questions and answers for a single document"""
        
        # Chunk the content if it's too long
        chunks = self._chunk_content(content, max_chunk_size=2000)
        
        items = []
        
        for i in range(num_questions):
            try:
                # Select difficulty level
                difficulty = random.choice(difficulty_levels)
                
                # Select relevant chunks (1-3 chunks per question)
                num_chunks = random.randint(1, min(3, len(chunks)))
                selected_chunks = random.sample(chunks, num_chunks)
                
                # Generate question and answer
                item = await self._generate_single_qa_pair(
                    chunks=selected_chunks,
                    document_metadata=document_metadata,
                    difficulty=difficulty,
                    model_name=model_name,
                    question_index=i + 1
                )
                
                if item:
                    items.append(item)
                
            except Exception as e:
                logger.error(f"Error generating question {i+1} for document: {e}")
                continue
        
        return items
    
    async def _generate_single_qa_pair(
        self,
        chunks: List[str],
        document_metadata: Dict[str, Any],
        difficulty: str,
        model_name: str,
        question_index: int
    ) -> Optional[DatasetItem]:
        """Generate a single question-answer pair"""
        
        # Combine chunks into context
        context = "\n\n".join(chunks)
        
        # Create prompt for question generation
        generation_prompt = f"""Based on the following text content, generate a {difficulty} difficulty question and its corresponding answer.

TEXT CONTENT:
{context[:3000]}  # Limit context length

REQUIREMENTS:
1. Generate a clear, specific question that can be answered using the provided text
2. Provide a comprehensive, accurate answer based on the text
3. The question should be at {difficulty} difficulty level
4. Question should be practical and useful for evaluation

DIFFICULTY GUIDELINES:
- Easy: Simple factual questions, direct information retrieval
- Medium: Questions requiring some understanding and connection of concepts
- Hard: Complex questions requiring analysis, synthesis, or deeper understanding

OUTPUT FORMAT (JSON):
{{
    "question": "Your generated question here",
    "answer": "Your comprehensive answer here based on the text",
    "confidence": "high/medium/low - how confident you are in this Q&A pair",
    "reasoning": "Brief explanation of why this is a good {difficulty} question"
}}

Generate only the JSON output, nothing else."""

        try:
            # Call Ollama API
            response = await self._call_ollama(
                model=model_name,
                prompt=generation_prompt,
                temperature=0.7  # Some creativity for question generation
            )
            
            if not response:
                return None
            
            # Parse JSON response
            try:
                qa_data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                qa_data = self._extract_json_from_response(response)
                if not qa_data:
                    logger.warning("Could not parse JSON from LLM response")
                    return None
            
            # Create expected chunks with metadata
            expected_chunks = []
            for chunk_idx, chunk in enumerate(chunks):
                expected_chunks.append({
                    "text": chunk,
                    "title": document_metadata.get('filename', f'Document Chunk {chunk_idx + 1}'),
                    "source": document_metadata.get('filename', 'Unknown'),
                    "chunk_index": chunk_idx,
                    "relevance_score": 1.0  # Since we selected these chunks
                })
            
            # Create dataset item
            item = DatasetItem(
                query=qa_data.get('question', '').strip(),
                expected_response=qa_data.get('answer', '').strip(),
                expected_chunks=expected_chunks,
                metadata={
                    "difficulty": difficulty,
                    "document_source": document_metadata.get('filename', 'Unknown'),
                    "document_id": document_metadata.get('id'),
                    "generation_model": model_name,
                    "generation_timestamp": datetime.now().isoformat(),
                    "question_index": question_index,
                    "confidence": qa_data.get('confidence', 'medium'),
                    "reasoning": qa_data.get('reasoning', ''),
                    "num_chunks_used": len(chunks)
                }
            )
            
            # Validate the generated item
            if len(item.query) < 10 or len(item.expected_response) < 20:
                logger.warning("Generated Q&A pair too short, skipping")
                return None
            
            return item
            
        except Exception as e:
            logger.error(f"Error generating Q&A pair: {e}")
            return None
    
    async def _call_ollama(self, model: str, prompt: str, temperature: float = 0.0) -> Optional[str]:
        """Call Ollama API to generate text"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "max_tokens": 1000
                }
            }
            
            response = await self.client.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return None
    
    def _chunk_content(self, content: str, max_chunk_size: int = 2000) -> List[str]:
        """Split content into manageable chunks"""
        if len(content) <= max_chunk_size:
            return [content]
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If chunks are still too large, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split by sentences
                sentences = chunk.split('. ')
                current_subchunk = ""
                for sentence in sentences:
                    if len(current_subchunk) + len(sentence) <= max_chunk_size:
                        current_subchunk += sentence + ". "
                    else:
                        if current_subchunk:
                            final_chunks.append(current_subchunk.strip())
                        current_subchunk = sentence + ". "
                if current_subchunk:
                    final_chunks.append(current_subchunk.strip())
        
        return final_chunks if final_chunks else [content[:max_chunk_size]]
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Try to extract JSON from LLM response that might have extra text"""
        try:
            # Look for JSON between curly braces
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            
            return None
            
        except Exception:
            return None
    
    async def save_dataset_to_file(self, dataset: GeneratedDataset, file_path: str):
        """Save generated dataset to JSON file"""
        try:
            dataset_dict = {
                "name": dataset.name,
                "description": dataset.description,
                "generation_metadata": dataset.generation_metadata,
                "items": []
            }
            
            for item in dataset.items:
                item_dict = {
                    "query": item.query,
                    "expected_response": item.expected_response,
                    "expected_chunks": item.expected_chunks,
                    "metadata": item.metadata
                }
                dataset_dict["items"].append(item_dict)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dataset saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving dataset to file: {e}")
            raise

    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Convenience function
async def generate_evaluation_dataset(
    documents: List[Dict[str, Any]],
    dataset_name: str,
    dataset_description: str,
    num_questions_per_doc: int = 3,
    model_name: str = "llama3",
    difficulty_levels: List[str] = ["easy", "medium", "hard"],
    ollama_base_url: str = "http://ollama:11434"
) -> GeneratedDataset:
    """
    Convenience function to generate evaluation dataset
    """
    generator = DatasetGenerator(ollama_base_url)
    try:
        dataset = await generator.generate_dataset_from_documents(
            documents=documents,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            num_questions_per_doc=num_questions_per_doc,
            model_name=model_name,
            difficulty_levels=difficulty_levels
        )
        return dataset
    finally:
        await generator.close()
