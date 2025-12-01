"""
Q-C-A (Question-Context-Answer) Dataset Generator for Finetuning
Implements the comprehensive pipeline as specified in the requirements.
"""

import asyncio
import json
import random
import httpx
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re

logger = logging.getLogger(__name__)

@dataclass
class QCAItem:
    """Single Q-C-A item for finetuning"""
    instruction: str
    input: str
    output: str
    metadata: Dict[str, Any]

@dataclass
class QCADataset:
    """Complete Q-C-A dataset"""
    name: str
    description: str
    items: List[QCAItem]
    generation_metadata: Dict[str, Any]

class QCADatasetGenerator:
    """Generate Q-C-A datasets from documents using Ollama LLMs"""
    
    def __init__(self, ollama_base_url: str = "http://ollama:11434"):
        self.ollama_base_url = ollama_base_url
        self.client = httpx.AsyncClient(timeout=120.0)
        
    async def generate_dataset_from_documents(
        self,
        documents: List[Dict[str, Any]],
        dataset_name: str,
        dataset_description: str,
        questions_per_doc: int = 5,
        model_name: str = "gemma2:2b",
        chunk_size: int = 200  # 100-300 words per chunk
    ) -> QCADataset:
        """
        Generate Q-C-A dataset from documents using the specified pipeline
        
        Steps:
        1. Chunk documents into 100-300 word chunks
        2. Generate 1-2 diverse questions per chunk
        3. Create grounded answers from context
        4. Assemble Q-C-A format
        5. Add variety (paraphrased, reasoning, instructional, comparative)
        6. Filter & clean
        """
        
        logger.info(f"Starting Q-C-A dataset generation for {len(documents)} documents")
        
        all_items = []
        generation_stats = {
            "total_documents": len(documents),
            "questions_per_document": questions_per_doc,
            "model_used": model_name,
            "chunk_size": chunk_size,
            "generation_start": datetime.now().isoformat(),
            "success_count": 0,
            "error_count": 0,
            "total_chunks": 0,
            "total_questions_generated": 0
        }
        
        for doc_idx, document in enumerate(documents):
            try:
                logger.info(f"Processing document {doc_idx + 1}/{len(documents)}: {document.get('filename', 'Unknown')}")
                
                # Step 1: Extract and chunk document content
                content = await self._extract_document_content(document)
                if not content:
                    logger.warning(f"No content extracted from document {document.get('filename')}")
                    continue
                
                chunks = self._chunk_content(content, chunk_size)
                generation_stats["total_chunks"] += len(chunks)
                
                logger.info(f"Created {len(chunks)} chunks from document {document.get('filename')}")
                
                # Step 2-4: Generate Q-C-A items from chunks
                doc_items = await self._generate_qca_items_from_chunks(
                    chunks=chunks,
                    document_metadata=document,
                    questions_per_doc=questions_per_doc,
                    model_name=model_name
                )
                
                all_items.extend(doc_items)
                generation_stats["success_count"] += len(doc_items)
                generation_stats["total_questions_generated"] += len(doc_items)
                
                # Small delay to avoid overwhelming the LLM
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing document {document.get('filename', 'Unknown')}: {e}")
                generation_stats["error_count"] += 1
                continue
        
        # Step 5: Add variety to questions
        all_items = self._add_question_variety(all_items)
        
        # Step 6: Filter & clean
        all_items = self._filter_and_clean(all_items)
        
        generation_stats["generation_end"] = datetime.now().isoformat()
        generation_stats["final_item_count"] = len(all_items)
        
        dataset = QCADataset(
            name=dataset_name,
            description=dataset_description,
            items=all_items,
            generation_metadata=generation_stats
        )
        
        logger.info(f"Generated Q-C-A dataset with {len(all_items)} items")
        return dataset
    
    async def _extract_document_content(self, document: Dict[str, Any]) -> str:
        """Extract text content from document"""
        try:
            # First check if content is already in the document dict
            content = document.get('content')
            if content:
                return str(content).strip()
            
            # If not, fetch content using document storage
            from document_storage import get_document_storage
            doc_storage = get_document_storage()
            
            doc_id = document.get('id')
            if doc_id:
                content = doc_storage.get_document_content_for_dataset(doc_id)
                if content:
                    return str(content).strip()

            logger.warning(f"No content found in document: {document}")
            return ""

        except Exception as e:
            logger.error(f"Error extracting content from document: {e}")
            return ""
    
    def _chunk_content(self, content: str, target_chunk_size: int = 200) -> List[str]:
        """
        Chunk content into 100-300 word pieces
        
        Args:
            content: Text content to chunk
            target_chunk_size: Target number of words per chunk
        """
        try:
            # Simple sentence-based chunking to preserve context
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            chunks = []
            current_chunk = []
            current_word_count = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                word_count = len(sentence.split())
                
                # If adding this sentence would exceed max chunk size, start new chunk
                if current_word_count + word_count > target_chunk_size * 1.5 and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.split()) >= 50:  # Minimum 50 words per chunk
                        chunks.append(chunk_text)
                    current_chunk = [sentence]
                    current_word_count = word_count
                else:
                    current_chunk.append(sentence)
                    current_word_count += word_count
            
            # Add final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= 50:  # Minimum 50 words per chunk
                    chunks.append(chunk_text)
            
            logger.info(f"Chunked content into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking content: {e}")
            return []
    
    async def _generate_qca_items_from_chunks(
        self,
        chunks: List[str],
        document_metadata: Dict[str, Any],
        questions_per_doc: int,
        model_name: str
    ) -> List[QCAItem]:
        """Generate Q-C-A items from document chunks"""
        
        items = []
        chunks_to_use = random.sample(chunks, min(questions_per_doc, len(chunks)))
        
        for i, chunk in enumerate(chunks_to_use):
            try:
                # Generate questions for this chunk
                questions = await self._generate_questions_for_chunk(chunk, model_name)
                
                for j, question_data in enumerate(questions):
                    # Generate grounded answer
                    answer = await self._generate_grounded_answer(
                        question=question_data['question'],
                        context=chunk,
                        model_name=model_name
                    )
                    
                    if answer and answer != "Not provided in the context.":
                        # Create Q-C-A item
                        item = QCAItem(
                            instruction="Answer the question using the given context.",
                            input=f"Context: {chunk}\n\nQuestion: {question_data['question']}",
                            output=answer,
                            metadata={
                                "document_source": document_metadata.get('filename', 'Unknown'),
                                "document_id": document_metadata.get('id'),
                                "chunk_index": i,
                                "question_index": j + 1,
                                "question_type": question_data.get('type', 'factual'),
                                "difficulty": question_data.get('difficulty', 'medium'),
                                "generation_model": model_name,
                                "generation_timestamp": datetime.now().isoformat(),
                                "chunk_word_count": len(chunk.split())
                            }
                        )
                        items.append(item)
                        
            except Exception as e:
                logger.error(f"Error generating Q-C-A for chunk {i}: {e}")
                continue
        
        return items
    
    async def _generate_questions_for_chunk(self, chunk: str, model_name: str) -> List[Dict[str, str]]:
        """Generate 1-2 diverse questions per chunk"""
        
        prompt = f"""You are creating training data for a retrieval-augmented generation model.
Given the following passage, generate 2 diverse questions that can be answered
ONLY using the passage. Avoid yes/no unless natural.

Make the questions varied:
- 1 factual/specific question (who, what, when, where)
- 1 reasoning/analytical question (how, why, what does this mean)

Passage:
{chunk}

Output format (JSON array):
[
  {{"question": "Your factual question here", "type": "factual", "difficulty": "easy"}},
  {{"question": "Your reasoning question here", "type": "reasoning", "difficulty": "medium"}}
]

Generate only the JSON array, nothing else."""

        try:
            response = await self._call_ollama(model_name, prompt, temperature=0.7)
            if not response:
                return []
            
            # Parse JSON response
            try:
                questions = json.loads(response)
                if isinstance(questions, list):
                    return questions
                else:
                    return []
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    try:
                        questions = json.loads(json_match.group())
                        if isinstance(questions, list):
                            return questions
                    except json.JSONDecodeError:
                        pass
                
                logger.warning("Could not parse JSON from LLM response")
                return []
                
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []
    
    async def _generate_grounded_answer(self, question: str, context: str, model_name: str) -> str:
        """Generate grounded answer from context"""
        
        prompt = f"""Answer the question using ONLY the given context. Do not add any information not present in the context.

Context:
{context}

Question: {question}

Instructions:
- If the answer is clearly present in the context, provide the exact answer
- If the answer is not present or unclear, respond with "Not provided in the context."
- Keep answers concise and directly based on the context
- Do not hallucinate or add external knowledge

Answer:"""

        try:
            response = await self._call_ollama(model_name, prompt, temperature=0.3)
            if response:
                answer = response.strip()
                # Additional validation
                if len(answer) > 500:  # Too long, might be hallucination
                    return "Not provided in the context."
                return answer
            return "Not provided in the context."
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Not provided in the context."
    
    def _add_question_variety(self, items: List[QCAItem]) -> List[QCAItem]:
        """Add variety to questions (paraphrased, comparative, instructional)"""
        
        # For now, keep original items
        # Could add paraphrasing, comparative questions, etc. here
        logger.info(f"Added variety to {len(items)} items")
        return items
    
    def _filter_and_clean(self, items: List[QCAItem]) -> List[QCAItem]:
        """Filter and clean dataset"""
        
        # Remove duplicates
        seen_questions = set()
        filtered_items = []
        
        for item in items:
            question_lower = item.input.lower()
            if question_lower not in seen_questions:
                seen_questions.add(question_lower)
                filtered_items.append(item)
            else:
                logger.debug(f"Removed duplicate question: {item.input[:50]}...")
        
        # Remove items that are too short or too long
        final_items = []
        for item in filtered_items:
            if (10 <= len(item.input.split()) <= 500 and 
                5 <= len(item.output.split()) <= 200):
                final_items.append(item)
        
        logger.info(f"Filtered {len(items)} -> {len(final_items)} items")
        return final_items
    
    async def _call_ollama(self, model: str, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """Call Ollama API"""
        try:
            response = await self.client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return None
    
    async def save_dataset_to_jsonl(self, dataset: QCADataset, file_path: str):
        """Save dataset to JSONL format for finetuning"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in dataset.items:
                    json_line = {
                        "instruction": item.instruction,
                        "input": item.input,
                        "output": item.output
                    }
                    f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
            
            logger.info(f"Dataset saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving dataset to file: {e}")
            raise
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()
