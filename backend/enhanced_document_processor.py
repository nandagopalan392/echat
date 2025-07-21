"""
Enhanced Document Processor with Multiple Chunking Methods
Handles different document types and chunking strategies
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import re
from dataclasses import dataclass

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from chunking_config import ChunkingMethod, ChunkingConfig, get_chunking_config_manager

logger = logging.getLogger(__name__)

# Try to import additional libraries for enhanced document processing
try:
    from langchain_community.document_loaders import Docx2txtLoader, CSVLoader, TextLoader
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
    ADVANCED_LOADERS_AVAILABLE = True
except ImportError:
    logger.warning("Advanced document loaders not available. Some features may be limited.")
    ADVANCED_LOADERS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available. Table processing may be limited.")
    PANDAS_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    logger.warning("OCR libraries not available. Image text extraction disabled.")
    OCR_AVAILABLE = False

@dataclass
class ChunkingResult:
    """Result of document chunking operation"""
    chunks: List[Document]
    metadata: Dict[str, Any]
    method_used: ChunkingMethod
    config_used: ChunkingConfig
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class EnhancedDocumentProcessor:
    """Enhanced document processor with multiple chunking methods"""
    
    def __init__(self):
        self.config_manager = get_chunking_config_manager()
        self.supported_extensions = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.doc': self._process_docx,
            '.txt': self._process_text,
            '.md': self._process_text,
            '.csv': self._process_csv,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.ppt': self._process_presentation,
            '.pptx': self._process_presentation,
            '.html': self._process_html,
            '.json': self._process_json,
            '.eml': self._process_email,
            '.jpg': self._process_image,
            '.jpeg': self._process_image,
            '.png': self._process_image,
            '.gif': self._process_image,
            '.tif': self._process_image,
            '.tiff': self._process_image
        }
    
    def process_document(self, file_path: str, method: ChunkingMethod = None, 
                        config: ChunkingConfig = None, user_id: str = None, original_filename: str = None) -> ChunkingResult:
        """
        Process document with specified chunking method and configuration
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Use original filename if provided, otherwise use the file path name
        source_filename = original_filename if original_filename else Path(file_path).name
        if method is None:
            from chunking_config import FileFormatSupport
            method = FileFormatSupport.get_optimal_method(file_ext[1:])  # Remove dot
        
        # Get configuration if not specified
        if config is None:
            config = self.config_manager.get_config(method, user_id)
        
        # Validate configuration
        warnings = self.config_manager.validate_config(config)
        
        logger.info(f"Processing {file_path} with method {method.value}")
        
        # Load document content
        loader_func = self.supported_extensions[file_ext]
        raw_documents = loader_func(file_path)
        
        if not raw_documents:
            raise ValueError(f"No content extracted from {file_path}")
        
        # Apply chunking method
        chunks = self._apply_chunking_method(raw_documents, method, config)
        
        # Add metadata to chunks - clear existing metadata first to avoid complex objects
        for i, chunk in enumerate(chunks):
            chunk.metadata = {  # Replace instead of update to avoid complex objects
                'source_file': source_filename,
                'chunk_index': i,
                'chunking_method': method.value,
                'total_chunks': len(chunks)
            }
        
        # Debug: Log metadata after setting
        if chunks:
            logger.info(f"Enhanced processor - chunk metadata set: {chunks[0].metadata}")
        
        # Create processing metadata
        metadata = {
            'source_file': source_filename,
            'file_size': os.path.getsize(file_path),
            'chunk_count': len(chunks),
            'method_used': method.value,
            'config_used': config.to_dict(),
            'total_tokens': sum(len(chunk.page_content.split()) for chunk in chunks)
        }
        
        return ChunkingResult(
            chunks=chunks,
            metadata=metadata,
            method_used=method,
            config_used=config,
            warnings=warnings
        )
    
    def _apply_chunking_method(self, documents: List[Document], method: ChunkingMethod, 
                              config: ChunkingConfig) -> List[Document]:
        """Apply specific chunking method to documents"""
        
        if method == ChunkingMethod.GENERAL:
            return self._chunk_general(documents, config)
        elif method == ChunkingMethod.QA:
            return self._chunk_qa(documents, config)
        elif method == ChunkingMethod.RESUME:
            return self._chunk_resume(documents, config)
        elif method == ChunkingMethod.TABLE:
            return self._chunk_table(documents, config)
        elif method == ChunkingMethod.PRESENTATION:
            return self._chunk_presentation(documents, config)
        elif method == ChunkingMethod.PICTURE:
            return self._chunk_picture(documents, config)
        elif method == ChunkingMethod.EMAIL:
            return self._chunk_email(documents, config)
        else:
            # Default to general chunking
            return self._chunk_general(documents, config)
    
    def _chunk_general(self, documents: List[Document], config: ChunkingConfig) -> List[Document]:
        """Standard recursive character text splitting"""
        # Parse delimiter string into list of separators
        separators = config.delimiter.split('|') if '|' in config.delimiter else [config.delimiter]
        separators = [sep.replace('\\n', '\n').replace('\\t', '\t') for sep in separators]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_token_num,
            chunk_overlap=config.chunk_overlap,
            separators=separators,
            length_function=len
        )
        
        return splitter.split_documents(documents)
    
    def _chunk_qa(self, documents: List[Document], config: ChunkingConfig) -> List[Document]:
        """Q&A format chunking - split on question/answer patterns"""
        chunks = []
        qa_patterns = [r'Q:', r'A:', r'Question:', r'Answer:', r'\d+\.', r'Q\d+', r'A\d+']
        
        for doc in documents:
            content = doc.page_content
            
            # Find Q&A boundaries
            boundaries = []
            for pattern in qa_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    boundaries.append(match.start())
            
            boundaries = sorted(set(boundaries))
            boundaries.append(len(content))
            
            # Create chunks based on Q&A boundaries
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                chunk_text = content[start:end].strip()
                
                if len(chunk_text) > 50:  # Minimum chunk size
                    chunk = Document(
                        page_content=chunk_text,
                        metadata={'qa_chunk': True}  # Only simple metadata, no complex inheritance
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _chunk_resume(self, documents: List[Document], config: ChunkingConfig) -> List[Document]:
        """Resume-specific chunking based on sections"""
        chunks = []
        section_patterns = [
            r'EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT',
            r'EDUCATION|ACADEMIC',
            r'SKILLS|TECHNICAL SKILLS|COMPETENCIES',
            r'PROJECTS|PROJECT EXPERIENCE',
            r'CERTIFICATIONS|CERTIFICATES',
            r'SUMMARY|OBJECTIVE|PROFILE'
        ]
        
        for doc in documents:
            content = doc.page_content
            
            # Find section boundaries
            boundaries = [0]  # Start with beginning
            for pattern in section_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    boundaries.append(match.start())
            
            boundaries = sorted(set(boundaries))
            boundaries.append(len(content))
            
            # Create chunks for each section
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                chunk_text = content[start:end].strip()
                
                if len(chunk_text) > 100:  # Minimum meaningful section size
                    chunk = Document(
                        page_content=chunk_text,
                        metadata={'resume_section': True}  # Only simple metadata
                    )
                    chunks.append(chunk)
        
        return chunks
    

    
    def _chunk_table(self, documents: List[Document], config: ChunkingConfig) -> List[Document]:
        """Table-aware chunking for structured data"""
        chunks = []
        
        for doc in documents:
            content = doc.page_content
            
            # Try to detect table structure
            lines = content.split('\n')
            
            # Check if content looks like CSV/TSV
            if any('\t' in line or ',' in line for line in lines[:5]):
                # Process as tabular data
                delimiter = '\t' if '\t' in content else ','
                
                # Group rows into chunks
                header = lines[0] if lines else ""
                data_lines = lines[1:] if len(lines) > 1 else []
                
                chunk_size = config.chunk_token_num // 50  # Rough estimate of rows per chunk
                
                for i in range(0, len(data_lines), chunk_size):
                    chunk_lines = [header] + data_lines[i:i + chunk_size]
                    chunk_content = '\n'.join(chunk_lines)
                    
                    chunk = Document(
                        page_content=chunk_content,
                        metadata={'table_chunk': True, 'chunk_rows': len(chunk_lines) - 1}  # Only simple metadata
                    )
                    chunks.append(chunk)
            else:
                # Not clearly tabular, use regular chunking
                chunks.extend(self._chunk_general([doc], config))
        
        return chunks
    

    
    def _chunk_presentation(self, documents: List[Document], config: ChunkingConfig) -> List[Document]:
        """Presentation chunking - typically one chunk per slide"""
        chunks = []
        
        for doc in documents:
            content = doc.page_content
            
            # Try to detect slide boundaries
            slide_patterns = [r'Slide\s+\d+', r'Page\s+\d+', r'^\d+\s*$']
            boundaries = [0]
            
            for pattern in slide_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    boundaries.append(match.start())
            
            # If no clear slide boundaries, split by double newlines
            if len(boundaries) <= 1:
                slide_breaks = [m.start() for m in re.finditer(r'\n\s*\n\s*\n', content)]
                boundaries.extend(slide_breaks)
            
            boundaries = sorted(set(boundaries))
            boundaries.append(len(content))
            
            # Create chunks for each slide
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                slide_content = content[start:end].strip()
                
                if slide_content:
                    chunk = Document(
                        page_content=slide_content,
                        metadata={'slide_chunk': True, 'slide_number': i}  # Only simple metadata
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _chunk_picture(self, documents: List[Document], config: ChunkingConfig) -> List[Document]:
        """Image content chunking - extract text via OCR if available"""
        chunks = []
        
        for doc in documents:
            # For image documents, the content might be OCR text
            content = doc.page_content
            
            if content and len(content.strip()) > 0:
                # Text was extracted from image
                if len(content) > config.chunk_token_num and config.chunk_token_num > 0:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=config.chunk_token_num,
                        chunk_overlap=config.chunk_overlap
                    )
                    image_chunks = splitter.split_documents([doc])
                    
                    # Distribute OCR bounding boxes among chunks if available
                    if 'ocr_bounding_boxes' in doc.metadata:
                        bounding_boxes = doc.metadata['ocr_bounding_boxes']
                        self._distribute_bounding_boxes_to_chunks(image_chunks, bounding_boxes)
                    
                    chunks.extend(image_chunks)
                else:
                    # Single chunk - include all bounding boxes
                    if 'ocr_bounding_boxes' in doc.metadata:
                        doc.metadata['chunk_bounding_boxes'] = doc.metadata['ocr_bounding_boxes']
                    chunks.append(doc)
            else:
                # No text content - create placeholder chunk
                chunk = Document(
                    page_content="[Image content - no text extracted]",
                    metadata={'image_only': True}  # Only simple metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _distribute_bounding_boxes_to_chunks(self, chunks: List[Document], bounding_boxes: List[dict]):
        """Distribute OCR bounding boxes among text chunks based on text content matching"""
        if not bounding_boxes:
            return
            
        # Create a mapping of words to their bounding boxes
        word_to_bbox = {}
        for bbox in bounding_boxes:
            word_key = bbox['text'].lower().strip()
            if word_key:
                if word_key not in word_to_bbox:
                    word_to_bbox[word_key] = []
                word_to_bbox[word_key].append(bbox)
        
        # For each chunk, find matching bounding boxes
        for chunk in chunks:
            chunk_bboxes = []
            chunk_text = chunk.page_content.lower()
            chunk_words = chunk_text.split()
            
            # Try to match words from chunk to bounding boxes
            used_bboxes = set()
            for word in chunk_words:
                clean_word = ''.join(c for c in word if c.isalnum()).lower()
                if clean_word in word_to_bbox:
                    # Find an unused bounding box for this word
                    for bbox in word_to_bbox[clean_word]:
                        bbox_id = f"{bbox['left']},{bbox['top']},{bbox['width']},{bbox['height']}"
                        if bbox_id not in used_bboxes:
                            chunk_bboxes.append(bbox)
                            used_bboxes.add(bbox_id)
                            break
            
            # Add bounding boxes to chunk metadata
            chunk.metadata['chunk_bounding_boxes'] = chunk_bboxes
    
    def _chunk_email(self, documents: List[Document], config: ChunkingConfig) -> List[Document]:
        """Email chunking based on email structure"""
        chunks = []
        email_patterns = [
            r'^From:', r'^To:', r'^Subject:', r'^Date:',
            r'^Reply-To:', r'^CC:', r'^BCC:'
        ]
        
        for doc in documents:
            content = doc.page_content
            
            # Find email header boundaries
            boundaries = [0]
            for pattern in email_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                    boundaries.append(match.start())
            
            # Also split on email separators
            email_separators = [
                r'^>{1,}\s*',  # Quote markers
                r'^-{3,}',     # Dash separators
                r'^={3,}',     # Equal separators
                r'On .* wrote:'  # Email thread markers
            ]
            
            for pattern in email_separators:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    boundaries.append(match.start())
            
            boundaries = sorted(set(boundaries))
            boundaries.append(len(content))
            
            # Create chunks for email sections
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                email_section = content[start:end].strip()
                
                if len(email_section) > 50:  # Minimum meaningful email section
                    chunk = Document(
                        page_content=email_section,
                        metadata={'email_section': True}  # Only simple metadata
                    )
                    chunks.append(chunk)
        
        return chunks
    

    
    # Document loading methods
    def _process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF files"""
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            return []
    
    def _process_docx(self, file_path: str) -> List[Document]:
        """Process DOCX files"""
        if ADVANCED_LOADERS_AVAILABLE:
            try:
                loader = Docx2txtLoader(file_path)
                return loader.load()
            except Exception as e:
                logger.error(f"Failed to process DOCX {file_path}: {e}")
        
        # Fallback to simple text reading
        return self._process_text(file_path)
    
    def _process_text(self, file_path: str) -> List[Document]:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return [Document(
                page_content=content,
                metadata={'source': Path(file_path).name}  # Use filename only
            )]
        except Exception as e:
            logger.error(f"Failed to process text file {file_path}: {e}")
            return []
    
    def _process_csv(self, file_path: str) -> List[Document]:
        """Process CSV files"""
        if ADVANCED_LOADERS_AVAILABLE:
            try:
                loader = CSVLoader(file_path)
                return loader.load()
            except Exception as e:
                logger.error(f"Failed to process CSV {file_path}: {e}")
        
        # Fallback to simple text reading
        return self._process_text(file_path)
    
    def _process_excel(self, file_path: str) -> List[Document]:
        """Process Excel files"""
        if PANDAS_AVAILABLE:
            try:
                df = pd.read_excel(file_path)
                content = df.to_string()
                
                return [Document(
                    page_content=content,
                    metadata={'source': Path(file_path).name, 'type': 'excel'}  # Use filename only
                )]
            except Exception as e:
                logger.error(f"Failed to process Excel file {file_path}: {e}")
        
        return []
    
    def _process_presentation(self, file_path: str) -> List[Document]:
        """Process PowerPoint files"""
        try:
            # Try using python-pptx first
            try:
                from pptx import Presentation
                return self._process_pptx_with_python_pptx(file_path)
            except ImportError:
                logger.warning("python-pptx not available, trying unstructured")
            
            # Fallback to unstructured if available
            if ADVANCED_LOADERS_AVAILABLE:
                try:
                    loader = UnstructuredPowerPointLoader(file_path)
                    return loader.load()
                except Exception as e:
                    logger.error(f"Failed to process presentation with unstructured {file_path}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to process presentation {file_path}: {e}")
        
        return []
    
    def _process_pptx_with_python_pptx(self, file_path: str) -> List[Document]:
        """Process PPTX files using python-pptx library"""
        try:
            from pptx import Presentation
            
            prs = Presentation(file_path)
            documents = []
            
            for slide_num, slide in enumerate(prs.slides, 1):
                slide_text = []
                
                # Extract text from shapes
                for shape in slide.shapes:
                    if hasattr(shape, 'text') and shape.text.strip():
                        slide_text.append(shape.text.strip())
                    
                    # Handle tables - check if shape actually contains a table
                    if hasattr(shape, 'table') and shape.has_table:
                        try:
                            table_text = []
                            for row in shape.table.rows:
                                row_text = []
                                for cell in row.cells:
                                    if cell.text.strip():
                                        row_text.append(cell.text.strip())
                                if row_text:
                                    table_text.append(' | '.join(row_text))
                            if table_text:
                                slide_text.append('\n'.join(table_text))
                        except Exception as e:
                            logger.warning(f"Could not process table in slide {slide_num}: {e}")
                            # Continue processing other shapes
                
                if slide_text:
                    slide_content = '\n\n'.join(slide_text)
                    documents.append(Document(
                        page_content=slide_content,
                        metadata={
                            'source': Path(file_path).name,
                            'slide_number': slide_num,
                            'type': 'presentation',
                            'total_slides': len(prs.slides)
                        }
                    ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process PPTX with python-pptx {file_path}: {e}")
            return []
    
    def _process_html(self, file_path: str) -> List[Document]:
        """Process HTML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple HTML tag removal
            import re
            clean_content = re.sub(r'<[^>]+>', '', content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            return [Document(
                page_content=clean_content,
                metadata={'source': Path(file_path).name, 'type': 'html'}  # Use filename only
            )]
        except Exception as e:
            logger.error(f"Failed to process HTML file {file_path}: {e}")
            return []
    
    def _process_json(self, file_path: str) -> List[Document]:
        """Process JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to readable text
            content = json.dumps(data, indent=2)
            
            return [Document(
                page_content=content,
                metadata={'source': Path(file_path).name, 'type': 'json'}  # Use filename only
            )]
        except Exception as e:
            logger.error(f"Failed to process JSON file {file_path}: {e}")
            return []
    
    def _process_email(self, file_path: str) -> List[Document]:
        """Process email files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return [Document(
                page_content=content,
                metadata={'source': Path(file_path).name, 'type': 'email'}  # Use filename only
            )]
        except Exception as e:
            logger.error(f"Failed to process email file {file_path}: {e}")
            return []
    
    def _process_image(self, file_path: str) -> List[Document]:
        """Process image files with OCR if available"""
        if OCR_AVAILABLE:
            try:
                logger.info(f"Processing image with OCR: {file_path}")
                image = Image.open(file_path)
                
                # Enhance image for better OCR results
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Get OCR text with better configuration
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(image, config=custom_config)
                
                # Get OCR data with bounding boxes
                ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
                
                # Process bounding box data with lower confidence threshold
                bounding_boxes = []
                image_width, image_height = image.size
                
                for i in range(len(ocr_data['text'])):
                    confidence = int(ocr_data['conf'][i])
                    if confidence > 10:  # Lower confidence threshold to capture more text
                        word_text = ocr_data['text'][i].strip()
                        if word_text and len(word_text) > 0:  # Only include non-empty text
                            bbox = {
                                'text': word_text,
                                'left': int(ocr_data['left'][i]),
                                'top': int(ocr_data['top'][i]),
                                'width': int(ocr_data['width'][i]),
                                'height': int(ocr_data['height'][i]),
                                'confidence': confidence,
                                'block_num': int(ocr_data['block_num'][i]),
                                'par_num': int(ocr_data['par_num'][i]),
                                'line_num': int(ocr_data['line_num'][i]),
                                'word_num': int(ocr_data['word_num'][i])
                            }
                            bounding_boxes.append(bbox)
                
                # Log OCR output
                logger.info(f"OCR extracted text from {Path(file_path).name}:")
                logger.info(f"--- OCR OUTPUT START ---")
                logger.info(text.strip())
                logger.info(f"--- OCR OUTPUT END ---")
                logger.info(f"OCR extracted {len(text.strip())} characters and {len(bounding_boxes)} bounding boxes from {Path(file_path).name}")
                
                # If no text was extracted, try different OCR settings
                if len(text.strip()) < 10 and len(bounding_boxes) < 5:
                    logger.warning(f"Poor OCR results, trying alternative settings for {Path(file_path).name}")
                    
                    # Try different PSM mode for single block of text
                    alt_config = r'--oem 3 --psm 8'
                    alt_text = pytesseract.image_to_string(image, config=alt_config)
                    alt_ocr_data = pytesseract.image_to_data(image, config=alt_config, output_type=pytesseract.Output.DICT)
                    
                    if len(alt_text.strip()) > len(text.strip()):
                        logger.info(f"Alternative OCR settings produced better results for {Path(file_path).name}")
                        text = alt_text
                        ocr_data = alt_ocr_data
                        
                        # Reprocess bounding boxes with alternative data
                        bounding_boxes = []
                        for i in range(len(ocr_data['text'])):
                            confidence = int(ocr_data['conf'][i])
                            if confidence > 10:
                                word_text = ocr_data['text'][i].strip()
                                if word_text and len(word_text) > 0:
                                    bbox = {
                                        'text': word_text,
                                        'left': int(ocr_data['left'][i]),
                                        'top': int(ocr_data['top'][i]),
                                        'width': int(ocr_data['width'][i]),
                                        'height': int(ocr_data['height'][i]),
                                        'confidence': confidence,
                                        'block_num': int(ocr_data['block_num'][i]),
                                        'par_num': int(ocr_data['par_num'][i]),
                                        'line_num': int(ocr_data['line_num'][i]),
                                        'word_num': int(ocr_data['word_num'][i])
                                    }
                                    bounding_boxes.append(bbox)
                
                return [Document(
                    page_content=text,
                    metadata={
                        'source': Path(file_path).name, 
                        'type': 'image', 
                        'extracted_via': 'ocr',
                        'image_width': image_width,
                        'image_height': image_height,
                        'ocr_bounding_boxes': bounding_boxes
                    }
                )]
            except Exception as e:
                logger.error(f"Failed to process image {file_path}: {e}")
        else:
            logger.warning(f"OCR not available for image processing: {file_path}")
        
        # Return placeholder if OCR not available
        return [Document(
            page_content="[Image file - OCR not available]",
            metadata={'source': Path(file_path).name, 'type': 'image', 'extracted_via': 'none'}  # Use filename only
        )]

# Global instance
_document_processor = None

def get_document_processor() -> EnhancedDocumentProcessor:
    """Get global document processor instance"""
    global _document_processor
    if _document_processor is None:
        _document_processor = EnhancedDocumentProcessor()
    return _document_processor
