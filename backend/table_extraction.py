"""
Table Extraction and Processing for Document Chunking
Handles table detection and extraction from various document formats
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)

# Try to import table extraction libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    logger.warning("pdfplumber not available. PDF table extraction will be limited.")
    PDFPLUMBER_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    logger.warning("tabula-py not available. Advanced PDF table extraction disabled.")
    TABULA_AVAILABLE = False

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    logger.warning("camelot-py not available. Advanced PDF table extraction disabled.")
    CAMELOT_AVAILABLE = False

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    logger.warning("python-pptx not available. PowerPoint table extraction disabled.")
    PPTX_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    logger.warning("python-docx not available. Word table extraction disabled.")
    DOCX_AVAILABLE = False


class TableExtractor:
    """Extract tables from various document formats"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self.extract_pdf_tables,
            '.pptx': self.extract_pptx_tables,
            '.ppt': self.extract_pptx_tables,
            '.docx': self.extract_docx_tables,
            '.doc': self.extract_docx_tables,
            '.csv': self.extract_csv_tables,
            '.xlsx': self.extract_excel_tables,
            '.xls': self.extract_excel_tables
        }
    
    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from a document file
        
        Returns:
            List of table dictionaries with structure:
            {
                'data': pandas.DataFrame or list of lists,
                'page_number': int or None,
                'slide_number': int or None,
                'table_number': int,
                'bbox': tuple or None (x1, y1, x2, y2),
                'extraction_method': str,
                'confidence': float
            }
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in self.supported_formats:
            logger.warning(f"File format {file_ext} not supported for table extraction")
            return []
        
        try:
            return self.supported_formats[file_ext](file_path)
        except Exception as e:
            logger.error(f"Error extracting tables from {file_path}: {e}")
            return []
    
    def extract_pdf_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF files using multiple methods"""
        tables = []
        
        # Method 1: pdfplumber (best for simple tables)
        if PDFPLUMBER_AVAILABLE:
            tables.extend(self._extract_pdf_tables_pdfplumber(file_path))
        
        # Method 2: tabula-py (good for complex tables)
        if TABULA_AVAILABLE and len(tables) == 0:
            tables.extend(self._extract_pdf_tables_tabula(file_path))
        
        # Method 3: camelot (most accurate but slower)
        if CAMELOT_AVAILABLE and len(tables) == 0:
            tables.extend(self._extract_pdf_tables_camelot(file_path))
        
        return tables
    
    def _extract_pdf_tables_pdfplumber(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber with improved settings"""
        tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Try different table extraction settings
                    extraction_settings = [
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "text", "horizontal_strategy": "text"},
                        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
                    ]
                    
                    page_tables_found = False
                    
                    for settings in extraction_settings:
                        try:
                            page_tables = page.extract_tables(table_settings=settings)
                            
                            for table_num, table_data in enumerate(page_tables, 1):
                                if table_data and len(table_data) > 1:  # Must have header + data
                                    # Clean the table data
                                    cleaned_table = self._clean_table_data(table_data)
                                    
                                    if cleaned_table and len(cleaned_table) > 1:
                                        # Convert to DataFrame
                                        try:
                                            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                                            
                                            # Clean empty rows/columns
                                            df = df.dropna(how='all').dropna(axis=1, how='all')
                                            
                                            # Remove columns that are mostly None or empty
                                            df = self._clean_dataframe(df)
                                            
                                            if not df.empty and len(df.columns) >= 2:
                                                tables.append({
                                                    'data': df,
                                                    'page_number': page_num,
                                                    'slide_number': None,
                                                    'table_number': table_num,
                                                    'bbox': None,
                                                    'extraction_method': f'pdfplumber_{settings["vertical_strategy"]}',
                                                    'confidence': 0.8
                                                })
                                                page_tables_found = True
                                                logger.info(f"Successfully extracted table {table_num} from page {page_num} using {settings}")
                                        except Exception as e:
                                            logger.warning(f"Failed to process table data on page {page_num}: {e}")
                                            continue
                            
                            if page_tables_found:
                                break  # Use the first successful extraction method
                                
                        except Exception as e:
                            logger.warning(f"Table extraction failed with settings {settings} on page {page_num}: {e}")
                            continue
                    
                    if not page_tables_found:
                        logger.info(f"No tables found on page {page_num} with pdfplumber")
                        
        except Exception as e:
            logger.error(f"pdfplumber table extraction failed: {e}")
        
        return tables

    def _clean_table_data(self, table_data: List[List]) -> List[List]:
        """Clean raw table data from PDF extraction"""
        if not table_data:
            return []
        
        cleaned_table = []
        
        for row in table_data:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Clean the cell content
                    cleaned_cell = str(cell).strip()
                    # Remove excessive whitespace
                    cleaned_cell = re.sub(r'\s+', ' ', cleaned_cell)
                    cleaned_row.append(cleaned_cell)
            
            # Only add row if it has some content
            if any(cell.strip() for cell in cleaned_row):
                cleaned_table.append(cleaned_row)
        
        return cleaned_table

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame by removing mostly empty columns and fixing headers"""
        # Ensure we have a valid pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning(f"Expected pandas DataFrame, got {type(df)}. Attempting to convert.")
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                logger.error(f"Failed to convert to DataFrame: {e}")
                return pd.DataFrame()  # Return empty DataFrame
        
        if df.empty:
            return df
        
        # Clean column names
        new_columns = []
        for i, col in enumerate(df.columns):
            if pd.isna(col) or str(col).strip() == '' or str(col).strip().lower() in ['none', 'nan']:
                new_columns.append(f'Column_{i+1}')
            else:
                new_columns.append(str(col).strip())
        df.columns = new_columns
        
        # Remove columns that are mostly empty
        threshold = 0.7  # Keep columns that have at least 30% non-empty values
        columns_to_drop = []
        for col in df.columns:
            try:
                # Ensure we have a pandas Series before calling .str methods
                series = df[col]
                if hasattr(series, 'astype') and hasattr(series, 'str'):
                    non_empty_ratio = series.astype(str).str.strip().str.len().gt(0).sum() / len(df)
                    if non_empty_ratio < (1 - threshold):
                        columns_to_drop.append(col)
                else:
                    logger.warning(f"Column {col} doesn't support string operations, skipping threshold check")
            except AttributeError as e:
                logger.warning(f"Column {col} doesn't support string operations: {e}")
                continue
        
        # Drop columns that are mostly empty
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        
        # Clean cell values
        for col in df.columns:
            try:
                series = df[col]
                if hasattr(series, 'astype') and hasattr(series, 'str'):
                    df[col] = series.astype(str).str.strip()
                    df[col] = df[col].replace(['None', 'nan', ''], pd.NA)
                else:
                    logger.warning(f"Column {col} doesn't support string operations, leaving as-is")
            except AttributeError as e:
                logger.warning(f"Column {col} doesn't support string operations: {e}")
                continue
        
        return df
    
    def _extract_pdf_tables_tabula(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables using tabula-py with multiple strategies"""
        tables = []
        
        try:
            # Strategy 1: Try lattice method (for tables with clear borders)
            try:
                dfs_lattice = tabula.read_pdf(
                    file_path, 
                    pages='all', 
                    multiple_tables=True,
                    lattice=True,
                    pandas_options={'header': 0}
                )
                
                for table_num, df in enumerate(dfs_lattice, 1):
                    if not df.empty:
                        # Clean the DataFrame
                        df = self._clean_dataframe(df)
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        
                        if not df.empty and len(df.columns) >= 2:
                            tables.append({
                                'data': df,
                                'page_number': None,  # tabula doesn't provide page info easily
                                'slide_number': None,
                                'table_number': table_num,
                                'bbox': None,
                                'extraction_method': 'tabula_lattice',
                                'confidence': 0.8
                            })
                            logger.info(f"Successfully extracted table {table_num} using tabula lattice method")
                
                if tables:
                    return tables  # If lattice worked, use those results
                    
            except Exception as e:
                logger.warning(f"tabula lattice method failed: {e}")
            
            # Strategy 2: Try stream method (for tables without clear borders)
            try:
                dfs_stream = tabula.read_pdf(
                    file_path, 
                    pages='all', 
                    multiple_tables=True,
                    stream=True,
                    pandas_options={'header': 0}
                )
                
                for table_num, df in enumerate(dfs_stream, 1):
                    if not df.empty:
                        # Clean the DataFrame
                        df = self._clean_dataframe(df)
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        
                        if not df.empty and len(df.columns) >= 2:
                            tables.append({
                                'data': df,
                                'page_number': None,
                                'slide_number': None,
                                'table_number': table_num,
                                'bbox': None,
                                'extraction_method': 'tabula_stream',
                                'confidence': 0.7
                            })
                            logger.info(f"Successfully extracted table {table_num} using tabula stream method")
                            
            except Exception as e:
                logger.warning(f"tabula stream method failed: {e}")
            
            # Strategy 3: Try basic extraction with guess
            if not tables:
                try:
                    dfs_guess = tabula.read_pdf(
                        file_path, 
                        pages='all', 
                        multiple_tables=True,
                        guess=True
                    )
                    
                    for table_num, df in enumerate(dfs_guess, 1):
                        if not df.empty:
                            # Clean the DataFrame
                            df = self._clean_dataframe(df)
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            if not df.empty and len(df.columns) >= 2:
                                tables.append({
                                    'data': df,
                                    'page_number': None,
                                    'slide_number': None,
                                    'table_number': table_num,
                                    'bbox': None,
                                    'extraction_method': 'tabula_guess',
                                    'confidence': 0.6
                                })
                                logger.info(f"Successfully extracted table {table_num} using tabula guess method")
                                
                except Exception as e:
                    logger.warning(f"tabula guess method failed: {e}")
        
        except Exception as e:
            logger.error(f"tabula table extraction failed: {e}")
        
        return tables
    
    def _extract_pdf_tables_camelot(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables using camelot-py"""
        tables = []
        
        try:
            # Extract tables with lattice method (for tables with lines)
            lattice_tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
            
            for table in lattice_tables:
                if table.parsing_report['accuracy'] > 50:  # Only use tables with decent accuracy
                    df = table.df
                    if not df.empty:
                        tables.append({
                            'data': df,
                            'page_number': table.page,
                            'slide_number': None,
                            'table_number': len(tables) + 1,
                            'bbox': table._bbox,
                            'extraction_method': 'camelot_lattice',
                            'confidence': table.parsing_report['accuracy'] / 100
                        })
            
            # If no lattice tables found, try stream method (for tables without lines)
            if not tables:
                stream_tables = camelot.read_pdf(file_path, pages='all', flavor='stream')
                
                for table in stream_tables:
                    if table.parsing_report['accuracy'] > 50:
                        df = table.df
                        if not df.empty:
                            tables.append({
                                'data': df,
                                'page_number': table.page,
                                'slide_number': None,
                                'table_number': len(tables) + 1,
                                'bbox': table._bbox,
                                'extraction_method': 'camelot_stream',
                                'confidence': table.parsing_report['accuracy'] / 100
                            })
        
        except Exception as e:
            logger.error(f"camelot table extraction failed: {e}")
        
        return tables
    
    def extract_pptx_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PowerPoint files"""
        tables = []
        
        if not PPTX_AVAILABLE:
            return tables
        
        try:
            prs = Presentation(file_path)
            
            for slide_num, slide in enumerate(prs.slides, 1):
                slide_tables = []
                
                for shape in slide.shapes:
                    if hasattr(shape, 'table') and shape.has_table:
                        table_data = []
                        
                        # Extract table data
                        for row in shape.table.rows:
                            row_data = []
                            for cell in row.cells:
                                row_data.append(cell.text.strip())
                            table_data.append(row_data)
                        
                        if table_data and len(table_data) > 1:
                            # Convert to DataFrame
                            try:
                                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                df = df.dropna(how='all').dropna(axis=1, how='all')
                                
                                if not df.empty:
                                    tables.append({
                                        'data': df,
                                        'page_number': None,
                                        'slide_number': slide_num,
                                        'table_number': len(slide_tables) + 1,
                                        'bbox': None,
                                        'extraction_method': 'python_pptx',
                                        'confidence': 0.9
                                    })
                                    slide_tables.append(df)
                            except Exception as e:
                                logger.warning(f"Failed to process table in slide {slide_num}: {e}")
        
        except Exception as e:
            logger.error(f"PowerPoint table extraction failed: {e}")
        
        return tables
    
    def extract_docx_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from Word documents"""
        tables = []
        
        if not DOCX_AVAILABLE:
            return tables
        
        try:
            doc = DocxDocument(file_path)
            
            for table_num, table in enumerate(doc.tables, 1):
                table_data = []
                
                for row in table.rows:
                    row_data = []
                    for cell in row.cells:
                        row_data.append(cell.text.strip())
                    table_data.append(row_data)
                
                if table_data and len(table_data) > 1:
                    try:
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        
                        if not df.empty:
                            tables.append({
                                'data': df,
                                'page_number': None,
                                'slide_number': None,
                                'table_number': table_num,
                                'bbox': None,
                                'extraction_method': 'python_docx',
                                'confidence': 0.9
                            })
                    except Exception as e:
                        logger.warning(f"Failed to process table {table_num}: {e}")
        
        except Exception as e:
            logger.error(f"Word document table extraction failed: {e}")
        
        return tables
    
    def extract_csv_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from CSV files"""
        tables = []
        
        try:
            df = pd.read_csv(file_path)
            
            if not df.empty:
                tables.append({
                    'data': df,
                    'page_number': None,
                    'slide_number': None,
                    'table_number': 1,
                    'bbox': None,
                    'extraction_method': 'pandas_csv',
                    'confidence': 1.0
                })
        
        except Exception as e:
            logger.error(f"CSV table extraction failed: {e}")
        
        return tables
    
    def extract_excel_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from Excel files"""
        tables = []
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_num, sheet_name in enumerate(excel_file.sheet_names, 1):
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                if not df.empty:
                    tables.append({
                        'data': df,
                        'page_number': None,
                        'slide_number': sheet_num,  # Use slide_number for sheet number
                        'table_number': 1,
                        'bbox': None,
                        'extraction_method': 'pandas_excel',
                        'confidence': 1.0,
                        'sheet_name': sheet_name
                    })
        
        except Exception as e:
            logger.error(f"Excel table extraction failed: {e}")
        
        return tables


def detect_table_in_text(text: str) -> bool:
    """
    Detect if text contains table-like structures with improved detection
    
    Args:
        text: Text content to analyze
        
    Returns:
        bool: True if table-like structure detected
    """
    lines = text.strip().split('\n')
    
    if len(lines) < 2:
        return False
    
    # Check for common table indicators
    table_indicators = [
        # Multiple consecutive lines with delimiters
        lambda: sum(1 for line in lines if '|' in line or '\t' in line) >= 2,
        
        # Lines with consistent column structure
        lambda: _has_consistent_columns(lines),
        
        # Table headers pattern
        lambda: any(re.search(r'^[\w\s]+\s*\|\s*[\w\s]+\s*\|', line) for line in lines[:3]),
        
        # CSV-like structure
        lambda: sum(1 for line in lines if line.count(',') >= 2) >= 2,
        
        # Tab-separated values
        lambda: sum(1 for line in lines if line.count('\t') >= 1) >= 2,
        
        # Patterns that suggest tabular data
        lambda: _detect_structured_data_patterns(lines),
        
        # Look for column-like spacing
        lambda: _detect_column_spacing(lines),
    ]
    
    return any(indicator() for indicator in table_indicators)


def _detect_structured_data_patterns(lines: List[str]) -> bool:
    """Detect patterns that suggest structured tabular data"""
    if len(lines) < 3:
        return False
    
    # Look for repeating patterns that suggest rows of data
    pattern_matches = 0
    
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if line:
            # Check for patterns like numbers, percentages, currency
            if re.search(r'\d+%|\$\d+|\d+\.\d+|\d+,\d+', line):
                pattern_matches += 1
            
            # Check for patterns with consistent separators
            if len(re.findall(r'\s{2,}|\t|\|', line)) >= 2:
                pattern_matches += 1
    
    return pattern_matches >= 3


def _detect_column_spacing(lines: List[str]) -> bool:
    """Detect consistent column-like spacing in text"""
    if len(lines) < 3:
        return False
    
    # Look for consistent spacing patterns across lines
    spacing_patterns = []
    
    for line in lines[:5]:
        if line.strip():
            # Find positions of text blocks separated by spaces
            words = []
            current_pos = 0
            for word in line.split():
                pos = line.find(word, current_pos)
                words.append(pos)
                current_pos = pos + len(word)
            
            if len(words) >= 3:  # At least 3 columns
                spacing_patterns.append(words)
    
    if len(spacing_patterns) >= 2:
        # Check if column positions are relatively consistent
        first_pattern = spacing_patterns[0]
        consistent_columns = 0
        
        for pattern in spacing_patterns[1:]:
            if len(pattern) == len(first_pattern):
                # Check if positions are within reasonable tolerance
                tolerance = 5  # characters
                matches = sum(1 for i, pos in enumerate(pattern) 
                            if abs(pos - first_pattern[i]) <= tolerance)
                if matches >= len(pattern) * 0.7:  # 70% of columns match
                    consistent_columns += 1
        
        return consistent_columns >= 1
    
    return False


def _has_consistent_columns(lines: List[str]) -> bool:
    """Check if lines have consistent column structure with improved detection"""
    if len(lines) < 3:
        return False
    
    # Check for pipe-separated columns
    pipe_counts = [line.count('|') for line in lines[:5] if line.strip()]
    if len(set(pipe_counts)) <= 2 and pipe_counts and max(pipe_counts) >= 1:
        return True
    
    # Check for tab-separated columns
    tab_counts = [line.count('\t') for line in lines[:5] if line.strip()]
    if len(set(tab_counts)) <= 2 and tab_counts and max(tab_counts) >= 1:
        return True
    
    # Check for consistent multi-space separators
    multi_space_counts = [len(re.findall(r'\s{2,}', line)) for line in lines[:5] if line.strip()]
    if len(set(multi_space_counts)) <= 2 and multi_space_counts and max(multi_space_counts) >= 1:
        return True
    
    # Check for space-separated columns (more complex)
    word_counts = [len(line.split()) for line in lines[:5] if line.strip()]
    if len(set(word_counts)) <= 3 and word_counts and max(word_counts) >= 3:
        # Additional check: ensure the words appear to be in columns
        if _check_column_alignment(lines[:5]):
            return True
    
    return False


def _check_column_alignment(lines: List[str]) -> bool:
    """Check if words in lines appear to be aligned in columns"""
    if len(lines) < 2:
        return False
    
    # Get word positions for each line
    line_positions = []
    for line in lines:
        if line.strip():
            positions = []
            words = line.split()
            current_pos = 0
            for word in words:
                pos = line.find(word, current_pos)
                positions.append(pos)
                current_pos = pos + len(word)
            if len(positions) >= 3:  # At least 3 words/columns
                line_positions.append(positions)
    
    if len(line_positions) < 2:
        return False
    
    # Check if column positions are reasonably consistent
    first_line_positions = line_positions[0]
    tolerance = 3  # Allow 3 character difference
    
    for positions in line_positions[1:]:
        if len(positions) != len(first_line_positions):
            continue
        
        aligned_columns = 0
        for i, pos in enumerate(positions):
            if abs(pos - first_line_positions[i]) <= tolerance:
                aligned_columns += 1
        
        if aligned_columns >= len(positions) * 0.6:  # At least 60% aligned
            return True
    
    return False


def format_table_for_chunking(table_data: pd.DataFrame, include_headers: bool = True) -> str:
    """
    Format a table DataFrame for text chunking - DEPRECATED
    Use create_row_based_chunks instead for proper table-aware chunking
    """
    if table_data.empty:
        return ""
    
    # Clean the DataFrame
    cleaned_df = table_data.copy()
    
    # Fill NaN values with empty string
    cleaned_df = cleaned_df.fillna('')
    
    # Convert all columns to string
    for col in cleaned_df.columns:
        cleaned_df[col] = cleaned_df[col].astype(str)
    
    # Format as pipe-separated text for better readability
    if include_headers:
        # Create header row
        headers = ' | '.join(str(col) for col in cleaned_df.columns)
        separator = ' | '.join('---' for _ in cleaned_df.columns)
        
        # Create data rows
        data_rows = []
        for _, row in cleaned_df.iterrows():
            data_rows.append(' | '.join(str(val) for val in row.values))
        
        return '\n'.join([headers, separator] + data_rows)
    else:
        # Only data rows
        data_rows = []
        for _, row in cleaned_df.iterrows():
            data_rows.append(' | '.join(str(val) for val in row.values))
        
        return '\n'.join(data_rows)


def create_row_based_chunks(table_data: pd.DataFrame, table_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create proper table-aware chunks: one chunk per row with full context
    
    This implements the right way to do table chunking:
    1. Parse the table structure explicitly (headers and rows)
    2. Create one chunk per row with column headers + cell values
    3. Format chunks cleanly with key-value pairs
    
    Args:
        table_data: pandas DataFrame containing table data
        table_info: Table metadata information
        
    Returns:
        List of chunk dictionaries with proper row-based content
    """
    if table_data.empty:
        return []
    
    chunks = []
    
    # Clean the DataFrame
    cleaned_df = table_data.copy()
    cleaned_df = cleaned_df.fillna('')
    
    # Convert all columns to string and clean them
    for col in cleaned_df.columns:
        try:
            # Ensure we have a proper pandas Series before applying .str operations
            series = cleaned_df[col]
            if hasattr(series, 'astype') and hasattr(series, 'str'):
                cleaned_df[col] = series.astype(str).str.strip()
            else:
                # Fallback for non-pandas series
                cleaned_df[col] = cleaned_df[col].apply(lambda x: str(x).strip() if x is not None else '')
        except AttributeError as e:
            logger.warning(f"Column {col} doesn't support string operations: {e}")
            # Try basic string conversion
            cleaned_df[col] = cleaned_df[col].apply(lambda x: str(x).strip() if x is not None else '')
    
    # Get table title/identifier
    table_identifier = f"Table {table_info.get('table_number', 1)}"
    if table_info.get('page_number'):
        table_identifier += f" (Page {table_info['page_number']})"
    elif table_info.get('slide_number'):
        table_identifier += f" (Slide {table_info['slide_number']})"
    
    # Create one chunk per row
    for row_idx, (_, row) in enumerate(cleaned_df.iterrows()):
        # Create key-value pairs for this row
        row_data = []
        primary_identifier = None
        
        for col_name, cell_value in row.items():
            if cell_value and str(cell_value).strip() and str(cell_value).strip().lower() not in ['nan', 'none', '']:
                clean_value = str(cell_value).strip()
                row_data.append(f"{col_name}: {clean_value}")
                
                # Use first meaningful column as primary identifier
                if primary_identifier is None:
                    primary_identifier = clean_value
        
        if row_data:  # Only create chunk if row has meaningful data
            # Format the chunk content
            chunk_title = f"{table_identifier} - Row {row_idx + 1}"
            if primary_identifier:
                chunk_title += f" ({primary_identifier})"
            
            chunk_content = f"{chunk_title}\n\n" + "\n".join(row_data)
            
            # Create metadata for this row chunk
            chunk_metadata = create_table_chunk_metadata(table_info, chunk_index=row_idx)
            chunk_metadata.update({
                'chunk_type': 'table_row',
                'row_index': row_idx,
                'primary_identifier': primary_identifier,
                'column_count': len(row_data),
                'chunking_strategy': 'row_based'
            })
            
            chunks.append({
                'content': chunk_content,
                'metadata': chunk_metadata
            })
    
    return chunks


def create_semantic_table_chunks(table_data: pd.DataFrame, table_info: Dict[str, Any], 
                                max_rows_per_chunk: int = 5) -> List[Dict[str, Any]]:
    """
    Create semantically grouped table chunks when rows are related
    
    Args:
        table_data: pandas DataFrame containing table data
        table_info: Table metadata information
        max_rows_per_chunk: Maximum number of rows to include in one chunk
        
    Returns:
        List of chunk dictionaries with semantically grouped content
    """
    if table_data.empty:
        return []
    
    chunks = []
    
    # Clean the DataFrame
    cleaned_df = table_data.copy()
    cleaned_df = cleaned_df.fillna('')
    
    # Convert all columns to string and clean them
    for col in cleaned_df.columns:
        try:
            series = cleaned_df[col]
            if hasattr(series, 'astype') and hasattr(series, 'str'):
                cleaned_df[col] = series.astype(str).str.strip()
            else:
                # Fallback for non-pandas series
                cleaned_df[col] = cleaned_df[col].apply(lambda x: str(x).strip() if x is not None else '')
        except AttributeError as e:
            logger.warning(f"Column {col} doesn't support string operations: {e}")
            # Try basic string conversion
            cleaned_df[col] = cleaned_df[col].apply(lambda x: str(x).strip() if x is not None else '')
    
    # Get table title/identifier
    table_identifier = f"Table {table_info.get('table_number', 1)}"
    if table_info.get('page_number'):
        table_identifier += f" (Page {table_info['page_number']})"
    elif table_info.get('slide_number'):
        table_identifier += f" (Slide {table_info['slide_number']})"
    
    # Split into chunks of max_rows_per_chunk
    total_rows = len(cleaned_df)
    
    for chunk_start in range(0, total_rows, max_rows_per_chunk):
        chunk_end = min(chunk_start + max_rows_per_chunk, total_rows)
        chunk_df = cleaned_df.iloc[chunk_start:chunk_end]
        
        # Create chunk content
        chunk_title = f"{table_identifier} - Rows {chunk_start + 1} to {chunk_end}"
        
        # Add column headers
        headers = " | ".join(chunk_df.columns)
        separator = " | ".join("---" for _ in chunk_df.columns)
        
        # Add data rows
        data_rows = []
        for _, row in chunk_df.iterrows():
            row_values = []
            for col_name, cell_value in row.items():
                clean_value = str(cell_value).strip() if cell_value else ""
                if clean_value.lower() in ['nan', 'none', '']:
                    clean_value = ""
                row_values.append(clean_value)
            data_rows.append(" | ".join(row_values))
        
        # Combine everything
        chunk_content = f"{chunk_title}\n\n{headers}\n{separator}\n" + "\n".join(data_rows)
        
        # Create metadata
        chunk_metadata = create_table_chunk_metadata(table_info, chunk_index=chunk_start // max_rows_per_chunk)
        chunk_metadata.update({
            'chunk_type': 'table_group',
            'row_start': chunk_start,
            'row_end': chunk_end - 1,
            'row_count': len(chunk_df),
            'chunking_strategy': 'semantic_group'
        })
        
        chunks.append({
            'content': chunk_content,
            'metadata': chunk_metadata
        })
    
    return chunks


def create_table_chunk_metadata(table_info: Dict[str, Any], chunk_index: int = 0) -> Dict[str, Any]:
    """
    Create metadata for a table chunk
    
    Args:
        table_info: Table information dictionary
        chunk_index: Index of the chunk within the table
        
    Returns:
        dict: Metadata for the chunk
    """
    metadata = {
        'chunk_type': 'table',
        'table_number': table_info.get('table_number', 1),
        'extraction_method': table_info.get('extraction_method', 'unknown'),
        'confidence': table_info.get('confidence', 0.5),
        'chunk_index': chunk_index
    }
    
    # Add page/slide information if available
    if table_info.get('page_number'):
        metadata['page_number'] = table_info['page_number']
    
    if table_info.get('slide_number'):
        metadata['slide_number'] = table_info['slide_number']
    
    if table_info.get('sheet_name'):
        metadata['sheet_name'] = table_info['sheet_name']
    
    # Add bounding box if available
    if table_info.get('bbox'):
        metadata['bbox'] = table_info['bbox']
    
    return metadata


# Global instance
_table_extractor = None

def get_table_extractor() -> TableExtractor:
    """Get global table extractor instance"""
    global _table_extractor
    if _table_extractor is None:
        _table_extractor = TableExtractor()
    return _table_extractor
