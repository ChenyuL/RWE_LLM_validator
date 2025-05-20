# src/utils/pdf_processor.py
import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import tempfile
from pathlib import Path

class PDFProcessor:
    """
    A utility class for processing PDF files in the LLM Validation Framework.
    Handles extraction of text, sectioning, and preprocessing for LLM analysis.
    """
    
    def __init__(self, use_ocr: bool = False):
        """
        Initialize the PDF processor.
        
        Args:
            use_ocr: Whether to use OCR for scanned PDFs (requires additional dependencies)
        """
        self.logger = logging.getLogger(__name__)
        self.use_ocr = use_ocr
        
        # Initialize PDF processing libraries
        try:
            import PyPDF2
            self.pdf_reader = PyPDF2
            self.logger.info("Initialized PyPDF2 for PDF processing")
        except ImportError:
            self.logger.warning("PyPDF2 not available, trying alternative")
            try:
                import fitz  # PyMuPDF
                self.pdf_reader = "pymupdf"
                self.logger.info("Initialized PyMuPDF for PDF processing")
            except ImportError:
                self.logger.error("No PDF processing library available")
                raise ImportError("Please install PyPDF2 or PyMuPDF (fitz) for PDF processing")
        
        # Initialize OCR if requested
        if self.use_ocr:
            try:
                import pytesseract
                self.ocr_engine = pytesseract
                self.logger.info("Initialized Tesseract OCR")
            except ImportError:
                self.logger.warning("OCR requested but pytesseract not available")
                self.use_ocr = False
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        self.logger.info(f"Extracting text from {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if self.pdf_reader == "pymupdf":
            return self._extract_with_pymupdf(pdf_path)
        else:
            return self._extract_with_pypdf2(pdf_path)
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """
        Extract text using PyPDF2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        text = ""
        reader = self.pdf_reader.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        
        for i in range(num_pages):
            try:
                page = reader.pages[i]
                page_text = page.extract_text()
                
                # Check if page is potentially scanned (very little text)
                if self.use_ocr and len(page_text.strip()) < 100:
                    page_text = self._process_with_ocr(pdf_path, i)
                
                text += f"\n\n--- Page {i+1} ---\n\n"
                text += page_text
            except Exception as e:
                self.logger.error(f"Error extracting text from page {i+1}: {e}")
                text += f"\n\n--- Page {i+1} (Error: {str(e)}) ---\n\n"
        
        return self._clean_text(text)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """
        Extract text using PyMuPDF (fitz).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        import fitz
        text = ""
        
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            
            for i in range(num_pages):
                try:
                    page = doc[i]
                    page_text = page.get_text()
                    
                    # Check if page is potentially scanned (very little text)
                    if self.use_ocr and len(page_text.strip()) < 100:
                        page_text = self._process_with_ocr(pdf_path, i)
                    
                    text += f"\n\n--- Page {i+1} ---\n\n"
                    text += page_text
                except Exception as e:
                    self.logger.error(f"Error extracting text from page {i+1}: {e}")
                    text += f"\n\n--- Page {i+1} (Error: {str(e)}) ---\n\n"
            
            doc.close()
        except Exception as e:
            self.logger.error(f"Error opening PDF with PyMuPDF: {e}")
            # Fallback to PyPDF2 if available
            if hasattr(self, 'pdf_reader') and self.pdf_reader != "pymupdf":
                self.logger.info("Falling back to PyPDF2")
                return self._extract_with_pypdf2(pdf_path)
            else:
                raise
        
        return self._clean_text(text)
    
    def _process_with_ocr(self, pdf_path: str, page_num: int) -> str:
        """
        Process a page with OCR if the page appears to be an image or has very little text.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to process
            
        Returns:
            Extracted text via OCR
        """
        if not self.use_ocr or not hasattr(self, 'ocr_engine'):
            return "[OCR not available for this page]"
        
        try:
            import fitz
            from PIL import Image
            import io
            
            # Open the PDF
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # Render page to an image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            
            # Process with OCR
            ocr_text = self.ocr_engine.image_to_string(img)
            
            doc.close()
            return ocr_text
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            return f"[OCR failed: {str(e)}]"
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing redundant whitespace, 
        fixing common PDF extraction issues, etc.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix hyphenated words split across lines
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Remove page markers from cleaned text (keep them in a more subtle form)
        text = re.sub(r'\n\n--- Page \d+ ---\n\n', r'\n\n', text)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections from the text based on common scientific paper structure.
        
        Args:
            text: Extracted text from the PDF
            
        Returns:
            Dictionary mapping section names to section content
        """
        # Common section headers in scientific papers
        section_patterns = [
            r'abstract', r'introduction', r'background', r'methods?', 
            r'materials\s+and\s+methods', r'results?', r'discussion', 
            r'conclusions?', r'references', r'acknowledgements?', 
            r'appendix', r'supplementary'
        ]
        
        # Combine patterns to find section headers
        section_pattern = r'(?i)^(' + '|'.join(section_patterns) + r')(?:\s|:|\.|\n)'
        
        # Find all section headers
        matches = list(re.finditer(section_pattern, text, re.MULTILINE))
        
        sections = {}
        for i, match in enumerate(matches):
            section_name = match.group(1).strip().lower()
            start_pos = match.start()
            
            # End of this section is the start of the next section or the end of the text
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
            
            # Extract section content
            section_content = text[start_pos:end_pos].strip()
            sections[section_name] = section_content
        
        # If no sections were found, return the entire text as "body"
        if not sections:
            sections["body"] = text
        
        return sections
    
    def chunk_text(self, text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for processing by LLMs.
        
        Args:
            text: Text to split into chunks
            chunk_size: Approximate size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        # If text is short enough, return as a single chunk
        if len(text) <= chunk_size:
            return [text]
        
        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for i, para in enumerate(paragraphs):
            # If adding this paragraph would exceed the limit, 
            # and we already have content in the current chunk
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Calculate how much of the previous chunk to include in the next one
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                # Otherwise, add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def extract_figures_tables(self, pdf_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract information about figures and tables in the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with lists of figure and table information
        """
        # This is a simplified implementation
        # A complete solution would need more sophisticated image and table extraction
        
        figures = []
        tables = []
        
        # Extract text to scan for figure and table references
        text = self.extract_text(pdf_path)
        
        # Find figure captions
        figure_matches = re.finditer(r'(?i)(?:figure|fig\.?)\s+(\d+[a-z]?)\.?\s+([^\n]+)', text)
        for match in figure_matches:
            figure_num = match.group(1)
            caption = match.group(2).strip()
            figures.append({
                "number": figure_num,
                "caption": caption,
                "page": self._estimate_page_number(match.start(), text)
            })
        
        # Find table captions
        table_matches = re.finditer(r'(?i)(?:table)\s+(\d+[a-z]?)\.?\s+([^\n]+)', text)
        for match in table_matches:
            table_num = match.group(1)
            caption = match.group(2).strip()
            tables.append({
                "number": table_num,
                "caption": caption,
                "page": self._estimate_page_number(match.start(), text)
            })
        
        return {
            "figures": figures,
            "tables": tables
        }
    
    def _estimate_page_number(self, position: int, text: str) -> int:
        """
        Estimate the page number for a given position in the text.
        
        Args:
            position: Character position in the text
            text: Full text of the document
            
        Returns:
            Estimated page number
        """
        # Find page markers before the position
        page_markers = list(re.finditer(r'--- Page (\d+) ---', text[:position]))
        
        if page_markers:
            # Return the last page marker before the position
            return int(page_markers[-1].group(1))
        else:
            return 1  # Default to page 1 if no marker found
    
    def get_metadata(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract metadata from the PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary of metadata fields
        """
        metadata = {
            "title": "",
            "authors": "",
            "year": "",
            "journal": "",
            "doi": ""
        }
        
        if self.pdf_reader == "pymupdf":
            try:
                import fitz
                doc = fitz.open(pdf_path)
                meta = doc.metadata
                
                if meta:
                    metadata["title"] = meta.get("title", "")
                    metadata["authors"] = meta.get("author", "")
                
                doc.close()
            except Exception as e:
                self.logger.error(f"Error extracting metadata with PyMuPDF: {e}")
        else:
            try:
                reader = self.pdf_reader.PdfReader(pdf_path)
                info = reader.metadata
                
                if info:
                    metadata["title"] = info.get("/Title", "")
                    metadata["authors"] = info.get("/Author", "")
            except Exception as e:
                self.logger.error(f"Error extracting metadata with PyPDF2: {e}")
        
        # If metadata extraction failed, try to extract from text
        if not metadata["title"] or not metadata["authors"]:
            text = self.extract_text(pdf_path)
            
            # Extract DOI if present
            doi_match = re.search(r'(?i)doi:?\s*(10\.\d+\/\S+)', text)
            if doi_match:
                metadata["doi"] = doi_match.group(1)
            
            # Try to extract year
            year_match = re.search(r'©?\s*(19|20)\d{2}', text)
            if year_match:
                metadata["year"] = year_match.group(0)
            
            # For other fields, would need more sophisticated extraction
        
        return metadata