import logging
import string
import unicodedata
from io import BytesIO
from typing import AsyncGenerator, Dict, Any

from pypdf import PdfReader

from ..base.parsers.base_parser import AsyncParser
from ..base.providers import CompletionProvider, DatabaseProvider, IngestionConfig

logger = logging.getLogger(__name__)


class BasicPDFParser(AsyncParser[str | bytes]):
    """A basic parser for PDF data using direct text extraction.
    
    This parser handles uploaded PDFs and extracts text without requiring
    external OCR or VLM services.
    """

    def __init__(
        self,
        config: IngestionConfig = None,
        database_provider: DatabaseProvider = None,
        llm_provider: CompletionProvider = None,
    ):
        self.database_provider = database_provider
        self.llm_provider = llm_provider
        self.config = config
        self.PdfReader = PdfReader

    async def ingest(
        self, data: str | bytes, **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Ingest PDF data and yield text from each page.
        
        Args:
            data: Either a file path (str) or binary PDF data (bytes)
            **kwargs: Additional parameters
            
        Yields:
            Dictionary with page content and page number
        """
        logger.info("Starting PDF ingestion using BasicPDFParser")
        
        try:
            if isinstance(data, str):
                # Handle file path input
                with open(data, 'rb') as file:
                    pdf_bytes = file.read()
                pdf = self.PdfReader(BytesIO(pdf_bytes))
            else:
                # Handle binary data input
                pdf = self.PdfReader(BytesIO(data))
                
            logger.info(f"PDF loaded with {len(pdf.pages)} pages")
                
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text is not None:
                    # Clean the text by removing non-printable characters
                    page_text = "".join(
                        filter(
                            lambda x: (
                                unicodedata.category(x)
                                in [
                                    "Ll", "Lu", "Lt", "Lm", "Lo", "Nl", "No",  # Letters and numbers
                                    "Pd", "Pc", "Pe", "Pf", "Pi", "Po", "Ps",  # Punctuation
                                    "Sc", "Sm", "Sk",  # Symbols
                                    "Zs",  # Spaces
                                ]
                                or "\u4e00" <= x <= "\u9fff"  # Chinese characters
                                or "\u0600" <= x <= "\u06ff"  # Arabic characters
                                or "\u0400" <= x <= "\u04ff"  # Cyrillic letters
                                or "\u0370" <= x <= "\u03ff"  # Greek letters
                                or "\u0e00" <= x <= "\u0e7f"  # Thai
                                or "\u3040" <= x <= "\u309f"  # Japanese Hiragana
                                or "\u30a0" <= x <= "\u30ff"  # Katakana
                                or "\uff00" <= x <= "\uffef"  # Halfwidth and Fullwidth Forms
                                or x in string.printable
                            ),
                            page_text,
                        )
                    )
                    
                    logger.debug(f"Processed page {i} with {len(page_text)} characters")
                    yield {
                        "content": page_text,
                        "page_number": i
                    }
                else:
                    logger.warning(f"No text content extracted from page {i}")
                    yield {
                        "content": "",
                        "page_number": i
                    }
        
        except Exception as e:
            logger.error(f"Error in BasicPDFParser: {str(e)}")
            raise