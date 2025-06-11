from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class IngestionConfig:
    """Configuration for ingestion process."""
    
    vlm: Optional[str] = None
    app: Any = None
    vlm_batch_size: int = 5
    vlm_max_tokens_to_sample: int = 1024
    max_concurrent_vlm_tasks: int = 5


class DatabaseProvider:
    """Interface for database operations."""
    
    class PromptsHandler:
        async def get_cached_prompt(self, prompt_name: str) -> str:
            raise NotImplementedError
    
    @property
    def prompts_handler(self) -> PromptsHandler:
        return self.PromptsHandler()


class CompletionProvider:
    """Interface for LLM completion operations."""
    
    async def aget_completion(self, messages: List[Dict], generation_config: Any, 
                             apply_timeout: bool = False, tools: Optional[List] = None, 
                             tool_choice: Optional[Dict] = None) -> Any:
        raise NotImplementedError


class OCRProvider:
    """Interface for OCR operations."""
    
    async def process_pdf(self, file_path: Optional[str] = None, 
                         file_content: Optional[bytes] = None) -> Any:
        raise NotImplementedError