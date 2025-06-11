# src/core/base/abstractions.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    model: str
    stream: bool = False
    max_tokens_to_sample: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None