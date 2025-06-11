# src/core/base/parsers/base_parser.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generic, TypeVar, Any # Add Any here

T = TypeVar("T")


class AsyncParser(Generic[T], ABC):
    """Base class for all async parsers."""
    
    @abstractmethod
    async def ingest(self, data: T, **kwargs) -> AsyncGenerator[Any, None]:
        """Ingest data and yield processed chunks."""
        pass