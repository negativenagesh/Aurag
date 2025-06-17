from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ChunkSearchResult(BaseModel):
    text: Optional[str] = None
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    file_name: Optional[str] = None
    doc_id: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index_in_page: Optional[int] = None

class KGEntity(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None

class KGRelationship(BaseModel):
    source_entity: Optional[str] = None
    target_entity: Optional[str] = None
    relation: Optional[str] = None
    relationship_description: Optional[str] = None
    relationship_weight: Optional[float] = None

class KGSearchResult(BaseModel):
    chunk_text: Optional[str] = None
    entities: List[KGEntity] = Field(default_factory=list)
    relationships: List[KGRelationship] = Field(default_factory=list)
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    file_name: Optional[str] = None
    doc_id: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index_in_page: Optional[int] = None

class AggregateSearchResult(BaseModel):
    query: Optional[str] = None
    chunk_search_results: List[ChunkSearchResult] = Field(default_factory=list)
    graph_search_results: List[KGSearchResult] = Field(default_factory=list)
    llm_formatted_context: Optional[str] = None # New field