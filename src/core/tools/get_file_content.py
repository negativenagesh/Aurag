import logging
from typing import Any, Optional, List
from uuid import UUID # Keep for potential future validation if needed, though not strictly used for ES query

# Import Elasticsearch directly if not available via context, or ensure context provides it.
# from elasticsearch import AsyncElasticsearch, NotFoundError # Example if direct import needed

logger = logging.getLogger(__name__)

# Assuming ELASTICSEARCH_INDEX_CHUNKS might be needed if not reliably from context
# However, best practice is to get it from context (agent -> retriever)
# from ..retrieval.rag_fusion_retriever import ELASTICSEARCH_INDEX_CHUNKS


class GetFileContentTool: # Removed (Tool) inheritance as we are not using its Pydantic model directly for return
    """
    A tool to fetch and concatenate all text chunks for a given document_id
    from Elasticsearch.
    """

    def __init__(self):
        self.name = "get_file_content"
        self.description = (
            "Fetches and concatenates all text chunks for a specified document_id from the knowledge base. "
            "Use this to retrieve the full available text content of a document when its ID is known."
        )
        self.parameters = { # OpenAPI schema for parameters
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The unique ID of the document to fetch all content for.",
                },
            },
            "required": ["document_id"],
        }
        self.context: Optional[Any] = None # To be set by the agent

    async def execute(
        self,
        document_id: str,
        # options: Optional[dict[str, Any]] = None, # Options not used in this direct ES query
        *args,
        **kwargs,
    ) -> str:
        """
        Fetches all text chunks for the given document_id from Elasticsearch
        and returns them as a single concatenated string.
        """
        if not self.context or not hasattr(self.context, "retriever") or \
           not hasattr(self.context.retriever, "es_client") or \
           not hasattr(self.context.retriever, "ELASTICSEARCH_INDEX_CHUNKS"):
            error_msg = "Tool context is not properly configured with Elasticsearch client or index information."
            logger.error(error_msg)
            return f"Error: {error_msg}"

        es_client = self.context.retriever.es_client
        index_name = self.context.retriever.ELASTICSEARCH_INDEX_CHUNKS

        if not es_client:
            error_msg = "Elasticsearch client is not available in the context."
            logger.error(error_msg)
            return f"Error: {error_msg}"
        if not document_id:
            return "Error: document_id parameter is required."

        logger.info(f"Executing GetFileContentTool for document_id: {document_id}")

        # Max chunks to retrieve. For very large documents, pagination (scroll/search_after) would be better.
        max_chunks_to_fetch = 1000 

        query_body = {
            "query": {
                "term": {
                    "metadata.doc_id.keyword": document_id # Use .keyword for exact match on keyword fields
                }
            },
            "sort": [
                {"metadata.page_number": "asc"},
                {"metadata.chunk_index_in_page": "asc"}
            ],
            "_source": ["chunk_text"],
            "size": max_chunks_to_fetch 
        }

        try:
            response = await es_client.search(
                index=index_name,
                body=query_body
            )
        except Exception as e:
            logger.error(f"Elasticsearch query failed for document_id '{document_id}': {e}", exc_info=True)
            return f"Error: Failed to query Elasticsearch for document_id '{document_id}'."

        hits = response.get("hits", {}).get("hits", [])

        if not hits:
            logger.warning(f"No content chunks found for document_id: {document_id} in index {index_name}")
            return f"No content found for document ID: {document_id}."
        
        if len(hits) == max_chunks_to_fetch:
            logger.warning(
                f"Retrieved the maximum configured number of chunks ({max_chunks_to_fetch}) for document_id '{document_id}'. "
                "The document might be larger, and some content might be truncated."
            )

        all_chunk_texts: List[str] = []
        for hit in hits:
            source = hit.get("_source", {})
            chunk_text = source.get("chunk_text")
            if chunk_text:
                all_chunk_texts.append(chunk_text)
        
        if not all_chunk_texts: # Should be redundant if hits is not empty and _source is requested
            logger.warning(f"Found hits for document_id '{document_id}', but no 'chunk_text' extracted.")
            return f"Content found for document ID: {document_id}, but text extraction failed."

        full_content = "\n\n---\n\n".join(all_chunk_texts) # Join chunks with a clear separator
        logger.info(f"Successfully retrieved and concatenated {len(all_chunk_texts)} chunks for document_id: {document_id}.")
        
        return full_content
