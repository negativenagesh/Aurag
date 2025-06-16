import logging
import os
import asyncio
import argparse
import sys
import json # Added for debug logging
from typing import Any, Optional, List
from uuid import UUID # Keep for potential future validation if needed, though not strictly used for ES query

from elasticsearch import AsyncElasticsearch, NotFoundError
from dotenv import load_dotenv

# We'll use this value directly to avoid circular imports
ELASTICSEARCH_INDEX_CHUNKS = os.getenv("ELASTICSEARCH_INDEX_CHUNKS", "r2rtest00")

# Logger for this module
logger = logging.getLogger(__name__)


class GetFileContentTool:
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
        *args,
        **kwargs,
    ) -> str:
        """
        Fetches all text chunks for the given document_id from Elasticsearch
        and returns them as a single concatenated string.
        """
        if not self.context or not hasattr(self.context, "es_client"):
            error_msg = "Tool context is not properly configured with Elasticsearch client."
            logger.error(error_msg)
            return f"Error: {error_msg}"

        es_client = self.context.es_client
        index_name = getattr(self.context, "index_name", ELASTICSEARCH_INDEX_CHUNKS)

        if not es_client:
            error_msg = "Elasticsearch client is not available in the context."
            logger.error(error_msg)
            return f"Error: {error_msg}"
        if not document_id:
            logger.warning("document_id parameter is required for GetFileContentTool.execute")
            return "Error: document_id parameter is required."

        logger.info(f"Executing GetFileContentTool for document_id: {document_id} on index {index_name}")

        max_chunks_to_fetch = 1000 

        query_body = {
            "query": {
                "term": {
                    # Assuming metadata.doc_id is mapped as a keyword directly.
                    # If it's a text field with a .keyword subfield, use "metadata.doc_id.keyword"
                    "metadata.doc_id": document_id
                }
            },
            "sort": [
                {"metadata.page_number": "asc"},
                {"metadata.chunk_index_in_page": "asc"}
            ],
            "_source": ["chunk_text"],
            "size": max_chunks_to_fetch 
        }
        
        logger.debug(f"Elasticsearch query body for document_id '{document_id}': {json.dumps(query_body, indent=2)}")

        try:
            response = await es_client.search(
                index=index_name,
                body=query_body
            )
            # Safely access response.body if it exists, otherwise use the response dict itself
            response_body_for_log = response.body if hasattr(response, 'body') and response.body else response
            logger.debug(f"Elasticsearch raw response for document_id '{document_id}': {json.dumps(response_body_for_log, indent=2, default=str)}")

        except Exception as e:
            logger.error(f"Elasticsearch query failed for document_id '{document_id}': {e}", exc_info=True)
            return f"Error: Failed to query Elasticsearch for document_id '{document_id}'."

        hits_data = response.get("hits", {})
        total_hits = hits_data.get("total", {}).get("value", 0)
        actual_hits_array = hits_data.get("hits", [])
        
        logger.info(f"Elasticsearch query for document_id '{document_id}' reported {total_hits} total matching documents. Retrieved {len(actual_hits_array)} hits for processing.")

        if not actual_hits_array:
            logger.warning(f"No content chunks found for document_id: {document_id} in index {index_name} (Total hits: {total_hits}). Check if the document_id is correct and if data is indexed properly under this ID.")
            return f"No content found for document ID: {document_id}."
        
        if len(actual_hits_array) == max_chunks_to_fetch:
            logger.warning(
                f"Retrieved the maximum configured number of chunks ({max_chunks_to_fetch}) for document_id '{document_id}'. "
                "The document might be larger, and some content might be truncated."
            )

        all_chunk_texts: List[str] = []
        for hit_idx, hit in enumerate(actual_hits_array):
            source = hit.get("_source", {})
            chunk_text = source.get("chunk_text")
            if chunk_text:
                all_chunk_texts.append(chunk_text)
            else:
                logger.warning(f"Hit {hit_idx} for document_id '{document_id}' (ES_ID: {hit.get('_id')}) is missing 'chunk_text' in _source.")
        
        if not all_chunk_texts:
            logger.warning(f"Found {len(actual_hits_array)} hits for document_id '{document_id}', but no 'chunk_text' could be extracted from any of them.")
            return f"Content found for document ID: {document_id}, but text extraction failed from all chunks."

        full_content = "\n\n---\n\n".join(all_chunk_texts)
        logger.info(f"Successfully retrieved and concatenated {len(all_chunk_texts)} chunks for document_id: {document_id}.")
        
        return full_content


async def list_available_documents(es_client: AsyncElasticsearch, index_name: str) -> List[str]:
    """List available document IDs in the index to help the user choose."""
    logger.info(f"Listing available documents from index: {index_name}")
    doc_ids: List[str] = []
    try:
        query = {
            "size": 0,
            "aggs": {
                "document_ids": {
                    "terms": {
                        # This field should match how doc_id is queryable for terms.
                        # If metadata.doc_id is keyword, this is fine.
                        # If it's text with .keyword subfield, metadata.doc_id.keyword is better here.
                        "field": "metadata.doc_id.keyword", # Using .keyword for aggregation is common
                        "size": 100,
                        "order": {"_key": "asc"}
                    }
                }
            }
        }
        logger.debug(f"List documents aggregation query: {json.dumps(query, indent=2)}")
        response = await es_client.search(index=index_name, body=query)
        response_body_for_log = response.body if hasattr(response, 'body') and response.body else response
        logger.debug(f"List documents aggregation response: {json.dumps(response_body_for_log, indent=2, default=str)}")
        
        doc_buckets = response.get("aggregations", {}).get("document_ids", {}).get("buckets", [])
        
        if not doc_buckets:
            print("No documents found in the index.")
            return []
            
        doc_ids = [bucket["key"] for bucket in doc_buckets]
        print(f"\nFound {len(doc_ids)} unique document ID(s) in the index '{index_name}'.")
        
        doc_details = []
        for doc_id_val in doc_ids:
            try:
                metadata_query = {
                    "query": {"term": {"metadata.doc_id.keyword": doc_id_val}}, # Consistent with agg
                    "_source": ["metadata.file_name"],
                    "size": 1
                }
                meta_response = await es_client.search(index=index_name, body=metadata_query)
                if meta_response.get("hits", {}).get("hits"):
                    file_name = meta_response["hits"]["hits"][0].get("_source", {}).get("metadata", {}).get("file_name", "Unknown File")
                    doc_details.append({"id": doc_id_val, "file_name": file_name})
                else:
                    doc_details.append({"id": doc_id_val, "file_name": "N/A (metadata not found)"})
            except Exception as e_meta:
                logger.warning(f"Could not fetch metadata for doc_id {doc_id_val}: {e_meta}")
                doc_details.append({"id": doc_id_val, "file_name": "N/A (error fetching metadata)"})
        
        print("\nAvailable documents (ID - File Name):")
        for i, detail in enumerate(doc_details, 1):
            print(f"{i}. {detail['id']} - {detail['file_name']}")
        return doc_ids # Return only the IDs for selection logic
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        print(f"Error listing documents: {e}")
        return []


async def main():
    load_dotenv()
    
    # Configure module-specific logger
    script_logger = logging.getLogger(__name__) # Gets the logger for this file (__main__ if run directly)
    script_logger.setLevel(logging.DEBUG) # Set to DEBUG to see all logs from this file

    # Ensure a handler is present if running as a script, to see output
    if not script_logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        script_logger.addHandler(handler)
        script_logger.propagate = False # Avoid double logging if root logger also has handlers

    # Configure Elasticsearch client logger to be less verbose unless needed
    logging.getLogger("elastic_transport.transport").setLevel(logging.INFO)


    parser = argparse.ArgumentParser(description="Retrieve document content from Elasticsearch")
    parser.add_argument("--document_id", "-d", type=str, help="Document ID to retrieve")
    parser.add_argument("--list", "-l", action="store_true", help="List available document IDs")
    args = parser.parse_args()
    
    es_client = None
    try:
        es_url = os.getenv("ELASTICSEARCH_URL","https://my-elasticsearch-project-c44c4f.es.us-east-1.aws.elastic.cloud:443")
        es_api_key = os.getenv("ELASTICSEARCH_API_KEY","ZFFvV2Q1Y0J5dkZQeWd2c3lhc086cXdhT3RYYUtlTXQ3N1ZSd3NWcmxQUQ==")
        if not es_url or not es_api_key:
            script_logger.error("ELASTICSEARCH_URL or ELASTICSEARCH_API_KEY environment variables not set.")
            print("Error: Elasticsearch connection details not found in environment variables.")
            return

        es_client = AsyncElasticsearch(es_url, api_key=es_api_key, request_timeout=30)
        await es_client.ping() # Verify connection
        script_logger.info(f"AsyncElasticsearch client initialized and connected to {es_url.split('@')[-1] if '@' in es_url else es_url}.")
    except Exception as e:
        script_logger.error(f"Failed to initialize or connect AsyncElasticsearch client: {e}", exc_info=True)
        print(f"Error: Could not connect to Elasticsearch: {e}")
        return
    
    context = type('SimpleContext', (), {'es_client': es_client, 'index_name': ELASTICSEARCH_INDEX_CHUNKS})
    tool = GetFileContentTool()
    tool.context = context
    
    document_id_to_fetch = None
    try:
        if args.list:
            listed_doc_ids = await list_available_documents(es_client, ELASTICSEARCH_INDEX_CHUNKS)
            if not listed_doc_ids:
                return
            
            try:
                selection = input("\nEnter document number from the list or full ID (or press Enter to exit): ").strip()
                if not selection:
                    print("Exiting.")
                    return
                
                selected_index = int(selection) - 1
                if 0 <= selected_index < len(listed_doc_ids):
                    document_id_to_fetch = listed_doc_ids[selected_index]
                else:
                    print("Invalid selection number. Exiting.")
                    return
            except ValueError: # User entered an ID directly
                document_id_to_fetch = selection
        else:
            document_id_to_fetch = args.document_id
            if not document_id_to_fetch:
                document_id_to_fetch = input("Enter document ID to retrieve: ").strip()
                if not document_id_to_fetch:
                    print("No document ID provided. Exiting.")
                    return
        
        script_logger.info(f"Attempting to retrieve content for document ID: {document_id_to_fetch}")
        result = await tool.execute(document_id_to_fetch)
        
        if result.startswith("Error:") or result.startswith("No content found"):
            print(f"\nResult: {result}")
        else:
            print("\n--- DOCUMENT CONTENT ---")
            print(result)
            print("------------------------\n")
            
            save_to_file = input("Save content to file? (y/n): ").strip().lower()
            if save_to_file in ('y', 'yes'):
                safe_doc_id = document_id_to_fetch.replace('/', '_').replace('\\', '_') # Basic sanitization
                file_name = f"{safe_doc_id}_content.txt"
                try:
                    with open(file_name, 'w', encoding='utf-8') as f:
                        f.write(result)
                    print(f"Content saved to {file_name}")
                except IOError as e_io:
                    print(f"Error saving file {file_name}: {e_io}")
                    
    except Exception as e_main:
        script_logger.error(f"An error occurred in main execution: {e_main}", exc_info=True)
        print(f"An unexpected error occurred: {e_main}")
    finally:
        if es_client:
            await es_client.close()
            script_logger.info("Elasticsearch client closed.")


if __name__ == "__main__":
    asyncio.run(main())
