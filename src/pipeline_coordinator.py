# src/pipeline_coordinator.py
import asyncio
import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional,List

from dotenv import load_dotenv

try:
    from .core.ingestion.chunking_pdf_processor import (
        ChunkingEmbeddingPDFProcessor,
        ensure_es_index_exists,
        CHUNKED_PDF_MAPPINGS,
        ELASTICSEARCH_INDEX_CHUNKS as INGESTION_ES_INDEX,
        es_client as ingestion_es_client,
        aclient_openai as ingestion_openai_client
    )
    from .core.retrieval.rag_fusion_retriever import (
        RAGFusionRetriever,
        ELASTICSEARCH_INDEX_CHUNKS as RETRIEVAL_ES_INDEX,
        es_client as retrieval_es_client,
        aclient_openai as retrieval_openai_client
    )
    from .utils.logging_config import setup_logger
except ImportError as e:
    print(f"ImportError: {e}. Please ensure you are running this script as a module from the project root, e.g., python -m src.pipeline_coordinator")
    print("Also, ensure __init__.py files exist in 'src', 'src/core', 'src/core/ingestion', 'src/core/retrieval', and 'src/utils'.")
    exit(1)


load_dotenv()
logger = setup_logger(__name__)

if INGESTION_ES_INDEX != RETRIEVAL_ES_INDEX:
    logger.warning(
        f"Mismatch in Elasticsearch index names: "
        f"Ingestion uses '{INGESTION_ES_INDEX}', Retrieval uses '{RETRIEVAL_ES_INDEX}'. "
        f"This will likely lead to issues. Using '{INGESTION_ES_INDEX}' for coordination."
    )
COORDINATOR_ES_INDEX = INGESTION_ES_INDEX

if ingestion_es_client is None:
    logger.error("Elasticsearch client from ingestion module is not initialized!")
    if retrieval_es_client:
        logger.warning("Falling back to Elasticsearch client from retrieval module.")
        es_coordinator_client = retrieval_es_client
    else:
        es_coordinator_client = None
else:
    es_coordinator_client = ingestion_es_client

# Determine a single OpenAI client instance to manage/close if they are shared
# For now, we assume they might be the same instance due to module-level initialization
# or that closing one (if they are the same) is sufficient.
# If they are distinct, both would ideally be closed if they were created by the coordinator.
# Since they are module-level, we'll pick one to try closing.
coordinator_openai_client = ingestion_openai_client if ingestion_openai_client else retrieval_openai_client

if coordinator_openai_client is None and os.getenv("OPENAI_API_KEY"):
    logger.error("OpenAI client is not initialized in either ingestion or retrieval modules, despite API key being present!")


class PipelineCoordinator:
    def __init__(self):
        if es_coordinator_client is None:
            raise ConnectionError("Elasticsearch client is not available for the coordinator.")
        
        self.ingestion_processor = ChunkingEmbeddingPDFProcessor()
        self.retriever = RAGFusionRetriever()
        logger.info("PipelineCoordinator initialized with Ingestion Processor and RAG Fusion Retriever.")

    async def run_ingestion(self, pdf_file_path: str, document_id: str, document_summary: str):
        logger.info(f"Starting ingestion for PDF: {pdf_file_path}, Doc ID: {document_id}")

        if not await ensure_es_index_exists(es_coordinator_client, COORDINATOR_ES_INDEX, CHUNKED_PDF_MAPPINGS):
            logger.error(f"Failed to ensure Elasticsearch index '{COORDINATOR_ES_INDEX}' exists. Aborting ingestion.")
            return False

        actions_for_es = []
        try:
            with open(pdf_file_path, "rb") as f:
                pdf_bytes_data = f.read()

            async for action in self.ingestion_processor.process_pdf(
                data=pdf_bytes_data,
                file_name=Path(pdf_file_path).name,
                doc_id=document_id,
                document_summary=document_summary
            ):
                if action:
                    actions_for_es.append(action)
            
            if actions_for_es:
                logger.info(f"Collected {len(actions_for_es)} actions for bulk ingestion into '{COORDINATOR_ES_INDEX}'.")
                from elasticsearch.helpers import async_bulk
                successes, errors = await async_bulk(es_coordinator_client, actions_for_es, raise_on_error=False, raise_on_exception=False)
                logger.info(f"Elasticsearch bulk ingestion: {successes} successes.")
                if errors:
                    logger.error(f"Elasticsearch bulk ingestion errors ({len(errors)}):")
                    for i, err_info in enumerate(errors):
                        error_details_dict = err_info.get('index', err_info.get('create', err_info.get('update', err_info.get('delete', {}))))
                        status = error_details_dict.get('status', 'N/A')
                        error_type = error_details_dict.get('error', {}).get('type', 'N/A')
                        error_reason = error_details_dict.get('error', {}).get('reason', 'N/A')
                        doc_id_errored = error_details_dict.get('_id', 'N/A')
                        logger.error(f"Error {i+1}: Doc ID '{doc_id_errored}', Status {status}, Type '{error_type}', Reason: {error_reason}")
                logger.info(f"Ingestion completed for {Path(pdf_file_path).name}.")
                return True
            else:
                logger.warning(f"No processable chunks found or generated for {Path(pdf_file_path).name}.")
                return False

        except FileNotFoundError:
            logger.error(f"PDF file not found at: {pdf_file_path}")
            return False
        except Exception as e:
            logger.error(f"An error occurred during ingestion for {pdf_file_path}: {e}", exc_info=True)
            return False

    async def run_retrieval(self, user_query: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Starting retrieval for query: '{user_query}'")
        try:
            search_results = await self.retriever.search(
                user_query=user_query,
                num_subqueries=2,
                top_k_chunks=3,
                top_k_kg=3
            )
            logger.info(f"Retrieval completed for query: '{user_query}'.")
            return search_results
        except Exception as e:
            logger.error(f"An error occurred during retrieval for query '{user_query}': {e}", exc_info=True)
            return None

async def run_pipeline():
    if not logging.getLogger(__name__).hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("src.pipeline_coordinator").setLevel(logging.DEBUG)

    if not es_coordinator_client:
        logger.critical("Elasticsearch client is not initialized. Cannot run pipeline.")
        return
    if coordinator_openai_client is None and os.getenv("OPENAI_API_KEY"):
         logger.critical("OpenAI client is not initialized in processors. Cannot run pipeline effectively.")
    elif not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is not set. LLM-dependent features will be skipped or fail.")

    coordinator = PipelineCoordinator()

    run_ingestion_q = input("Do you want to run the ingestion process for a PDF? (yes/no): ").strip().lower()
    if run_ingestion_q == 'yes':
        pdf_path_input = input("Enter the full path to your PDF file: ").strip()
        if not pdf_path_input:
            logger.error("No PDF path provided. Skipping ingestion.")
        else:
            pdf_file_path = Path(pdf_path_input)
            if not pdf_file_path.is_file():
                logger.error(f"File not found: {pdf_file_path}. Skipping ingestion.")
            else:
                doc_id_input = input(f"Enter a unique document ID for '{pdf_file_path.name}': ").strip()
                if not doc_id_input:
                    doc_id_input = pdf_file_path.stem.replace(" ", "_").lower() + "_id"
                    logger.info(f"Using generated document ID: {doc_id_input}")
                
                doc_summary_input = input(f"Enter a brief summary for '{pdf_file_path.name}': ").strip()
                if not doc_summary_input:
                    doc_summary_input = f"Document: {pdf_file_path.name}"
                    logger.info(f"Using placeholder summary: '{doc_summary_input}'")

                await coordinator.run_ingestion(str(pdf_file_path), doc_id_input, doc_summary_input)
    else:
        logger.info("Skipping ingestion process.")

    run_retrieval_q = input("\nDo you want to run the retrieval (search) process? (yes/no): ").strip().lower()
    if run_retrieval_q == 'yes':
        user_query = input("Enter your search query: ").strip()
        if not user_query:
            logger.warning("No search query entered. Skipping retrieval.")
        else:
            results = await coordinator.run_retrieval(user_query)
            if results:
                print("\n--- Search Results ---")
                print(json.dumps(results, indent=2, default=str))
            else:
                print("No results returned or an error occurred during search.")
    else:
        logger.info("Skipping retrieval process.")

    if es_coordinator_client:
        try:
            await es_coordinator_client.close()
            logger.info("Elasticsearch client closed.")
        except Exception as e:
            logger.warning(f"Error closing Elasticsearch client: {e}")
    
    if coordinator_openai_client and hasattr(coordinator_openai_client, "aclose"):
        try:
            await coordinator_openai_client.aclose()
            logger.info("OpenAI client closed.")
        except Exception as e:
            logger.warning(f"Error closing OpenAI client: {e}")
    elif coordinator_openai_client:
        logger.info("OpenAI client instance does not have 'aclose' method; manual closing might not be needed or is handled differently.")


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(run_pipeline())