# src/core/retrieval/rag_fusion_retriever.py
import logging
import os
import asyncio
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from elasticsearch import AsyncElasticsearch
from elasticsearch import exceptions as es_exceptions
from sentence_transformers.cross_encoder import CrossEncoder

from ...utils.logging_config import setup_logger

load_dotenv()
logger = setup_logger(__name__)

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini") # Corrected from gpt-4.1-mini to gpt-4o-mini as per chunking_pdf_processor
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_BASE_DIMENSION", 3072))
ELASTICSEARCH_INDEX_CHUNKS = os.getenv("ELASTICSEARCH_INDEX_CHUNKS", "r2rtest0")
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "mixedbread-ai/mxbai-rerank-large-v1")


# --- Initialize Clients ---
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found. OpenAI client will not be functional.")
    aclient_openai = None
else:
    aclient_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

try:
    es_client = AsyncElasticsearch(
        os.getenv("ELASTICSEARCH_URL", "https://my-elasticsearch-project-c44c4f.es.us-east-1.aws.elastic.cloud:443"),
        api_key=os.getenv("ELASTICSEARCH_API_KEY", "cWR0WFU1Y0I3YS1td2g2cWdpV186VTdnNFNVWmRmWVI5dnd3WEwwOWdMQQ=="),
        request_timeout=30
    )
    logger.info("AsyncElasticsearch client initialized for RAGFusionRetriever.")
except Exception as e:
    logger.error(f"Failed to initialize AsyncElasticsearch client for RAGFusionRetriever: {e}")
    es_client = None


class RAGFusionRetriever:
    def __init__(self):
        if aclient_openai is None:
            raise ValueError("AsyncOpenAI client is not initialized. Check OPENAI_API_KEY.")
        if es_client is None:
            raise ValueError("AsyncElasticsearch client is not initialized. Check Elasticsearch connection details.")
            
        self.aclient_openai = aclient_openai
        self.es_client = es_client
        self.rag_fusion_prompt_template = self._load_prompt_template("rag_fusion")
        
        self.reranker = None
        if RERANKER_MODEL_ID:
            try:
                self.reranker = CrossEncoder(RERANKER_MODEL_ID)
                logger.info(f"CrossEncoder reranker initialized with model: {RERANKER_MODEL_ID}")
            except Exception as e:
                logger.error(f"Failed to initialize CrossEncoder reranker with model {RERANKER_MODEL_ID}: {e}", exc_info=True)

    def _load_prompt_template(self, prompt_name: str) -> str:
        try:
            prompt_file_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.yaml"
            with open(prompt_file_path, 'r') as f:
                prompt_data = yaml.safe_load(f)
            
            if prompt_data and prompt_name in prompt_data and "template" in prompt_data[prompt_name]:
                template_content = prompt_data[prompt_name]["template"]
                logger.info(f"Successfully loaded prompt template for '{prompt_name}'.")
                return template_content
            else:
                logger.error(f"Prompt template for '{prompt_name}' not found or invalid in {prompt_file_path}.")
                raise ValueError(f"Invalid prompt structure for {prompt_name}")
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt '{prompt_name}': {e}")
            raise

    async def _generate_subqueries(self, original_query: str, num_subqueries: int = 2) -> List[str]:
        if not self.rag_fusion_prompt_template:
            logger.error("RAG Fusion prompt template not loaded.")
            return []

        formatted_prompt = self.rag_fusion_prompt_template.format(
            num_outputs=num_subqueries, message=original_query
        )
        
        logger.info(f"Generating {num_subqueries} subqueries for: '{original_query}' using model {OPENAI_CHAT_MODEL}")
        try:
            response = await self.aclient_openai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that generates multiple search queries based on a single user query."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.5,
                max_tokens=500 
            )
            llm_response_content = response.choices[0].message.content.strip()
            subqueries = [sq.strip() for sq in llm_response_content.split('\n\n') if sq.strip()]
            logger.info(f"Successfully generated {len(subqueries)} subqueries: {subqueries}")
            return subqueries[:num_subqueries]
        except Exception as e:
            logger.error(f"Error generating subqueries: {e}", exc_info=True)
            return []

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text.strip():
            logger.warning("Attempted to generate embedding for empty text.")
            return None
        logger.debug(f"Generating embedding for text (first 50 chars): '{text[:50]}...'")
        try:
            response = await self.aclient_openai.embeddings.create(
                input=[text],
                model=OPENAI_EMBEDDING_MODEL,
                dimensions=OPENAI_EMBEDDING_DIMENSIONS
            )
            logger.debug(f"Embedding generated successfully for text: '{text[:50]}...'")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text '{text[:50]}...': {e}", exc_info=True)
            return None

    async def _semantic_search_chunks(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        if not query_embedding:
            logger.warning("Semantic search skipped: No query embedding provided.")
            return []
        
        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": top_k * 10 
        }
        logger.debug(f"Performing semantic search for chunks with top_k={top_k}.")
        try:
            response = await self.es_client.search(
                index=ELASTICSEARCH_INDEX_CHUNKS,
                knn=knn_query,
                size=top_k,
                _source_includes=["chunk_text", "metadata.file_name", "metadata.doc_id", "metadata.page_number", "metadata.chunk_index_in_page"]
            )
            results = []
            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                metadata = source.get('metadata', {})
                results.append({
                    "text": source.get('chunk_text'),
                    "score": hit.get('_score'),
                    "file_name": metadata.get('file_name'),
                    "doc_id": metadata.get('doc_id'),
                    "page_number": metadata.get('page_number'),
                    "chunk_index_in_page": metadata.get('chunk_index_in_page')
                })
            logger.info(f"Semantic search found {len(results)} chunks.")
            return results
        except es_exceptions as e: # Changed exception type
            logger.error(f"Elasticsearch semantic search error: {e}", exc_info=True)
            return []

    async def _structured_kg_search(self, subquery_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not subquery_text.strip():
            logger.warning("Structured KG search skipped: No subquery text provided.")
            return []

        entity_fields = ["metadata.entities.name", "metadata.entities.type", "metadata.entities.description"]
        relationship_fields = [
            "metadata.relationships.source_entity", "metadata.relationships.target_entity",
            "metadata.relationships.relation", "metadata.relationships.relationship_description"
        ]
        query_body = {
            "query": {
                "bool": {
                    "should": [
                        {"nested": {"path": "metadata.entities", "query": {"multi_match": {"query": subquery_text, "fields": entity_fields, "fuzziness": "AUTO"}}}},
                        {"nested": {"path": "metadata.relationships", "query": {"multi_match": {"query": subquery_text, "fields": relationship_fields, "fuzziness": "AUTO"}}}}
                    ],
                    "minimum_should_match": 1 
                }
            },
            "size": top_k,
            "_source_includes": ["chunk_text", "metadata"] # Ensure metadata is included
        }
        logger.debug(f"Performing structured KG search for subquery '{subquery_text[:50]}...' with top_k={top_k}.")
        try:
            response = await self.es_client.search(index=ELASTICSEARCH_INDEX_CHUNKS, body=query_body)
            results = []
            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                metadata = source.get('metadata', {})
                results.append({
                    "chunk_text": source.get('chunk_text'), # Text of the chunk containing the KG
                    "entities": metadata.get('entities', []), # KG entities from metadata
                    "relationships": metadata.get('relationships', []), # KG relationships from metadata
                    "score": hit.get('_score'),
                    "file_name": metadata.get('file_name'),
                    "doc_id": metadata.get('doc_id'),
                    "page_number": metadata.get('page_number'),
                    "chunk_index_in_page": metadata.get('chunk_index_in_page')
                })
            logger.info(f"Structured KG search found {len(results)} documents for subquery '{subquery_text[:50]}...'.")
            return results
        except es_exceptions as e: # Changed exception type
            logger.error(f"Elasticsearch structured KG search error for subquery '{subquery_text[:50]}...': {e}", exc_info=True)
            return []

    async def _rerank_documents(self, query: str, documents: List[Dict[str, Any]], doc_type: str) -> List[Dict[str, Any]]:
        if not documents:
            logger.debug(f"No {doc_type} documents to rerank for query: '{query[:50]}...'")
            return []
        if not self.reranker:
            logger.warning(f"Reranker not initialized. Skipping reranking for {doc_type} documents.")
            return documents

        logger.info(f"Reranking {len(documents)} {doc_type} documents for query: '{query[:50]}...'")
        
        if doc_type == "chunk":
            pairs = [(query, doc.get("text", "")) for doc in documents]
            text_key_for_logging = "text"
        elif doc_type == "kg": 
            pairs = [(query, doc.get("chunk_text", "")) for doc in documents]
            text_key_for_logging = "chunk_text"
        else:
            logger.warning(f"Unknown document type '{doc_type}' for reranking. Skipping.")
            return documents

        try:
            scores = await asyncio.to_thread(self.reranker.predict, pairs)
            logger.debug(f"Successfully got scores for {len(pairs)} pairs for {doc_type} reranking.")
        except Exception as e:
            logger.error(f"Error during reranking {doc_type} documents with CrossEncoder: {e}", exc_info=True)
            return documents 

        docs_with_scores = [{'doc': doc, 'score': scores[i]} for i, doc in enumerate(documents)]
        docs_with_scores.sort(key=lambda x: x['score'], reverse=True)

        reranked_docs = []
        for item in docs_with_scores:
            doc = item['doc']
            doc['rerank_score'] = float(item['score']) 
            reranked_docs.append(doc)
            logger.debug(f"Reranked {doc_type} doc (source: {doc.get('file_name', 'N/A')}, page: {doc.get('page_number', 'N/A')}, content: {doc.get(text_key_for_logging, '')[:30]}...) new score: {doc['rerank_score']:.4f}")

        logger.info(f"Successfully reranked {len(reranked_docs)} {doc_type} documents.")
        return reranked_docs

    async def _perform_semantic_search_for_subquery(self, subquery_text: str, top_k: int) -> List[Dict[str, Any]]:
        logger.debug(f"Performing semantic search for subquery: '{subquery_text}'")
        embedding = await self._generate_embedding(subquery_text)
        if not embedding:
            logger.warning(f"Could not generate embedding for subquery: '{subquery_text}'. Semantic search will yield no results.")
            return []
        return await self._semantic_search_chunks(embedding, top_k)

    async def _perform_kg_search_for_subquery(self, subquery_text: str, top_k: int) -> List[Dict[str, Any]]:
        logger.debug(f"Performing KG search for subquery: '{subquery_text}'")
        return await self._structured_kg_search(subquery_text, top_k)

    async def search(
        self, 
        user_query: str, 
        num_subqueries: int = 2, 
        top_k_chunks: int = 10, 
        top_k_kg: int = 10
    ) -> Dict[str, Any]:
        logger.info(f"Starting RAG Fusion search for user query: '{user_query}'")
        subqueries = await self._generate_subqueries(user_query, num_subqueries)
        if not subqueries:
            logger.warning("No subqueries generated from user query. Returning empty results.")
            return {"original_query": user_query, "sub_queries_results": []}

        processed_subquery_results = []

        for sq_text in subqueries:
            logger.info(f"Processing subquery: '{sq_text}'")
            retrieved_chunks: List[Dict[str, Any]] = []
            retrieved_kg_evidence: List[Dict[str, Any]] = []

            logger.debug(f"Initiating concurrent retrieval for subquery: '{sq_text}'")
            try:
                retrieval_results = await asyncio.gather(
                    self._perform_semantic_search_for_subquery(sq_text, top_k_chunks),
                    self._perform_kg_search_for_subquery(sq_text, top_k_kg),
                    return_exceptions=True
                )
                retrieved_chunks = retrieval_results[0] if not isinstance(retrieval_results[0], Exception) and retrieval_results[0] is not None else []
                if isinstance(retrieval_results[0], Exception):
                    logger.error(f"Semantic search task failed for subquery '{sq_text}': {retrieval_results[0]}")
                
                retrieved_kg_evidence = retrieval_results[1] if not isinstance(retrieval_results[1], Exception) and retrieval_results[1] is not None else []
                if isinstance(retrieval_results[1], Exception):
                    logger.error(f"KG search task failed for subquery '{sq_text}': {retrieval_results[1]}")
                
                logger.info(f"Initial retrieval for subquery '{sq_text}': {len(retrieved_chunks)} chunks, {len(retrieved_kg_evidence)} KG evidences.")
            except Exception as e:
                logger.error(f"Unexpected error during concurrent retrieval for subquery '{sq_text}': {e}", exc_info=True)
            
            reranked_chunks_for_sq = retrieved_chunks 
            
            # KG evidence will not be reranked.
            final_kg_evidence_for_sq = retrieved_kg_evidence 

            if self.reranker and retrieved_chunks: 
                logger.debug(f"Initiating reranking for chunks for subquery: '{sq_text}'")
                try:
                    reranked_chunks_for_sq = await self._rerank_documents(sq_text, retrieved_chunks, "chunk")
                    logger.info(f"Chunk reranking for subquery '{sq_text}' completed: {len(reranked_chunks_for_sq)} chunks.")
                except Exception as e:
                    logger.error(f"Chunk reranking task failed for subquery '{sq_text}': {e}. Using original chunks.", exc_info=True)
                    reranked_chunks_for_sq = retrieved_chunks 
            elif not self.reranker:
                logger.warning("Reranker not available. Using original retrieved chunks without reranking.")
            elif not retrieved_chunks:
                 logger.debug(f"No chunks to rerank for subquery '{sq_text}'.")
            
            if retrieved_kg_evidence:
                logger.info(f"KG evidence for subquery '{sq_text}' will not be reranked. Using {len(final_kg_evidence_for_sq)} originally retrieved KG evidences.")
            else:
                logger.debug(f"No KG evidence to process for subquery '{sq_text}'.")

            processed_subquery_results.append({
                "sub_query_text": sq_text,
                "reranked_chunks": reranked_chunks_for_sq, 
                "retrieved_kg_data": final_kg_evidence_for_sq # Changed key here
            })

        final_results = {"original_query": user_query, "sub_queries_results": processed_subquery_results}
        logger.info(f"RAG Fusion search fully completed for user query: '{user_query}'.")
        return final_results

async def main_example_search():
    # Ensure root logger is configured if no handlers are present
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Set specific logger level for this module
    logging.getLogger("src.core.retrieval.rag_fusion_retriever").setLevel(logging.DEBUG)
    
    if not aclient_openai or not es_client:
        logger.error("OpenAI or Elasticsearch client not initialized. Cannot run example.")
        return

    retriever = RAGFusionRetriever()
    user_query_input = input("Enter your search query: ").strip()
    if not user_query_input:
        logger.warning("No query entered. Exiting.")
    else:
        logger.info(f"\n--- Running RAG Fusion Search for: '{user_query_input}' ---")
        search_results = await retriever.search(
            user_query=user_query_input, num_subqueries=2, top_k_chunks=3, top_k_kg=3
        )
        import json
        print("\n--- Search Results (RAG Fusion: Chunks Reranked, KG Raw) ---") # Updated print message
        print(json.dumps(search_results, indent=2, default=str)) 
    
    if es_client:
        await es_client.close()
        logger.info("Elasticsearch client closed.")
    if aclient_openai and hasattr(aclient_openai, "aclose"): 
        await aclient_openai.aclose()
        logger.info("OpenAI client closed.")

if __name__ == "__main__":
    # This basicConfig will only take effect if no handlers are already configured on the root logger.
    # This is useful for direct script execution.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main_example_search())