import logging
import os
import asyncio
import yaml
import json # Added for logging query bodies
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import TransportError
from sentence_transformers.cross_encoder import CrossEncoder

from ...utils.logging_config import setup_logger

load_dotenv()
logger = setup_logger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_BASE_DIMENSION", 3072))
ELASTICSEARCH_INDEX_CHUNKS = os.getenv("ELASTICSEARCH_INDEX_CHUNKS", "messbill")
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "mixedbread-ai/mxbai-rerank-large-v1")
ELASTICSEARCH_API_KEY=os.getenv("ELASTICSEARCH_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found. OpenAI client will not be functional.")
    aclient_openai = None
else:
    aclient_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

try:
    es_client = AsyncElasticsearch(
        os.getenv("ELASTICSEARCH_URL"),
        api_key=os.getenv("ELASTICSEARCH_API_KEY"),
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
            logger.warning("Semantic search for chunks skipped: No query embedding provided.")
            return []

        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": top_k * 10
        }
        logger.debug(f"Performing semantic search for chunks with top_k={top_k}. Query: {json.dumps(knn_query)}")
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
            logger.info(f"Semantic chunk search found {len(results)} documents.")
            return results
        except TransportError as e:
            logger.error(f"Elasticsearch semantic chunk search error: {e}", exc_info=True)
            return []

    async def _structured_kg_search(self, subquery_embedding: List[float], top_k: int = 3, top_k_entities: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search on nested entity description embeddings, then filters
        for the top N entities within each returned document based on cosine similarity.
        """
        if not subquery_embedding:
            logger.warning("Structured KG search skipped: No subquery embedding provided.")
            return []

        knn_query = {
            "field": "metadata.entities.description_embedding",
            "query_vector": subquery_embedding,
            "k": top_k,
            "num_candidates": top_k * 10
        }

        logger.debug(f"Performing semantic KG search. KNN Query: {json.dumps(knn_query, indent=2)}")
        try:
            response = await self.es_client.search(
                index=ELASTICSEARCH_INDEX_CHUNKS,
                knn=knn_query,
                size=top_k,
                _source=["chunk_text", "metadata"]
            )
            
            final_results = []
            query_vec = np.array(subquery_embedding)

            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                metadata_from_hit = source.get('metadata', {})
                all_entities = metadata_from_hit.get('entities', [])
                all_relationships = metadata_from_hit.get('relationships', [])

                if not all_entities:
                    continue

                # Score each entity in the document against the subquery embedding
                scored_entities = []
                for entity in all_entities:
                    entity_embedding_list = entity.get("description_embedding")
                    if entity_embedding_list:
                        entity_vec = np.array(entity_embedding_list)
                        # Calculate cosine similarity, handle potential norm=0
                        norm_query = np.linalg.norm(query_vec)
                        norm_entity = np.linalg.norm(entity_vec)
                        if norm_query > 0 and norm_entity > 0:
                            similarity = np.dot(query_vec, entity_vec) / (norm_query * norm_entity)
                            scored_entities.append((similarity, entity))

                # Sort entities by score and take top N
                scored_entities.sort(key=lambda x: x[0], reverse=True)
                top_entities = [entity for score, entity in scored_entities[:top_k_entities]]
                top_entity_names = {entity['name'] for entity in top_entities if entity.get('name')}

                # Filter relationships based on the top entities
                filtered_relationships = []
                if top_entity_names:
                    for rel in all_relationships:
                        if rel.get('source_entity') in top_entity_names or rel.get('target_entity') in top_entity_names:
                            filtered_relationships.append(rel)

                # If we found any top entities, create a result object for this document
                if top_entities:
                    final_results.append({
                        "chunk_text": source.get('chunk_text'),
                        "entities": top_entities,
                        "relationships": filtered_relationships,
                        "score": hit.get('_score'), # This is the parent document's score
                        "file_name": metadata_from_hit.get('file_name'),
                        "doc_id": metadata_from_hit.get('doc_id'),
                        "page_number": metadata_from_hit.get('page_number'),
                        "chunk_index_in_page": metadata_from_hit.get('chunk_index_in_page')
                    })
            
            logger.info(f"Semantic KG search found and processed {len(final_results)} documents.")
            return final_results
        except TransportError as e:
            logger.error(f"Elasticsearch semantic KG search error: {e}", exc_info=True)
            return []

    async def _rerank_documents(self, query: str, documents: List[Dict[str, Any]], doc_type: str) -> List[Dict[str, Any]]:
        if not documents:
            logger.debug(f"No {doc_type} documents to rerank for query: '{query[:50]}...'")
            return []
        if not self.reranker:
            logger.warning(f"Reranker not initialized. Skipping reranking for {doc_type} documents.")
            # Ensure 'rerank_score' is present if not reranking, set to original score or None
            for doc in documents:
                if 'rerank_score' not in doc:
                    doc['rerank_score'] = doc.get('score') 
            return documents

        logger.info(f"Reranking {len(documents)} {doc_type} documents for query: '{query[:50]}...'")

        if doc_type == "chunk":
            pairs = [(query, doc.get("text", "")) for doc in documents]
            text_key_for_logging = "text"
        elif doc_type == "kg": # KG evidences are reranked based on their associated chunk_text
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
            # If reranking fails, ensure 'rerank_score' is present, set to original score or None
            for doc in documents:
                if 'rerank_score' not in doc:
                    doc['rerank_score'] = doc.get('score')
            return documents

        docs_with_scores = [{'doc': doc, 'score': scores[i]} for i, doc in enumerate(documents)]
        docs_with_scores.sort(key=lambda x: x['score'], reverse=True)

        reranked_docs = []
        for item in docs_with_scores:
            doc = item['doc']
            doc['rerank_score'] = float(item['score']) # Store the reranker's score
            reranked_docs.append(doc)
            content_to_log = doc.get(text_key_for_logging, '')
            logger.debug(f"Reranked {doc_type} doc (source: {doc.get('file_name', 'N/A')}, page: {doc.get('page_number', 'N/A')}, content: {content_to_log[:30]}...) new score: {doc['rerank_score']:.4f}")

        logger.info(f"Successfully reranked {len(reranked_docs)} {doc_type} documents.")
        return reranked_docs

    async def _perform_semantic_search_for_subquery(self, subquery_text: str, top_k: int) -> List[Dict[str, Any]]:
        logger.debug(f"Performing semantic search for subquery: '{subquery_text}'")
        embedding = await self._generate_embedding(subquery_text)
        if not embedding:
            logger.warning(f"Could not generate embedding for subquery: '{subquery_text}'. Semantic search will yield no results.")
            return []
        return await self._semantic_search_chunks(embedding, top_k)

    async def _perform_kg_search_for_subquery(self, subquery_text: str, top_k: int, top_k_entities: int) -> List[Dict[str, Any]]:
        logger.debug(f"Performing semantic KG search for subquery: '{subquery_text}'")
        embedding = await self._generate_embedding(subquery_text)
        if not embedding:
            logger.warning(f"Could not generate embedding for subquery: '{subquery_text}'. KG search will yield no results.")
            return []
        return await self._structured_kg_search(embedding, top_k, top_k_entities)

    # --- New Helper Methods for Formatting ---
    def _generate_shorthand_id(self, item: Dict[str, Any], prefix: str, index: int) -> str:
        doc_id_part = "unknown"
        if item.get("doc_id"):
            doc_id_part = str(item["doc_id"]).replace('-', '')[:6]
        
        page_num_val = item.get("page_number")
        page_num_part = str(page_num_val) if page_num_val is not None else "NA"
        
        chunk_idx_val = item.get("chunk_index_in_page")
        chunk_idx_part = str(chunk_idx_val) if chunk_idx_val is not None else str(index)
        
        return f"{prefix}_{doc_id_part}_p{page_num_part}_i{chunk_idx_part}"

    def _format_search_results_for_llm(self, original_query: str, sub_queries_results: List[Dict[str, Any]]) -> str:
        lines = [f"Original Query: {original_query}\n"]
        
        if not sub_queries_results:
            lines.append("No search results found.")
            return "\n".join(lines)

        for sq_idx, sq_result in enumerate(sub_queries_results):
            sub_query_text = sq_result.get("sub_query_text", f"Sub-query {sq_idx + 1}")
            lines.append(f"--- Results for Sub-query: \"{sub_query_text}\" ---")

            # 1) Chunk search results (reranked_chunks)
            reranked_chunks = sq_result.get("reranked_chunks", [])
            if reranked_chunks:
                lines.append("\nVector Search Results (Chunks):")
                for chunk_idx, chunk in enumerate(reranked_chunks):
                    if not isinstance(chunk, dict):
                        logger.warning(f"Skipping non-dict chunk item during formatting: {chunk}")
                        continue

                    shorthand_id = self._generate_shorthand_id(chunk, "c", chunk_idx)
                    score_val = chunk.get('rerank_score', chunk.get('score'))
                    score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
                    lines.append(f"Source ID [{shorthand_id}]: (Score: {score_str})")
                    
                    text_content = chunk.get("text", "N/A")
                    lines.append(text_content)
                    lines.append(f"  File: {chunk.get('file_name', 'N/A')}, Page: {chunk.get('page_number', 'N/A')}, Chunk Index in Page: {chunk.get('chunk_index_in_page', 'N/A')}")
            else:
                lines.append("\nNo vector search results for this sub-query.")

            # 2) Graph search results (retrieved_kg_data)
            retrieved_kg_data = sq_result.get("retrieved_kg_data", [])
            if retrieved_kg_data:
                lines.append("\nKnowledge Graph Results:")
                for kg_idx, kg_item in enumerate(retrieved_kg_data):
                    if not isinstance(kg_item, dict):
                        logger.warning(f"Skipping non-dict kg_item during formatting: {kg_item}")
                        continue
                    
                    shorthand_id = self._generate_shorthand_id(kg_item, "kg", kg_idx)
                    score_val = kg_item.get('rerank_score', kg_item.get('score'))
                    score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
                    lines.append(f"Source ID [{shorthand_id}]: (Score: {score_str})")
                    lines.append(f"  File: {kg_item.get('file_name', 'N/A')}, Page: {kg_item.get('page_number', 'N/A')}, Chunk Index in Page: {kg_item.get('chunk_index_in_page', 'N/A')}")

                    entities = kg_item.get("entities", [])
                    if entities:
                        lines.append("  Entities:")
                        for entity in entities:
                            if not isinstance(entity, dict): continue
                            lines.append(f"    - Name: {entity.get('name', 'N/A')}, Type: {entity.get('type', 'N/A')}")
                            entity_desc = entity.get('description', '')
                            if entity_desc:
                                lines.append(f"      Description: {entity_desc}")
                    
                    relationships = kg_item.get("relationships", [])
                    if relationships:
                        lines.append("  Relationships:")
                        for rel in relationships:
                            if not isinstance(rel, dict): continue
                            lines.append(f"    - {rel.get('source_entity', 'S')} -> {rel.get('relation', 'R')} -> {rel.get('target_entity', 'T')} (Weight: {rel.get('relationship_weight', 'N/A')})")
                            rel_desc = rel.get('relationship_description', '')
                            if rel_desc:
                                lines.append(f"      Description: {rel_desc}")

            else:
                lines.append("\nNo knowledge graph results for this sub-query.")
            
            lines.append("")

        return "\n".join(lines)

    async def search(
        self,
        user_query: str,
        num_subqueries: int = 2,
        top_k_chunks: int = 10,
        top_k_kg: int = 10,
        top_k_kg_entities: int = 5
    ) -> Dict[str, Any]:
        logger.info(f"Starting RAG Fusion search for user query: '{user_query}'")
        subqueries = await self._generate_subqueries(user_query, num_subqueries)
        if not subqueries:
            logger.warning("No subqueries generated from user query. Using original query for search.")
            subqueries = [user_query] # Fallback to original query

        processed_subquery_results = []

        for sq_text in subqueries:
            logger.info(f"Processing subquery: '{sq_text}'")
            retrieved_chunks: List[Dict[str, Any]] = []
            # KG search returns items with 'chunk_text', 'entities', 'relationships'
            retrieved_kg_evidence_with_chunk_text: List[Dict[str, Any]] = []


            logger.debug(f"Initiating concurrent retrieval for subquery: '{sq_text}'")
            try:
                retrieval_results = await asyncio.gather(
                    self._perform_semantic_search_for_subquery(sq_text, top_k_chunks),
                    self._perform_kg_search_for_subquery(sq_text, top_k_kg, top_k_kg_entities), # This returns KG items with chunk_text
                    return_exceptions=True
                )
                retrieved_chunks = retrieval_results[0] if not isinstance(retrieval_results[0], Exception) and retrieval_results[0] is not None else []
                if isinstance(retrieval_results[0], Exception):
                    logger.error(f"Semantic search task failed for subquery '{sq_text}': {retrieval_results[0]}", exc_info=retrieval_results[0])
                
                retrieved_kg_evidence_with_chunk_text = retrieval_results[1] if not isinstance(retrieval_results[1], Exception) and retrieval_results[1] is not None else []
                if isinstance(retrieval_results[1], Exception):
                    logger.error(f"KG search task failed for subquery '{sq_text}': {retrieval_results[1]}", exc_info=retrieval_results[1])

                logger.info(f"Initial retrieval for subquery '{sq_text}': {len(retrieved_chunks)} chunks, {len(retrieved_kg_evidence_with_chunk_text)} KG evidences (with chunk_text).")
            except Exception as e:
                logger.error(f"Unexpected error during concurrent retrieval for subquery '{sq_text}': {e}", exc_info=True)

            # Reranking
            reranked_chunks_for_sq = retrieved_chunks
            # Keep chunk_text for KG items for reranking, then decide whether to pop for final output
            processed_kg_evidence_for_reranking = retrieved_kg_evidence_with_chunk_text 

            if self.reranker:
                if retrieved_chunks:
                    logger.debug(f"Initiating reranking for chunks for subquery: '{sq_text}'")
                    try:
                        reranked_chunks_for_sq = await self._rerank_documents(sq_text, retrieved_chunks, "chunk")
                        logger.info(f"Chunk reranking for subquery '{sq_text}' completed: {len(reranked_chunks_for_sq)} chunks.")
                    except Exception as e:
                        logger.error(f"Chunk reranking task failed for subquery '{sq_text}': {e}. Using original chunks.", exc_info=True)
                else:
                    logger.debug(f"No chunks to rerank for subquery '{sq_text}'.")

                if processed_kg_evidence_for_reranking: # Rerank KG items based on their chunk_text
                    logger.debug(f"Initiating reranking for KG evidence for subquery: '{sq_text}'")
                    try:
                        # _rerank_documents expects 'chunk_text' for 'kg' type
                        processed_kg_evidence_for_reranking = await self._rerank_documents(sq_text, processed_kg_evidence_for_reranking, "kg")
                        logger.info(f"KG evidence reranking for subquery '{sq_text}' completed: {len(processed_kg_evidence_for_reranking)} KG evidences.")
                    except Exception as e:
                        logger.error(f"KG evidence reranking task failed for subquery '{sq_text}': {e}. Using original KG evidence.", exc_info=True)
                else:
                    logger.debug(f"No KG evidence to rerank for subquery '{sq_text}'.")
            else:
                logger.warning("Reranker not available. Using original retrieved documents without explicit reranking scores (scores will be original ES scores).")
                # Ensure rerank_score field exists if reranker is off
                for chunk_doc in reranked_chunks_for_sq:
                    if 'rerank_score' not in chunk_doc: chunk_doc['rerank_score'] = chunk_doc.get('score')
                for kg_doc in processed_kg_evidence_for_reranking:
                    if 'rerank_score' not in kg_doc: kg_doc['rerank_score'] = kg_doc.get('score')


            # Prepare final KG data for output by removing 'chunk_text'.
            # The 'chunk_text' was used for contextual reranking but is removed from the
            # final KG output to avoid confusion with chunk search results.
            final_kg_evidence_for_output = []
            for doc in processed_kg_evidence_for_reranking:
                doc_copy = doc.copy()
                doc_copy.pop('chunk_text', None)
                final_kg_evidence_for_output.append(doc_copy)
            
            if not self.reranker and not reranked_chunks_for_sq:
                 logger.debug(f"No chunks to process (reranker off) for subquery '{sq_text}'.")

            if final_kg_evidence_for_output:
                logger.info(f"Using {len(final_kg_evidence_for_output)} KG evidences for subquery '{sq_text}' (post-processing).")
            else:
                logger.debug(f"No KG evidence to include in final output for subquery '{sq_text}'.")

            processed_subquery_results.append({
                "sub_query_text": sq_text,
                "reranked_chunks": reranked_chunks_for_sq,
                "retrieved_kg_data": final_kg_evidence_for_output
            })

        final_results_dict = {
            "original_query": user_query,
            "sub_queries_results": processed_subquery_results
        }
        
        # Generate and add the LLM-formatted context string
        llm_formatted_context = self._format_search_results_for_llm(
            original_query=user_query,
            sub_queries_results=processed_subquery_results 
        )
        final_results_dict["llm_formatted_context"] = llm_formatted_context
        
        logger.info(f"RAG Fusion search fully completed for user query: '{user_query}'. Formatted context generated.")
        return final_results_dict

async def main_example_search():
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("src.core.retrieval.new_rag_fusion").setLevel(logging.DEBUG) # Set to DEBUG to see more logs

    if not aclient_openai or not es_client:
        logger.error("OpenAI or Elasticsearch client not initialized. Cannot run example.")
        return

    retriever = RAGFusionRetriever()
    user_query_input = input("Enter your search query: ").strip()
    if not user_query_input:
        logger.warning("No query entered. Exiting.")
    else:
        logger.info(f"\n--- Running RAG Fusion Search for: '{user_query_input}' ---")
        search_results_dict = await retriever.search(
            user_query=user_query_input, num_subqueries=2, top_k_chunks=3, top_k_kg=3, top_k_kg_entities=5
        )
        print("\n--- Search Results Dictionary (RAG Fusion: Chunks & KG Reranked if applicable) ---")
        print(json.dumps(search_results_dict, indent=2, default=str))
        
        print("\n--- LLM Formatted Context ---")
        print(search_results_dict.get("llm_formatted_context", "No formatted context generated."))


    if es_client and hasattr(es_client, 'close'):
        await es_client.close()
        logger.info("Elasticsearch client closed.")
    if aclient_openai and hasattr(aclient_openai, "aclose"):
        try:
            await aclient_openai.aclose()
            logger.info("OpenAI client closed.")
        except Exception as e:
            logger.warning(f"Error closing OpenAI client: {e}")

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main_example_search())