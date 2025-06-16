import logging
import os
import asyncio
import yaml
import json # Added for logging query bodies
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import TransportError
from sentence_transformers.cross_encoder import CrossEncoder

from ...utils.logging_config import setup_logger

load_dotenv()
logger = setup_logger(__name__)

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_BASE_DIMENSION", 3072))
ELASTICSEARCH_INDEX_CHUNKS = os.getenv("ELASTICSEARCH_INDEX_CHUNKS", "r2rtest00")
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
            # Assuming prompts are in src/core/prompts/
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

    async def _extract_entities_from_text(self, text: str) -> List[str]:
        if not text.strip() or not self.aclient_openai:
            logger.debug("Entity extraction skipped: Empty text or OpenAI client not available.")
            return []

        entity_extraction_prompt = """You are a highly specialized AI assistant for precise Named Entity Recognition (NER) from search queries. Your primary task is to identify and extract ALL relevant named entities from the given text. These entities are crucial for querying a knowledge graph effectively.

**Entity Types to Extract:**
Focus on, but do not limit yourself to, the following types. Be discerning and prioritize entities that would typically be indexed and searchable in a knowledge graph:
-   **People:** (e.g., "Elon Musk", "Dr. Jane Goodall", "Marie Curie")
-   **Organizations/Companies/Institutions:** (e.g., "OpenAI", "Google", "NASA", "United Nations", "Stanford University", "NIST")
-   **Locations:** (Specific cities, countries, geographical features, e.g., "Paris", "Mount Everest", "Silicon Valley", "Mars", "California")
-   **Products/Services/Technologies:** (e.g., "iPhone 15", "GPT-4", "AWS S3", "Tesla Model S", "H100 GPU", "TPU v5", "Llama 2")
-   **Specific Scientific or Technical Concepts/Fields:** (When they are well-defined and central to the query, e.g., "Quantum Computing", "Machine Learning", "CRISPR", "Blockchain", "General Relativity", "cryptographic security protocols")
-   **Named Events:** (e.g., "Apollo 11 Moon Landing", "WWDC 2023", "COP28 Summit", "French Revolution")
-   **Specific Dates/Time Periods:** (If they are specific, named, and central, e.g., "Q4 2023 earnings", "Industrial Revolution", "Renaissance period")
-   **Key Acronyms/Abbreviations:** (If they represent specific named entities and are common identifiers, e.g., "NASA", "WHO", "IBM", "GDPR")
-   **Specific Laws/Regulations/Treaties/Standards:** (e.g., "GDPR", "Paris Agreement", "ISO 27001")
-   **Named Projects/Initiatives/Programs:** (e.g., "Human Genome Project", "Project Artemis", "Manhattan Project")
-   **Works of Art/Literature/Music:** (e.g., "Mona Lisa", "The Great Gatsby", "Bohemian Rhapsody")

**Output Format:**
Return the extracted entities as a single, comma-separated string. Each entity should be a distinct string.
Example: "Entity One,Entity Two,Another Entity"

**Critical Instructions for Accuracy & Comprehensiveness:**
1.  **Extract ALL:** Your primary goal is to extract *every* identifiable named entity that would be useful for a knowledge graph query. Do not be conservative; if it's a named entity relevant to the query's core, extract it.
2.  **Precision and Fullness:** Capture the most complete and precise form of the entity as it appears or is commonly known.
3.  **Synonyms and Alternative Names for KG Matching:** For certain types of entities, especially **Locations** and **Organizations**, if there are very common alternative names or synonyms that are likely to be used interchangeably in a knowledge graph, include them. Your goal is to maximize the chances of matching an entity in the KG.
    *   For example:
        *   If the query mentions 'Bangalore', and 'Bengaluru' is a common alternative, output: `Bangalore,Bengaluru`.
        *   If the query mentions 'USA', and 'United States', 'United States of America', or 'America' are common alternatives, consider outputting: `USA,United States,America` (choose the most relevant and common ones for KG lookup).
        *   If the query mentions 'WHO', also include `World Health Organization` if it's a likely synonym in the KG: `WHO,World Health Organization`.
    *   Be judicious. Only include highly common and relevant alternatives. Do not generate an excessive number of synonyms for every entity.
4.  **Multi-Word Entities:** Correctly identify and group multi-word entities (e.g., "New York City" not "New", "York", "City"; "Chief Executive Officer" if it refers to a specific role being discussed as an entity type, though usually you'd extract the person's name holding that role).
5.  **Specificity is Key:** Distinguish between general nouns/concepts and specific *named* entities.
    *   "artificial intelligence" is a broad field. However, if the query is "OpenAI's work in artificial intelligence", extract "OpenAI". If the query is "advancements in artificial intelligence by DeepMind", extract "artificial intelligence" (as a key concept for KG) and "DeepMind".
    *   "database systems" is general. "PostgreSQL" is a specific named entity.
6.  **Contextual Relevance:** Ensure extracted entities are central to the query's intent.
7.  **No Entities Found:** If, after careful analysis, absolutely no distinct named entities suitable for a knowledge graph query are identified, return an **empty string** (e.g., ""). Do NOT invent entities or return general terms if they aren't acting as specific identifiers in the query.

**Examples to Guide Extraction (incorporating synonyms):**

**Example 1 (Standard Query with Multiple Entity Types & Location Synonym):**
Query: "What were the key findings of the Human Genome Project regarding genetic research in Bangalore and its impact on companies like Genentech?"
Output: "Human Genome Project,genetic research,Bangalore,Bengaluru,Genentech"

**Example 2 (Technical Query with Products and Organizations):**
Query: "Analyze the performance differences between NVIDIA's A100 and AMD's MI250X for deep learning workloads, specifically citing benchmarks from MLPerf."
Output: "NVIDIA,A100,AMD,MI250X,deep learning,MLPerf"

**Example 3 (Query with Acronyms, Locations, Concepts & Organization Synonym):**
Query: "How is the EU's GDPR affecting data privacy policies of tech companies in California, particularly concerning AI development by the WHO?"
Output: "EU,European Union,GDPR,data privacy,California,AI development,WHO,World Health Organization"

**Example 4 (Query with No Specific Named Entities for KG):**
Query: "What are some general tips for improving public speaking skills?"
Output: ""

**Example 5 (Historical Event with People and Locations):**
Query: "Describe the role of Winston Churchill during the Blitz in London in World War II."
Output: "Winston Churchill,The Blitz,London,World War II"

---
Now, process the following search query according to all the above instructions.
Query: {text}
Output:
"""

        formatted_prompt_for_extraction = entity_extraction_prompt.format(text=text)
        logger.debug(f"Extracting entities from text: '{text[:100]}...' using advanced prompt.")

        try:
            response = await self.aclient_openai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "user", "content": formatted_prompt_for_extraction}
                ],
                temperature=0.0,
                max_tokens=200
            )
            entities_str = response.choices[0].message.content.strip()

            if not entities_str:
                logger.info(f"No entities extracted by LLM for text: '{text[:100]}...'")
                return []

            entities = [
                e.strip().strip('"').strip("'")
                for e in entities_str.split(',')
                if e.strip().strip('"').strip("'")
            ]
            logger.info(f"Extracted entities (cleaned): {entities} from text: '{text[:100]}...'")
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities from text '{text[:100]}...': {e}", exc_info=True)
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
            logger.info(f"Semantic search found {len(results)} chunks.")
            return results
        except TransportError as e:
            logger.error(f"Elasticsearch semantic search error: {e}", exc_info=True)
            return []

    async def _structured_kg_search(self, original_subquery_text: str, extracted_entities: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        if not original_subquery_text.strip():
            logger.warning("Structured KG search skipped: No subquery text provided.")
            return []

        should_clauses = []

        if extracted_entities:
            logger.info(f"Performing KG search using extracted entities: {extracted_entities} for subquery: '{original_subquery_text[:50]}...'")
            should_clauses.append({
                "nested": {
                    "path": "metadata.entities",
                    "query": {"terms": {"metadata.entities.name.keyword": extracted_entities}}
                }
            })
            should_clauses.append({
                "nested": {
                    "path": "metadata.relationships",
                    "query": {
                        "bool": {
                            "should": [
                                {"terms": {"metadata.relationships.source_entity.keyword": extracted_entities}},
                                {"terms": {"metadata.relationships.target_entity.keyword": extracted_entities}}
                            ],
                            "minimum_should_match": 1
                        }
                    }
                }
            })
            entity_text_fields = ["metadata.entities.name", "metadata.entities.type", "metadata.entities.description"]
            should_clauses.append(
                {"nested": {"path": "metadata.entities", "query": {"multi_match": {"query": original_subquery_text, "fields": entity_text_fields, "fuzziness": "AUTO"}}}}
            )
            relationship_text_fields = [
                "metadata.relationships.source_entity", "metadata.relationships.target_entity",
                "metadata.relationships.relation", "metadata.relationships.relationship_description"
            ]
            should_clauses.append(
                {"nested": {"path": "metadata.relationships", "query": {"multi_match": {"query": original_subquery_text, "fields": relationship_text_fields, "fuzziness": "AUTO"}}}}
            )
        else:
            logger.info(f"No specific entities extracted for '{original_subquery_text[:50]}...'. Performing broader KG search using full subquery text.")
            entity_text_fields = ["metadata.entities.name", "metadata.entities.type", "metadata.entities.description"]
            should_clauses.append(
                {"nested": {"path": "metadata.entities", "query": {"multi_match": {"query": original_subquery_text, "fields": entity_text_fields, "fuzziness": "AUTO"}}}}
            )
            relationship_text_fields = [
                "metadata.relationships.source_entity", "metadata.relationships.target_entity",
                "metadata.relationships.relation", "metadata.relationships.relationship_description"
            ]
            should_clauses.append(
                {"nested": {"path": "metadata.relationships", "query": {"multi_match": {"query": original_subquery_text, "fields": relationship_text_fields, "fuzziness": "AUTO"}}}}
            )

        query_body = {
            "query": {"bool": {"should": should_clauses, "minimum_should_match": 1}},
            "size": top_k,
            "_source": ["chunk_text", "metadata"]
        }

        logger.debug(f"Performing structured KG search. Query body: {json.dumps(query_body, indent=2)}")
        try:
            response = await self.es_client.search(index=ELASTICSEARCH_INDEX_CHUNKS, body=query_body)
            results = []

            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                metadata_from_hit = source.get('metadata', {})

                original_entities_in_hit = metadata_from_hit.get('entities', [])
                original_relationships_in_hit = metadata_from_hit.get('relationships', [])

                current_hit_filtered_entities = []
                current_hit_filtered_relationships = []

                if extracted_entities:
                    for entity_obj in original_entities_in_hit:
                        if entity_obj.get("name") in extracted_entities:
                            current_hit_filtered_entities.append(entity_obj)

                    for rel_obj in original_relationships_in_hit:
                        if rel_obj.get("source_entity") in extracted_entities or \
                           rel_obj.get("target_entity") in extracted_entities:
                            current_hit_filtered_relationships.append(rel_obj)

                    if current_hit_filtered_entities or current_hit_filtered_relationships:
                        results.append({
                            "chunk_text": source.get('chunk_text'),
                            "entities": current_hit_filtered_entities,
                            "relationships": current_hit_filtered_relationships,
                            "score": hit.get('_score'),
                            "file_name": metadata_from_hit.get('file_name'),
                            "doc_id": metadata_from_hit.get('doc_id'),
                            "page_number": metadata_from_hit.get('page_number'),
                            "chunk_index_in_page": metadata_from_hit.get('chunk_index_in_page')
                        })
                    else:
                        logger.debug(f"Skipping hit (Doc ID: {metadata_from_hit.get('doc_id')}) for KG search with extracted entities, as it doesn't contain the specifically extracted entities/relationships after filtering.")
                else:
                    if original_entities_in_hit or original_relationships_in_hit:
                        results.append({
                            "chunk_text": source.get('chunk_text'),
                            "entities": original_entities_in_hit,
                            "relationships": original_relationships_in_hit,
                            "score": hit.get('_score'),
                            "file_name": metadata_from_hit.get('file_name'),
                            "doc_id": metadata_from_hit.get('doc_id'),
                            "page_number": metadata_from_hit.get('page_number'),
                            "chunk_index_in_page": metadata_from_hit.get('chunk_index_in_page')
                        })
                    else:
                        logger.debug(f"Skipping hit (Doc ID: {metadata_from_hit.get('doc_id')}) for broad KG search as it lacks any entities/relationships.")

            logger.info(f"Structured KG search (with filtering if applicable) found {len(results)} documents with relevant KG data for subquery '{original_subquery_text[:50]}...'.")
            return results
        except TransportError as e:
            logger.error(f"Elasticsearch structured KG search error for subquery '{original_subquery_text[:50]}...': {e}", exc_info=True)
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

    async def _perform_kg_search_for_subquery(self, subquery_text: str, top_k: int) -> List[Dict[str, Any]]:
        logger.debug(f"Performing KG search for subquery: '{subquery_text}'")
        extracted_entities = await self._extract_entities_from_text(subquery_text)
        return await self._structured_kg_search(subquery_text, extracted_entities, top_k)

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
            logger.warning("No subqueries generated from user query. Using original query for search.")
            subqueries = [user_query]

        processed_subquery_results = []

        for sq_text in subqueries:
            logger.info(f"Processing subquery: '{sq_text}'")
            retrieved_chunks: List[Dict[str, Any]] = []
            retrieved_kg_evidence_with_chunk_text: List[Dict[str, Any]] = []

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

                retrieved_kg_evidence_with_chunk_text = retrieval_results[1] if not isinstance(retrieval_results[1], Exception) and retrieval_results[1] is not None else []
                if isinstance(retrieval_results[1], Exception):
                    logger.error(f"KG search task failed for subquery '{sq_text}': {retrieval_results[1]}")

                logger.info(f"Initial retrieval for subquery '{sq_text}': {len(retrieved_chunks)} chunks, {len(retrieved_kg_evidence_with_chunk_text)} KG evidences (with chunk_text).")
            except Exception as e:
                logger.error(f"Unexpected error during concurrent retrieval for subquery '{sq_text}': {e}", exc_info=True)

            reranked_chunks_for_sq = retrieved_chunks
            processed_kg_evidence_for_sq = retrieved_kg_evidence_with_chunk_text

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

                if processed_kg_evidence_for_sq:
                    logger.debug(f"Initiating reranking for KG evidence for subquery: '{sq_text}'")
                    try:
                        processed_kg_evidence_for_sq = await self._rerank_documents(sq_text, processed_kg_evidence_for_sq, "kg")
                        logger.info(f"KG evidence reranking for subquery '{sq_text}' completed: {len(processed_kg_evidence_for_sq)} KG evidences.")
                    except Exception as e:
                        logger.error(f"KG evidence reranking task failed for subquery '{sq_text}': {e}. Using original KG evidence.", exc_info=True)
                else:
                    logger.debug(f"No KG evidence to rerank for subquery '{sq_text}'.")
            else:
                logger.warning("Reranker not available. Using original retrieved documents without reranking.")

            final_kg_evidence_for_output = []
            if processed_kg_evidence_for_sq:
                for item in processed_kg_evidence_for_sq:
                    item_copy = item.copy()
                    item_copy.pop("chunk_text", None)
                    final_kg_evidence_for_output.append(item_copy)

            if not self.reranker and not retrieved_chunks:
                 logger.debug(f"No chunks to process (reranker off) for subquery '{sq_text}'.")

            if final_kg_evidence_for_output:
                logger.info(f"Using {len(final_kg_evidence_for_output)} KG evidences for subquery '{sq_text}' (post-processing, chunk_text removed for final output).")
            else:
                logger.debug(f"No KG evidence to include in final output for subquery '{sq_text}'.")

            processed_subquery_results.append({
                "sub_query_text": sq_text,
                "reranked_chunks": reranked_chunks_for_sq,
                "retrieved_kg_data": final_kg_evidence_for_output
            })

        final_results = {"original_query": user_query, "sub_queries_results": processed_subquery_results}
        logger.info(f"RAG Fusion search fully completed for user query: '{user_query}'.")
        return final_results

async def main_example_search():
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        print("\n--- Search Results (RAG Fusion: Chunks & KG Reranked if applicable) ---")
        print(json.dumps(search_results, indent=2, default=str))

    if es_client:
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