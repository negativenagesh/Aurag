import logging
import os
import asyncio
from io import BytesIO
from typing import AsyncGenerator, Dict, Any, List, Tuple
import yaml 
from pathlib import Path 
import copy 
import xml.etree.ElementTree as ET 
import re
import hashlib # Added for SHA256 hashing

from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from openai import AsyncOpenAI
from pypdf import PdfReader

from ...utils.logging_config import setup_logger

load_dotenv()
logger = setup_logger(__name__) 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini") 
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_BASE_DIMENSION", 3072))

ELASTICSEARCH_INDEX_CHUNKS = os.getenv("ELASTICSEARCH_INDEX_CHUNKS", "r2rtest00") # Ensure this matches your target index

CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE", 3072)) 
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP", 512))
FORWARD_CHUNKS = int(os.getenv("FORWARD_CHUNKS", 3))
BACKWARD_CHUNKS = int(os.getenv("BACKWARD_CHUNKS", 3))
CHARS_PER_TOKEN_ESTIMATE = 4 
SUMMARY_MAX_TOKENS = 512


# --- Initialize Clients ---
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in .env. OpenAI client will not be functional.")
    aclient_openai = None
else:
    aclient_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

try:
    es_client = AsyncElasticsearch(
        os.getenv("ELASTICSEARCH_URL", "https://my-elasticsearch-project-c44c4f.es.us-east-1.aws.elastic.cloud:443"),
        api_key=os.getenv("ELASTICSEARCH_API_KEY", "cWR0WFU1Y0I3YS1td2g2cWdpV186VTdnNFNVWmRmWVI5dnd3WEwwOWdMQQ=="),
        request_timeout=30  # Increased timeout from default 10s
    )
    logger.info("AsyncElasticsearch client initialized.")
except Exception as e:
    logger.error(f"Failed to initialize AsyncElasticsearch client: {e}")
    es_client = None

# --- Tokenizer Function ---
def get_tokenizer_for_model(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"Tokenizer for model '{model_name}' not found. Using 'cl100k_base'.")
        return tiktoken.get_encoding("cl100k_base")

tokenizer = get_tokenizer_for_model(OPENAI_EMBEDDING_MODEL)

def tiktoken_len(text: str) -> int:
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# --- Elasticsearch Mapping Definition ---
CHUNKED_PDF_MAPPINGS = {
    "mappings": {
        "properties": {
            "chunk_text": {"type": "text"}, 
            "embedding": {
                "type": "dense_vector",
                "dims": OPENAI_EMBEDDING_DIMENSIONS,
                "index": True,
                "similarity": "cosine"
            },
            "metadata": {
                "properties": {
                    "file_name": {"type": "keyword"},
                    "doc_id": {"type": "keyword"}, 
                    "page_number": {"type": "integer"},
                    "chunk_index_in_page": {"type": "integer"},
                    "document_summary": {"type": "text"}, 
                    "entities": {
                        "type": "nested",
                        "properties": {
                            "name": {"type": "keyword"},
                            "type": {"type": "keyword"},
                            "description": {"type": "text"}
                        }
                    },
                    "relationships": {
                        "type": "nested",
                        "properties": {
                            "source_entity": {"type": "keyword"},
                            "target_entity": {"type": "keyword"},
                            "relation": {"type": "keyword"},
                            "relationship_description": {"type": "text"},
                            "relationship_weight": {"type": "float"}
                        }
                    }
                }
            }
        }
    }
}

class ChunkingEmbeddingPDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            length_function=tiktoken_len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.pdf_reader = PdfReader
        self.enrich_prompt_template = self._load_prompt_template("chunk_enrichment")
        self.graph_extraction_prompt_template = self._load_prompt_template("graph_extraction")
        self.summary_prompt_template = self._load_prompt_template("summary") 

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

    def _parse_graph_xml(self, xml_string: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        entities = []
        relationships = []
        string_to_parse_for_log = "" 
        try:
            stripped_xml = xml_string.strip()
            if stripped_xml.startswith("```xml"):
                stripped_xml = stripped_xml[len("```xml"):].strip()
            elif stripped_xml.startswith("```"): 
                stripped_xml = stripped_xml[len("```"):].strip()
            
            if stripped_xml.endswith("```"):
                stripped_xml = stripped_xml[:-len("```")].strip()

            if stripped_xml.startswith("<?xml"):
                end_decl = stripped_xml.find("?>")
                if end_decl != -1:
                    stripped_xml = stripped_xml[end_decl + 2:].lstrip()
            
            if not stripped_xml:
                logger.warning("Empty XML string after stripping common artifacts and XML declaration.")
                return [], []

            first_tag_start = stripped_xml.find("<")
            if first_tag_start == -1:
                logger.warning(f"No '<' (start of tag) found in XML string after stripping: {stripped_xml[:200]}...")
                return [], []
            
            content_to_parse = stripped_xml[first_tag_start:]
            if not content_to_parse.startswith("<root>") and not content_to_parse.startswith("<graph>"): 
                 string_to_parse_for_log = f"<root_wrapper>{content_to_parse}</root_wrapper>"
            else:
                 string_to_parse_for_log = content_to_parse
            
            try:
                root = ET.fromstring(string_to_parse_for_log)
                
                for entity_elem in root.findall(".//entity"): 
                    name = entity_elem.get("name")
                    ent_type_elem = entity_elem.find("type")
                    ent_desc_elem = entity_elem.find("description")
                    ent_type = ent_type_elem.text.strip() if ent_type_elem is not None and ent_type_elem.text else "Unknown"
                    ent_desc = ent_desc_elem.text.strip() if ent_desc_elem is not None and ent_desc_elem.text else ""
                    if name:
                        entities.append({"name": name.strip(), "type": ent_type, "description": ent_desc})

                for rel_elem in root.findall(".//relationship"): 
                    source_elem = rel_elem.find("source")
                    target_elem = rel_elem.find("target")
                    rel_type_elem = rel_elem.find("type")
                    rel_desc_elem = rel_elem.find("description")
                    rel_weight_elem = rel_elem.find("weight")
                    source = source_elem.text.strip() if source_elem is not None and source_elem.text else None
                    target = target_elem.text.strip() if target_elem is not None and target_elem.text else None
                    rel_type = rel_type_elem.text.strip() if rel_type_elem is not None and rel_type_elem.text else "RELATED_TO"
                    rel_desc = rel_desc_elem.text.strip() if rel_desc_elem is not None and rel_desc_elem.text else ""
                    weight = None
                    if rel_weight_elem is not None and rel_weight_elem.text:
                        try:
                            weight = float(rel_weight_elem.text.strip())
                        except ValueError:
                            logger.warning(f"Could not parse relationship weight '{rel_weight_elem.text}' as float.")
                    if source and target:
                        relationships.append({
                            "source_entity": source, "target_entity": target, "relation": rel_type,
                            "relationship_description": rel_desc, "relationship_weight": weight
                        })
                
            except ET.ParseError as e:
                err_line, err_col = e.position
                log_message = (
                    f"XML parsing error: {e}\n"
                    f"Error at line {err_line}, column {err_col}. Trying regex-based extraction as fallback.\n"
                    f"Original XML content snippet: {xml_string[:500]}"
                )
                logger.warning(log_message)
                
                entities = [] 
                relationships = [] 

                entity_pattern = r'<entity\s+name="([^"]+)"\s*>(?:<type>([^<]+)</type>)?(?:<description>([^<]+)</description>)?</entity>'
                for match in re.finditer(entity_pattern, content_to_parse): 
                    name, entity_type, description = match.groups()
                    if name:
                        entities.append({
                            "name": name.strip(),
                            "type": entity_type.strip() if entity_type else "Unknown",
                            "description": description.strip() if description else ""
                        })
                
                rel_pattern = r'<relationship>(?:<source>([^<]+)</source>)?(?:<target>([^<]+)</target>)?(?:<type>([^<]+)</type>)?(?:<description>([^<]+)</description>)?(?:<weight>([^<]+)</weight>)?</relationship>'
                for match in re.finditer(rel_pattern, content_to_parse): 
                    source, target, rel_type, description, weight_str = match.groups()
                    if source and target:
                        weight = None
                        if weight_str:
                            try:
                                weight = float(weight_str.strip())
                            except ValueError:
                                pass
                        
                        relationships.append({
                            "source_entity": source.strip(),
                            "target_entity": target.strip(),
                            "relation": rel_type.strip() if rel_type else "RELATED_TO",
                            "relationship_description": description.strip() if description else "",
                            "relationship_weight": weight
                        })
            
            logger.debug(f"Parsed {len(entities)} entities and {len(relationships)} relationships from XML.")
        
        except Exception as e: 
            logger.error(f"An unexpected error occurred during XML parsing: {e}\n"
                        f"Original XML content from LLM (first 500 chars):\n{xml_string[:500]}\n"
                        f"Content attempted for parsing (if available):\n{string_to_parse_for_log}", exc_info=True)
        
        return entities, relationships

    async def _extract_knowledge_graph(
        self, chunk_text: str, document_summary: str 
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not aclient_openai or not self.graph_extraction_prompt_template:
            logger.warning("OpenAI client or graph extraction prompt not available. Skipping graph extraction.")
            return [], []
        
        formatted_prompt = self.graph_extraction_prompt_template.format(
            document_summary=document_summary, 
            input=chunk_text, 
            entity_types=str([]), 
            relation_types=str([]) 
        )
        logger.debug(f"Formatted prompt for graph extraction (chunk-level, to {OPENAI_CHAT_MODEL}): First 200 chars: {formatted_prompt[:200]}...")
        try:
            response = await aclient_openai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert assistant that extracts entities and relationships from text and formats them as XML according to the provided schema. Ensure all tags are correctly opened and closed."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.1, 
                max_tokens=4000 
            )
            xml_response = response.choices[0].message.content.strip()
            logger.debug(f"Raw XML response from LLM for chunk-level graph extraction (first 500 chars):\n{xml_response[:500]}")
            return self._parse_graph_xml(xml_response)
        except Exception as e:
            logger.error(f"Error during chunk-level graph extraction API call or parsing: {e}", exc_info=True)
            return [], []

    async def _enrich_chunk_content(
        self, chunk_text: str, document_summary: str, 
        preceding_chunks_texts: List[str], succeeding_chunks_texts: List[str],
    ) -> str:
        if not aclient_openai or not self.enrich_prompt_template:
            logger.warning("OpenAI client or enrichment prompt not available. Skipping enrichment.")
            return chunk_text 
        
        preceding_context = "\n---\n".join(preceding_chunks_texts)
        succeeding_context = "\n---\n".join(succeeding_chunks_texts)
        max_output_chars = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN_ESTIMATE
        
        formatted_prompt = self.enrich_prompt_template.format(
            document_summary=document_summary, preceding_chunks=preceding_context,
            succeeding_chunks=succeeding_context, chunk=chunk_text, chunk_size=max_output_chars 
        )
        logger.debug(f"Formatted prompt for enrichment (to be sent to {OPENAI_CHAT_MODEL}): ...")
        try:
            response = await aclient_openai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert assistant that refines and enriches text chunks according to specific guidelines."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3, 
                max_tokens=min(CHUNK_SIZE_TOKENS + CHUNK_OVERLAP_TOKENS, 4000) 
            )
            enriched_text = response.choices[0].message.content.strip()
            logger.debug(f"Chunk enriched. Original length: {len(chunk_text)}, Enriched length: {len(enriched_text)}")
            return enriched_text
        except Exception as e:
            logger.error(f"Error during chunk enrichment API call: {e}", exc_info=True)
            return chunk_text

    async def _generate_document_summary(self, full_document_text: str) -> str: 
        if not aclient_openai or not self.summary_prompt_template:
            logger.warning("OpenAI client or summary prompt not available. Skipping document summary generation.")
            return "Summary generation skipped due to missing configuration."
        if not full_document_text.strip():
            logger.warning("Full document text is empty. Skipping summary generation.")
            return "Document is empty, no summary generated."

        formatted_prompt = self.summary_prompt_template.format(document=full_document_text)
        logger.info(f"Generating document summary using {OPENAI_CHAT_MODEL}...")
        try:
            response = await aclient_openai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,
                max_tokens=SUMMARY_MAX_TOKENS 
            )
            summary_text = response.choices[0].message.content.strip()
            logger.info(f"Document summary generated. Length: {len(summary_text)} chars.")
            return summary_text
        except Exception as e:
            logger.error(f"Error generating document summary: {e}", exc_info=True)
            return f"Error during summary generation: {e}"

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        if not aclient_openai:
            logger.warning("OpenAI client not available. Cannot generate embeddings.")
            return [[] for _ in texts] 
        
        all_embeddings = []
        openai_batch_size = 2048 
        try:
            for i in range(0, len(texts), openai_batch_size):
                batch_texts = texts[i:i + openai_batch_size]
                processed_batch_texts = [text if text.strip() else " " for text in batch_texts]
                
                response = await aclient_openai.embeddings.create(
                    input=processed_batch_texts, 
                    model=OPENAI_EMBEDDING_MODEL, 
                    dimensions=OPENAI_EMBEDDING_DIMENSIONS
                )
                all_embeddings.extend([item.embedding for item in response.data])
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            return [[] for _ in texts] 

    async def _generate_all_raw_chunks_from_pages(
        self, 
        pdf_pages_data: List[Tuple[int, str]], 
        file_name: str, 
        doc_id: str
    ) -> List[Dict[str, Any]]:
        all_raw_chunks_with_meta: List[Dict[str, Any]] = []
        for page_num, page_text in pdf_pages_data:
            if not page_text or not page_text.strip(): 
                logger.debug(f"Skipping empty page {page_num} for raw chunk generation.")
                continue 
            
            raw_chunks_on_page = self.text_splitter.split_text(page_text)
            for chunk_idx_on_page, raw_chunk_text in enumerate(raw_chunks_on_page):
                logger.debug(f"RAW CHUNK (File: {file_name}, Page {page_num}, Idx {chunk_idx_on_page}, Len {len(raw_chunk_text)}): '''{raw_chunk_text[:100].strip()}...'''")
                all_raw_chunks_with_meta.append({
                    "text": raw_chunk_text, "page_num": page_num,
                    "chunk_idx_on_page": chunk_idx_on_page,
                    "file_name": file_name, "doc_id": doc_id
                })
        logger.info(f"Generated {len(all_raw_chunks_with_meta)} raw chunks from {file_name}.")
        return all_raw_chunks_with_meta

    async def _process_individual_chunk_pipeline(
        self, 
        raw_chunk_info: Dict[str, Any], 
        user_provided_doc_summary: str, 
        llm_generated_doc_summary: str, 
        all_raw_texts: List[str], 
        global_idx: int, 
        file_name: str, 
        doc_id: str 
    ) -> Dict[str, Any] | None: 
        chunk_text = raw_chunk_info["text"]
        page_num = raw_chunk_info["page_num"]
        chunk_idx_on_page = raw_chunk_info["chunk_idx_on_page"]
        
        logger.debug(f"Starting pipeline for chunk: File {file_name}, Page {page_num}, Index {chunk_idx_on_page}")

        preceding_indices = range(max(0, global_idx - BACKWARD_CHUNKS), global_idx)
        succeeding_indices = range(global_idx + 1, min(len(all_raw_texts), global_idx + 1 + FORWARD_CHUNKS))
        preceding_texts = [all_raw_texts[i] for i in preceding_indices]
        succeeding_texts = [all_raw_texts[i] for i in succeeding_indices]

        kg_task = asyncio.create_task(
            self._extract_knowledge_graph(chunk_text, user_provided_doc_summary)
        )
        enrich_task = asyncio.create_task(
            self._enrich_chunk_content(
                chunk_text, user_provided_doc_summary, preceding_texts, succeeding_texts
            )
        )

        results = await asyncio.gather(kg_task, enrich_task, return_exceptions=True)
        
        kg_result_or_exc = results[0]
        enrich_result_or_exc = results[1]

        chunk_entities, chunk_relationships = [], []
        if isinstance(kg_result_or_exc, Exception):
            logger.error(f"KG extraction failed for chunk (Page {page_num}, Index {chunk_idx_on_page}) for '{file_name}': {kg_result_or_exc}")
        elif kg_result_or_exc: 
            chunk_entities, chunk_relationships = kg_result_or_exc
            logger.debug(f"KG extracted for chunk (Page {page_num}, Index {chunk_idx_on_page}): {len(chunk_entities)} entities, {len(chunk_relationships)} relationships.")

        enriched_text: str
        if isinstance(enrich_result_or_exc, Exception):
            logger.error(f"Enrichment failed for chunk (Page {page_num}, Index {chunk_idx_on_page}) for '{file_name}': {enrich_result_or_exc}. Using original text.")
            enriched_text = chunk_text 
        else:
            enriched_text = enrich_result_or_exc
            logger.debug(f"Enrichment successful for chunk (Page {page_num}, Index {chunk_idx_on_page}).")
        
        embedding_list = await self._generate_embeddings([enriched_text]) 
        
        embedding_vector = []
        if embedding_list and embedding_list[0]: 
            embedding_vector = embedding_list[0]
        
        if not embedding_vector: 
            logger.warning(f"Skipping chunk from page {page_num}, index {chunk_idx_on_page} for '{file_name}' due to missing embedding.")
            return None

        es_doc_id = f"{doc_id}_p{page_num}_c{chunk_idx_on_page}"
        metadata_payload = {
            "file_name": file_name, 
            "doc_id": doc_id,
            "page_number": page_num,
            "chunk_index_in_page": chunk_idx_on_page,
            "document_summary": llm_generated_doc_summary, 
            "entities": chunk_entities, 
            "relationships": chunk_relationships 
        }
        
        action = {
            "_index": ELASTICSEARCH_INDEX_CHUNKS, 
            "_id": es_doc_id,
            "_source": {
                "chunk_text": enriched_text, 
                "embedding": embedding_vector, 
                "metadata": metadata_payload
            }
        }
        logger.debug(f"Pipeline complete for chunk (Page {page_num}, Index {chunk_idx_on_page}). ES action prepared.")
        return action

    async def process_pdf(
        self, data: str | bytes, file_name: str, doc_id: str, user_provided_document_summary: str 
    ) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"Processing PDF: {file_name} (Doc ID: {doc_id}) with user-provided summary: '{user_provided_document_summary[:100]}...'")
        
        try:
            if isinstance(data, str): 
                with open(data, 'rb') as f_in:
                    pdf_stream = BytesIO(f_in.read())
            else: 
                pdf_stream = BytesIO(data)
            
            pdf = self.pdf_reader(pdf_stream)
            logger.info(f"Loaded {len(pdf.pages)} pages from '{file_name}'.")

            pdf_pages_data: List[Tuple[int, str]] = []
            full_document_text_parts: List[str] = []
            for page_num, page in enumerate(pdf.pages, 1): 
                page_text = page.extract_text()
                if not page_text or not page_text.strip(): 
                    logger.warning(f"Page {page_num} in '{file_name}' has no extractable text during initial scan.")
                    pdf_pages_data.append((page_num, "")) 
                    full_document_text_parts.append("") 
                    continue
                pdf_pages_data.append((page_num, page_text))
                full_document_text_parts.append(page_text)

            if not any(pt for _, pt in pdf_pages_data if pt.strip()):
                logger.warning(f"No text extracted from any page in '{file_name}'. Aborting processing for this PDF.")
                return

            full_document_text = "\n\n".join(full_document_text_parts)
            llm_generated_doc_summary = await self._generate_document_summary(full_document_text)

            all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_pages(
                pdf_pages_data, file_name, doc_id
            )
            
            if not all_raw_chunks_with_meta:
                logger.warning(f"No raw chunks were generated from '{file_name}'. Aborting further processing.")
                return

            all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]
            
            logger.info(f"Starting concurrent processing for {len(all_raw_chunks_with_meta)} raw chunks from '{file_name}'.")

            processing_tasks = []
            for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
                task = asyncio.create_task(
                    self._process_individual_chunk_pipeline(
                        raw_chunk_info=raw_chunk_info_item,
                        user_provided_doc_summary=user_provided_document_summary, 
                        llm_generated_doc_summary=llm_generated_doc_summary, 
                        all_raw_texts=all_raw_texts,
                        global_idx=i,
                        file_name=file_name, 
                        doc_id=doc_id        
                    )
                )
                processing_tasks.append(task)
            
            num_successfully_processed = 0
            for future in asyncio.as_completed(processing_tasks):
                try:
                    es_action = await future 
                    if es_action: 
                        yield es_action
                        num_successfully_processed += 1
                except Exception as e:
                    logger.error(f"Error processing a chunk future for '{file_name}': {e}", exc_info=True)
            
            logger.info(f"Finished processing for '{file_name}'. Successfully processed and yielded {num_successfully_processed}/{len(all_raw_chunks_with_meta)} chunks.")

        except Exception as e:
            logger.error(f"Major failure in process_pdf for '{file_name}': {e}", exc_info=True)
            raise 

async def ensure_es_index_exists(client: AsyncElasticsearch, index_name: str, mappings_body: Dict):
    if client is None:
        logger.error("Elasticsearch client is not initialized. Cannot ensure index.")
        return False
    try:
        if not await client.indices.exists(index=index_name):
            await client.indices.create(index=index_name, body=mappings_body)
            logger.info(f"Elasticsearch index '{index_name}' created with specified mappings.")
            return True
        else: 
            current_mapping = await client.indices.get_mapping(index=index_name)
            current_properties = current_mapping.get(index_name, {}).get('mappings', {}).get('properties', {}).get('metadata', {}).get('properties', {})
            expected_metadata_properties = mappings_body.get('mappings', {}).get('properties', {}).get('metadata', {}).get('properties', {})
            
            if "document_summary" not in current_properties:
                logger.warning(f"Field 'document_summary' missing in index '{index_name}'. Attempting to update mapping.")
                try:
                    update_body = {"properties": {"metadata": {"properties": {"document_summary": expected_metadata_properties["document_summary"]}}}}
                    await client.indices.put_mapping(index=index_name, body=update_body)
                    logger.info(f"Successfully added 'document_summary' field to mapping of index '{index_name}'.")
                except Exception as map_e:
                    logger.error(f"Failed to update mapping for index '{index_name}' to add 'document_summary': {map_e}. This might cause issues.", exc_info=True)
            elif current_properties.get("document_summary") != expected_metadata_properties.get("document_summary"):
                 logger.warning(f"Elasticsearch index '{index_name}' exists but 'document_summary' mapping differs. This might cause issues.")
                 logger.debug(f"Current 'document_summary' mapping: {current_properties.get('document_summary')}")
                 logger.debug(f"Expected 'document_summary' mapping: {expected_metadata_properties.get('document_summary')}")
            else:
                logger.info(f"Elasticsearch index '{index_name}' already exists and 'document_summary' field is consistent.")
            return True 
    except Exception as e:
        logger.error(f"Error with Elasticsearch index '{index_name}': {e}", exc_info=True)
        return False

async def example_run_pdf_processing(pdf_data: str | bytes, original_file_name: str, document_id: str, user_provided_doc_summary: str): 
    if not es_client:
        logger.error("Elasticsearch client not configured. Aborting example run.")
        return
    if not aclient_openai: 
        logger.error("OpenAI client not configured. Aborting example run.")
        return

    if not await ensure_es_index_exists(es_client, ELASTICSEARCH_INDEX_CHUNKS, CHUNKED_PDF_MAPPINGS):
        logger.error(f"Failed to ensure Elasticsearch index '{ELASTICSEARCH_INDEX_CHUNKS}' exists or is compatible. Aborting.")
        return

    processor = ChunkingEmbeddingPDFProcessor()
    actions_for_es = []
    
    logger.info(f"\n--- Starting PDF Processing for: {original_file_name} (Doc ID: {document_id}) ---")
    try:
        async for action in processor.process_pdf(pdf_data, original_file_name, document_id, user_provided_doc_summary):
            if action: 
                actions_for_es.append(action)

        if actions_for_es:
            logger.info(f"Collected {len(actions_for_es)} actions for bulk ingestion into '{ELASTICSEARCH_INDEX_CHUNKS}'.")
            
            if actions_for_es: 
                logger.info("Sample document to be indexed (first one, embedding vector omitted if long):")
                sample_action_copy = copy.deepcopy(actions_for_es[0]) 
                if "_source" in sample_action_copy and "embedding" in sample_action_copy["_source"]:
                    embedding_val = sample_action_copy["_source"]["embedding"]
                    if isinstance(embedding_val, list) and embedding_val:
                        sample_action_copy["_source"]["embedding"] = f"<embedding_vector_dim_{len(embedding_val)}>"
                    elif not embedding_val: 
                         sample_action_copy["_source"]["embedding"] = "<empty_embedding_vector>"
                    else: 
                        sample_action_copy["_source"]["embedding"] = f"<embedding_vector_unexpected_format: {type(embedding_val).__name__}>"
                logger.info(json.dumps(sample_action_copy, indent=2, default=str)) # Ensure json is imported if used here

            successes, errors = await async_bulk(es_client, actions_for_es, raise_on_error=False, raise_on_exception=False)
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
        else:
            logger.info(f"No chunks generated or processed for ingestion from '{original_file_name}'.")
            
    except Exception as e:
        logger.error(f"An error occurred during the example run for '{original_file_name}': {e}", exc_info=True)
    logger.info(f"--- Finished PDF Processing for: {original_file_name} ---\n")

def _generate_doc_id_from_content(content_bytes: bytes) -> str:
    """Generates a SHA256 hash for the given byte content."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content_bytes)
    return sha256_hash.hexdigest()

if __name__ == "__main__":
    import json # Added for sample document logging
    logging.getLogger("src.core.ingestion.chunking_pdf_processor").setLevel(logging.DEBUG)
    logging.getLogger("__main__").setLevel(logging.DEBUG) 
    logging.getLogger("elasticsearch").setLevel(logging.WARNING) 

    async def main_example():
        if not es_client:
            logger.error("Elasticsearch client failed to initialize in main. Exiting.")
            return
        if not aclient_openai and OPENAI_API_KEY: 
            logger.error("OpenAI client failed to initialize in main despite API key. Exiting.")
            return
        if not OPENAI_API_KEY: 
            logger.warning("OPENAI_API_KEY is not set. Enrichment, Embeddings, and Graph/Summary Extraction will be skipped or fail.")

        pdf_path_input = input("Enter the full path to your PDF file: ").strip()
        if not pdf_path_input:
            logger.error("No PDF path provided. Exiting.")
            return
        
        pdf_path = Path(pdf_path_input)
        if not pdf_path.is_file():
            logger.error(f"File not found at path: {pdf_path}. Please check the path and try again.")
            return

        original_file_name = pdf_path.name
        
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes_data = f.read()
            logger.info(f"Successfully read PDF file: {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to read PDF file '{pdf_path}': {e}")
            return

        # --- MODIFIED: Automatic doc_id generation ---
        generated_doc_id = _generate_doc_id_from_content(pdf_bytes_data)
        logger.info(f"Generated Document ID (SHA256 of content) for '{original_file_name}': {generated_doc_id}")
        # --- End MODIFIED ---

        user_provided_summary_input = input("Enter a brief summary for the document (for contextual prompts during processing, optional): ").strip()
        if not user_provided_summary_input:
            user_provided_summary_input = f"Content from {original_file_name}" 
            logger.info(f"No user-provided summary given, using placeholder: '{user_provided_summary_input}'")


        await example_run_pdf_processing(
            pdf_data=pdf_bytes_data, 
            original_file_name=original_file_name,
            document_id=generated_doc_id, # Use generated doc_id
            user_provided_doc_summary=user_provided_summary_input 
        )
        
        if es_client:
            await es_client.close()
            logger.info("Elasticsearch client closed.")
        if aclient_openai and hasattr(aclient_openai, "aclose"): 
             try:
                 await aclient_openai.aclose() 
                 logger.info("OpenAI client closed.")
             except Exception as e:
                 logger.warning(f"Error closing OpenAI client: {e}")

    asyncio.run(main_example())