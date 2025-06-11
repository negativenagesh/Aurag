# src/core/ingestion/chunking_pdf_processor.py
import logging
import os
import asyncio
from io import BytesIO
from typing import AsyncGenerator, Dict, Any, List, Tuple
import yaml 
from pathlib import Path 
import copy 
import xml.etree.ElementTree as ET 

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

# --- Configuration from .env or defaults ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini") 
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_BASE_DIMENSION", 3072))

ELASTICSEARCH_INDEX_CHUNKS = "r2rtest2" 

CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE", 3072)) 
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP", 512))
FORWARD_CHUNKS = int(os.getenv("FORWARD_CHUNKS", 3))
BACKWARD_CHUNKS = int(os.getenv("BACKWARD_CHUNKS", 3))
CHARS_PER_TOKEN_ESTIMATE = 4 


# --- Initialize Clients ---
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in .env. OpenAI client will not be functional.")
    aclient_openai = None
else:
    aclient_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

try:
    es_client = AsyncElasticsearch(
        "https://my-elasticsearch-project-c44c4f.es.us-east-1.aws.elastic.cloud:443", 
        api_key="cWR0WFU1Y0I3YS1td2g2cWdpV186VTdnNFNVWmRmWVI5dnd3WEwwOWdMQQ==",
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
        processed_xml_string = xml_string # Keep a copy of what's being parsed for logging
        try:
            # Remove common LLM artifacts like markdown code block fences
            stripped_xml = xml_string.strip()
            if stripped_xml.startswith("```xml"):
                stripped_xml = stripped_xml[len("```xml"):].strip()
            elif stripped_xml.startswith("```"):
                stripped_xml = stripped_xml[len("```"):].strip()
            
            if stripped_xml.endswith("```"):
                stripped_xml = stripped_xml[:-len("```")].strip()

            # Ensure there's some content left
            if not stripped_xml:
                logger.warning("Empty XML string after stripping common artifacts.")
                return [], []

            # Find the first actual XML tag
            first_tag_start = stripped_xml.find("<")
            if first_tag_start == -1:
                logger.warning(f"No '<' found in XML string: {stripped_xml[:200]}...")
                return [], []
            
            processed_xml_string = stripped_xml[first_tag_start:]

            # Ensure it's wrapped in a single root element if not already
            if not processed_xml_string.strip().startswith("<root>"):
                processed_xml_string = f"<root>{processed_xml_string}</root>"
            
            # It's possible the LLM includes a </root> prematurely or an extra one.
            # This is harder to fix reliably without a more complex parser.
            # For now, we rely on ET.fromstring to catch structural errors.

            root = ET.fromstring(processed_xml_string)

            for entity_elem in root.findall(".//entity"): 
                name = entity_elem.get("name")
                ent_type_elem = entity_elem.find("type")
                ent_desc_elem = entity_elem.find("description")
                ent_type = ent_type_elem.text if ent_type_elem is not None else "Unknown"
                ent_desc = ent_desc_elem.text if ent_desc_elem is not None else ""
                if name:
                    entities.append({"name": name, "type": ent_type, "description": ent_desc})

            for rel_elem in root.findall(".//relationship"): 
                source_elem = rel_elem.find("source")
                target_elem = rel_elem.find("target")
                rel_type_elem = rel_elem.find("type")
                rel_desc_elem = rel_elem.find("description")
                rel_weight_elem = rel_elem.find("weight")
                source = source_elem.text if source_elem is not None else None
                target = target_elem.text if target_elem is not None else None
                rel_type = rel_type_elem.text if rel_type_elem is not None else "RELATED_TO"
                rel_desc = rel_desc_elem.text if rel_desc_elem is not None else ""
                weight = None
                if rel_weight_elem is not None and rel_weight_elem.text:
                    try:
                        weight = float(rel_weight_elem.text)
                    except ValueError:
                        logger.warning(f"Could not parse relationship weight '{rel_weight_elem.text}' as float.")
                if source and target:
                    relationships.append({
                        "source_entity": source, "target_entity": target, "relation": rel_type,
                        "relationship_description": rel_desc, "relationship_weight": weight
                    })
            logger.debug(f"Parsed {len(entities)} entities and {len(relationships)} relationships from XML.")
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML for graph data: {e}\nProcessed XML content was:\n{processed_xml_string}\nOriginal XML content from LLM (first 500 chars):\n{xml_string[:500]}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during XML parsing: {e}\nProcessed XML content was:\n{processed_xml_string}\nOriginal XML content from LLM (first 500 chars):\n{xml_string[:500]}")
        return entities, relationships

    async def _extract_knowledge_graph(
        self, chunk_text: str, document_summary: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not aclient_openai or not self.graph_extraction_prompt_template:
            logger.warning("OpenAI client or graph extraction prompt not available. Skipping graph extraction.")
            return [], []
        
        formatted_prompt = self.graph_extraction_prompt_template.format(
            document_summary=document_summary, input=chunk_text, entity_types=[], relation_types=[]
        )
        logger.debug(f"Formatted prompt for graph extraction (chunk-level, to {OPENAI_CHAT_MODEL}): ...")
        try:
            response = await aclient_openai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert assistant that extracts entities and relationships from text and formats them as XML."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.2, 
                max_tokens=15000
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
        document_summary: str,
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
            self._extract_knowledge_graph(chunk_text, document_summary)
        )
        enrich_task = asyncio.create_task(
            self._enrich_chunk_content(
                chunk_text, document_summary, preceding_texts, succeeding_texts
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
        self, data: str | bytes, file_name: str, doc_id: str, document_summary: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"Processing PDF: {file_name} (Doc ID: {doc_id}) with summary: '{document_summary[:100]}...'")
        
        try:
            if isinstance(data, str): 
                with open(data, 'rb') as f_in:
                    pdf_stream = BytesIO(f_in.read())
            else: 
                pdf_stream = BytesIO(data)
            
            pdf = self.pdf_reader(pdf_stream)
            logger.info(f"Loaded {len(pdf.pages)} pages from '{file_name}'.")

            pdf_pages_data: List[Tuple[int, str]] = []
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if not page_text or not page_text.strip():
                    logger.warning(f"Page {page_num} in '{file_name}' has no extractable text during initial scan.")
                    pdf_pages_data.append((page_num, "")) 
                    continue
                pdf_pages_data.append((page_num, page_text))

            if not any(pt for _, pt in pdf_pages_data if pt.strip()):
                logger.warning(f"No text extracted from any page in '{file_name}'. Aborting processing for this PDF.")
                return

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
                        document_summary=document_summary,
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
            if current_mapping.get(index_name, {}).get('mappings', {}).get('properties') != mappings_body.get('mappings', {}).get('properties'):
                 logger.warning(f"Elasticsearch index '{index_name}' exists but mapping differs. This might cause issues.")
                 logger.debug(f"Current mapping properties: {current_mapping.get(index_name, {}).get('mappings', {}).get('properties')}")
                 logger.debug(f"Expected mapping properties: {mappings_body.get('mappings', {}).get('properties')}")
            else:
                logger.info(f"Elasticsearch index '{index_name}' already exists and mapping properties are consistent.")
            return True
    except Exception as e:
        logger.error(f"Error with Elasticsearch index '{index_name}': {e}")
        return False

async def example_run_pdf_processing(pdf_data: str | bytes, original_file_name: str, document_id: str, doc_summary: str):
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
    
    logger.info(f"\n--- Starting PDF Processing for: {original_file_name} ---")
    try:
        async for action in processor.process_pdf(pdf_data, original_file_name, document_id, doc_summary):
            if action: 
                actions_for_es.append(action)

        if actions_for_es:
            logger.info(f"Collected {len(actions_for_es)} actions for bulk ingestion into '{ELASTICSEARCH_INDEX_CHUNKS}'.")
            
            if actions_for_es: 
                logger.info("Sample document to be indexed (first one, embedding vector omitted):")
                import json
                sample_action_copy = copy.deepcopy(actions_for_es[0]) 
                if "_source" in sample_action_copy and "embedding" in sample_action_copy["_source"]:
                    if isinstance(sample_action_copy["_source"]["embedding"], list) and sample_action_copy["_source"]["embedding"]:
                        sample_action_copy["_source"]["embedding"] = f"<embedding_vector_dim_{len(sample_action_copy['_source']['embedding'])}>"
                    elif not sample_action_copy["_source"]["embedding"]:
                         sample_action_copy["_source"]["embedding"] = "<empty_embedding_vector>"
                    else:
                        sample_action_copy["_source"]["embedding"] = f"<embedding_vector_unexpected_format: {type(sample_action_copy['_source']['embedding']).__name__}>"
                logger.info(json.dumps(sample_action_copy, indent=2, default=str))

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


if __name__ == "__main__":
    logging.getLogger("src.core.ingestion.chunking_pdf_processor").setLevel(logging.DEBUG)
    logging.getLogger("__main__").setLevel(logging.DEBUG) 

    async def main_example():
        if not es_client:
            logger.error("Elasticsearch client failed to initialize in main. Exiting.")
            return
        if not aclient_openai and OPENAI_API_KEY: 
            logger.error("OpenAI client failed to initialize in main despite API key. Exiting.")
            return
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set. Enrichment, Embeddings, and Graph Extraction will be skipped or fail.")

        pdf_path_input = input("Enter the full path to your PDF file: ").strip()
        if not pdf_path_input:
            logger.error("No PDF path provided. Exiting.")
            return
        
        pdf_path = Path(pdf_path_input)
        if not pdf_path.is_file():
            logger.error(f"File not found at path: {pdf_path}. Please check the path and try again.")
            return

        original_file_name = pdf_path.name
        
        document_id_input = input(f"Enter a document ID for '{original_file_name}' (e.g., my_doc_001): ").strip()
        if not document_id_input:
            document_id_input = original_file_name.rsplit('.', 1)[0].replace(" ", "_").lower() + "_id"
            logger.info(f"No document ID provided, using generated ID: {document_id_input}")

        doc_summary_input = input("Enter a brief summary for the document: ").strip()
        if not doc_summary_input:
            doc_summary_input = f"Summary for {original_file_name}" 
            logger.info(f"No summary provided, using placeholder: '{doc_summary_input}'")

        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes_data = f.read()
            logger.info(f"Successfully read PDF file: {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to read PDF file '{pdf_path}': {e}")
            return

        await example_run_pdf_processing(
            pdf_data=pdf_bytes_data, 
            original_file_name=original_file_name,
            document_id=document_id_input, 
            doc_summary=doc_summary_input 
        )
        
        if es_client:
            await es_client.close()
            logger.info("Elasticsearch client closed.")

    asyncio.run(main_example())