import asyncio
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

try:
    from ..retrieval.rag_fusion_retriever import RAGFusionRetriever
    from ..tools.search_file_knowledge import SearchFileKnowledgeTool
    from ..tools.search_file_descriptions import SearchFileDescriptionsTool
    from ..tools.get_file_content import GetFileContentTool
    from ...utils.logging_config import setup_logger
    from ...core.base.abstractions import AggregateSearchResult, ChunkSearchResult, KGSearchResult, KGEntity, KGRelationship # MODIFIED: Import new models
except ImportError as e:
    print(f"ImportError in static_research_agent.py: {e}. Please ensure all dependencies are correctly placed and __init__.py files exist.")
    print("This agent expects RAGFusionRetriever, tools, and AggregateSearchResult to be importable from its location.")
    raise

load_dotenv()
logger = setup_logger(__name__)

DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_AGENT_CONFIG_PATH = Path(__file__).parent.parent / "prompts" / "static_research_agent.yaml"

class StaticResearchAgent:
    def __init__(
        self,
        llm_client: Optional[AsyncOpenAI] = None,
        retriever: Optional[RAGFusionRetriever] = None,
        config_path: Optional[Union[str, Path]] = None,
        llm_model: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ):
        if llm_client is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found and llm_client not provided.")
            self.llm_client = AsyncOpenAI(api_key=openai_api_key)
            logger.info("Initialized default AsyncOpenAI client for StaticResearchAgent.")
        else:
            self.llm_client = llm_client

        if retriever is None:
            self.retriever = RAGFusionRetriever()
            logger.info("Initialized default RAGFusionRetriever for StaticResearchAgent.")
        else:
            self.retriever = retriever
            
        self.config_path = Path(config_path) if config_path else DEFAULT_AGENT_CONFIG_PATH
        self.config = self._load_config()
        
        self.llm_model = llm_model or self.config.get("model", DEFAULT_LLM_MODEL)
        self.max_iterations = max_iterations or self.config.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        
        self.system_prompt_template = self._load_prompt_template_from_config("static_research_agent")

        # Initialize tools
        self.tools = {
            "search_file_knowledge": SearchFileKnowledgeTool(),
            "search_file_descriptions": SearchFileDescriptionsTool(),
            "get_file_content": GetFileContentTool(),
        }
        # Set context for tools if they need it (e.g., for calling agent's methods)
        for tool_instance in self.tools.values():
            if hasattr(tool_instance, 'set_context'):
                tool_instance.set_context(self)
        logger.info(f"StaticResearchAgent initialized with model: {self.llm_model}, max_iterations: {self.max_iterations}")
        logger.debug(f"Tools available: {list(self.tools.keys())}")


    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            if not isinstance(config_data, dict):
                logger.error(f"Config file {self.config_path} did not load as a dictionary.")
                return {}
            logger.info(f"Successfully loaded agent config from {self.config_path}")
            return config_data
        except FileNotFoundError:
            logger.error(f"Agent config file not found: {self.config_path}. Using empty config.")
            return {}
        except Exception as e:
            logger.error(f"Error loading agent config from {self.config_path}: {e}", exc_info=True)
            return {}

    def _load_prompt_template_from_config(self, prompt_key: str) -> str:
        prompt_details = self.config.get(prompt_key, {})
        if isinstance(prompt_details, dict) and "template" in prompt_details:
            logger.info(f"Successfully loaded prompt template for '{prompt_key}' from agent config.")
            return prompt_details["template"]
        else:
            logger.warning(f"Prompt template for '{prompt_key}' not found directly in agent config. Attempting to load from default prompts location.")
            try:
                prompt_file_path = Path(__file__).parent.parent / "prompts" / f"{prompt_key}.yaml"
                with open(prompt_file_path, 'r') as f:
                    data = yaml.safe_load(f)
                if data and prompt_key in data and "template" in data[prompt_key]:
                    logger.info(f"Successfully loaded prompt template for '{prompt_key}' from {prompt_file_path}.")
                    return data[prompt_key]["template"]
                else:
                    logger.error(f"Prompt template for '{prompt_key}' not found or invalid in {prompt_file_path}.")
                    raise ValueError(f"Invalid or missing prompt structure for {prompt_key}")
            except Exception as e:
                logger.error(f"Failed to load fallback prompt for {prompt_key}: {e}", exc_info=True)
                return "You are a helpful assistant. Answer the user's query based on the provided context. Today's date is {date}."


    def _format_search_results(self, search_results: Optional[Dict[str, Any]], subquery_limit: int = 2) -> str:
        if not search_results or "sub_queries_results" not in search_results:
            return "No initial search results found."

        formatted_texts = []
        sq_count = 0
        for i, sq_result in enumerate(search_results["sub_queries_results"]):
            if sq_count >= subquery_limit and subquery_limit > 0: 
                break
            
            sq_text = sq_result.get("sub_query_text", f"Sub-query {i+1}")
            formatted_texts.append(f"\n--- Results for Sub-query: \"{sq_text}\" ---")
            
            chunks = sq_result.get("reranked_chunks", [])
            kg_data = sq_result.get("retrieved_kg_data", []) 

            if chunks:
                formatted_texts.append("\nVector Search Results (Chunks):")
                for idx, chunk in enumerate(chunks[:3]):
                    doc_id = chunk.get('doc_id', 'unknown_doc')
                    page_num = chunk.get('page_number', 'N/A')
                    chunk_idx_page = chunk.get('chunk_index_in_page', idx)
                    source_id = f"c_{doc_id.replace('-', '')[:6]}_p{page_num}_i{chunk_idx_page}" 
                    
                    text_snippet = chunk.get('text', 'N/A')
                    if len(text_snippet) > 300: 
                        text_snippet = text_snippet[:297] + "..."
                    score_display = chunk.get('rerank_score', chunk.get('score')) # Prefer rerank_score
                    score_str = f"{score_display:.4f}" if score_display is not None else "N/A"
                    formatted_texts.append(f"Source ID [{source_id}]: {text_snippet} (Score: {score_str})")
            
            if kg_data:
                formatted_texts.append("\nKnowledge Graph Results:")
                for idx, kg_item in enumerate(kg_data[:2]): # Limit displayed KG items per subquery
                    doc_id = kg_item.get('doc_id', 'unknown_kg_doc')
                    page_num = kg_item.get('page_number', 'N/A')
                    chunk_idx_page = kg_item.get('chunk_index_in_page', idx)
                    source_id = f"kg_{doc_id.replace('-', '')[:6]}_p{page_num}_i{chunk_idx_page}"

                    entities_str = ", ".join([e.get('name', 'N/A') for e in kg_item.get('entities', [])[:3]]) 
                    if len(kg_item.get('entities', [])) > 3: entities_str += "..."
                    
                    rels_str = ", ".join([f"{r.get('source_entity','S')}->{r.get('relation','R')}->{r.get('target_entity','T')}" for r in kg_item.get('relationships', [])[:2]])
                    if len(kg_item.get('relationships', [])) > 2: rels_str += "..."
                    
                    chunk_text_snippet = kg_item.get('chunk_text', '')[:100] + "..." if kg_item.get('chunk_text') else "N/A"
                    formatted_texts.append(f"Source ID [{source_id}]: Entities: [{entities_str if entities_str else 'None'}]. Relationships: [{rels_str if rels_str else 'None'}]. (Associated chunk: {chunk_text_snippet})")
            
            if not chunks and not kg_data:
                 formatted_texts.append("No specific chunks or KG data found for this sub-query.")
            sq_count +=1

        return "\n".join(formatted_texts)

    def _parse_llm_tool_calls(self, response_content: str) -> List[Dict[str, Any]]:
        tool_calls = []
        try:
            tool_calls_match = re.search(r"<ToolCalls>(.*?)</ToolCalls>", response_content, re.DOTALL)
            if not tool_calls_match:
                return []

            tool_calls_xml_str = tool_calls_match.group(1)
            if not tool_calls_xml_str.strip().startswith("<root>"):
                 tool_calls_xml_str = f"<root>{tool_calls_xml_str}</root>"

            root = ET.fromstring(tool_calls_xml_str)
            for tool_call_elem in root.findall(".//ToolCall"):
                name_elem = tool_call_elem.find("Name")
                params_elem = tool_call_elem.find("Parameters")
                if name_elem is not None and name_elem.text and params_elem is not None and params_elem.text:
                    try:
                        parameters = json.loads(params_elem.text)
                        tool_calls.append({"tool_name": name_elem.text.strip(), "parameters": parameters})
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON parameters for tool {name_elem.text.strip()}: {params_elem.text}. Error: {e}")
        except ET.ParseError as e:
            logger.error(f"XML parsing error for tool calls: {e}. Content: {response_content[:500]}")
        except Exception as e:
            logger.error(f"Unexpected error parsing tool calls: {e}. Content: {response_content[:500]}", exc_info=True)
        return tool_calls

    async def _execute_tool_call(self, tool_name: str, parameters: Dict[str, Any], current_config: Dict[str, Any]) -> str:
        logger.info(f"Attempting to execute tool: {tool_name} with parameters: {parameters}")
        tool_instance = self.tools.get(tool_name)
        if not tool_instance:
            return f"Error: Tool '{tool_name}' not found."

        try:
            tool_output: Any

            if tool_name == "search_file_knowledge":
                query = parameters.get("query")
                if not query: return "Error: 'query' parameter missing for search_file_knowledge."
                
                tool_output_obj = await self.knowledge_search_method(query=query, agent_config=current_config) 
                
                formatted_output_parts = []
                if tool_output_obj.chunk_search_results:
                    formatted_output_parts.append("Chunk Results:")
                    max_display = current_config.get("tool_max_chunk_results_display", 2)
                    for idx, r in enumerate(tool_output_obj.chunk_search_results[:max_display]):
                        doc_id_str = r.doc_id.replace('-', '')[:6] if r.doc_id else 'unknown'
                        page_num_str = str(r.page_number) if r.page_number is not None else 'NA'
                        chunk_idx_str = str(r.chunk_index_in_page) if r.chunk_index_in_page is not None else str(idx)
                        source_id = f"c_{doc_id_str}_p{page_num_str}_i{chunk_idx_str}"
                        
                        text_snippet = r.text or 'N/A'
                        if len(text_snippet) > 200: text_snippet = text_snippet[:197] + "..."
                        
                        score_val = r.rerank_score if r.rerank_score is not None else r.score
                        score_disp = f"{score_val:.4f}" if score_val is not None else "N/A"
                        formatted_output_parts.append(f"  Source ID [{source_id}]: {text_snippet} (Score: {score_disp})")
                
                if tool_output_obj.graph_search_results:
                    formatted_output_parts.append("\nKnowledge Graph Results:")
                    max_display = current_config.get("tool_max_kg_results_display", 2)
                    for idx, kg_item in enumerate(tool_output_obj.graph_search_results[:max_display]):
                        doc_id_str = kg_item.doc_id.replace('-', '')[:6] if kg_item.doc_id else 'unknown'
                        page_num_str = str(kg_item.page_number) if kg_item.page_number is not None else 'NA'
                        chunk_idx_str = str(kg_item.chunk_index_in_page) if kg_item.chunk_index_in_page is not None else str(idx)
                        source_id = f"kg_{doc_id_str}_p{page_num_str}_i{chunk_idx_str}"
                        
                        entities_str = ", ".join([e.name for e in kg_item.entities[:2] if e.name])
                        if len(kg_item.entities) > 2: entities_str += "..."
                        
                        rels_str = ", ".join([f"{r.source_entity}->{r.relation}->{r.target_entity}" for r in kg_item.relationships[:1] if r.source_entity and r.relation and r.target_entity])
                        if len(kg_item.relationships) > 1: rels_str += "..."
                        
                        chunk_text_snippet = kg_item.chunk_text or ""
                        if len(chunk_text_snippet) > 100: chunk_text_snippet = chunk_text_snippet[:97] + "..."
                        formatted_output_parts.append(f"  Source ID [{source_id}]: Entities: [{entities_str or 'None'}]. Relationships: [{rels_str or 'None'}]. (Chunk: {chunk_text_snippet})")

                if not formatted_output_parts:
                    return "No relevant information found by search_file_knowledge."
                return "\n".join(formatted_output_parts)

            elif tool_name == "search_file_descriptions":
                query = parameters.get("query")
                if not query: return "Error: 'query' parameter missing for search_file_descriptions."
                logger.warning("search_file_descriptions tool is using general RAG search, not a dedicated description search.")
                search_results_dict = await self.file_search_method(query=query, agent_config=current_config)
                return self._format_search_results(search_results_dict, subquery_limit=1)


            elif tool_name == "get_file_content":
                doc_id = parameters.get("document_id")
                if not doc_id: return "Error: 'document_id' parameter missing for get_file_content."
                logger.warning(f"get_file_content tool is simulating full doc retrieval using RAG search for doc_id: {doc_id}.")
                content_results_dict = await self.content_method(filters={"id": {"$eq": doc_id}}, agent_config=current_config)
                # Format this dict.
                return self._format_search_results(content_results_dict, subquery_limit=1) # Assuming similar structure for now

            else: 
                if hasattr(tool_instance, 'execute'):
                    if asyncio.iscoroutinefunction(tool_instance.execute):
                        tool_output = await tool_instance.execute(**parameters)
                    else: # Synchronous execute
                        tool_output = tool_instance.execute(**parameters)
                else:
                    return f"Error: Tool '{tool_name}' has no execute method."
            
            # General string conversion for other tool outputs if they aren't AggregateSearchResult
            # or haven't been handled specifically above.
            if isinstance(tool_output, str):
                return tool_output # Already a string
            elif isinstance(tool_output, (dict, list)):
                try:
                    return json.dumps(tool_output, indent=2)
                except TypeError:
                    return str(tool_output) # Fallback if not JSON serializable
            else:
                return str(tool_output)

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return f"Error: Failed to execute tool {tool_name}. Details: {str(e)}"

    async def arun(self, query: str, agent_config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        current_config = {**self.config, **(agent_config_override or {})}
        current_llm_model = current_config.get("model", self.llm_model)
        current_max_iterations = current_config.get("max_iterations", self.max_iterations)
        initial_retrieval_num_sq = current_config.get("initial_retrieval_subqueries", 2)
        initial_retrieval_top_k_chunks = current_config.get("initial_retrieval_top_k_chunks", 3)
        initial_retrieval_top_k_kg = current_config.get("initial_retrieval_top_k_kg", 2)
        
        logger.info(f"StaticResearchAgent starting 'arun' for query: \"{query}\" with model {current_llm_model}, max_iter={current_max_iterations}")

        messages: List[Dict[str, Any]] = []
        
        # 1. Initial Retrieval (Optional, based on config)
        initial_context = "No initial search performed." # Default
        if current_config.get("perform_initial_retrieval", True):
            try:
                logger.info("Performing initial retrieval...")
                initial_search_results = await self.retriever.search(
                    user_query=query, 
                    num_subqueries=initial_retrieval_num_sq,
                    top_k_chunks=initial_retrieval_top_k_chunks, 
                    top_k_kg=initial_retrieval_top_k_kg
                )
                initial_context = self._format_search_results(initial_search_results, subquery_limit=initial_retrieval_num_sq)
                logger.debug(f"Initial context formatted (first 500 chars): {initial_context[:500]}")
            except Exception as e:
                logger.error(f"Error during initial retrieval: {e}", exc_info=True)
                initial_context = "Error during initial search."
        
        # 2. Prepare System Prompt
        # Assuming date is passed or obtained somehow; for now, a placeholder.
        current_date_str = current_config.get("current_date", "today") # Could be dynamic
        
        # Construct tool definitions string for the prompt
        tool_definitions_str = "<AvailableTools>\n"
        for tool_name, tool_instance in self.tools.items():
            tool_definitions_str += f"  <ToolDefinition>\n"
            tool_definitions_str += f"    <Name>{tool_instance.name}</Name>\n"
            tool_definitions_str += f"    <Description>{tool_instance.description}</Description>\n"
            # Parameters need to be in the specific XML format expected by the prompt
            # Assuming tool_instance.parameters is a JSON schema dict
            params_xml = "    <Parameters>\n"
            if tool_instance.parameters and "properties" in tool_instance.parameters:
                for param_name, param_details in tool_instance.parameters["properties"].items():
                    param_type = param_details.get("type", "string")
                    param_desc = param_details.get("description", "")
                    required = param_name in tool_instance.parameters.get("required", [])
                    params_xml += f'      <Parameter name="{param_name}" type="{param_type}" required="{str(required).lower()}">{param_desc}</Parameter>\n'
            params_xml += "    </Parameters>\n"
            tool_definitions_str += params_xml
            tool_definitions_str += f"  </ToolDefinition>\n"
        tool_definitions_str += "</AvailableTools>"

        system_prompt = self.system_prompt_template.format(
            date=current_date_str,
            initial_context=initial_context, # Pass initial search results into the system prompt
            tool_definitions=tool_definitions_str # Pass tool definitions
        )
        messages.append({"role": "system", "content": system_prompt})
        
        # 3. Add User Query
        messages.append({"role": "user", "content": query})

        # 4. Main Interaction Loop
        iterations_count = 0
        while iterations_count < current_max_iterations:
            iterations_count += 1
            logger.info(f"Agent Iteration {iterations_count}/{current_max_iterations}")
            
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"Messages to LLM:\n{json.dumps(messages, indent=2, default=str)}")

            try:
                llm_response = await self.llm_client.chat.completions.create(
                    model=current_llm_model,
                    messages=messages,
                    temperature=current_config.get("temperature", 0.2),
                    max_tokens=current_config.get("max_tokens_llm_response", 1500) 
                )
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}", exc_info=True)
                return {"answer": "Error: LLM API call failed.", "history": messages, "error": str(e)}

            assistant_message = llm_response.choices[0].message
            response_content = assistant_message.content if assistant_message.content else ""
            finish_reason = llm_response.choices[0].finish_reason
            
            logger.debug(f"LLM Raw Response Content:\n{response_content}")
            logger.debug(f"LLM Finish Reason: {finish_reason}")

            current_assistant_response_message = {"role": "assistant", "content": response_content}
            
            parsed_tool_calls = self._parse_llm_tool_calls(response_content)

            if parsed_tool_calls:
                logger.info(f"LLM requested tool calls: {parsed_tool_calls}")
                messages.append(current_assistant_response_message) 
                
                for tool_call in parsed_tool_calls:
                    tool_name = tool_call["tool_name"]
                    tool_params = tool_call["parameters"]
                    
                    tool_result_str = await self._execute_tool_call(tool_name, tool_params, current_config)
                    
                    messages.append({
                        "role": "tool", 
                        "name": tool_name, 
                        "content": tool_result_str 
                    })
                    logger.debug(f"Appended tool result for {tool_name}: {tool_result_str[:200]}...")
            else: 
                if finish_reason == "stop":
                    logger.info("LLM indicated 'stop' and no tool calls parsed. Returning final answer.")
                    messages.append(current_assistant_response_message)
                    return {"answer": response_content, "history": messages}
                elif response_content: 
                    logger.warning(f"LLM provided content but finish_reason was '{finish_reason}' and no tool calls. Assuming it's a partial/final answer.")
                    messages.append(current_assistant_response_message)
                    return {"answer": response_content, "history": messages, "warning": f"Finished due to {finish_reason}"}
                else: 
                    logger.error(f"LLM stopped for reason '{finish_reason}' without tool calls or content.")
                    messages.append(current_assistant_response_message) 
                    return {"answer": "Agent could not produce a final answer.", "error": f"LLM stopped unexpectedly: {finish_reason}", "history": messages}
        
        logger.warning(f"Max iterations ({current_max_iterations}) reached.")
        last_llm_content = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else "Max iterations reached without a conclusive answer."
        return {"answer": last_llm_content, "history": messages, "warning": "Max iterations reached"}

    async def knowledge_search_method(self, query: str, agent_config: Optional[Dict[str, Any]] = None) -> AggregateSearchResult:
        effective_config = agent_config or self.config # Use passed config or agent's default
        logger.debug(f"Agent's knowledge_search_method called with query: {query}")
        
        num_sq = effective_config.get("tool_num_subqueries", 1)
        top_k_c = effective_config.get("tool_top_k_chunks", 3)
        top_k_kg_val = effective_config.get("tool_top_k_kg", 2)

        raw_results_dict = await self.retriever.search(
            user_query=query, 
            num_subqueries=num_sq,
            top_k_chunks=top_k_c,
            top_k_kg=top_k_kg_val
        )

        agg_result = AggregateSearchResult(query=query)
        
        if raw_results_dict and "sub_queries_results" in raw_results_dict:
            for sq_res in raw_results_dict["sub_queries_results"]:
                for chunk_data in sq_res.get("reranked_chunks", []):
                    # Ensure all keys expected by ChunkSearchResult are present or have defaults
                    mapped_chunk_data = {
                        "text": chunk_data.get("text"),
                        "score": chunk_data.get("score"),
                        "rerank_score": chunk_data.get("rerank_score"),
                        "file_name": chunk_data.get("file_name"),
                        "doc_id": chunk_data.get("doc_id"),
                        "page_number": chunk_data.get("page_number"),
                        "chunk_index_in_page": chunk_data.get("chunk_index_in_page"),
                    }
                    agg_result.chunk_search_results.append(ChunkSearchResult(**mapped_chunk_data))
                
                for kg_data_item in sq_res.get("retrieved_kg_data", []):
                    entities = [KGEntity(**e_data) for e_data in kg_data_item.get("entities", [])]
                    relationships = [KGRelationship(**r_data) for r_data in kg_data_item.get("relationships", [])]
                    
                    mapped_kg_data = {
                        "chunk_text": kg_data_item.get("chunk_text"),
                        "entities": entities,
                        "relationships": relationships,
                        "score": kg_data_item.get("score"),
                        "rerank_score": kg_data_item.get("rerank_score"), # Though KG not typically reranked by score
                        "file_name": kg_data_item.get("file_name"),
                        "doc_id": kg_data_item.get("doc_id"),
                        "page_number": kg_data_item.get("page_number"),
                        "chunk_index_in_page": kg_data_item.get("chunk_index_in_page"),
                    }
                    agg_result.graph_search_results.append(KGSearchResult(**mapped_kg_data))
        return agg_result

    async def file_search_method(self, query: str, agent_config: Optional[Dict[str, Any]] = None):
        effective_config = agent_config or self.config
        logger.debug(f"Agent's file_search_method called with query: {query}")
        # This tool is for descriptions, so focus on text chunks, less on KG.
        results = await self.retriever.search(
            user_query=query, 
            num_subqueries=effective_config.get("tool_file_search_subqueries", 1), 
            top_k_chunks=effective_config.get("tool_file_search_top_k_chunks", 5), 
            top_k_kg=0 # No KG for file description search
        )
        return results 

    async def content_method(self, filters: Dict, agent_config: Optional[Dict[str, Any]] = None, options: Optional[Dict] = None):
        effective_config = agent_config or self.config
        logger.debug(f"Agent's content_method called with filters: {filters}")
        doc_id_filter = filters.get("id", {}).get("$eq") # Assuming filter format like {"id": {"$eq": "doc_id_value"}}
        if doc_id_filter:
            # Simulate by searching for this doc_id, get more chunks
            query = f"Retrieve all content for document ID {doc_id_filter}" 
            results = await self.retriever.search(
                user_query=query, 
                num_subqueries=1, # Focus on the specific doc
                top_k_chunks=effective_config.get("tool_content_top_k_chunks", 10), 
                top_k_kg=0 # Less emphasis on KG for "get content"
            )
            return results
        return {"error": "Document ID filter not correctly processed or missing in agent's content_method"}

    @property
    def search_settings(self): 
        # This might be used by tools if they expect a specific settings object.
        # For now, RAGFusionRetriever parameters are controlled via its method calls.
        return {
            "dummy_setting": "placeholder_value" 
            # Add actual relevant settings if tools evolve to use this
        }

async def main_research_agent_example():
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("src.core.agents.static_research_agent").setLevel(logging.DEBUG)
    logging.getLogger("src.core.retrieval.rag_fusion_retriever").setLevel(logging.INFO) # Keep retriever a bit quieter for agent logs

    try:
        agent = StaticResearchAgent()
        
        # Example query
        # query = "What are the recent advancements in quantum computing and their potential impact on cryptography?"
        query = input("Enter your research query for the StaticResearchAgent: ").strip()
        if not query:
            print("No query entered. Exiting.")
            return

        print(f"\n--- Running StaticResearchAgent for query: '{query}' ---")
        result = await agent.arun(query)
        
        print("\n--- Agent's Final Answer ---")
        print(result.get("answer", "No answer provided."))
        
        if result.get("warning"):
            print(f"\nWarning: {result['warning']}")
        if result.get("error"):
            print(f"\nError: {result['error']}")

        if logger.isEnabledFor(logging.DEBUG) and result.get("history"):
            print("\n--- Full Conversation History (Debug) ---")
            print(json.dumps(result["history"], indent=2, default=str))

    except ValueError as ve:
        logger.error(f"Initialization error for StaticResearchAgent: {ve}")
    except Exception as e:
        logger.error(f"An error occurred during the agent example run: {e}", exc_info=True)
    finally:
        # Clean up clients if they were created by the agent's default constructor
        if hasattr(agent, 'llm_client') and isinstance(agent.llm_client, AsyncOpenAI):
             if hasattr(agent.llm_client, "aclose"): await agent.llm_client.aclose()
        if hasattr(agent, 'retriever') and hasattr(agent.retriever, 'es_client') and agent.retriever.es_client:
             if hasattr(agent.retriever.es_client, "close"): await agent.retriever.es_client.close()
        # Note: RAGFusionRetriever's OpenAI client is shared (aclient_openai module global),
        # it should be closed at the application's very end if created globally.
        # If agent created its own retriever which created its own clients, they'd be closed above.

if __name__ == "__main__":
    asyncio.run(main_research_agent_example())
