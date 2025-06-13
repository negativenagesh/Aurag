import asyncio
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

try:
    from ..retrieval.rag_fusion_retriever import RAGFusionRetriever
    from ..tools.search_file_knowledge import SearchFileKnowledgeTool
    from ..tools.search_file_descriptions import SearchFileDescriptionsTool
    from ..tools.get_file_content import GetFileContentTool
    from ...utils.logging_config import setup_logger
    from ...core.base.abstractions import AggregateSearchResult # Assuming this path
except ImportError as e:
    print(f"ImportError in static_research_agent.py: {e}. Please ensure all dependencies are correctly placed and __init__.py files exist.")
    print("This agent expects RAGFusionRetriever, tools, and AggregateSearchResult to be importable from its location.")
    raise

load_dotenv()
logger = setup_logger(__name__)

# --- Agent Configuration ---
DEFAULT_AGENT_MODEL = "gpt-4o-mini" # Cost-effective reasoning model
DEFAULT_MAX_ITERATIONS = 5
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Helper to get attributes from objects or keys from dicts
def get_attribute_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    elif isinstance(obj, dict) and key in obj:
        return obj[key]
    return default

class StaticResearchAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm_model = self.config.get("model", DEFAULT_AGENT_MODEL)
        self.max_iterations = self.config.get("max_iterations", DEFAULT_MAX_ITERATIONS)

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. The agent cannot function.")
        self.llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        self.retriever = RAGFusionRetriever()
        self.prompt_template_str = self._load_prompt_template("static_research_agent")

        self.tools: Dict[str, Any] = {
            "search_file_knowledge": SearchFileKnowledgeTool(),
            "search_file_descriptions": SearchFileDescriptionsTool(),
            "get_file_content": GetFileContentTool(),
        }
        # Set a simplified context for tools; they might need adaptation
        # for full functionality outside the R2R framework.
        for tool_instance in self.tools.values():
            tool_instance.context = self # Agent itself acts as a basic context

        self.tool_system_prompt = self._construct_tool_use_system_prompt()
        logger.info(f"StaticResearchAgent initialized with model {self.llm_model} and tools: {list(self.tools.keys())}")

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

    def _construct_format_tool_for_prompt(
        self, name: str, description: str, parameters_schema: Dict[str, Any]
    ) -> str:
        """Formats a single tool's schema into the XML string for the prompt."""
        param_xml_list = []
        if parameters_schema and "properties" in parameters_schema:
            required_params = parameters_schema.get("required", [])
            for param_name, param_details in parameters_schema["properties"].items():
                param_xml = f"<parameter>\n<name>{param_name}</name>\n"
                param_xml += f"<type>{param_details.get('type', 'string')}</type>\n"
                param_xml += f"<description>{param_details.get('description', '')}</description>\n"
                if param_name in required_params:
                    param_xml += f"<required>true</required>\n"
                param_xml += "</parameter>"
                param_xml_list.append(param_xml)

        return (
            "<tool_description>\n"
            f"<tool_name>{name}</tool_name>\n"
            f"<description>{description}</description>\n"
            "<parameters>\n" + "\n".join(param_xml_list) + "\n</parameters>\n"
            "</tool_description>"
        )

    def _construct_tool_use_system_prompt(self) -> str:
        tool_str_list = []
        for tool_name, tool_instance in self.tools.items():
            # Assuming tools have 'name', 'description', and 'parameters' (OpenAPI schema) attributes
            description = get_attribute_or_key(tool_instance, "description", f"Tool named {tool_name}")
            parameters_schema = get_attribute_or_key(tool_instance, "parameters", {})
            tool_str = self._construct_format_tool_for_prompt(tool_name, description, parameters_schema)
            tool_str_list.append(tool_str)

        return (
            "In this environment you have access to a set of tools you can use to answer the user's question.\n"
            "\n"
            "You may call them like this by including the following XML structure in your response:\n"
            "<function_calls>\n"
            "  <invoke>\n"
            "    <tool_name>$TOOL_NAME</tool_name>\n"
            "    <parameters>\n"
            "      <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n"
            "      <!-- ... more parameters ... -->\n"
            "    </parameters>\n"
            "  </invoke>\n"
            "  <!-- ... more invokes for parallel calls ... -->\n"
            "</function_calls>\n"
            "\n"
            "Here are the tools available:\n"
            "<tools>\n" + "\n".join(tool_str_list) + "\n</tools>"
        )

    def _format_search_results(self, search_results: Optional[Dict[str, Any]], subquery_limit: int = 2) -> str:
        if not search_results or "sub_queries_results" not in search_results:
            return "No initial search results found."

        formatted_texts = []
        sq_count = 0
        for i, sq_result in enumerate(search_results["sub_queries_results"]):
            if sq_count >= subquery_limit and subquery_limit > 0: # Limit displayed subquery results
                break
            
            sq_text = sq_result.get("sub_query_text", f"Sub-query {i+1}")
            formatted_texts.append(f"\n--- Results for Sub-query: \"{sq_text}\" ---")
            
            chunks = sq_result.get("reranked_chunks", [])
            kg_data = sq_result.get("retrieved_kg_data", []) # Key updated as per previous request

            if chunks:
                formatted_texts.append("\nVector Search Results (Chunks):")
                for idx, chunk in enumerate(chunks):
                    doc_id = chunk.get('doc_id', 'unknown_doc')
                    page_num = chunk.get('page_number', 'N/A')
                    chunk_idx = chunk.get('chunk_index_in_page', idx)
                    source_id = f"c_{doc_id}_p{page_num}_i{chunk_idx}" # Construct a unique-ish ID
                    
                    text_snippet = chunk.get('text', 'N/A')
                    if len(text_snippet) > 300: # Snippet for brevity
                        text_snippet = text_snippet[:297] + "..."
                    formatted_texts.append(f"Source ID [{source_id}]: {text_snippet}")
            
            if kg_data:
                formatted_texts.append("\nKnowledge Graph Results:")
                for idx, kg_item in enumerate(kg_data):
                    doc_id = kg_item.get('doc_id', 'unknown_kg_doc')
                    page_num = kg_item.get('page_number', 'N/A')
                    chunk_idx = kg_item.get('chunk_index_in_page', idx)
                    source_id = f"kg_{doc_id}_p{page_num}_i{chunk_idx}"

                    entities_str = ", ".join([e.get('name', 'N/A') for e in kg_item.get('entities', [])][:3]) # Show a few entities
                    if len(kg_item.get('entities', [])) > 3: entities_str += "..."
                    
                    rels_str = ", ".join([f"{r.get('source_entity','S')}->{r.get('relation','R')}->{r.get('target_entity','T')}" for r in kg_item.get('relationships', [])][:2]) # Show a few relationships
                    if len(kg_item.get('relationships', [])) > 2: rels_str += "..."

                    formatted_texts.append(f"Source ID [{source_id}]: Entities: [{entities_str if entities_str else 'None'}]. Relationships: [{rels_str if rels_str else 'None'}]. (Associated chunk: {kg_item.get('chunk_text', '')[:100]}...)")
            
            if not chunks and not kg_data:
                 formatted_texts.append("No specific chunks or KG data found for this sub-query.")
            sq_count +=1

        return "\n".join(formatted_texts)

    def _parse_llm_tool_calls(self, llm_response_content: str) -> Optional[List[Dict[str, Any]]]:
        """Parses XML tool calls from the LLM's response content."""
        tool_calls = []
        try:
            # Look for the <function_calls> block
            match = re.search(r"<function_calls>(.*?)</function_calls>", llm_response_content, re.DOTALL)
            if not match:
                return None
            
            function_calls_xml_str = match.group(1)
            # Wrap in a root element for robust parsing if it's just a sequence of <invoke>
            root = ET.fromstring(f"<root_wrapper>{function_calls_xml_str}</root_wrapper>") 
            
            for invoke_element in root.findall("invoke"):
                tool_name_element = invoke_element.find("tool_name")
                parameters_element = invoke_element.find("parameters")
                
                if tool_name_element is not None and tool_name_element.text:
                    tool_name = tool_name_element.text.strip()
                    params = {}
                    if parameters_element is not None:
                        for param_child in parameters_element:
                            # Tag name is $PARAMETER_NAME, text is $PARAMETER_VALUE
                            # Strip potential '$' if model includes it literally
                            param_key = param_child.tag.lstrip('$')
                            params[param_key] = param_child.text.strip() if param_child.text else ""
                    
                    tool_calls.append({"tool_name": tool_name, "parameters": params})
            
            return tool_calls if tool_calls else None
        except ET.ParseError as e:
            logger.error(f"XML parsing error for tool calls: {e}\nContent: {llm_response_content}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing tool calls: {e}\nContent: {llm_response_content}")
            return None

    async def _execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        logger.info(f"Attempting to execute tool: {tool_name} with parameters: {parameters}")
        tool_instance = self.tools.get(tool_name)
        if not tool_instance:
            return f"Error: Tool '{tool_name}' not found."

        try:
            # Simplified execution based on tool name
            # This part needs to be robust and aligned with how tools are designed/modified
            if tool_name == "search_file_knowledge":
                query = parameters.get("query")
                if not query: return "Error: 'query' parameter missing for search_file_knowledge."
                # The RAGFusionRetriever's search method is more comprehensive than a simple knowledge_search_method
                # We'll use it and format its output.
                search_results = await self.retriever.search(user_query=query, num_subqueries=1, top_k_chunks=3, top_k_kg=2)
                return self._format_search_results(search_results, subquery_limit=1) # Limit to 1 subquery for tool call
            
            elif tool_name == "search_file_descriptions":
                query = parameters.get("query")
                if not query: return "Error: 'query' parameter missing for search_file_descriptions."
                # As noted, RAGFusionRetriever doesn't have a dedicated description search.
                # We'll use its general search.
                logger.warning("search_file_descriptions tool is using general RAG search, not a dedicated description search.")
                search_results = await self.retriever.search(user_query=query, num_subqueries=1, top_k_chunks=3, top_k_kg=0) # Focus on chunks
                return self._format_search_results(search_results, subquery_limit=1)

            elif tool_name == "get_file_content":
                doc_id = parameters.get("document_id")
                if not doc_id: return "Error: 'document_id' parameter missing for get_file_content."
                # Simulate fetching content by querying RAG retriever for a specific doc_id
                # This won't be the "full document" unless the document is small and fully chunked.
                logger.warning(f"get_file_content tool is simulating full doc retrieval using RAG search for doc_id: {doc_id}.")
                # Construct a query that might hit the document_id if it's in the text or metadata
                query = f"Retrieve content for document ID {doc_id}"
                # A more direct way would be to filter Elasticsearch if doc_id is a metadata field.
                # For now, using the retriever's search:
                search_results = await self.retriever.search(user_query=query, num_subqueries=1, top_k_chunks=5, top_k_kg=0)
                
                # Filter results to only include those matching the doc_id (if retriever doesn't do exact match)
                filtered_chunks_text = []
                if search_results and "sub_queries_results" in search_results:
                    for sq_res in search_results["sub_queries_results"]:
                        for chunk in sq_res.get("reranked_chunks", []):
                            if chunk.get("doc_id") == doc_id:
                                filtered_chunks_text.append(chunk.get("text", ""))
                
                if filtered_chunks_text:
                    return f"Content chunks for document ID '{doc_id}':\n\n" + "\n\n---\n\n".join(filtered_chunks_text)
                else:
                    return f"No specific content chunks found directly matching document ID '{doc_id}' through general search."

            else:
                # Fallback if direct mapping isn't implemented or tool has its own execute
                if hasattr(tool_instance, 'execute') and asyncio.iscoroutinefunction(tool_instance.execute):
                    # Ensure parameters are passed as keyword arguments
                    tool_output = await tool_instance.execute(**parameters) 
                elif hasattr(tool_instance, 'execute'): # Synchronous execute
                     tool_output = tool_instance.execute(**parameters)
                else:
                    return f"Error: Tool '{tool_name}' has no execute method."

                if isinstance(tool_output, AggregateSearchResult):
                    # Basic formatting for AggregateSearchResult
                    # This might need to be more sophisticated depending on AggregateSearchResult structure
                    formatted_output_parts = []
                    if tool_output.chunk_search_results:
                        formatted_output_parts.append("Chunk Results:\n" + json.dumps([r.dict() for r in tool_output.chunk_search_results[:2]], indent=2)) # Show first 2
                    if tool_output.graph_search_results:
                         formatted_output_parts.append("Graph Results:\n" + json.dumps([r.dict() for r in tool_output.graph_search_results[:2]], indent=2)) # Show first 2
                    if tool_output.document_search_results: # For file description search
                        formatted_output_parts.append("Document Description Results:\n" + json.dumps([r.dict() for r in tool_output.document_search_results[:2]], indent=2))

                    return "\n".join(formatted_output_parts) if formatted_output_parts else "Tool executed, no specific results to format."
                return str(tool_output) # Default to string representation

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return f"Error during execution of tool {tool_name}: {str(e)}"

    async def arun(self, query: str, agent_config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        current_config = {**self.config, **(agent_config_override or {})}
        current_llm_model = current_config.get("model", self.llm_model)
        current_max_iterations = current_config.get("max_iterations", self.max_iterations)
        
        logger.info(f"StaticResearchAgent starting 'arun' for query: \"{query}\" with model {current_llm_model}, max_iter={current_max_iterations}")

        messages: List[Dict[str, Any]] = []
        
        # 1. Initial Retrieval
        try:
            initial_search_results = await self.retriever.search(
                user_query=query, 
                num_subqueries=2, # As per RAGFusionRetriever example
                top_k_chunks=3, 
                top_k_kg=2
            )
            formatted_initial_context = self._format_search_results(initial_search_results)
        except Exception as e:
            logger.error(f"Error during initial retrieval: {e}", exc_info=True)
            formatted_initial_context = "Error: Could not perform initial information retrieval."

        # 2. System Prompt Construction
        # The static_research_agent.yaml prompt is the main system message.
        # Tool definitions are appended to it.
        # Date formatting for the prompt
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        system_prompt_content = self.prompt_template_str.format(date=current_date) + "\n\n" + self.tool_system_prompt
        messages.append({"role": "system", "content": system_prompt_content})

        # 3. Add Initial Context as Assistant Message
        messages.append({"role": "assistant", "content": f"I have performed an initial search. Here's what I found:\n{formatted_initial_context}"})

        # 4. Add User Query
        messages.append({"role": "user", "content": query})

        # 5. Main Interaction Loop
        iterations_count = 0
        while iterations_count < current_max_iterations:
            iterations_count += 1
            logger.info(f"Agent Iteration {iterations_count}/{current_max_iterations}")
            
            if logger.isEnabledFor(logging.DEBUG): # Avoid expensive ops if not debugging
                 logger.debug(f"Messages to LLM:\n{json.dumps(messages, indent=2)}")

            try:
                llm_response = await self.llm_client.chat.completions.create(
                    model=current_llm_model,
                    messages=messages,
                    temperature=0.2, # Lower temperature for more factual/tool-driven responses
                    # max_tokens can be set if needed
                )
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}", exc_info=True)
                return {"answer": "Error: LLM API call failed.", "history": messages, "error": str(e)}

            assistant_message = llm_response.choices[0].message
            response_content = assistant_message.content if assistant_message.content else ""
            finish_reason = llm_response.choices[0].finish_reason
            
            logger.debug(f"LLM Raw Response Content:\n{response_content}")
            logger.debug(f"LLM Finish Reason: {finish_reason}")

            # Append LLM's response (which might contain tool calls or final answer)
            # If it contains tool calls, we'll add it as assistant's turn before tool results.
            # If it's a final answer, this is the message we'll return.
            current_assistant_response_message = {"role": "assistant", "content": response_content}
            
            # Check for tool calls in the response content (XML parsing)
            parsed_tool_calls = self._parse_llm_tool_calls(response_content)

            if parsed_tool_calls:
                logger.info(f"LLM requested tool calls: {parsed_tool_calls}")
                messages.append(current_assistant_response_message) # Add the message that contained the tool call
                
                all_tool_results_appended = True
                for tool_call in parsed_tool_calls:
                    tool_name = tool_call["tool_name"]
                    tool_params = tool_call["parameters"]
                    
                    # Using a generic tool_call_id as we are not using OpenAI's native tool_calls array
                    # This ID is mostly for structuring the conversation history.
                    tool_call_id_for_history = f"tool_call_{iterations_count}_{tool_name}" 

                    tool_result_str = await self._execute_tool_call(tool_name, tool_params)
                    
                    # Append tool result message
                    # Note: OpenAI's native tool usage expects `tool_call_id` and `name` in the tool message.
                    # Since we parse from content, we simulate this structure.
                    messages.append({
                        "role": "tool", 
                        # "tool_call_id": tool_call_id_for_history, # Not strictly needed if LLM doesn't use it
                        "name": tool_name, # Function name
                        "content": tool_result_str
                    })
                    logger.debug(f"Appended tool result for {tool_name}: {tool_result_str[:200]}...")
                
                # If finish_reason was 'stop' but there were tool calls, we continue to process tools.
                # If finish_reason was specific to tool usage (like 'tool_calls' if OpenAI ever uses that for content-parsed tools),
                # we also continue.
                # The primary driver is whether parsed_tool_calls is not empty.
                # Loop will continue to get LLM's next thought after tool execution.

            else: # No tool calls parsed
                if finish_reason == "stop":
                    logger.info("LLM indicated 'stop' and no tool calls parsed. Returning final answer.")
                    # The current_assistant_response_message is the final answer
                    messages.append(current_assistant_response_message) # Ensure final answer is in history
                    return {"answer": response_content, "history": messages}
                elif response_content: # LLM provided content, no tool calls, but finish_reason wasn't 'stop' (e.g. 'length')
                    logger.warning(f"LLM provided content but finish_reason was '{finish_reason}' and no tool calls. Assuming it's a partial/final answer.")
                    messages.append(current_assistant_response_message)
                    return {"answer": response_content, "history": messages, "warning": f"Finished due to {finish_reason}"}
                else: # No content and no tool calls, unusual state
                    logger.error(f"LLM stopped for reason '{finish_reason}' without tool calls or content.")
                    messages.append(current_assistant_response_message) # Log what was (not) said
                    return {"answer": "Agent could not produce a final answer.", "error": f"LLM stopped unexpectedly: {finish_reason}", "history": messages}
        
        logger.warning(f"Max iterations ({current_max_iterations}) reached.")
        # Return the last piece of content the LLM generated, if any, or an error.
        last_llm_content = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else "Max iterations reached without a conclusive answer."
        return {"answer": last_llm_content, "history": messages, "warning": "Max iterations reached"}

    # --- Methods for tools to call if agent is their context ---
    # These are simplified proxies or would need full implementation if tools strictly
    # rely on the R2R app-like context methods.
    async def knowledge_search_method(self, query: str, settings: Optional[Any] = None):
        # This is what SearchFileKnowledgeTool expects.
        # We map it to the retriever's search.
        logger.debug(f"Agent's knowledge_search_method called with query: {query}")
        # `settings` from tool context is not directly used here, RAGFusionRetriever has its own params.
        results = await self.retriever.search(user_query=query, num_subqueries=1, top_k_chunks=3, top_k_kg=2)
        # The tool expects AggregateSearchResult or a dict. RAGFusionRetriever returns a dict.
        # We need to map it or ensure the tool can handle this dict.
        # For now, returning the dict and the tool's execute method will format it.
        return results # This will be a dict like RAGFusionRetriever.search output

    async def file_search_method(self, query: str, settings: Optional[Any] = None):
        # For SearchFileDescriptionsTool
        logger.debug(f"Agent's file_search_method called with query: {query}")
        results = await self.retriever.search(user_query=query, num_subqueries=1, top_k_chunks=3, top_k_kg=0) # Focus on text
        # This also returns the RAGFusionRetriever's dict format.
        # The tool expects a list of DocumentFragment. This is a mismatch.
        # The _execute_tool_call handles formatting for now.
        return results 

    async def content_method(self, filters: Dict, options: Optional[Dict] = None):
        # For GetFileContentTool
        logger.debug(f"Agent's content_method called with filters: {filters}")
        doc_id_filter = filters.get("id", {}).get("$eq")
        if doc_id_filter:
            # Simulate by searching for this doc_id
            query = f"Retrieve all content for document ID {doc_id_filter}"
            results = await self.retriever.search(user_query=query, num_subqueries=1, top_k_chunks=10, top_k_kg=0) # Get more chunks
            # This needs significant adaptation if the tool expects full Document objects.
            # _execute_tool_call handles formatting.
            return results
        return {"error": "Document ID filter not correctly processed by agent's content_method"}

    @property
    def search_settings(self): # Property tools might look for in context
        return {} # Placeholder


async def main_example_agent_run():
    # Ensure root logger is configured if no handlers are present
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set specific logger levels
    logging.getLogger("src.core.agents.static_research_agent").setLevel(logging.DEBUG)
    logging.getLogger("src.core.retrieval.rag_fusion_retriever").setLevel(logging.INFO) # Keep retriever less verbose unless debugging it

    try:
        agent = StaticResearchAgent()
    except Exception as e:
        logger.critical(f"Failed to initialize StaticResearchAgent: {e}", exc_info=True)
        return

    user_query = input("Enter your research query: ").strip()
    if not user_query:
        logger.warning("No query entered. Exiting.")
        return

    logger.info(f"\n--- Running Static Research Agent for: '{user_query}' ---")
    
    results = await agent.arun(user_query)
    
    print("\n\n--- Agent Final Response ---")
    print(results.get("answer", "No answer provided."))
    
    if results.get("warning"):
        print(f"\nWarning: {results.get('warning')}")
    if results.get("error"):
        print(f"\nError: {results.get('error')}")

    # Optionally print full history for debugging
    # print("\n--- Full Conversation History ---")
    # print(json.dumps(results.get("history", []), indent=2))

    # Clean up (if clients were created in agent, not module-level)
    # Since RAGFusionRetriever uses module-level clients, they are closed by its own __main__ or pipeline_coordinator
    # The agent's own OpenAI client is instance-level.
    if agent.llm_client and hasattr(agent.llm_client, "close"): # AsyncOpenAI client doesn't have close, it uses atexit
        pass # await agent.llm_client.close() 


if __name__ == "__main__":
    asyncio.run(main_example_agent_run())
