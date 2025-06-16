import logging
from typing import Any, Dict, Optional

# --- Base Tool Definition ---
# Since shared.abstractions.tool.Tool does not exist in the provided codebase,
# we define a base Tool class here for SearchFileKnowledgeTool to inherit from.
# In a larger system, this Tool base class would ideally live in a shared abstractions module.

class Tool:
    """
    Base class for tools that can be used by an agent.
    """
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        """
        Initializes the Tool.

        Args:
            name: The name of the tool.
            description: A description of what the tool does.
            parameters: A dictionary defining the parameters the tool accepts,
                        typically following an OpenAPI schema-like structure.
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.context: Optional[Any] = None

    def set_context(self, context: Any):
        """
        Sets a context for the tool, which might be the calling agent
        or any other relevant environment information.

        Args:
            context: The context to set.
        """
        self.context = context
        logger.debug(f"Context set for tool '{self.name}': {type(context).__name__ if context else 'None'}")

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Executes the tool's main functionality.
        This method should be overridden by subclasses.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"The 'execute' method must be implemented by subclasses of Tool (e.g., {self.__class__.__name__})."
        )

# --- End Base Tool Definition ---

from ...core.base.abstractions import AggregateSearchResult # Relative import

logger = logging.getLogger(__name__)


class SearchFileKnowledgeTool(Tool):
    """
    A tool to perform semantic and knowledge graph searches on the local knowledge base.
    It retrieves relevant text chunks and associated knowledge graph (KG) snippets.
    """

    def __init__(self):
        super().__init__(
            name="search_file_knowledge",
            description=(
                "Search your local knowledge base (text chunks and knowledge graph data) "
                "using the R2R system. Use this when you need to find relevant information, "
                "including text snippets and structured entity/relationship data, based on a query."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The user's query to search for in the local knowledge base. "
                            "E.g., 'details about Project Phoenix', 'information on quantum entanglement', "
                            "'who is related to John Doe?'."
                        ),
                    }
                },
                "required": ["query"],
            },
        )
        # self.context is inherited from the base Tool class

    # set_context method is inherited from the base Tool class

    async def execute(self, query: str, *args: Any, **kwargs: Any) -> AggregateSearchResult:
        """
        Calls the knowledge_search_method from the provided context (e.g., an agent)
        to search for chunks and KG data related to the query.

        Args:
            query: The search query string.

        Returns:
            An AggregateSearchResult object containing chunk and graph search results.
        """
        logger.info(f"Executing {self.name} with query: '{query}'")

        if not self.context:
            logger.error(f"No context provided for {self.name}. Cannot execute search.")
            return AggregateSearchResult(query=query) # Return empty result with query

        if not hasattr(self.context, "knowledge_search_method"):
            logger.error(
                f"'knowledge_search_method' not found in the context ({type(self.context).__name__}) for {self.name}."
            )
            return AggregateSearchResult(query=query) # Return empty result with query

        try:
            # Retrieve agent's configuration to pass to the search method if needed
            # The StaticResearchAgent.knowledge_search_method expects an agent_config parameter.
            agent_config_for_tool = getattr(self.context, 'config', {})
            if not agent_config_for_tool and hasattr(self.context, 'get_config'): # Fallback if config is a method
                agent_config_for_tool = self.context.get_config()


            # Call the agent's knowledge_search_method
            # This method is expected to return an AggregateSearchResult object.
            result_object: AggregateSearchResult = await self.context.knowledge_search_method(
                query=query,
                agent_config=agent_config_for_tool
            )

            logger.info(
                f"{self.name} executed successfully for query '{query}'. "
                f"Found {len(result_object.chunk_search_results)} chunk results and "
                f"{len(result_object.graph_search_results)} graph results."
            )

            # If the context (agent) has a results collector, add the structured result
            # This is optional and depends on the agent's design for internal state/logging.
            if hasattr(self.context, "search_results_collector") and \
               hasattr(self.context.search_results_collector, "add_aggregate_result"):
                try:
                    # Assuming add_aggregate_result can take the AggregateSearchResult directly
                    self.context.search_results_collector.add_aggregate_result(result_object)
                    logger.debug(f"Result for query '{query}' added to agent's search_results_collector.")
                except Exception as e_collector:
                    logger.warning(
                        f"Failed to add result to search_results_collector for query '{query}': {e_collector}",
                        exc_info=True
                    )
            
            return result_object

        except Exception as e:
            logger.error(
                f"Error during {self.name} execution for query '{query}': {e}",
                exc_info=True
            )
            # Return an empty AggregateSearchResult associated with the query in case of an error
            return AggregateSearchResult(query=query)