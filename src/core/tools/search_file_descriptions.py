import logging
from typing import Any, Dict, Optional

# --- Base Tool Definition ---
# Since shared.abstractions.tool.Tool does not exist in the provided codebase,
# we define a base Tool class here for SearchFileDescriptionsTool to inherit from.
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

# Use relative import to match project structure
from ...core.base.abstractions import AggregateSearchResult

logger = logging.getLogger(__name__)


class SearchFileDescriptionsTool(Tool):
    """
    A tool to search over high-level document data (titles, descriptions, etc.)
    """

    def __init__(self):
        super().__init__(
            name="search_file_descriptions",
            description=(
                "Semantic search over AI-generated summaries of stored documents. "
                "This does NOT retrieve chunk-level contents or knowledge-graph relationships. "
                "Use this when you need a broad overview of which documents (files) might be relevant."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query string to semantic search over available files 'list documents about XYZ'.",
                    }
                },
                "required": ["query"],
            },
        )
        # self.context is inherited from the base Tool class

    # set_context method is inherited from the base Tool class

    async def execute(self, query: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Calls the file_search_method from the provided context (e.g., an agent)
        to search for file descriptions related to the query.

        Args:
            query: The search query string.

        Returns:
            The search results dictionary from the file_search_method call.
        """
        logger.info(f"Executing {self.name} with query: '{query}'")

        if not self.context:
            logger.error(f"No context provided for {self.name}. Cannot execute search.")
            return {"error": "No context provided for search_file_descriptions"}

        if not hasattr(self.context, "file_search_method"):
            logger.error(
                f"'file_search_method' not found in the context ({type(self.context).__name__}) for {self.name}."
            )
            return {"error": f"'file_search_method' not available in {type(self.context).__name__}"}

        try:
            # Retrieve agent's configuration to pass to the search method
            # This matches how StaticResearchAgent.file_search_method expects parameters
            agent_config_for_tool = getattr(self.context, 'config', {})
            if not agent_config_for_tool and hasattr(self.context, 'get_config'): # Fallback if config is a method
                agent_config_for_tool = self.context.get_config()

            # Call the agent's file_search_method with proper parameters
            search_results = await self.context.file_search_method(
                query=query,
                agent_config=agent_config_for_tool
            )
            
            logger.info(f"{self.name} executed successfully for query '{query}'")
            
            # If the context (agent) has a results collector, add the result
            if hasattr(self.context, "search_results_collector") and \
               hasattr(self.context.search_results_collector, "add_aggregate_result"):
                try:
                    # The original code created an AggregateSearchResult here, but that's not needed
                    # since file_search_method returns the appropriate dictionary structure already
                    self.context.search_results_collector.add_aggregate_result(search_results)
                    logger.debug(f"Result for query '{query}' added to agent's search_results_collector.")
                except Exception as e_collector:
                    logger.warning(
                        f"Failed to add result to search_results_collector for query '{query}': {e_collector}",
                        exc_info=True
                    )
            
            return search_results

        except Exception as e:
            logger.error(
                f"Error during {self.name} execution for query '{query}': {e}",
                exc_info=True
            )
            return {"error": f"Failed to execute search: {str(e)}"}