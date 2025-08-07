from agno.agent import Agent, RunResponse
from agno.utils.log import logger

from agno.workflow import Workflow
from instructions import Instructions
from utils import getModel, getSearchTool
from typing import Iterator
from agno.tools.firecrawl import FirecrawlTools
import json



class ShoppingTeam(Workflow):
    """
    A workflow that orchestrates a team of agents to assist users in shopping.
    """

    def __init__(
        self,
        api_key_llm: str,
        api_key_search_tool: str,
        search_tool: str,
        llm_mode: str,
        firecrawl_api_key: str,
    ):
        """
        Initializes the ShoppingTeam with agents for query enhancement, product link finding, scraping, and comparison.

        Args:
            api_key_llm (str): API key for the language model.
            api_key_search_tool (str): API key for the search tool (e.g., Tavily, SerpApi).
            search_tool (str): The search tool to use (e.g., "Tavily", "SerpApi").
            llm_mode (str): The mode of the language model (e.g., "Groq", "OpenAI").
            firecrawl_api_key (str): API key for Firecrawl.
        """
        super().__init__()

        # Agent to enhance product search queries
        self.site_finder: Agent = Agent(
            name="Product Search Query Enhancer",
            description="Refines and expands the product search query for finding relevant product links",
            instructions=Instructions.SITE_FINDER_INSTRUCTIONS,
            model=getModel(llm_mode, api_key_llm),
            debug_mode=False,
            add_datetime_to_instructions=True,
        )

        # Agent to perform web searches and collect product links
        self.researcher: Agent = Agent(
            name="Product Link Finder",
            description="Collects relevant product links based on the query",
            instructions=Instructions.RESEARCH_INSTRUCTIONS,
            tools=[
                getSearchTool(
                    search_tool=search_tool, api_key_search_tool=api_key_search_tool
                )
            ],
            model=getModel(llm_mode, api_key_llm),
            debug_mode=False,
            add_datetime_to_instructions=True,
        )

        # Agent to scrape product details from URLs
        self.scraping_agent: Agent = Agent(
            instructions=Instructions.SCRAPING_INSTRUCTIONS,
            tools=[FirecrawlTools(api_key=firecrawl_api_key)],
            show_tool_calls=False,
            model=getModel(llm_mode, api_key_llm),
            debug_mode=False,
            add_datetime_to_instructions=True,
        )

        # Agent to compare products and generate HTML
        self.product_comparison_agent: Agent = Agent(
            instructions=Instructions.PRODUCT_COMPARISON_INSTRUCTIONS,
            show_tool_calls=False,
            model=getModel(llm_mode, api_key_llm),
            debug_mode=False,
            add_datetime_to_instructions=True,
        )


    def run(self,payload:dict)-> RunResponse:
    
        logger.info("Starting Shopping Workflow...")
    
        payload_json_string = json.dumps(payload)
        enhanced_query_with_sites_response = self.site_finder.run(payload_json_string)
        enhanced_query_with_sites_response = (
            enhanced_query_with_sites_response.content
            if hasattr(enhanced_query_with_sites_response, "content")
            else str(enhanced_query_with_sites_response)
        )
       
        #Step 3: Use the researcher agent to search for product links
        search_results = self.researcher.run(enhanced_query_with_sites_response)
        logger.info(f"Search Results: {search_results.content}")

        # Step 4: Scrape product details from the links
        scraping_results = self.scraping_agent.run(search_results.content)
        logger.info(f"Scraping Results: {scraping_results.content}")

        # Step 5: Compare products and generate HTML
        comparison_result = self.product_comparison_agent.run(scraping_results.content)
        logger.info(f"Comparison Result: {comparison_result.content}")

        logger.info("Shopping Workflow Completed.")
        return comparison_result
