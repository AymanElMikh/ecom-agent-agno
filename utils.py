from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.tools.tavily import TavilyTools
from agno.tools.serpapi import SerpApiTools
from agno.models.google import Gemini

def getSearchTool(search_tool,api_key_search_tool:str):
        if search_tool == 'Tavily':
            return TavilyTools(api_key=api_key_search_tool)
            
        if search_tool == 'SerpApi':
            return SerpApiTools(api_key=api_key_search_tool)
        
def getModel(llm_mode:str,api_key_llm:str):
    if llm_mode == 'OpenAI':
        return OpenAIChat(id="gpt-4o",api_key=api_key_llm)
        
    if llm_mode == 'Groq':
        return Groq(
            id="llama-3.3-70b-versatile",
            api_key=api_key_llm
        )
    
    if llm_mode == 'Gemini':
        return Gemini(
            id="gemini-2.0-flash",
            api_key=api_key_llm
        )
        
        