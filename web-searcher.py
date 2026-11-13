from dotenv import load_dotenv
load_dotenv()
#import os

from typing import List
from pydantic import BaseModel, Field

from langchain.agents import create_agent
#from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
#from firecrawl import Firecrawl
from langchain_firecrawl import FirecrawlSearch, FirecrawlScrape

web_search = FirecrawlSearch()
read_webpage = FirecrawlScrape()

# fc = Firecrawl(api_key=os.environ.get("FIRECRAWL_API_KEY"))

# @tool
# def web_search(query: str) -> str:
#     """Search the web for current information and return titles, URLs, and snippets."""
#     results = fc.search(query, limit=3)
    
#     formatted_results = []
#     for item in results.web or []:
#         formatted_results.append(
#             f"Title: {item.title}\nURL: {item.url}\nSnippet: {item.description}"
#         )
#     return "\n\n".join(formatted_results) if formatted_results else "No results found."

# @tool
# def read_webpage(url: str) -> str:
#     """Scrape a specific webpage URL and return its content in clean Markdown."""
#     doc = fc.scrape(url, formats=["markdown"])
#     # Truncate very long pages to avoid exceeding your LLM's context window
#     return doc.markdown[:8000] if doc.markdown else "Could not extract page content."

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

llm = ChatOpenAI(temperature=0, model_name="gpt-5")
tools = [web_search, read_webpage]
search_agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    result = search_agent.invoke({"messages":HumanMessage(content="How's the weather in Gwacheon?")})
    print(result['structured_response'].answer)
    print(result['structured_response'].sources)

if __name__ == "__main__":
    main()