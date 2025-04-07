from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from opendeepsearch import OpenDeepSearchTool
from opendeepsearch.wolfram_tool import WolframAlphaTool
from opendeepsearch.prompts import REACT_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent, CodeAgent
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Standalone OpenDeepSearch AI Service",
    description="A fully standalone AI service leveraging all OpenDeepSearch capabilities, automatically selecting tools based on query content.",
    version="1.0.0"
)

# Configuration class
class Config:
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    SEARXNG_INSTANCE_URL = os.getenv("SEARXNG_INSTANCE_URL", "https://searxng.example.com")
    SEARXNG_API_KEY = os.getenv("SEARXNG_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    WOLFRAM_ALPHA_APP_ID = os.getenv("WOLFRAM_ALPHA_APP_ID")
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
    DEFAULT_MODEL = os.getenv("LITELLM_MODEL_ID", "openrouter/google/gemini-2.0-flash-001")
    SEARCH_MODEL = os.getenv("LITELLM_SEARCH_MODEL_ID", "openrouter/google/gemini-2.0-flash-001")
    ORCHESTRATOR_MODEL = os.getenv("LITELLM_ORCHESTRATOR_MODEL_ID", "openrouter/google/gemini-2.0-flash-001")
    CURRENT_DATE = datetime.now().strftime("%B %d, %Y")

# Request model
class QueryRequest(BaseModel):
    query: str

# Response model with metadata
class QueryResponse(BaseModel):
    response: str
    metadata: dict

# AI Service class
class AIService:
    def __init__(self):
        self.search_agents = {}
        self.react_agent = None
        self.code_agent = None
        self._initialize_all_tools()

    def _initialize_all_tools(self):
        try:
            # Initialize Serper-based search agent with Jina reranker
            serper_jina = OpenDeepSearchTool(
                model_name=Config.SEARCH_MODEL,
                reranker="jina",
                search_provider="serper"
            )
            if not serper_jina.is_initialized:
                serper_jina.setup()
            self.search_agents["serper_jina"] = serper_jina

            # Initialize SearXNG-based search agent with Infinity reranker (if configured)
            if Config.SEARXNG_INSTANCE_URL:
                searxng_infinity = OpenDeepSearchTool(
                    model_name=Config.SEARCH_MODEL,
                    reranker="infinity",
                    search_provider="searxng",
                    searxng_instance_url=Config.SEARXNG_INSTANCE_URL,
                    searxng_api_key=Config.SEARXNG_API_KEY
                )
                if not searxng_infinity.is_initialized:
                    searxng_infinity.setup()
                self.search_agents["searxng_infinity"] = searxng_infinity

            # LiteLLM model for orchestration
            lite_model = LiteLLMModel(
                model_name=Config.ORCHESTRATOR_MODEL,
                temperature=0.7
            )

            # CodeAgent for code-related tasks
            self.code_agent = CodeAgent(
                tools=[serper_jina],
                model=lite_model
            )

            # ReAct agent with Wolfram Alpha (if available)
            if Config.WOLFRAM_ALPHA_APP_ID:
                wolfram_tool = WolframAlphaTool(app_id=Config.WOLFRAM_ALPHA_APP_ID)
                self.react_agent = ToolCallingAgent(
                    tools=[serper_jina, wolfram_tool],
                    model=lite_model,
                    prompt_templates=REACT_PROMPT
                )

            logger.info("All AI tools initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tools: {str(e)}")
            raise

    async def process_query(self, query: str) -> tuple[str, dict]:
        try:
            dated_query = f"Today is {Config.CURRENT_DATE}. {query}"
            metadata = {"query": query, "date": Config.CURRENT_DATE}

            # Determine query type and select appropriate tools
            query_lower = query.lower()

            # Code-related query
            if "write code" in query_lower or "code" in query_lower:
                metadata["tool_used"] = "CodeAgent"
                metadata["search_provider"] = "serper"
                metadata["reranker"] = "jina"
                result = self.code_agent.run(dated_query)

            # Math or reasoning query (if Wolfram is available)
            elif (self.react_agent and 
                  any(keyword in query_lower for keyword in ["calculate", "distance", "how many", "what is the"])):
                metadata["tool_used"] = "ReActAgent"
                metadata["search_provider"] = "serper"
                metadata["reranker"] = "jina"
                metadata["additional_tool"] = "WolframAlpha"
                result = self.react_agent.run(dated_query)

            # Complex or multi-hop query (use SearXNG if available)
            elif ("compare" in query_lower or "and" in query_lower or "?" in query) and "searxng_infinity" in self.search_agents:
                metadata["tool_used"] = "OpenDeepSearch"
                metadata["search_provider"] = "searxng"
                metadata["reranker"] = "infinity"
                result = self.search_agents["searxng_infinity"].forward(dated_query)

            # Default simple search (using Serper with Jina)
            else:
                metadata["tool_used"] = "OpenDeepSearch"
                metadata["search_provider"] = "serper"
                metadata["reranker"] = "jina"
                result = self.search_agents["serper_jina"].forward(dated_query)

            return result, metadata
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

ai_service = AIService()

# Routes
@app.get("/health",
    summary="Check service health",
    description="Returns the operational status of the service along with the current date.",
    response_description="A JSON object containing the service status and current date."
)
async def health_check():
    return {"status": "healthy", "date": Config.CURRENT_DATE}

@app.post("/query",
    summary="Process a search query",
    description="Processes a user query using all available OpenDeepSearch tools, automatically selecting the best approach based on query content. Returns the response with metadata.",
    response_description="A JSON object containing the response and metadata about the tools used."
)
async def process_query(request: QueryRequest):
    """
    Parameters:
    - query: The search query or question to process (e.g., 'fastest animal on earth', 'write code to sort an array', 'distance between Paris and London').
    """
    try:
        result, metadata = await ai_service.process_query(request.query)
        return QueryResponse(response=result, metadata=metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config",
    summary="Get service configuration",
    description="Returns the current configuration details of the service, including available tools and models.",
    response_description="A JSON object containing configuration details."
)
async def get_config():
    return {
        "default_model": Config.DEFAULT_MODEL,
        "search_model": Config.SEARCH_MODEL,
        "orchestrator_model": Config.ORCHESTRATOR_MODEL,
        "available_tools": [
            "OpenDeepSearch (Serper + Jina)",
            "CodeAgent",
            "ReActAgent" if Config.WOLFRAM_ALPHA_APP_ID else None,
            "OpenDeepSearch (SearXNG + Infinity)" if Config.SEARXNG_INSTANCE_URL else None
        ],
        "version": "1.0.0"
    }