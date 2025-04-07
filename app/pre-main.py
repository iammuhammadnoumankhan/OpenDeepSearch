from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
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
    title="Advanced OpenDeepSearch AI Service",
    description="A comprehensive AI service leveraging OpenDeepSearch with full capabilities, including multiple search providers, reranking options, and agent integrations.",
    version="1.0.0"
)

# Configuration class
class Config:
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    SEARXNG_INSTANCE_URL = os.getenv("SEARXNG_INSTANCE_URL")
    SEARXNG_API_KEY = os.getenv("SEARXNG_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    WOLFRAM_ALPHA_APP_ID = os.getenv("WOLFRAM_ALPHA_APP_ID")
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
    DEFAULT_MODEL = os.getenv("LITELLM_MODEL_ID", "openrouter/google/gemini-2.0-flash-001")
    SEARCH_MODEL = os.getenv("LITELLM_SEARCH_MODEL_ID", "openrouter/google/gemini-2.0-flash-001")
    ORCHESTRATOR_MODEL = os.getenv("LITELLM_ORCHESTRATOR_MODEL_ID", "openrouter/google/gemini-2.0-flash-001")
    CURRENT_DATE = datetime.now().strftime("%B %d, %Y")

# Request models
class QueryRequest(BaseModel):
    query: str
    mode: Optional[Literal["default", "pro"]] = "default"
    search_provider: Optional[Literal["serper", "searxng"]] = "serper"
    reranker: Optional[Literal["jina", "infinity"]] = "jina"
    use_react: Optional[bool] = False  # Whether to use ReAct agent with Wolfram

class AgentConfigRequest(BaseModel):
    model_name: Optional[str] = Config.DEFAULT_MODEL
    search_provider: Optional[Literal["serper", "searxng"]] = "serper"
    reranker: Optional[Literal["jina", "infinity"]] = "jina"
    searxng_instance_url: Optional[str] = Config.SEARXNG_INSTANCE_URL
    searxng_api_key: Optional[str] = Config.SEARXNG_API_KEY

# AI Service class
class AIService:
    def __init__(self):
        self.search_agents = {}
        self.react_agent = None
        self.code_agent = None
        self._initialize_default_agents()

    def _initialize_default_agents(self):
        try:
            # Default OpenDeepSearch agent with Serper
            serper_agent = OpenDeepSearchTool(
                model_name=Config.SEARCH_MODEL,
                reranker="jina",
                search_provider="serper"
            )
            if not serper_agent.is_initialized:
                serper_agent.setup()
            self.search_agents["serper_jina"] = serper_agent

            # LiteLLM model for agents
            lite_model = LiteLLMModel(
                model_name=Config.ORCHESTRATOR_MODEL,
                temperature=0.7
            )

            # CodeAgent for code-related queries
            self.code_agent = CodeAgent(
                tools=[serper_agent],
                model=lite_model
            )

            # ReAct agent with Wolfram Alpha
            if Config.WOLFRAM_ALPHA_APP_ID:
                wolfram_tool = WolframAlphaTool(app_id=Config.WOLFRAM_ALPHA_APP_ID)
                self.react_agent = ToolCallingAgent(
                    tools=[serper_agent, wolfram_tool],
                    model=lite_model,
                    prompt_templates=REACT_PROMPT
                )

            logger.info("Default AI agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize default agents: {str(e)}")
            raise

    def get_search_agent(self, provider: str, reranker: str, searxng_url: str = None, searxng_key: str = None):
        key = f"{provider}_{reranker}"
        if key not in self.search_agents:
            try:
                params = {
                    "model_name": Config.SEARCH_MODEL,
                    "reranker": reranker,
                    "search_provider": provider
                }
                if provider == "searxng":
                    params["searxng_instance_url"] = searxng_url or Config.SEARXNG_INSTANCE_URL
                    params["searxng_api_key"] = searxng_key or Config.SEARXNG_API_KEY
                
                agent = OpenDeepSearchTool(**params)
                if not agent.is_initialized:
                    agent.setup()
                self.search_agents[key] = agent
            except Exception as e:
                logger.error(f"Failed to initialize search agent {key}: {str(e)}")
                raise
        return self.search_agents[key]

    async def process_query(self, query: str, mode: str, provider: str, reranker: str, use_react: bool, 
                          searxng_url: str = None, searxng_key: str = None) -> str:
        try:
            dated_query = f"Today is {Config.CURRENT_DATE}. {query}"
            search_agent = self.get_search_agent(provider, reranker, searxng_url, searxng_key)

            if use_react and self.react_agent:
                result = self.react_agent.run(dated_query)
            elif query.lower().startswith("write code") or "code" in query.lower():
                result = self.code_agent.run(dated_query)
            else:
                result = search_agent.forward(dated_query, deep_search=(mode == "pro"))
            
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

ai_service = AIService()

# Routes
# @app.get("/health")
@app.get("/health",
    summary="Check service health",
    description="Returns the operational status of the service along with the current date.",
    response_description="A JSON object containing the service status and current date."
)
async def health_check():
    return {"status": "healthy", "date": Config.CURRENT_DATE}

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        result = await ai_service.process_query(
            query=request.query,
            mode=request.mode,
            provider=request.search_provider,
            reranker=request.reranker,
            use_react=request.use_react
        )
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/configure-agent")
async def configure_agent(config: AgentConfigRequest):
    try:
        ai_service.get_search_agent(
            config.search_provider,
            config.reranker,
            config.searxng_instance_url,
            config.searxng_api_key
        )
        return {"message": f"Agent configured with {config.search_provider} and {config.reranker}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    return {
        "default_model": Config.DEFAULT_MODEL,
        "search_model": Config.SEARCH_MODEL,
        "orchestrator_model": Config.ORCHESTRATOR_MODEL,
        "available_providers": ["serper", "searxng"],
        "available_rerankers": ["jina", "infinity"],
        "react_available": bool(Config.WOLFRAM_ALPHA_APP_ID),
        "version": "1.0.0"
    }