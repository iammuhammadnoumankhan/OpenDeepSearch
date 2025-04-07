from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
from opendeepsearch import OpenDeepSearchTool
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent, Tool
from datetime import datetime
import logging
from opendeepsearch.prompts import REACT_PROMPT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenDeepSearch AI Service",
    description="Standalone AI service powered by OpenDeepSearch and extensible tools",
    version="1.0.0"
)

# Configuration class
class Config:
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    DEFAULT_MODEL = "openrouter/google/gemini-2.0-flash-001"
    CURRENT_DATE = datetime.now().strftime("%B %d, %Y")

# Request model
class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "default"  # "default" or "pro"

# AI Service class with extensible tools
class AIService:
    def __init__(self):
        try:
            # Initialize core search tool
            self.search_agent = OpenDeepSearchTool(
                model_name=Config.DEFAULT_MODEL,
                reranker="jina"
            )
            
            # Initialize language model
            self.model = LiteLLMModel(
                Config.DEFAULT_MODEL,
                temperature=0.2
            )
            
            # Initialize tools list (can be extended later)
            self.tools: List = [self.search_agent]
            
            # Initialize CodeAgent with tools
            self.code_agent = CodeAgent(tools=self.tools, model=self.model)

            # Initialize the React Agent with search and wolfram tools
            self.react_agent = ToolCallingAgent(
                tools=[self.search_agent],
                model=self.model,
                prompt_templates=REACT_PROMPT # Using REACT_PROMPT as system prompt
            )
            
            if not self.search_agent.is_initialized:
                self.search_agent.setup()
            
            logger.info("AI Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI Service: {str(e)}")
            raise

    def add_tool(self, tool):
        """Method to add new tools dynamically"""
        self.tools.append(tool)
        self.code_agent = CodeAgent(tools=self.tools, model=self.model)
        self.react_agent = ToolCallingAgent(
                tools=self.tools,
                model=self.model,
                prompt_templates=REACT_PROMPT # Using REACT_PROMPT as system prompt
            )
        logger.info(f"Added new tool: {tool.__class__.__name__}")

    async def process_query(self, query: str, mode: str) -> str:
        try:
            # Add current date context
            dated_query = f"Today is {Config.CURRENT_DATE}. {query}"
            
            if mode == "pro":
                # Pro mode uses CodeAgent with all tools
                result = self.react_agent.run(dated_query)
            elif mode == "code":
                # Pro mode uses CodeAgent with all tools
                result = self.code_agent.run(dated_query)
            else:
                # Default mode uses basic search
                result = self.search_agent.forward(dated_query)
                
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

# Initialize service
ai_service = AIService()


# # ----------------- add new tools ----------------------
# # Initialize the Wolfram Alpha tool
# from opendeepsearch.wolfram_tool import WolframAlphaTool
# wolfram_tool = WolframAlphaTool(app_id=os.environ["WOLFRAM_ALPHA_APP_ID"])
# ai_service.add_tool(wolfram_tool)


# Routes
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "date": Config.CURRENT_DATE,
        "active_tools": [tool.__class__.__name__ for tool in ai_service.tools]
    }

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        result = await ai_service.process_query(request.query, request.mode)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    return {
        "model": Config.DEFAULT_MODEL,
        "search_provider": "serper",
        "reranker": "jina",
        "version": "1.0.0",
        "active_tools": [tool.__class__.__name__ for tool in ai_service.tools]
    }

# Example of adding a new tool (uncomment to use)
# from some_new_tool import NewTool
# ai_service.add_tool(NewTool())

# Initialize the Wolfram Alpha tool
from opendeepsearch.wolfram_tool import WolframAlphaTool
wolfram_tool = WolframAlphaTool(app_id=os.environ["WOLFRAM_ALPHA_APP_ID"])
ai_service.add_tool(wolfram_tool)
