# OpenDeepSearch AI Service

A standalone AI service powered by OpenDeepSearch, FastAPI, and extensible tools for processing queries with advanced search capabilities.

## Features

- **FastAPI Backend**: RESTful API with endpoints for health checks, configuration, and query processing
- **Extensible Tools**: Modular design allowing addition of new tools via `CodeAgent` and `ToolCallingAgent`
- **Multi Mode Operation**:
  - Default: Basic search using OpenDeepSearch
  - Pro: Enhanced processing with all available tools
  - code: write code to get answer.
- **Environment Configuration**: Managed via `.env` file
- **Docker Support**: Containerized deployment ready

## Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- API Keys:
  - Serper API Key
  - OpenRouter API Key
  - Jina API Key

## Setup

### Local Development

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd opendeepsearch-ai-service
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   Create a `.env` file in the root directory with:
   ```
   SERPER_API_KEY=your-serper-api-key-here
   OPENROUTER_API_KEY=your-openrouter-api-key-here
   JINA_API_KEY=your-jina-api-key-here
   ```

4. **Run the Application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Deployment

1. **Build the Docker Image**
   ```bash
   docker build -t opendeepsearch-ai .
   ```

2. **Run the Container**
   Ensure your `.env` file is present in the current directory:
   ```bash
   docker run -d -p 8000:8000 --name opendeepsearch-ai opendeepsearch-ai
   ```

   Alternatively, pass environment variables directly:
   ```bash
   docker run -d -p 8000:8000      -e SERPER_API_KEY=your-serper-api-key-here      -e OPENROUTER_API_KEY=your-openrouter-api-key-here      -e JINA_API_KEY=your-jina-api-key-here      --name opendeepsearch-ai      opendeepsearch-ai
   ```

## Usage

### Endpoints

- **Health Check**
  ```
  GET /health
  ```
  Response:
  ```json
  {
    "status": "healthy",
    "date": "April 07, 2025",
    "active_tools": ["OpenDeepSearchTool"]
  }
  ```

- **Process Query**
  ```
  POST /query
  ```
  Request Body:
  ```json
  {
    "query": "How long would a cheetah run across Pont Alexandre III?",
    "mode": "pro"  // or "default" or "code"
  }
  ```
  Response:
  ```json
  {
    "response": "A cheetah at full speed (60-70 mph) would take approximately 2-3 seconds to cross Pont Alexandre III (160 meters)."
  }
  ```

- **Configuration**
  ```
  GET /config
  ```
  Response:
  ```json
  {
    "model": "openrouter/google/gemini-2.0-flash-001",
    "search_provider": "serper",
    "reranker": "jina",
    "version": "1.0.0",
    "active_tools": ["OpenDeepSearchTool"]
  }
  ```

### Adding New Tools

To extend functionality with new tools:
```python
from your_new_tool import NewTool
ai_service.add_tool(NewTool())
```

## Project Structure

```
opendeepsearch-ai-service/
├── main.py           # FastAPI application
├── Dockerfile        # Docker configuration
├── requirements.txt  # Python dependencies
├── .env             # Environment variables (not in version control)
├── .dockerignore    # Files to ignore in Docker build
└── README.md        # This file
```

## Dependencies

See `requirements.txt` for a full list. Key dependencies include:
- FastAPI: Web framework
- OpenDeepSearch: Search tool
- Smolagents: CodeAgent for tool management
- Python-dotenv: Environment variable management

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a Pull Request

## License

[MIT License](LICENSE) (Note: Add a LICENSE file if applicable)

## Troubleshooting

- **API Key Errors**: Ensure all required keys are set in `.env`
- **Docker Issues**: Verify port 8000 is available and Docker daemon is running
- **Dependency Conflicts**: Update `requirements.txt` versions as needed

For additional support, open an issue in the repository.