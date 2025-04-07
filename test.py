import requests
import time
import json
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
TEST_QUERY = "How long would a cheetah at full speed take to run the length of Pont Alexandre III?"
MODES = ["default", "code", "pro"]  # Added "code" as a third mode

def test_health_endpoint() -> None:
    """Test the /health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        assert data["status"] == "healthy", "Health check failed"
        assert "date" in data, "Date missing in health response"
        assert "active_tools" in data, "Active tools missing in health response"
        
        print("Health endpoint test: PASSED")
    except Exception as e:
        print(f"Health endpoint test: FAILED - {str(e)}")

def test_config_endpoint() -> None:
    """Test the /config endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/config")
        response.raise_for_status()
        
        data = response.json()
        assert "model" in data, "Model missing in config response"
        assert "search_provider" in data, "Search provider missing in config response"
        assert "version" in data, "Version missing in config response"
        
        print("Config endpoint test: PASSED")
    except Exception as e:
        print(f"Config endpoint test: FAILED - {str(e)}")

def test_query_endpoint(mode: str) -> None:
    """Test the /query endpoint with a specific mode"""
    try:
        payload = {
            "query": TEST_QUERY,
            "mode": mode
        }
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        
        data = response.json()
        elapsed_time = time.time() - start_time
        
        assert "response" in data, f"Response missing in {mode} mode"
        assert isinstance(data["response"], str), f"Response not a string in {mode} mode"
        assert len(data["response"]) > 0, f"Empty response in {mode} mode"
        
        print(f"Query test ({mode} mode): PASSED")
        print(f"Response time: {elapsed_time:.2f} seconds")
        print(f"Response: {data['response'][:100]}{'...' if len(data['response']) > 100 else ''}")
    except Exception as e:
        print(f"Query test ({mode} mode): FAILED - {str(e)}")

def run_tests() -> None:
    """Run all tests"""
    print("Starting AI Service Tests...\n")
    
    # Test health endpoint
    test_health_endpoint()
    print("-" * 50)
    
    # Test config endpoint
    test_config_endpoint()
    print("-" * 50)
    
    # Test query endpoint for each mode
    for mode in MODES:
        test_query_endpoint(mode)
        print("-" * 50)
    
    print("Tests completed!")

if __name__ == "__main__":
    # Note: Ensure your FastAPI server is running before executing tests
    run_tests()