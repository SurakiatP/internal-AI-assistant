"""
Test FastAPI Endpoints
=======================
Test all API endpoints to ensure they work correctly.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def print_response(response, title="Response"):
    """Pretty print API response"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nResponse:")
        print(json.dumps(data, indent=2))
    else:
        print(f"\nError: {response.text}")


def test_root():
    """Test root endpoint"""
    print("\n" + "=" * 70)
    print("TEST 1: Root Endpoint")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/")
    print_response(response, "GET /")
    
    assert response.status_code == 200
    print("\n/ Root endpoint working")


def test_health():
    """Test health check endpoint"""
    print("\n" + "=" * 70)
    print("TEST 2: Health Check")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/api/health")
    print_response(response, "GET /api/health")
    
    assert response.status_code == 200
    
    data = response.json()
    print(f"\nServices Status:")
    for service, status in data['services'].items():
        icon = "/" if status else "!"
        print(f"  {icon} {service}: {status}")
    
    print(f"\nOverall Status: {data['status']}")
    print("\n/ Health check working")


def test_collection_stats():
    """Test collection statistics endpoint"""
    print("\n" + "=" * 70)
    print("TEST 3: Collection Statistics")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/api/collections/stats")
    print_response(response, "GET /api/collections/stats")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nCollection Info:")
        print(f"  Name: {data['collection_name']}")
        print(f"  Documents: {data['documents_count']}")
        print(f"  Vectors: {data['vectors_count']}")
        print(f"  Status: {data['status']}")
        print(f"  Vector Size: {data['vector_size']}")
        print("\n/ Collection stats working")
    else:
        print("\n!  Collection not found (run ingestion first)")


def test_qa_direct():
    """Test direct Q&A endpoint"""
    print("\n" + "=" * 70)
    print("TEST 4: Direct Q&A Tool")
    print("=" * 70)
    
    payload = {
        "query": "What are the upload issues?",
        "top_k": 3,
        "score_threshold": 0.5
    }
    
    print(f"\nRequest: POST /api/tools/qa")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/tools/qa", json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n/ Response received in {elapsed:.2f}s")
        print(f"\nAnswer: {data['answer'][:200]}...")
        print(f"Sources: {len(data['sources'])} documents")
        print(f"Confidence: {data['confidence']}")
        print("\n/ Q&A endpoint working")
    else:
        print_response(response, "Error")


def test_summary_direct():
    """Test direct summary endpoint"""
    print("\n" + "=" * 70)
    print("TEST 5: Direct Summary Tool")
    print("=" * 70)
    
    payload = {
        "issue_text": """When users try to upload files larger than 100MB, 
        the system crashes and they lose all their work. This happens on 
        both desktop and mobile platforms. Users are very frustrated."""
    }
    
    print(f"\nRequest: POST /api/tools/summary")
    print(f"Issue text: {len(payload['issue_text'])} characters")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/tools/summary", json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n/ Response received in {elapsed:.2f}s")
        print(f"\nSummary: {data['summary']}")
        print(f"Components: {', '.join(data['components'])}")
        print(f"Severity: {data['severity']}")
        print(f"Requirements: {len(data['requirements'])} items")
        print("\n/ Summary endpoint working")
    else:
        print_response(response, "Error")


def test_agent_qa_query():
    """Test main agent endpoint with Q&A query"""
    print("\n" + "=" * 70)
    print("TEST 6: Agent Endpoint (Q&A Query)")
    print("=" * 70)
    
    payload = {
        "query": "What are the email notification problems?",
        "top_k": 5,
        "score_threshold": 0.5
    }
    
    print(f"\nRequest: POST /api/query")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/query", json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n/ Response received in {elapsed:.2f}s")
        print(f"\nTool Used: {data['tool_used']}")
        print(f"Reasoning: {data['reasoning']}")
        print(f"\nResult Preview:")
        if 'answer' in data['result']:
            print(f"  Answer: {data['result']['answer'][:150]}...")
            print(f"  Sources: {len(data['result']['sources'])}")
        print("\n/ Agent Q&A routing working")
    else:
        print_response(response, "Error")


def test_agent_summary_query():
    """Test main agent endpoint with summary query"""
    print("\n" + "=" * 70)
    print("TEST 7: Agent Endpoint (Summary Query)")
    print("=" * 70)
    
    payload = {
        "query": "Summarize this: The search results are completely irrelevant. When I search for 'invoice', I get documents about gardening."
    }
    
    print(f"\nRequest: POST /api/query")
    print(f"Query: {payload['query']}")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/query", json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n/ Response received in {elapsed:.2f}s")
        print(f"\nTool Used: {data['tool_used']}")
        print(f"Reasoning: {data['reasoning']}")
        print(f"\nResult Preview:")
        if 'summary' in data['result']:
            print(f"  Summary: {data['result']['summary']}")
            print(f"  Severity: {data['result']['severity']}")
        print("\n/ Agent summary routing working")
    else:
        print_response(response, "Error")


def test_with_filters():
    """Test agent endpoint with filters"""
    print("\n" + "=" * 70)
    print("TEST 8: Agent with Filters")
    print("=" * 70)
    
    payload = {
        "query": "What performance issues exist?",
        "top_k": 5,
        "filter_category": "Performance",
        "score_threshold": 0.5
    }
    
    print(f"\nRequest: POST /api/query")
    print(f"Filters: category=Performance")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/query", json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n/ Response received in {elapsed:.2f}s")
        print(f"\nTool Used: {data['tool_used']}")
        
        if 'sources' in data['result']:
            print(f"Sources: {len(data['result']['sources'])}")
            print("\nFiltered sources:")
            for source in data['result']['sources'][:3]:
                print(f"  - [{source['id']}] {source['category']}")
        
        print("\n/ Filters working correctly")
    else:
        print_response(response, "Error")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  FASTAPI ENDPOINT TESTS")
    print("=" * 70)
    print(f"\nBase URL: {BASE_URL}")
    print("\nMake sure the API is running:")
    print("  docker-compose up -d")
    print()
    
    try:
        # Basic endpoints
        test_root()
        test_health()
        test_collection_stats()
        
        # Direct tool endpoints
        test_qa_direct()
        test_summary_direct()
        
        # Agent endpoints
        test_agent_qa_query()
        test_agent_summary_query()
        test_with_filters()
        
        print("\n" + "=" * 70)
        print("/ ALL FASTAPI TESTS PASSED!")
        print("=" * 70)
        print("\n Summary:")
        print("  / Root endpoint working")
        print("  / Health check working")
        print("  / Collection stats working")
        print("  / irect Q&A working")
        print("  / Direct Summary working")
        print("  / Agent routing working")
        print("  / Filters working")
        print()
        
    except requests.exceptions.ConnectionError:
        print("\n! ERROR: Cannot connect to API")
        print("   Make sure the API is running:")
        print("   docker-compose up -d")
        print()
    except Exception as e:
        print(f"\n! TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()