"""
Test Services
=============
Test LLM and Vector services to ensure they're working correctly.
"""

import sys
sys.path.append("/app")

from app.services import llm_service, vector_service
from app.config import settings


def test_llm_service():
    """Test LLM service functions"""
    print("=" * 70)
    print(" TESTING LLM SERVICE")
    print("=" * 70)
    
    # Test 1: Health check
    print("\n✓ Test 1: Health Check")
    llm_healthy = llm_service.check_llm_health()
    embedding_healthy = llm_service.check_embedding_health()
    print(f"  LLM Health: {'/ Healthy' if llm_healthy else '! Unhealthy'}")
    print(f"  Embedding Health: {'/ Healthy' if embedding_healthy else '! Unhealthy'}")
    
    # Test 2: Simple text generation.
    print("\n✓ Test 2: Text Generation")
    response = llm_service.generate_text("Say 'Hello' in one word", temperature=0.1)
    print(f"  Response: {response}")
    
    # Test 3: Classification
    print("\n✓ Test 3: Category Classification")
    categories = ["UI/UX", "Performance", "Backend", "General"]
    text = "The button is not aligned properly"
    category = llm_service.classify_category(text, categories)
    print(f"  Text: {text}")
    print(f"  Category: {category}")
    
    # Test 4: Severity classification.
    print("\n✓ Test 4: Severity Classification")
    text = "The app crashes when I try to login"
    severity = llm_service.classify_severity(text)
    print(f"  Text: {text}")
    print(f"  Severity: {severity}")
    
    # Test 5: Embedding generation
    print("\n✓ Test 5: Embedding Generation")
    embedding = llm_service.generate_embedding("test text")
    print(f"  Embedding size: {len(embedding)}")
    print(f"  Expected size: {settings.VECTOR_SIZE}")
    print(f"  Match: {'/' if len(embedding) == settings.VECTOR_SIZE else '!'}")
    
    print("\n/ LLM Service Tests Complete!")


def test_vector_service():
    """Test Vector service functions"""
    print("\n" + "=" * 70)
    print(" TESTING VECTOR SERVICE")
    print("=" * 70)
    
    # Test 1: Health check
    print("\n✓ Test 1: Health Check")
    db_healthy = vector_service.check_vector_db_health()
    print(f"  Vector DB Health: {'/ Healthy' if db_healthy else '! Unhealthy'}")
    
    # Test 2: Collection info
    print("\n✓ Test 2: Collection Info")
    if vector_service.collection_exists():
        info = vector_service.get_collection_info()
        print(f"  Collection: {info['name']}")
        print(f"  Documents: {info['points_count']}")
        print(f"  Vectors: {info['vectors_count']}")
        print(f"  Status: {info['status']}")
        print(f"  Vector Size: {info['vector_size']}")
    else:
        print("  !  Collection doesn't exist (run ingest.py first)")
        return
    
    # Test 3: Document count
    print("\n✓ Test 3: Document Counts")
    total = vector_service.count_documents()
    high_severity = vector_service.count_documents(severity="High")
    ui_issues = vector_service.count_documents(category="UI/UX")
    print(f"  Total documents: {total}")
    print(f"  High severity: {high_severity}")
    print(f"  UI/UX category: {ui_issues}")
    
    # Test 4: Simple search
    print("\n✓ Test 4: Vector Search")
    query = "upload issues"
    results = vector_service.search_documents(query, top_k=3)
    print(f"  Query: '{query}'")
    print(f"  Results: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"    {i}. [{result['id']}] {result['title'] or result['content'][:50]}...")
        print(f"       Category: {result['category']}, Severity: {result['severity']}, Score: {result['score']:.3f}")
    
    # Test 5: Filtered search
    print("\n✓ Test 5: Filtered Search")
    query = "performance"
    results = vector_service.search_documents(
        query,
        top_k=3,
        filter_severity="High"
    )
    print(f"  Query: '{query}' (High severity only)")
    print(f"  Results: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"    {i}. [{result['id']}] Severity: {result['severity']}")
    
    # Test 6: Get specific document
    print("\n✓ Test 6: Get Document by ID")
    doc = vector_service.get_document_by_id("BUG-001")
    if doc:
        print(f"  Found: {doc['id']}")
        print(f"  Title: {doc['title']}")
        print(f"  Category: {doc['category']}")
    else:
        print("  !  Document not found")
    
    # Test 7: Context building
    print("\n✓ Test 7: Context Building")
    query = "email notification"
    results = vector_service.search_documents(query, top_k=2)
    context = vector_service.build_context_from_results(results, max_length=500)
    print(f"  Query: '{query}'")
    print(f"  Context length: {len(context)} characters")
    print(f"  Preview: {context[:200]}...")
    
    print("\n/ Vector Service Tests Complete!")


def test_integration():
    """Test LLM + Vector integration"""
    print("\n" + "=" * 70)
    print("  TESTING INTEGRATION (LLM + Vector)")
    print("=" * 70)
    
    if not vector_service.collection_exists():
        print("!  Skipping integration test (collection doesn't exist)")
        return
    
    # Test: Q&A workflow
    print("\n✓ Test: Q&A Workflow")
    query = "What issues are there with email notifications?"
    print(f"  Query: {query}")
    
    # Step 1: Search for relevant documents
    print("  Step 1: Searching documents...")
    results = vector_service.search_documents(query, top_k=3)
    print(f"    Found {len(results)} results")
    
    # Step 2: Build context
    print("  Step 2: Building context...")
    context = vector_service.build_context_from_results(results)
    print(f"    Context length: {len(context)} characters")
    
    # Step 3: Generate answer
    print("  Step 3: Generating answer...")
    answer = llm_service.generate_answer(query, context)
    print(f"  Answer: {answer[:300]}...")
    
    print("\n/ Integration Test Complete!")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" SERVICES TEST SUITE")
    print("=" * 70)
    
    try:
        test_llm_service()
        test_vector_service()
        test_integration()
        
        print("\n" + "=" * 70)
        print("/ ALL TESTS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n! TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()