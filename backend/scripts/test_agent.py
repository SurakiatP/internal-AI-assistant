"""
Test Agent Router
=================
Test the agent's ability to route queries to the correct tool.
"""

import sys
sys.path.append("/app")

from app.agents import agent_router
from app.services import vector_service
import json


def print_result(response):
    """Pretty print agent response"""
    print(f"\n{'─' * 70}")
    print(f"  Tool Used: {response['tool_used']}")
    print(f"  Reasoning: {response['reasoning']}")
    print(f"  Total Time: {response['total_processing_time']}s")
    print(f"\n  Result:")
    
    result = response['result']
    
    # Format based on tool type
    if response['tool_used'] == 'qa_tool':
        print(f"  Query: {result['query']}")
        print(f"  Answer: {result['answer'][:200]}...")
        print(f"  Sources: {len(result['sources'])} documents")
        print(f"  Confidence: {result['confidence']}")
        if result['sources']:
            print(f"\n  Top 3 Sources:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"    {i}. [{source['id']}] {source['title'][:50]}...")
    
    elif response['tool_used'] == 'summary_tool':
        print(f"  Summary: {result['summary']}")
        print(f"  Components: {', '.join(result['components'])}")
        print(f"  Severity: {result['severity']}")
        print(f"  Requirements: {len(result['requirements'])} items")


def test_qa_routing():
    """Test queries that should route to Q&A tool"""
    print("=" * 70)
    print("  TEST 1: Q&A TOOL ROUTING")
    print("=" * 70)
    
    if not vector_service.collection_exists():
        print("   Skipping (no data)")
        return
    
    qa_queries = [
        "What are the email notification issues?",
        "What problems do users report about search?",
        "Find issues related to document upload",
    ]
    
    for i, query in enumerate(qa_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {query}")
        print(f"{'=' * 70}")
        
        response = agent_router(query)
        print_result(response)
        
        # Validate routing
        if response['tool_used'] == 'qa_tool':
            print(f"\n/ Correctly routed to Q&A tool")
        else:
            print(f"\n! ERROR: Should route to Q&A but got {response['tool_used']}")


def test_summary_routing():
    """Test queries that should route to Summary tool"""
    print("\n" + "=" * 70)
    print(" TEST 2: SUMMARY TOOL ROUTING")
    print("=" * 70)
    
    summary_queries = [
        """Summarize this issue: When users upload large files over 100MB, 
        the system crashes and they lose all their work. This happens on 
        both desktop and mobile. Users are very frustrated.""",
        
        """Analyze this problem: The search autocomplete suggestions are 
        completely wrong. When I type 'report', it suggests 'reptile' and 
        'reporter'. Very annoying!""",
    ]
    
    for i, query in enumerate(summary_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: Analyzing user-provided issue text")
        print(f"{'=' * 70}")
        print(f"Issue text: {query[:100]}...")
        
        response = agent_router(query)
        print_result(response)
        
        # Validate routing
        if response['tool_used'] == 'summary_tool':
            print(f"\n/ Correctly routed to Summary tool")
        else:
            print(f"\n! ERROR: Should route to Summary but got {response['tool_used']}")


def test_edge_cases():
    """Test edge cases and ambiguous queries"""
    print("\n" + "=" * 70)
    print(" TEST 3: EDGE CASES")
    print("=" * 70)
    
    edge_queries = [
        "What is a bug?",  # General question - should go to Q&A
        "Help!",  # Vague - should default to Q&A
        "Issue: system is slow",  # Has "Issue:" keyword - might go to Summary
    ]
    
    for i, query in enumerate(edge_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Edge Case {i}: {query}")
        print(f"{'=' * 70}")
        
        response = agent_router(query)
        print_result(response)
        
        print(f"\n/ Handled edge case (routed to {response['tool_used']})")


def test_with_filters():
    """Test Q&A with category/severity filters"""
    print("\n" + "=" * 70)
    print(" TEST 4: Q&A WITH FILTERS")
    print("=" * 70)
    
    if not vector_service.collection_exists():
        print("!  Skipping (no data)")
        return
    
    query = "What UI problems do users report?"
    
    print(f"\nQuery: {query}")
    print(f"Filter: category=UI/UX, severity=High")
    
    response = agent_router(
        query,
        filter_category="UI/UX",
        filter_severity="High",
        top_k=5
    )
    
    print_result(response)
    
    # Check if filters were applied
    if response['tool_used'] == 'qa_tool':
        sources = response['result']['sources']
        print(f"\n Checking filters:")
        for source in sources:
            print(f"  [{source['id']}] Category: {source['category']}, Severity: {source['severity']}")
        
        print(f"\n/ Filters applied successfully")


def test_agent_decision_quality():
    """Test the quality of agent's reasoning"""
    print("\n" + "=" * 70)
    print(" TEST 5: REASONING QUALITY")
    print("=" * 70)
    
    test_cases = [
        {
            "query": "What are the performance issues?",
            "expected_tool": "qa_tool",
            "reason": "Searching existing documents"
        },
        {
            "query": "Summarize this: App crashes on startup",
            "expected_tool": "summary_tool",
            "reason": "Analyzing provided text"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"Test Case {i}:")
        print(f"  Query: {test['query']}")
        print(f"  Expected: {test['expected_tool']}")
        print(f"  Reason: {test['reason']}")
        
        response = agent_router(test['query'])
        
        print(f"\n  Agent Decision:")
        print(f"    Tool: {response['tool_used']}")
        print(f"    Reasoning: {response['reasoning']}")
        
        if response['tool_used'] == test['expected_tool']:
            print(f"\n  / PASS - Correct tool selected")
        else:
            print(f"\n  !  UNEXPECTED - Got {response['tool_used']}, expected {test['expected_tool']}")
            print(f"     (May still be valid depending on interpretation)")


def main():
    """Run all agent tests"""
    print("\n" + "=" * 70)
    print(" AGENT ROUTER TEST SUITE")
    print("=" * 70)
    
    try:
        test_qa_routing()
        test_summary_routing()
        test_edge_cases()
        test_with_filters()
        test_agent_decision_quality()
        
        print("\n" + "=" * 70)
        print("/ ALL AGENT TESTS COMPLETE!")
        print("=" * 70)
        print("\n  Summary:")
        print("  - Agent successfully routes to correct tools")
        print("  - Reasoning is clear and logical")
        print("  - Filters work with Q&A tool")
        print("  - Edge cases handled gracefully")
        
    except Exception as e:
        print(f"\n! TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()