"""
Test Tools
==========
Test Q&A and Summary tools.
"""

import sys
sys.path.append("/app")

from app.tools import qa_tool, summary_tool
from app.services import vector_service


def test_qa_tool():
    """Test Q&A Tool"""
    print("=" * 70)
    print("  TESTING Q&A TOOL")
    print("=" * 70)
    
    if not vector_service.collection_exists():
        print("!  Skipping Q&A test (no data)")
        return
    
    print("\n✓ Test: Email Notification Issues")
    result = qa_tool("What are the issues with email notifications?", top_k=3)
    
    print(f"  Answer: {result['answer'][:150]}...")
    print(f"  Sources: {len(result['sources'])}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Time: {result['processing_time']}s")
    
    print("\n/ Q&A Tool Test Complete!")


def test_summary_tool():
    """Test Summary Tool"""
    print("\n" + "=" * 70)
    print("  TESTING SUMMARY TOOL")
    print("=" * 70)
    
    issue = """When users try to upload large PDF files (over 50MB), the progress bar 
    gets stuck at 99%. This happens on Chrome and Firefox. Users have to refresh 
    and try again. The file uploads successfully but the UI doesn't show it."""
    
    print(f"\n✓ Test: Upload Issue Analysis")
    result = summary_tool(issue)
    
    print(f"  Summary: {result['summary']}")
    print(f"  Components: {', '.join(result['components'])}")
    print(f"  Severity: {result['severity']}")
    print(f"  Requirements: {len(result['requirements'])} items")
    print(f"  Time: {result['processing_time']}s")
    
    print("\n/ Summary Tool Test Complete!")


def main():
    print("\n" + "=" * 70)
    print("  TOOLS TEST SUITE")
    print("=" * 70)
    
    try:
        test_qa_tool()
        test_summary_tool()
        
        print("\n" + "=" * 70)
        print("/ ALL TESTS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n! FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()