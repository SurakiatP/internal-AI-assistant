"""
Test Score Thresholds
=====================
Compare results with different score thresholds.
"""

import sys
sys.path.append("/app")

from app.tools import qa_tool


def test_thresholds():
    """Test different score thresholds"""
    print("=" * 70)
    print(" TESTING SCORE THRESHOLDS")
    print("=" * 70)
    
    query = "What are the upload issues?"
    thresholds = [0.0, 0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\n{'=' * 70}")
        print(f"Threshold: {threshold}")
        print(f"{'=' * 70}")
        
        result = qa_tool(query, top_k=10, score_threshold=threshold)
        
        print(f"\nResults: {len(result['sources'])} documents")
        print(f"Confidence: {result['confidence']}")
        
        if result['sources']:
            print("\nTop 3 Sources:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"  {i}. [{source['id']}] Score: {source['relevance_score']}")
                print(f"     {source['title'][:60]}...")
        else:
            print("  No results found!")
        
        print(f"\nAnswer preview:")
        print(f"  {result['answer'][:150]}...")


if __name__ == "__main__":
    test_thresholds()