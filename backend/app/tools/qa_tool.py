"""
Q&A Tool
========
Internal Q&A tool that searches documents and answers questions.

Uses:
- Vector search (Qdrant)
- Context building
- LLM answer generation
"""

import time
from typing import Dict, Any, List

from app.config import settings
from app.services.llm_service import generate_with_template
from app.services.vector_service import (
    search_documents,
    build_context_from_results
)


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

QA_PROMPT_TEMPLATE = """You are an internal AI assistant helping the product and engineering team.

Answer the following question based ONLY on the provided context from internal documents.
Be specific and cite relevant information when possible.
If the answer is not in the context, say "I don't have enough information to answer that question based on the available documents."

Context from internal documents:
{context}

Question: {question}

Provide a clear, concise answer:"""


# ============================================================================
# Q&A TOOL IMPLEMENTATION
# ============================================================================

def qa_tool(
    query: str,
    top_k: int = 10,
    filter_category: str = None,
    filter_severity: str = None,
    score_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Q&A Tool: Answer questions using internal documents.
    
    Process:
    1. Search vector database for relevant documents
    2. Filter by score threshold (quality control)
    3. Build context from search results
    4. Generate answer using LLM
    5. Return structured JSON response
    
    Args:
        query: User question
        top_k: Maximum number of documents to retrieve
        filter_category: Optional category filter
        filter_severity: Optional severity filter
        score_threshold: Minimum relevance score (0.0-1.0, default: 0.5)
    
    Returns:
        Structured Q&A response with answer and sources
    """
    start_time = time.time()
    
    try:
        # Step 1: Search for relevant documents
        print(f"   Searching for: '{query}'")
        print(f"   Filters: category={filter_category}, severity={filter_severity}")
        print(f"   Params: top_k={top_k}, score_threshold={score_threshold}")
        
        search_results = search_documents(
            query=query,
            top_k=top_k,
            filter_category=filter_category,
            filter_severity=filter_severity,
            score_threshold=score_threshold
        )
        
        print(f"   Found {len(search_results)} documents (score >= {score_threshold})")
        
        if not search_results:
            return {
                "query": query,
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }
        
        # Step 2: Build context from results
        print(f"  Building context...")
        context = build_context_from_results(
            search_results,
            max_length=2000
        )
        
        # Step 3: Generate answer using LLM
        print(f"  Generating answer...")
        answer = generate_with_template(
            template=QA_PROMPT_TEMPLATE,
            variables={
                "context": context,
                "question": query
            },
            temperature=0.7
        )
        
        # Step 4: Format sources
        sources = []
        for result in search_results:
            source = {
                "id": result["id"],
                "title": result.get("title") or result["content"][:50] + "...",
                "category": result["category"],
                "severity": result["severity"],
                "relevance_score": round(result["score"], 3),
                "source_type": result["source_type"]
            }
            sources.append(source)
        
        # Calculate confidence based on top result score
        confidence = min(search_results[0]["score"], 1.0) if search_results else 0.0
        
        processing_time = time.time() - start_time
        
        print(f"/ Answer generated (confidence: {confidence:.2f}, time: {processing_time:.2f}s)")
        
        # Return structured response
        return {
            "query": query,
            "answer": answer.strip(),
            "sources": sources,
            "confidence": round(confidence, 3),
            "processing_time": round(processing_time, 2)
        }
    
    except Exception as e:
        print(f"! Q&A Tool error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "query": query,
            "answer": f"An error occurred while processing your question: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "processing_time": time.time() - start_time
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_qa_response(response: Dict[str, Any]) -> bool:
    """Validate Q&A response structure"""
    required_keys = ["query", "answer", "sources", "confidence", "processing_time"]
    return all(key in response for key in required_keys)


def format_sources_for_display(sources: List[Dict[str, Any]]) -> str:
    """Format sources for human-readable display"""
    if not sources:
        return "No sources found."
    
    formatted = []
    for i, source in enumerate(sources, 1):
        formatted.append(
            f"{i}. [{source['id']}] {source['title']}\n"
            f"   Category: {source['category']}, "
            f"Severity: {source['severity']}, "
            f"Score: {source['relevance_score']}"
        )
    
    return "\n".join(formatted)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "qa_tool",
    "validate_qa_response",
    "format_sources_for_display",
]