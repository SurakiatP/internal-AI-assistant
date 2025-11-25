"""
Vector Service
==============
Handles all interactions with Qdrant vector database using LangChain.

Features:
- Vector similarity search
- Metadata filtering
- Document retrieval
- Collection management
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import settings
from app.services.llm_service import embeddings


# ============================================================================
# INITIALIZE QDRANT CLIENT & LANGCHAIN VECTOR STORE
# ============================================================================

# Direct Qdrant client for low-level operations
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    timeout=120
)

# LangChain Qdrant vector store
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=settings.QDRANT_COLLECTION,
    embeddings=embeddings,
)


# ============================================================================
# SEARCH OPERATIONS
# ============================================================================

def search_documents(
    query: str,
    top_k: int = 5,
    filter_category: Optional[str] = None,
    filter_severity: Optional[str] = None,
    score_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Search documents by semantic similarity with optional filtering.
    
    Args:
        query: Search query
        top_k: Number of results to return
        filter_category: Filter by category
        filter_severity: Filter by severity
        score_threshold: Minimum similarity score (0-1)
    
    Returns:
        List of matching documents with metadata and scores
    """
    try:
        # Build filter
        filter_conditions = []
        
        if filter_category:
            filter_conditions.append(
                FieldCondition(
                    key="category",
                    match=MatchValue(value=filter_category)
                )
            )
        
        if filter_severity:
            filter_conditions.append(
                FieldCondition(
                    key="severity",
                    match=MatchValue(value=filter_severity)
                )
            )
        
        # Create filter object if conditions exist
        qdrant_filter = None
        if filter_conditions:
            qdrant_filter = Filter(must=filter_conditions)
        
        # Perform search using LangChain
        if qdrant_filter:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=qdrant_filter
            )
        else:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
        
        # Format results
        formatted_results = []
        for doc, score in results:
            if score >= score_threshold:
                formatted_results.append({
                    "id": doc.metadata.get("id", ""),
                    "content": doc.metadata.get("content", ""),
                    "title": doc.metadata.get("title", ""),
                    "category": doc.metadata.get("category", ""),
                    "severity": doc.metadata.get("severity", ""),
                    "source_type": doc.metadata.get("source_type", ""),
                    "environment": doc.metadata.get("environment", ""),
                    "score": float(score),
                    "metadata": doc.metadata
                })
        
        return formatted_results
    
    except Exception as e:
        print(f"! Search error: {e}")
        raise


def search_by_embedding(
    embedding: List[float],
    top_k: int = 5,
    filter_category: Optional[str] = None,
    filter_severity: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search documents using a pre-computed embedding vector.
    
    Args:
        embedding: Embedding vector
        top_k: Number of results to return
        filter_category: Filter by category
        filter_severity: Filter by severity
    
    Returns:
        List of matching documents with metadata and scores
    """
    try:
        # Build filter
        qdrant_filter = None
        if filter_category or filter_severity:
            conditions = []
            if filter_category:
                conditions.append(
                    FieldCondition(key="category", match=MatchValue(value=filter_category))
                )
            if filter_severity:
                conditions.append(
                    FieldCondition(key="severity", match=MatchValue(value=filter_severity))
                )
            qdrant_filter = Filter(must=conditions)
        
        # Search using Qdrant client directly
        search_results = qdrant_client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=embedding,
            limit=top_k,
            query_filter=qdrant_filter
        )
        
        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "id": result.payload.get("id", ""),
                "content": result.payload.get("content", ""),
                "title": result.payload.get("title", ""),
                "category": result.payload.get("category", ""),
                "severity": result.payload.get("severity", ""),
                "source_type": result.payload.get("source_type", ""),
                "score": float(result.score),
                "metadata": result.payload
            })
        
        return formatted_results
    
    except Exception as e:
        print(f"! Embedding search error: {e}")
        raise


def search_by_metadata(
    category: Optional[str] = None,
    severity: Optional[str] = None,
    source_type: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Search documents by metadata only (no semantic search).
    
    Args:
        category: Filter by category
        severity: Filter by severity
        source_type: Filter by source type
        limit: Maximum number of results
    
    Returns:
        List of matching documents
    """
    try:
        # Build filter
        conditions = []
        
        if category:
            conditions.append(
                FieldCondition(key="category", match=MatchValue(value=category))
            )
        
        if severity:
            conditions.append(
                FieldCondition(key="severity", match=MatchValue(value=severity))
            )
        
        if source_type:
            conditions.append(
                FieldCondition(key="source_type", match=MatchValue(value=source_type))
            )
        
        if not conditions:
            raise ValueError("At least one filter must be provided")
        
        qdrant_filter = Filter(must=conditions)
        
        # Scroll through collection with filter
        results, _ = qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=qdrant_filter,
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.payload.get("id", ""),
                "content": result.payload.get("content", ""),
                "title": result.payload.get("title", ""),
                "category": result.payload.get("category", ""),
                "severity": result.payload.get("severity", ""),
                "source_type": result.payload.get("source_type", ""),
                "metadata": result.payload
            })
        
        return formatted_results
    
    except Exception as e:
        print(f"! Metadata search error: {e}")
        raise


# ============================================================================
# DOCUMENT RETRIEVAL
# ============================================================================

def get_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific document by ID.
    
    Args:
        doc_id: Document ID (e.g., "BUG-001", "FEEDBACK-050")
    
    Returns:
        Document data or None if not found
    """
    try:
        # Get all documents and search for matching ID
        results, _ = qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=1000  # Get all documents
        )
        
        for result in results:
            if result.payload.get("id") == doc_id:
                return {
                    "id": result.payload.get("id", ""),
                    "content": result.payload.get("content", ""),
                    "title": result.payload.get("title", ""),
                    "category": result.payload.get("category", ""),
                    "severity": result.payload.get("severity", ""),
                    "source_type": result.payload.get("source_type", ""),
                    "metadata": result.payload
                }
        
        return None
    
    except Exception as e:
        print(f"! Document retrieval error: {e}")
        return None


def get_documents_batch(doc_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieve multiple documents by IDs.
    
    Args:
        doc_ids: List of document IDs
    
    Returns:
        List of document data
    """
    results = []
    for doc_id in doc_ids:
        doc = get_document_by_id(doc_id)
        if doc:
            results.append(doc)
    return results


# ============================================================================
# COLLECTION MANAGEMENT
# ============================================================================

def get_collection_info() -> Dict[str, Any]:
    """
    Get information about the collection.
    
    Returns:
        Collection statistics
    """
    try:
        collection_info = qdrant_client.get_collection(
            collection_name=settings.QDRANT_COLLECTION
        )
        
        return {
            "name": settings.QDRANT_COLLECTION,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
            "status": collection_info.status.value,
            "vector_size": settings.VECTOR_SIZE,
        }
    
    except Exception as e:
        print(f"! Collection info error: {e}")
        raise


def collection_exists() -> bool:
    """
    Check if the collection exists.
    
    Returns:
        True if collection exists, False otherwise
    """
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        return settings.QDRANT_COLLECTION in collection_names
    except:
        return False


def count_documents(
    category: Optional[str] = None,
    severity: Optional[str] = None
) -> int:
    """
    Count documents with optional filtering.
    
    Args:
        category: Filter by category
        severity: Filter by severity
    
    Returns:
        Document count
    """
    try:
        if category or severity:
            results = search_by_metadata(
                category=category,
                severity=severity,
                limit=10000  # Large limit to get all
            )
            return len(results)
        else:
            collection_info = get_collection_info()
            return collection_info["points_count"]
    
    except Exception as e:
        print(f"! Count error: {e}")
        return 0


# ============================================================================
# CONTEXT BUILDING (for RAG)
# ============================================================================

def build_context_from_results(
    results: List[Dict[str, Any]],
    max_length: int = 2000
) -> str:
    """
    Build context string from search results for LLM.
    
    Args:
        results: Search results
        max_length: Maximum context length in characters
    
    Returns:
        Formatted context string
    """
    context_parts = []
    current_length = 0
    
    for i, result in enumerate(results, 1):
        # Format document
        doc_text = f"""Document {i} [{result['id']}]:
Title: {result.get('title', 'N/A')}
Category: {result['category']}
Severity: {result['severity']}
Content: {result['content']}
"""
        
        # Check if adding this would exceed limit
        if current_length + len(doc_text) > max_length:
            break
        
        context_parts.append(doc_text)
        current_length += len(doc_text)
    
    return "\n---\n".join(context_parts)


# ============================================================================
# HEALTH CHECK
# ============================================================================

def check_vector_db_health() -> bool:
    """
    Check if Qdrant is available.
    
    Returns:
        True if healthy, False otherwise
    """
    try:
        collections = qdrant_client.get_collections()
        return True
    except:
        return False


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Search
    "search_documents",
    "search_by_embedding",
    "search_by_metadata",
    
    # Retrieval
    "get_document_by_id",
    "get_documents_batch",
    
    # Collection
    "get_collection_info",
    "collection_exists",
    "count_documents",
    
    # Context
    "build_context_from_results",
    
    # Health
    "check_vector_db_health",
    
    # Clients (if needed directly)
    "qdrant_client",
    "vector_store",
]