"""
FastAPI Backend
===============
REST API for Internal AI Assistant.

Endpoints:
- POST /api/query - Main agent endpoint
- GET /api/health - Health check
- GET /api/collections/stats - Qdrant statistics
- POST /api/tools/qa - Direct Q&A tool access
- POST /api/tools/summary - Direct summary tool access
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import time

from app.config import settings
from app.agents import agent_router
from app.tools import qa_tool, summary_tool
from app.services import (
    check_llm_health,
    check_embedding_health,
    check_vector_db_health,
    get_collection_info,
    collection_exists
)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Internal AI Assistant API",
    description="Agentic AI system for answering questions and summarizing issues",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CORS MIDDLEWARE (for Gradio frontend)
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for main query endpoint"""
    query: str = Field(..., description="User query", min_length=1)
    top_k: Optional[int] = Field(10, description="Number of documents to retrieve", ge=1, le=20)
    filter_category: Optional[str] = Field(None, description="Filter by category")
    filter_severity: Optional[str] = Field(None, description="Filter by severity (Low/Medium/High)")
    score_threshold: Optional[float] = Field(0.5, description="Minimum relevance score", ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the email notification issues?",
                "top_k": 10,
                "filter_category": "Notifications",
                "filter_severity": "High",
                "score_threshold": 0.5
            }
        }


class QARequest(BaseModel):
    """Request model for Q&A tool"""
    query: str = Field(..., description="Question to answer", min_length=1)
    top_k: Optional[int] = Field(10, ge=1, le=20)
    filter_category: Optional[str] = None
    filter_severity: Optional[str] = None
    score_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class SummaryRequest(BaseModel):
    """Request model for summary tool"""
    issue_text: str = Field(..., description="Issue text to analyze", min_length=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "issue_text": "When users upload files larger than 100MB, the system crashes and they lose all their work. This happens on both desktop and mobile."
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    services: Dict[str, bool]
    timestamp: float


class StatsResponse(BaseModel):
    """Collection statistics response"""
    collection_name: str
    documents_count: int
    vectors_count: int
    status: str
    vector_size: int


# ============================================================================
# MAIN ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Internal AI Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.post("/api/query")
async def query_agent(request: QueryRequest) -> Dict[str, Any]:
    """
    Main agent endpoint - Routes query to appropriate tool.
    
    This endpoint uses the agent router to intelligently decide
    whether to use Q&A or Summary tool based on the query.
    """
    try:
        print(f"\n{'=' * 70}")
        print(f"  API Request: /api/query")
        print(f"Query: {request.query}")
        print(f"Filters: category={request.filter_category}, severity={request.filter_severity}")
        print(f"{'=' * 70}\n")
        
        # Call agent router
        response = agent_router(
            query=request.query,
            top_k=request.top_k,
            filter_category=request.filter_category,
            filter_severity=request.filter_severity,
            score_threshold=request.score_threshold
        )
        
        print(f"\n/ Response ready: tool={response['tool_used']}\n")
        
        return response
    
    except Exception as e:
        print(f"! Error in /api/query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Checks status of all services:
    - LLM (Ollama)
    - Embeddings
    - Vector DB (Qdrant)
    """
    try:
        services = {
            "llm": check_llm_health(),
            "embeddings": check_embedding_health(),
            "vector_db": check_vector_db_health(),
            "collection_exists": collection_exists()
        }
        
        # Overall status
        status = "healthy" if all(services.values()) else "degraded"
        
        return HealthResponse(
            status=status,
            services=services,
            timestamp=time.time()
        )
    
    except Exception as e:
        print(f"! Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collections/stats")
async def collection_stats() -> StatsResponse:
    """
    Get Qdrant collection statistics.
    
    Returns information about the document collection:
    - Number of documents
    - Number of vectors
    - Collection status
    - Vector dimensionality
    """
    try:
        if not collection_exists():
            raise HTTPException(
                status_code=404,
                detail="Collection not found. Run data ingestion first."
            )
        
        info = get_collection_info()
        
        return StatsResponse(
            collection_name=info["name"],
            documents_count=info["points_count"],
            vectors_count=info["vectors_count"] or info["points_count"],  # Fallback to points_count
            status=info["status"],
            vector_size=info["vector_size"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"! Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DIRECT TOOL ENDPOINTS (Optional)
# ============================================================================

@app.post("/api/tools/qa")
async def qa_endpoint(request: QARequest) -> Dict[str, Any]:
    """
    Direct Q&A tool access.
    
    Use this endpoint to directly call the Q&A tool without agent routing.
    Useful for when you know you want to search documents.
    """
    try:
        print(f"\n  Direct Q&A Request: {request.query}\n")
        
        result = qa_tool(
            query=request.query,
            top_k=request.top_k,
            filter_category=request.filter_category,
            filter_severity=request.filter_severity,
            score_threshold=request.score_threshold
        )
        
        return result
    
    except Exception as e:
        print(f"! Error in Q&A tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/summary")
async def summary_endpoint(request: SummaryRequest) -> Dict[str, Any]:
    """
    Direct Summary tool access.
    
    Use this endpoint to directly call the Summary tool without agent routing.
    Useful for when you know you want to analyze specific text.
    """
    try:
        print(f"\n  Direct Summary Request ({len(request.issue_text)} chars)\n")
        
        result = summary_tool(issue_text=request.issue_text)
        
        return result
    
    except Exception as e:
        print(f"! Error in Summary tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    print("\n" + "=" * 70)
    print("  STARTING INTERNAL AI ASSISTANT API")
    print("=" * 70)
    print(f"  API URL: http://localhost:8000")
    print(f"  Docs: http://localhost:8000/docs")
    print(f"  LLM Model: {settings.LLM_MODEL}")
    print(f"  Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"  Collection: {settings.QDRANT_COLLECTION}")
    print("=" * 70 + "\n")
    
    # Check services
    print("  Checking services...")
    services = {
        "LLM": check_llm_health(),
        "Embeddings": check_embedding_health(),
        "Vector DB": check_vector_db_health(),
        "Collection": collection_exists()
    }
    
    for service, status in services.items():
        icon = "/" if status else "!"
        print(f"   {icon} {service}: {'OK' if status else 'FAILED'}")
    
    if not all(services.values()):
        print("\n!  WARNING: Some services are not available!")
    else:
        print("\n/ All services ready!")
    
    print()


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    print("\n" + "=" * 70)
    print("!!! SHUTTING DOWN INTERNAL AI ASSISTANT API")
    print("=" * 70 + "\n")


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "GET /",
            "POST /api/query",
            "GET /api/health",
            "GET /api/collections/stats",
            "POST /api/tools/qa",
            "POST /api/tools/summary"
        ]
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Check server logs for details."
    }


# ============================================================================
# RUN APP (for development only)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )