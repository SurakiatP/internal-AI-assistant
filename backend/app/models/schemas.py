"""
Pydantic models for data validation and serialization.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


# ============================================================================
# UNIFIED DOCUMENT SCHEMA (for both Bug Reports and User Feedback)
# ============================================================================

class UnifiedDocument(BaseModel):
    """
    Unified schema for both bug reports and user feedback.
    This ensures consistency across different data sources.
    """
    # Core fields
    id: str = Field(..., description="Unique identifier (e.g., 'BUG-001', 'FEEDBACK-001')")
    source_type: Literal["bug_report", "user_feedback"] = Field(..., description="Source of the document")
    content: str = Field(..., description="Main content/description")
    
    # Metadata fields
    title: Optional[str] = Field(None, description="Title or summary")
    category: str = Field(..., description="Category (e.g., 'Document Management', 'UI/UX')")
    severity: Literal["Low", "Medium", "High"] = Field(..., description="Issue severity")
    
    # Additional fields
    environment: Optional[str] = Field(None, description="Environment info (for bug reports)")
    steps_to_reproduce: Optional[str] = Field(None, description="Reproduction steps (for bug reports)")
    proposed_fix: Optional[str] = Field(None, description="Proposed fix (for bug reports)")
    
    # For vector search
    embedding_text: str = Field(..., description="Text used for embedding generation")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "BUG-001",
                "source_type": "bug_report",
                "content": "When uploading large PDF documents...",
                "title": "Document Upload Stuck at 99%",
                "category": "Document Management",
                "severity": "Medium",
                "embedding_text": "Document Management | Medium | Document Upload Stuck at 99%: When uploading large PDF..."
            }
        }


# ============================================================================
# API REQUEST/RESPONSE SCHEMAS
# ============================================================================

class QueryRequest(BaseModel):
    """Request schema for Q&A queries"""
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results to return")
    filter_category: Optional[str] = Field(None, description="Filter by category")
    filter_severity: Optional[Literal["Low", "Medium", "High"]] = Field(None, description="Filter by severity")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the issues with email notifications?",
                "top_k": 5,
                "filter_severity": "High"
            }
        }


class SummaryRequest(BaseModel):
    """Request schema for issue summarization"""
    issue_text: str = Field(..., min_length=10, max_length=5000, description="Issue text to summarize")
    
    class Config:
        json_schema_extra = {
            "example": {
                "issue_text": "Users are reporting that email notifications are not being sent..."
            }
        }


class Source(BaseModel):
    """Source document metadata"""
    id: str
    title: Optional[str]
    category: str
    severity: str
    relevance_score: float
    source_type: str


class QAResponse(BaseModel):
    """Response schema for Q&A queries"""
    query: str
    answer: str
    sources: List[Source]
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the issues with email notifications?",
                "answer": "Based on the internal documents, there are several issues with email notifications...",
                "sources": [
                    {
                        "id": "BUG-007",
                        "title": "Email Notifications Not Being Sent",
                        "category": "Notifications",
                        "severity": "High",
                        "relevance_score": 0.95,
                        "source_type": "bug_report"
                    }
                ],
                "confidence": 0.92,
                "processing_time": 1.23
            }
        }


class IssueSummary(BaseModel):
    """Response schema for issue summarization"""
    reported_issues: List[str] = Field(..., description="List of main issues identified")
    affected_features: List[str] = Field(..., description="Features/components affected")
    severity: Literal["Low", "Medium", "High"] = Field(..., description="Overall severity")
    summary: str = Field(..., description="Brief summary of the issues")
    recommendations: Optional[List[str]] = Field(None, description="Suggested fixes")
    processing_time: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "reported_issues": [
                    "Email notifications not being sent",
                    "Notification settings not saving"
                ],
                "affected_features": [
                    "Email Service",
                    "User Settings"
                ],
                "severity": "High",
                "summary": "Critical issues with the notification system affecting user communication",
                "recommendations": [
                    "Investigate email service integration",
                    "Review notification queue processing"
                ],
                "processing_time": 0.85
            }
        }


class AgentResponse(BaseModel):
    """Response schema from the agent router"""
    tool_used: Literal["qa_tool", "summary_tool"]
    reasoning: str
    result: Dict[str, Any]
    processing_time: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "tool_used": "qa_tool",
                "reasoning": "User is asking about existing issues in the knowledge base",
                "result": {
                    "answer": "...",
                    "sources": [...]
                },
                "processing_time": 1.45
            }
        }


# ============================================================================
# INTERNAL SCHEMAS (for data processing)
# ============================================================================

class BugReport(BaseModel):
    """Raw bug report structure"""
    bug_id: str
    title: str
    description: str
    steps_to_reproduce: str
    environment: str
    severity: str
    proposed_fix: str


class UserFeedback(BaseModel):
    """Raw user feedback structure"""
    feedback_id: str
    content: str


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings"""
    text: str
    model: str = "nomic-embed-text:latest"


class EmbeddingResponse(BaseModel):
    """Response containing embedding vector"""
    embedding: List[float]
    model: str
    dimensions: int