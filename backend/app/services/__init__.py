"""
Services for external dependencies (LLM, Vector DB)
"""

from .llm_service import (
    generate_text,
    generate_with_template,
    generate_structured_output,
    generate_embedding,
    generate_embeddings_batch,
    classify_category,
    classify_severity,
    summarize_text,
    generate_answer,
    check_llm_health,
    check_embedding_health,
)

from .vector_service import (
    search_documents,
    search_by_embedding,
    search_by_metadata,
    get_document_by_id,
    get_documents_batch,
    get_collection_info,
    collection_exists,
    count_documents,
    build_context_from_results,
    check_vector_db_health,
)

__all__ = [
    # LLM Service
    "generate_text",
    "generate_with_template",
    "generate_structured_output",
    "generate_embedding",
    "generate_embeddings_batch",
    "classify_category",
    "classify_severity",
    "summarize_text",
    "generate_answer",
    "check_llm_health",
    "check_embedding_health",
    
    # Vector Service
    "search_documents",
    "search_by_embedding",
    "search_by_metadata",
    "get_document_by_id",
    "get_documents_batch",
    "get_collection_info",
    "collection_exists",
    "count_documents",
    "build_context_from_results",
    "check_vector_db_health",
]