"""
LLM Service
===========
Handles all interactions with Ollama LLM using LangChain.

Features:
- Text generation
- Embedding generation
- JSON extraction
- Streaming support
"""

import json
import re
from typing import List, Dict, Any, Optional
import httpx

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from app.config import settings


# ============================================================================
# INITIALIZE LANGCHAIN MODELS
# ============================================================================

# LLM for text generation
llm = Ollama(
    base_url=settings.OLLAMA_URL,
    model=settings.LLM_MODEL,
    temperature=settings.TEMPERATURE,
)

# Embeddings model
embeddings = OllamaEmbeddings(
    base_url=settings.OLLAMA_URL,
    model=settings.EMBEDDING_MODEL
)


# ============================================================================
# TEXT GENERATION
# ============================================================================

def generate_text(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> str:
    """
    Generate text using LLM.
    
    Args:
        prompt: Input prompt
        temperature: Sampling temperature (default from settings)
        max_tokens: Maximum tokens to generate (default from settings)
    
    Returns:
        Generated text
    """
    try:
        # Create temporary LLM with custom parameters if provided
        if temperature is not None or max_tokens is not None:
            temp_llm = Ollama(
                base_url=settings.OLLAMA_URL,
                model=settings.LLM_MODEL,
                temperature=temperature or settings.TEMPERATURE,
            )
            response = temp_llm.invoke(prompt)
        else:
            response = llm.invoke(prompt)
        
        return response.strip()
    
    except Exception as e:
        print(f"! LLM generation error: {e}")
        raise


def generate_with_template(
    template: str,
    variables: Dict[str, Any],
    temperature: Optional[float] = None
) -> str:
    """
    Generate text using a prompt template with LangChain.
    
    Args:
        template: Prompt template string
        variables: Dictionary of variables to fill in template
        temperature: Optional temperature override
    
    Returns:
        Generated text
    """
    try:
        prompt_template = PromptTemplate(
            template=template,
            input_variables=list(variables.keys())
        )
        
        # Create chain
        if temperature is not None:
            temp_llm = Ollama(
                base_url=settings.OLLAMA_URL,
                model=settings.LLM_MODEL,
                temperature=temperature,
            )
            chain = LLMChain(llm=temp_llm, prompt=prompt_template)
        else:
            chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Run chain
        result = chain.run(**variables)
        return result.strip()
    
    except Exception as e:
        print(f"! Template generation error: {e}")
        raise


# ============================================================================
# JSON EXTRACTION
# ============================================================================

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response, handling markdown code blocks.
    
    Args:
        response: Raw LLM response that should contain JSON
    
    Returns:
        Parsed JSON dictionary
    """
    try:
        # Remove markdown code blocks
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*", "", response)
        response = response.strip()
        
        # Parse JSON
        return json.loads(response)
    
    except json.JSONDecodeError as e:
        print(f"! JSON parsing error: {e}")
        print(f"Response: {response[:200]}...")
        raise ValueError(f"Failed to parse JSON from response: {e}")


def generate_structured_output(
    prompt: str,
    output_schema: Dict[str, Any],
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Generate structured JSON output from LLM.
    
    Args:
        prompt: Input prompt
        output_schema: Expected JSON schema
        temperature: Lower temperature for more consistent output
    
    Returns:
        Parsed JSON matching schema
    """
    # Add JSON format instruction to prompt
    schema_str = json.dumps(output_schema, indent=2)
    full_prompt = f"""{prompt}

Respond ONLY with valid JSON in this exact format:
{schema_str}

Do not include any text before or after the JSON.
Do not use markdown code blocks.

JSON:"""
    
    try:
        response = generate_text(full_prompt, temperature=temperature)
        return extract_json_from_response(response)
    
    except Exception as e:
        print(f"! Structured output error: {e}")
        raise


# ============================================================================
# EMBEDDINGS
# ============================================================================

def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding vector for text using LangChain.
    
    Args:
        text: Input text to embed
    
    Returns:
        Embedding vector (list of floats)
    """
    try:
        embedding_vector = embeddings.embed_query(text)
        return embedding_vector
    
    except Exception as e:
        print(f"! Embedding generation error: {e}")
        raise


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts (batch processing).
    
    Args:
        texts: List of texts to embed
    
    Returns:
        List of embedding vectors
    """
    try:
        embedding_vectors = embeddings.embed_documents(texts)
        return embedding_vectors
    
    except Exception as e:
        print(f"! Batch embedding error: {e}")
        raise


# ============================================================================
# CLASSIFICATION & ANALYSIS
# ============================================================================

def classify_category(text: str, categories: List[str]) -> str:
    """
    Classify text into one of the given categories.
    
    Args:
        text: Text to classify
        categories: List of possible categories
    
    Returns:
        Selected category
    """
    template = """Classify the following text into ONE of these categories:
{categories}

Text: "{text}"

Respond with ONLY the category name, nothing else.

Category:"""
    
    try:
        result = generate_with_template(
            template=template,
            variables={
                "categories": ", ".join(categories),
                "text": text
            },
            temperature=0.3
        )
        
        # Clean and validate result
        result = result.strip()
        if result in categories:
            return result
        
        # Try to find matching category
        for cat in categories:
            if cat.lower() in result.lower():
                return cat
        
        # Default to first category if no match
        return categories[0]
    
    except Exception as e:
        print(f"! Classification error: {e}")
        return categories[0]


def classify_severity(text: str) -> str:
    """
    Classify text severity as Low, Medium, or High.
    
    Args:
        text: Text to analyze
    
    Returns:
        Severity level (Low, Medium, or High)
    """
    template = """Analyze the severity of this issue/feedback:

Text: "{text}"

Classify as:
- High: Critical issues, system failures, data loss, security problems
- Medium: Functionality impaired, workarounds exist, moderate impact
- Low: Cosmetic issues, minor inconvenience, polish items

Respond with ONLY one word: Low, Medium, or High

Severity:"""
    
    try:
        result = generate_with_template(
            template=template,
            variables={"text": text},
            temperature=0.3
        )
        
        result = result.strip()
        
        # Validate
        if result in ["Low", "Medium", "High"]:
            return result
        
        # Try to extract
        if "high" in result.lower():
            return "High"
        elif "medium" in result.lower():
            return "Medium"
        elif "low" in result.lower():
            return "Low"
        
        # Default
        return "Medium"
    
    except Exception as e:
        print(f"! Severity classification error: {e}")
        return "Medium"


# ============================================================================
# SUMMARIZATION
# ============================================================================

def summarize_text(
    text: str,
    max_length: int = 200
) -> str:
    """
    Summarize text to specified length.
    
    Args:
        text: Text to summarize
        max_length: Maximum summary length in words
    
    Returns:
        Summary text
    """
    template = """Summarize the following text in {max_length} words or less.
Be concise and focus on key points.

Text: "{text}"

Summary:"""
    
    try:
        result = generate_with_template(
            template=template,
            variables={
                "text": text,
                "max_length": max_length
            },
            temperature=0.5
        )
        return result
    
    except Exception as e:
        print(f"! Summarization error: {e}")
        raise


# ============================================================================
# ANSWER GENERATION (for Q&A)
# ============================================================================

def generate_answer(
    query: str,
    context: str,
    max_tokens: int = 500
) -> str:
    """
    Generate answer based on query and context.
    
    Args:
        query: User question
        context: Retrieved context from documents
        max_tokens: Maximum answer length
    
    Returns:
        Generated answer
    """
    template = """You are an internal AI assistant helping the product and engineering team.

Answer the following question based ONLY on the provided context.
Be specific and cite relevant information.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        result = generate_with_template(
            template=template,
            variables={
                "query": query,
                "context": context
            },
            temperature=0.7
        )
        return result
    
    except Exception as e:
        print(f"! Answer generation error: {e}")
        raise


# ============================================================================
# HEALTH CHECK
# ============================================================================

def check_llm_health() -> bool:
    """
    Check if LLM service is available.
    
    Returns:
        True if healthy, False otherwise
    """
    try:
        response = httpx.get(f"{settings.OLLAMA_URL}/api/tags", timeout=5.0)
        return response.status_code == 200
    except:
        return False


def check_embedding_health() -> bool:
    """
    Check if embedding service is available.
    
    Returns:
        True if healthy, False otherwise
    """
    try:
        # Try to generate a small embedding
        test_embedding = generate_embedding("test")
        return len(test_embedding) == settings.VECTOR_SIZE
    except:
        return False


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Text Generation
    "generate_text",
    "generate_with_template",
    "generate_structured_output",
    
    # JSON
    "extract_json_from_response",
    
    # Embeddings
    "generate_embedding",
    "generate_embeddings_batch",
    
    # Classification
    "classify_category",
    "classify_severity",
    
    # Summarization
    "summarize_text",
    
    # Q&A
    "generate_answer",
    
    # Health
    "check_llm_health",
    "check_embedding_health",
    
    # LangChain objects (if needed directly)
    "llm",
    "embeddings",
]