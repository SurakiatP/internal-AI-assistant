"""
Summary Tool
============
Issue summarization tool that analyzes text without searching documents.

Uses:
- LLM analysis only (no vector search)
- Structured output extraction
- Component identification
"""

import time
import json
from typing import Dict, Any

from app.services.llm_service import generate_structured_output, generate_with_template


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

SUMMARY_PROMPT_TEMPLATE = """You are an expert at analyzing technical issues and bug reports.

Analyze the following issue text and provide a comprehensive summary.

Issue Text:
{issue_text}

Extract the following information:
1. **Summary**: A concise overview of the main issues (2-3 sentences)
2. **Components**: List of affected system components or features
3. **Severity**: Overall severity level (High, Medium, or Low)
   - High: Critical issues, system failures, data loss, security problems
   - Medium: Functionality impaired, workarounds exist, moderate impact
   - Low: Cosmetic issues, minor inconvenience, polish items
4. **Requirements**: Recommended actions or fixes needed

Provide your analysis in the following JSON format:
{{
  "summary": "Brief overview of the issues",
  "components": ["Component 1", "Component 2"],
  "severity": "High/Medium/Low",
  "requirements": ["Action 1", "Action 2"]
}}

Respond ONLY with valid JSON. Do not include any other text."""


# ============================================================================
# SUMMARY TOOL IMPLEMENTATION
# ============================================================================

def summary_tool(issue_text: str) -> Dict[str, Any]:
    """
    Summary Tool: Analyze and summarize issue text.
    
    Process:
    1. Receive issue text from user
    2. Analyze with LLM (no vector search needed)
    3. Extract structured information
    4. Return JSON response
    
    Note: Does NOT search documents or save to database.
    
    Args:
        issue_text: The issue text to analyze
    
    Returns:
        Structured summary with components, severity, and requirements
    """
    start_time = time.time()
    
    try:
        print(f"  Analyzing issue text ({len(issue_text)} characters)...")
        
        # Validate input
        if not issue_text or len(issue_text.strip()) < 10:
            return {
                "summary": "Issue text is too short to analyze.",
                "components": [],
                "severity": "Low",
                "requirements": ["Provide more detailed information"],
                "processing_time": time.time() - start_time
            }
        
        # Generate structured analysis using LLM
        print(f"  Generating structured summary...")
        
        # Use structured output generation
        output_schema = {
            "summary": "string",
            "components": ["string"],
            "severity": "string",
            "requirements": ["string"]
        }
        
        result = generate_structured_output(
            prompt=SUMMARY_PROMPT_TEMPLATE.format(issue_text=issue_text),
            output_schema=output_schema,
            temperature=0.3  # Lower temperature for consistent analysis
        )
        
        # Validate severity
        valid_severities = ["Low", "Medium", "High"]
        if result.get("severity") not in valid_severities:
            # Try to fix common variations
            severity_lower = result.get("severity", "").lower()
            if "high" in severity_lower:
                result["severity"] = "High"
            elif "medium" in severity_lower:
                result["severity"] = "Medium"
            elif "low" in severity_lower:
                result["severity"] = "Low"
            else:
                result["severity"] = "Medium"  # Default
        
        # Ensure lists are not empty
        if not result.get("components"):
            result["components"] = ["General System"]
        
        if not result.get("requirements"):
            result["requirements"] = ["Further investigation needed"]
        
        # Add processing time
        processing_time = time.time() - start_time
        result["processing_time"] = round(processing_time, 2)
        
        print(f"/ Summary generated (severity: {result['severity']}, time: {processing_time:.2f}s)")
        
        return result
    
    except json.JSONDecodeError as e:
        print(f"!  JSON parsing error: {e}")
        print(f"   Using fallback method...")
        # Fallback: Try alternative method
        return summary_tool_fallback(issue_text, start_time)
    
    except Exception as e:
        print(f"! Summary Tool error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "summary": f"An error occurred while analyzing the issue: {str(e)}",
            "components": ["Error Handler"],
            "severity": "Medium",
            "requirements": ["Retry analysis", "Check input format"],
            "processing_time": time.time() - start_time
        }


def summary_tool_fallback(issue_text: str, start_time: float) -> Dict[str, Any]:
    """
    Fallback method if structured output fails.
    Uses simpler prompt without JSON format enforcement.
    
    Args:
        issue_text: Issue text to analyze
        start_time: Start timestamp
    
    Returns:
        Structured summary (best effort)
    """
    print(f"!  Using fallback method...")
    
    try:
        # Simpler prompts for each component
        summary = generate_with_template(
            template="Summarize this issue in 2-3 sentences:\n\n{issue_text}\n\nSummary:",
            variables={"issue_text": issue_text},
            temperature=0.3
        )
        
        components_text = generate_with_template(
            template="List the affected system components from this issue (comma-separated):\n\n{issue_text}\n\nComponents:",
            variables={"issue_text": issue_text},
            temperature=0.3
        )
        components = [c.strip() for c in components_text.split(",") if c.strip()]
        
        severity = generate_with_template(
            template="Rate the severity of this issue as High, Medium, or Low:\n\n{issue_text}\n\nSeverity:",
            variables={"issue_text": issue_text},
            temperature=0.3
        ).strip()
        
        if severity not in ["High", "Medium", "Low"]:
            severity = "Medium"
        
        requirements_text = generate_with_template(
            template="List 2-3 recommended actions to address this issue:\n\n{issue_text}\n\nActions:",
            variables={"issue_text": issue_text},
            temperature=0.3
        )
        requirements = [r.strip().lstrip("-•").strip() for r in requirements_text.split("\n") if r.strip()]
        
        return {
            "summary": summary.strip(),
            "components": components if components else ["General System"],
            "severity": severity,
            "requirements": requirements if requirements else ["Further investigation needed"],
            "processing_time": round(time.time() - start_time, 2)
        }
    
    except Exception as e:
        print(f"! Fallback method also failed: {e}")
        return {
            "summary": "Unable to analyze the issue text.",
            "components": ["Unknown"],
            "severity": "Medium",
            "requirements": ["Manual review required"],
            "processing_time": time.time() - start_time
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_summary_response(response: Dict[str, Any]) -> bool:
    """
    Validate summary response structure.
    
    Args:
        response: Summary response dictionary
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["summary", "components", "severity", "requirements"]
    return all(key in response for key in required_keys)


def format_summary_for_display(response: Dict[str, Any]) -> str:
    """
    Format summary for human-readable display.
    
    Args:
        response: Summary response dictionary
    
    Returns:
        Formatted string
    """
    output = []
    output.append(f"  Issue Summary")
    output.append(f"{'=' * 50}")
    output.append(f"\n  Summary:\n{response['summary']}")
    output.append(f"\n  Affected Components:")
    for comp in response['components']:
        output.append(f"  • {comp}")
    output.append(f"\n   Severity: {response['severity']}")
    output.append(f"\n/  Recommended Actions:")
    for i, req in enumerate(response['requirements'], 1):
        output.append(f"  {i}. {req}")
    
    return "\n".join(output)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "summary_tool",
    "validate_summary_response",
    "format_summary_for_display",
]