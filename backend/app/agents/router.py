"""
Agent Router
============
LLM-powered agent that decides which tool to use.

Uses LangChain for:
- Tool selection
- Reasoning explanation
- Structured output
"""

import time
import json
from typing import Dict, Any, Literal

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from app.config import settings
from app.services.llm_service import llm, generate_structured_output, generate_with_template
from app.tools import qa_tool, summary_tool


# ============================================================================
# AGENT PROMPT TEMPLATE
# ============================================================================

AGENT_ROUTING_PROMPT = """You are an AI agent router that decides which tool to use based on the user's query.

Available Tools:
1. **qa_tool**: Use this when the user wants to SEARCH for information from existing internal documents.
   - Questions about what's in the documents
   - "What are the issues with...?"
   - "What did users say about...?"
   - "Find information about..."
   
2. **summary_tool**: Use this when the user wants to ANALYZE/SUMMARIZE new text they provide.
   - "Summarize this issue: [text]"
   - "Analyze this problem: [text]"
   - User provides issue text to analyze
   - NOT searching documents, just analyzing given text

User Query: "{query}"

Analyze the query and decide:
1. Which tool should be used? (qa_tool or summary_tool)
2. Why did you choose this tool? (1-2 sentences)

Respond with valid JSON:
{{
  "tool": "qa_tool" or "summary_tool",
  "reasoning": "explanation here"
}}

JSON:"""


# ============================================================================
# AGENT ROUTER IMPLEMENTATION
# ============================================================================

def agent_router(query: str, **kwargs) -> Dict[str, Any]:
    """
    Agent Router: Intelligent tool selection with LLM reasoning.
    
    Process:
    1. Analyze user query
    2. Decide which tool to use (Q&A or Summary)
    3. Explain reasoning
    4. Route to selected tool
    5. Combine tool result with agent decision
    
    Args:
        query: User input
        **kwargs: Additional parameters for tools
    
    Returns:
        Final response with tool_used, reasoning, and result
    """
    start_time = time.time()
    
    try:
        # Step 1: Agent Decision - Which tool to use?
        print(f"\n  AGENT ROUTER")
        print(f"{'=' * 70}")
        print(f"Query: {query}")
        print(f"\n  Analyzing query to select tool...")
        
        # Use LLM to decide
        decision = make_routing_decision(query)
        
        tool_name = decision["tool"]
        reasoning = decision["reasoning"]
        
        print(f"✓ Decision: {tool_name}")
        print(f"✓ Reasoning: {reasoning}")
        
        # Step 2: Route to selected tool
        print(f"\n  Executing {tool_name}...")
        
        if tool_name == "qa_tool":
            result = qa_tool(
                query=query,
                top_k=kwargs.get("top_k", 10),
                filter_category=kwargs.get("filter_category"),
                filter_severity=kwargs.get("filter_severity"),
                score_threshold=kwargs.get("score_threshold", 0.5)
            )
        elif tool_name == "summary_tool":
            # For summary tool, the query itself is the issue text
            result = summary_tool(issue_text=query)
        else:
            # Fallback to Q&A
            print(f"!  Unknown tool '{tool_name}', defaulting to qa_tool")
            result = qa_tool(query=query)
            tool_name = "qa_tool"
            reasoning = "Defaulted to Q&A tool due to routing error"
        
        # Step 3: Combine agent decision with tool result
        processing_time = time.time() - start_time
        
        final_response = {
            "tool_used": tool_name,
            "reasoning": reasoning,
            "result": result,
            "total_processing_time": round(processing_time, 2)
        }
        
        print(f"\n/ Agent completed (total time: {processing_time:.2f}s)")
        print(f"{'=' * 70}\n")
        
        return final_response
    
    except Exception as e:
        print(f"! Agent Router error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to Q&A tool
        return {
            "tool_used": "qa_tool",
            "reasoning": f"Agent routing failed, defaulted to Q&A tool: {str(e)}",
            "result": qa_tool(query=query),
            "total_processing_time": time.time() - start_time
        }


def make_routing_decision(query: str) -> Dict[str, str]:
    """
    Use LLM to decide which tool to use.
    
    Args:
        query: User query
    
    Returns:
        Dictionary with 'tool' and 'reasoning'
    """
    try:
        # Generate structured decision
        output_schema = {
            "tool": "string",
            "reasoning": "string"
        }
        
        decision = generate_structured_output(
            prompt=AGENT_ROUTING_PROMPT.format(query=query),
            output_schema=output_schema,
            temperature=0.3  # Low temperature for consistent routing
        )
        
        # Validate tool name
        valid_tools = ["qa_tool", "summary_tool"]
        if decision.get("tool") not in valid_tools:
            # Try to fix common variations
            tool_lower = decision.get("tool", "").lower()
            if "summary" in tool_lower:
                decision["tool"] = "summary_tool"
            else:
                decision["tool"] = "qa_tool"  # Default
        
        return decision
    
    except Exception as e:
        print(f"!  Routing decision failed: {e}, using fallback")
        # Fallback: Simple keyword-based routing
        return fallback_routing(query)


def fallback_routing(query: str) -> Dict[str, str]:
    """
    Fallback routing based on keywords if LLM fails.
    
    Args:
        query: User query
    
    Returns:
        Dictionary with 'tool' and 'reasoning'
    """
    query_lower = query.lower()
    
    # Keywords for summary tool
    summary_keywords = [
        "summarize", "analyze", "issue:", "problem:", 
        "here is", "here's", "following issue",
        "this issue", "this problem"
    ]
    
    # Check if it's a summary request
    if any(keyword in query_lower for keyword in summary_keywords):
        return {
            "tool": "summary_tool",
            "reasoning": "Query contains issue text to analyze (keyword-based fallback)"
        }
    
    # Default to Q&A
    return {
        "tool": "qa_tool",
        "reasoning": "Query appears to be asking for information from documents (keyword-based fallback)"
    }


# ============================================================================
# LANGCHAIN-BASED AGENT (Alternative Advanced Implementation)
# ============================================================================

def create_langchain_agent():
    """
    Create a LangChain agent with tools.
    
    This is a more advanced implementation using LangChain's agent framework.
    Currently not used, but available for future enhancement.
    """
    from langchain.agents import Tool, AgentExecutor, create_react_agent
    from langchain.prompts import PromptTemplate
    
    # Define tools
    tools = [
        Tool(
            name="qa_tool",
            func=lambda q: qa_tool(q),
            description="Search internal documents to answer questions about existing issues, bugs, and feedback."
        ),
        Tool(
            name="summary_tool",
            func=lambda text: summary_tool(text),
            description="Analyze and summarize issue text provided by the user."
        )
    ]
    
    # Create agent
    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
    )
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_agent_response(response: Dict[str, Any]) -> bool:
    """
    Validate agent response structure.
    
    Args:
        response: Agent response dictionary
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["tool_used", "reasoning", "result"]
    return all(key in response for key in required_keys)


def format_agent_response(response: Dict[str, Any]) -> str:
    """
    Format agent response for human-readable display.
    
    Args:
        response: Agent response dictionary
    
    Returns:
        Formatted string
    """
    output = []
    output.append(f"  Agent Decision")
    output.append(f"{'=' * 70}")
    output.append(f"\n  Tool Used: {response['tool_used']}")
    output.append(f"  Reasoning: {response['reasoning']}")
    output.append(f"\n  Result:")
    output.append(f"{json.dumps(response['result'], indent=2)}")
    
    return "\n".join(output)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "agent_router",
    "make_routing_decision",
    "validate_agent_response",
    "format_agent_response",
]