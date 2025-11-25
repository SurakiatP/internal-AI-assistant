"""
Gradio Frontend
===============
Web UI for Internal AI Assistant with JSON response display.

Features:
- Clean chat interface
- Advanced filter options
- JSON response display (for technical assessment)
- Real-time processing
"""

import gradio as gr
import requests
import json
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = "http://localhost:8000"


# ============================================================================
# API CALL FUNCTIONS
# ============================================================================

def call_agent_api(query, top_k, category, severity, threshold):
    """
    Call the main agent API endpoint.
    
    Returns:
        tuple: (json_response, formatted_display, status_message)
    """
    try:
        # Prepare request
        payload = {
            "query": query,
            "top_k": int(top_k),
            "score_threshold": float(threshold)
        }
        
        # Add filters if provided
        if category and category != "All":
            payload["filter_category"] = category
        
        if severity and severity != "All":
            payload["filter_severity"] = severity
        
        # Call API
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/api/query",
            json=payload,
            timeout=180  # 3 minutes timeout
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Format JSON for display
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            
            # Create formatted display
            formatted = format_response(data, elapsed)
            
            status = f"✅ Success (API: {elapsed:.2f}s, Total: {data.get('total_processing_time', 0):.2f}s)"
            
            return json_str, formatted, status
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            return json.dumps({"error": error_msg}, indent=2), error_msg, "❌ Error"
    
    except requests.exceptions.Timeout:
        error = {"error": "Request timeout (>3 minutes). Try a simpler query."}
        return json.dumps(error, indent=2), "⏱️ Timeout - Request took too long", "❌ Timeout"
    
    except requests.exceptions.ConnectionError:
        error = {"error": "Cannot connect to API. Make sure backend is running."}
        return json.dumps(error, indent=2), "🔌 Connection Error - Backend not reachable", "❌ Connection Error"
    
    except Exception as e:
        error = {"error": str(e)}
        return json.dumps(error, indent=2), f"❌ Error: {str(e)}", "❌ Error"


def format_response(data, api_time):
    """
    Format response for human-readable display.
    
    Args:
        data: API response data
        api_time: API call time
    
    Returns:
        Formatted markdown string
    """
    output = []
    
    # Header
    output.append("# 🤖 AI Assistant Response")
    output.append("")
    
    # Tool used
    tool = data.get('tool_used', 'unknown')
    tool_emoji = "🔍" if tool == "qa_tool" else "📊"
    output.append(f"## {tool_emoji} Tool Used: `{tool}`")
    output.append("")
    
    # Reasoning
    output.append(f"## 💭 Reasoning")
    output.append(f"> {data.get('reasoning', 'N/A')}")
    output.append("")
    
    # Result
    result = data.get('result', {})
    
    if tool == "qa_tool":
        output.append("## 💬 Answer")
        output.append(result.get('answer', 'No answer provided'))
        output.append("")
        
        # Sources
        sources = result.get('sources', [])
        if sources:
            output.append(f"## 📚 Sources ({len(sources)} documents)")
            output.append("")
            for i, source in enumerate(sources, 1):
                output.append(f"**{i}. [{source['id']}]** {source.get('title', 'No title')}")
                output.append(f"   - Category: {source['category']}")
                output.append(f"   - Severity: {source['severity']}")
                output.append(f"   - Relevance: {source['relevance_score']:.3f}")
                output.append("")
        
        # Confidence
        output.append(f"**Confidence:** {result.get('confidence', 0):.3f}")
        output.append("")
    
    elif tool == "summary_tool":
        output.append("## 📝 Summary")
        output.append(result.get('summary', 'No summary provided'))
        output.append("")
        
        # Components
        components = result.get('components', [])
        if components:
            output.append("## 🔧 Affected Components")
            for comp in components:
                output.append(f"- {comp}")
            output.append("")
        
        # Severity
        severity = result.get('severity', 'N/A')
        severity_emoji = {
            'High': '🔴',
            'Medium': '🟡',
            'Low': '🟢'
        }.get(severity, '⚪')
        output.append(f"## {severity_emoji} Severity: {severity}")
        output.append("")
        
        # Requirements
        requirements = result.get('requirements', [])
        if requirements:
            output.append("## ✅ Requirements")
            for i, req in enumerate(requirements, 1):
                output.append(f"{i}. {req}")
            output.append("")
    
    # Timing
    total_time = data.get('total_processing_time', 0)
    output.append("---")
    output.append(f"⏱️ **Processing Time:** API={api_time:.2f}s, Total={total_time:.2f}s")
    
    return "\n".join(output)


def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'healthy':
                return "✅ All services operational"
            else:
                services = data.get('services', {})
                failed = [k for k, v in services.items() if not v]
                return f"⚠️ Degraded: {', '.join(failed)}"
        else:
            return f"❌ API returned {response.status_code}"
    except:
        return "❌ Cannot connect to backend"


def get_collection_stats():
    """Get collection statistics"""
    try:
        response = requests.get(f"{API_URL}/api/collections/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"📊 {data['documents_count']} documents indexed"
        else:
            return "⚠️ Collection not found"
    except:
        return "❌ Cannot fetch stats"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(
        title="Internal AI Assistant",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate"
        )
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🤖 Internal AI Assistant
        ### Agentic AI system for answering questions and summarizing issues
        
        Ask questions about internal documents or provide issue text to analyze.
        """)
        
        # Status indicators
        with gr.Row():
            health_status = gr.Textbox(
                value=check_api_health(),
                label="API Status",
                interactive=False,
                scale=1
            )
            collection_status = gr.Textbox(
                value=get_collection_stats(),
                label="Collection Status",
                interactive=False,
                scale=1
            )
        
        # Main interface
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="💬 Your Query",
                    placeholder="Examples:\n- What are the email notification issues?\n- Summarize this: System crashes when uploading files...",
                    lines=5,
                    max_lines=10
                )
                
                # Advanced options (collapsible)
                with gr.Accordion("⚙️ Advanced Options", open=False):
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Top K Results",
                        info="Number of documents to retrieve (for Q&A)"
                    )
                    
                    category_filter = gr.Dropdown(
                        choices=["All", "Document Management", "Search & Filtering", 
                                "UI/UX", "Performance", "Mobile", "Authentication",
                                "Notifications", "Collaboration", "Backend", "General"],
                        value="All",
                        label="Category Filter",
                        info="Filter by category (for Q&A)"
                    )
                    
                    severity_filter = gr.Dropdown(
                        choices=["All", "High", "Medium", "Low"],
                        value="All",
                        label="Severity Filter",
                        info="Filter by severity (for Q&A)"
                    )
                    
                    threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Score Threshold",
                        info="Minimum relevance score (for Q&A)"
                    )
                
                submit_btn = gr.Button("🚀 Submit", variant="primary", size="lg")
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=True
                )
            
            # Right column - Output
            with gr.Column(scale=2):
                # Tabs for different views
                with gr.Tabs():
                    with gr.Tab("📋 JSON Response"):
                        json_output = gr.Code(
                            label="Raw JSON Response",
                            language="json",
                            lines=25,
                            interactive=False
                        )
                    
                    with gr.Tab("👁️ Formatted View"):
                        formatted_output = gr.Markdown(
                            label="Formatted Response"
                        )
        
        # Examples
        gr.Markdown("### 💡 Example Queries")
        
        gr.Examples(
            examples=[
                ["What are the email notification issues?"],
                ["Find problems related to document upload"],
                ["What UI issues do users report?"],
                ["Summarize this: When users try to upload large files over 100MB, the system crashes and they lose all their work. This happens on both desktop and mobile."],
                ["Analyze this problem: The search autocomplete is completely wrong. When I type 'report', it suggests 'reptile'."],
            ],
            inputs=query_input
        )
        
        # Event handlers
        submit_btn.click(
            fn=call_agent_api,
            inputs=[query_input, top_k, category_filter, severity_filter, threshold],
            outputs=[json_output, formatted_output, status_text]
        )
        
        # Enter key support
        query_input.submit(
            fn=call_agent_api,
            inputs=[query_input, top_k, category_filter, severity_filter, threshold],
            outputs=[json_output, formatted_output, status_text]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **API Endpoints:**
        - 📚 Documentation: http://localhost:8000/docs
        - 🏥 Health Check: http://localhost:8000/api/health
        - 📊 Collection Stats: http://localhost:8000/api/collections/stats
        """)
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎨 LAUNCHING GRADIO FRONTEND")
    print("=" * 70)
    print(f"Backend API: {API_URL}")
    print(f"Frontend URL: http://localhost:7860")
    print("=" * 70 + "\n")
    
    # Check backend
    health = check_api_health()
    print(f"Backend Status: {health}\n")
    
    if "Cannot connect" in health:
        print("⚠️  WARNING: Cannot connect to backend!")
        print("   Make sure the API is running:")
        print("   docker-compose up -d\n")
    
    # Launch
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )