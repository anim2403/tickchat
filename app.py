from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import json
import os
from groq import Groq
from models import TicketClassification
from llm_utils import classify_ticket
from rag_pipeline import AtlanRAGPipeline

st.set_page_config(page_title="Ticket Classifier with AI Help", layout="wide")
st.title("AI-Powered Ticket Classifier with Atlan Documentation Support")

# Load sample tickets from file
with open("sample_tickets.json") as f:
    tickets = json.load(f)

# Initialize session state
if "classifications" not in st.session_state:
    st.session_state.classifications = {}
if "new_ticket_classification" not in st.session_state:
    st.session_state.new_ticket_classification = None
if "ai_help_results" not in st.session_state:
    st.session_state.ai_help_results = {}

# Initialize clients
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize RAG pipeline (with error handling)
@st.cache_resource
def initialize_rag_pipeline():
    try:
        # Initialize Pinecone-based RAG pipeline
        rag_pipeline = AtlanRAGPipeline(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        return rag_pipeline, None
    except Exception as e:
        return None, str(e)

rag_pipeline, rag_error = initialize_rag_pipeline()

if rag_error:
    st.error(f"⚠️ RAG Pipeline initialization failed: {rag_error}")
    st.info("Please ensure your Pinecone API key is set and the 'atlan-docs' index exists.")

def get_classification(ticket):
    if ticket["id"] in st.session_state.classifications:
        return st.session_state.classifications[ticket["id"]]
    try:
        analysis, classification = classify_ticket(client, ticket["subject"], ticket["body"])
    except Exception as e:
        analysis = {
            "prompt": {
                "system": "",
                "user": ""
            },
            "raw_output": "",
            "error": f"Classification failed for {ticket['id']}: {e}",
        }
        classification = TicketClassification(topic_tags=[], topic_tag_confidence={}, core_problem="", priority="", sentiment="")
    st.session_state.classifications[ticket["id"]] = (analysis, classification)
    return analysis, classification

def get_ai_help(ticket_id, subject, body, topic_tags):
    """Get AI help for a ticket using RAG pipeline"""
    if not rag_pipeline:
        return {"error": "RAG pipeline not available"}
    
    if ticket_id in st.session_state.ai_help_results:
        return st.session_state.ai_help_results[ticket_id]
    
    try:
        result = rag_pipeline.get_ai_help(subject, body, topic_tags)
        st.session_state.ai_help_results[ticket_id] = result
        return result
    except Exception as e:
        error_result = {"error": f"AI Help failed: {str(e)}"}
        st.session_state.ai_help_results[ticket_id] = error_result
        return error_result

def badge(text, color="#e63946"):
    return f"""<span style="
        display:inline-block;
        background:{color};
        color:white;
        font-weight:bold;
        border-radius:20px;
        padding:6px 16px;
        margin:2px 6px 2px 0;
        font-size:0.96em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    ">{text}</span>"""

def display_ai_help_result(result):
    """Display the AI help result in a nice format"""
    if "error" in result:
        st.error(f"❌ {result['error']}")
        return
    
    if not result["use_rag"]:
        st.info(f"📋 {result['message']}")
        return
    
    if "answer" in result:
        st.success("🤖 AI Generated Solution:")
        st.markdown(result["answer"])
        
        # Show confidence indicator
        st.caption(f"📚 Generated from {result['num_sources']} documentation sources")
        
        # Option to show source details
        with st.expander("🔍 View Source Documents", expanded=False):
            for i, doc in enumerate(result["relevant_docs"]):
                st.markdown(f"**Source {i+1}** (Confidence: {doc['score']:.3f})")
                st.markdown(f"**URL:** [{doc['url']}]({doc['url']})")
                st.markdown(f"**Content Preview:**")
                st.text(doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"])
                st.markdown("---")
    else:
        st.warning("⚠️ No relevant documentation found for this query.")

def ticket_menu_row(ticket, analysis, classification):
    # Create two columns for ticket info and AI help button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.write(f"**ID:** {ticket['id']}")
        st.write(f"**Subject:** {ticket['subject']}")
        st.write(f"**Body:** {ticket['body']}")

        # --- Internal Backend View (AI's Internal Analysis) ---
        with st.expander("🔍 AI's Internal Analysis (Back-end view)", expanded=False):
            st.markdown("**Prompt sent to LLM:**")
            st.code(json.dumps(analysis.get("prompt", {}), indent=2), language="json")
            st.markdown("**Raw Output from LLM:**")
            st.code(analysis.get("raw_output", "") or "No output", language="json")
            if analysis.get("error"):
                st.markdown("**Error / Reasoning:**")
                st.code(analysis["error"], language="text")

            # Show confidence scores for top 3-4 topic_tags
            if classification.topic_tag_confidence:
                sorted_tags = sorted(
                    classification.topic_tag_confidence.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                st.markdown("**Top Topic Tag Confidence Scores:**")
                for tag, score in sorted_tags[:4]:
                    st.markdown(f"- `{tag}`: **{score:.2f}**")

        # --- Front-end View (Final Classification) ---
        topic_tags_html = "".join([badge(topic_tag) for topic_tag in classification.topic_tags])
        core_problem_html = badge(classification.core_problem)
        priority_html = badge(classification.priority)
        sentiment_html = badge(classification.sentiment)
        st.markdown(f"**Topic_Tags:** {topic_tags_html}", unsafe_allow_html=True)
        st.markdown(f"**Core_Problem:** {core_problem_html}", unsafe_allow_html=True)
        st.markdown(f"**Priority:** {priority_html}", unsafe_allow_html=True)
        st.markdown(f"**Sentiment:** {sentiment_html}", unsafe_allow_html=True)
    
    with col2:
        # AI Help button
        if st.button(f"🤖 Get AI Help", key=f"help_{ticket['id']}", disabled=not rag_pipeline):
            with st.spinner("Getting AI help..."):
                result = get_ai_help(
                    ticket['id'],
                    ticket['subject'], 
                    ticket['body'], 
                    classification.topic_tags
                )
    
    # Display AI help result if available
    if ticket['id'] in st.session_state.ai_help_results:
        st.markdown("### 🚀 AI Help Result")
        result = st.session_state.ai_help_results[ticket['id']]
        display_ai_help_result(result)
        
        # Add a button to clear the result
        if st.button(f"🗑️ Clear AI Help", key=f"clear_{ticket['id']}"):
            del st.session_state.ai_help_results[ticket['id']]
            st.rerun()
    
    st.markdown("---")

# Main interface
option = st.radio(
    "What would you like to do?",
    ("Categorise sample tickets", "Enter a new ticket")
)

if option == "Categorise sample tickets":
    st.subheader("Sample Tickets and Classification")
    
    # Show RAG status
    if rag_pipeline:
        st.success("✅ RAG Pipeline Ready - AI Help available for How-to, Product, Best practices, API/SDK, and SSO topics")
        
        # Show Pinecone connection status
        try:
            stats = rag_pipeline.index.describe_index_stats()
            st.info(f"🔗 Connected to Pinecone index 'atlan-docs' with {stats['total_vector_count']} documents")
        except:
            st.warning("⚠️ Could not retrieve Pinecone index stats")
    else:
        st.warning("⚠️ RAG Pipeline not available - AI Help disabled")
    
    for ticket in tickets:
        analysis, classification = get_classification(ticket)
        ticket_menu_row(ticket, analysis, classification)

elif option == "Enter a new ticket":
    st.subheader("Add and Classify a New Ticket")
    
    with st.form("new_ticket"):
        new_id = st.text_input("Ticket ID")
        new_subject = st.text_input("Subject")
        new_body = st.text_area("Body")
        submitted = st.form_submit_button("Classify Ticket")
    
    if submitted and new_id and new_subject and new_body:
        new_ticket = {"id": new_id, "subject": new_subject, "body": new_body}
        try:
            analysis, classification = classify_ticket(client, new_subject, new_body)
            st.session_state.new_ticket_classification = (new_ticket, analysis, classification)
            st.success("Ticket classified!")
        except Exception as e:
            st.session_state.new_ticket_classification = None
            st.error(f"Classification failed: {e}")
    
    if st.session_state.new_ticket_classification:
        st.markdown("### New Ticket Classification")
        new_ticket, analysis, classification = st.session_state.new_ticket_classification
        ticket_menu_row(new_ticket, analysis, classification)

# Sidebar with information
with st.sidebar:
    st.markdown("### 📖 About AI Help")
    st.markdown("""
    The **Get AI Help** feature uses RAG (Retrieval-Augmented Generation) to provide solutions based on Atlan's official documentation.
    
    **When AI Help is used:**
    - How-to questions
    - Product inquiries  
    - Best practices
    - API/SDK issues
    - SSO problems
    
    **For other topics:**
    - Tickets are routed to appropriate teams
    """)
    
    if rag_pipeline:
        try:
            stats = rag_pipeline.index.describe_index_stats()
            st.success(f"📊 Pinecone Index: atlan-docs")
            st.success(f"📄 Documents loaded: {stats['total_vector_count']}")
            st.info(f"🤖 Embedding Model: {rag_pipeline.embedding_model}")
        except:
            st.success("✅ RAG Pipeline connected")
            st.info(f"🤖 Embedding Model: {rag_pipeline.embedding_model}")
    else:
        st.error("❌ RAG Pipeline not loaded")
        st.markdown("""
        **To enable AI Help:**
        1. Set your PINECONE_API_KEY in .env file
        2. Ensure 'atlan-docs' index exists in Pinecone
        3. Upload your documents to the index
        4. Restart the application
        """)
    
    st.markdown("---")
    st.markdown("### 🔧 Configuration")
    
    # Show environment variable status
    if os.getenv("PINECONE_API_KEY"):
        st.success("✅ Pinecone API Key loaded")
    else:
        st.error("❌ Pinecone API Key missing")
    
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Groq API Key loaded")
    else:
        st.error("❌ Groq API Key missing")
    
    # Add a button to refresh RAG pipeline connection
    if st.button("🔄 Refresh RAG Pipeline"):
        st.cache_resource.clear()
        st.rerun()
