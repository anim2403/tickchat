from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import json
import os
from groq import Groq
from models2 import TicketClassification
from llm_utils2 import classify_ticket

st.set_page_config(page_title="Ticket Classifier (Groq)", layout="wide")
st.title("AI-Powered Ticket Classifier (Groq SDK)")

# Load sample tickets from file
with open("sample_tickets.json") as f:
    tickets = json.load(f)

if "classifications" not in st.session_state:
    st.session_state.classifications = {}
if "new_ticket_classification" not in st.session_state:
    st.session_state.new_ticket_classification = None

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

def ticket_menu_row(ticket, analysis, classification):
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
    st.markdown("---")

option = st.radio(
    "What would you like to do?",
    ("Categorise sample tickets", "Enter a new ticket")
)

if option == "Categorise sample tickets":
    st.subheader("Sample Tickets and Classification")
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
