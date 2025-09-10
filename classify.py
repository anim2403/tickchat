import os
import json
from enum import Enum
from typing import List, Optional

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from groq import Groq


# =========================
# 1) Enums & Pydantic model
# =========================

class TicketTopic(str, Enum):
    HOW_TO = "how_to"
    PRODUCT = "product"
    CONNECTOR = "connector"
    LINEAGE = "lineage"
    API_SDK = "api_sdk"
    SSO = "sso"
    GLOSSARY = "glossary"
    BEST_PRACTICES = "best_practices"
    SENSITIVE_DATA = "sensitive_data"
    OTHER = "other"

class CustomerSentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    CURIOUS = "curious"

class TicketUrgency(str, Enum):
    P2 = "low"     # P2
    P1 = "medium"  # P1
    P0 = "high"    # P0

class TicketClassification(BaseModel):
    topic: TicketTopic
    urgency: TicketUrgency
    sentiment: CustomerSentiment
    confidence: float = Field(ge=0, le=1, description="Confidence score for the classification")


TOPIC_LABELS = {
    "how_to": "How-to",
    "product": "Product",
    "connector": "Connector",
    "lineage": "Lineage",
    "api_sdk": "API/SDK",
    "sso": "SSO",
    "glossary": "Glossary",
    "best_practices": "Best practices",
    "sensitive_data": "Sensitive data",
    "other": "Other",
}

URGENCY_LABELS = {"low": "P2 (Low)", "medium": "P1 (Medium)", "high": "P0 (High)"}
SENTIMENT_LABELS = {
    "angry": "Angry",
    "frustrated": "Frustrated",
    "neutral": "Neutral",
    "curious": "Curious",
}


# =========================
# 2) Groq client (no OpenAI)
# =========================

@st.cache_resource(show_spinner=False)
def get_groq_client() -> Groq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Set it in your environment or Streamlit secrets."
        )
    return Groq(api_key=api_key)


SYSTEM_INSTRUCTIONS = """\
You are a ticket triage assistant.
Return STRICT JSON (no extra text) that matches this schema exactly:
{
  "topic": one of ["how_to","product","connector","lineage","api_sdk","sso","glossary","best_practices","sensitive_data","other"],
  "urgency": one of ["low","medium","high"],   // maps to P2,P1,P0 respectively
  "sentiment": one of ["angry","frustrated","neutral","curious"],
  "confidence": number between 0 and 1,
}

Rules:
- Pick the single best 'topic' label from the list above (use "other" if unclear).
- 'urgency' reflects business criticality implied by the text (low=P2, medium=P1, high=P0).
- 'sentiment' is the customer's tone.
- 'confidence' is your self-estimated probability of correctness (0..1).
Return ONLY valid JSON, with double-quoted keys and string values. No markdown, no commentary, no code fences.
"""


def _parse_json_strict(text: str) -> dict:
    """
    Best-effort strict JSON parser. Groq doesn't have a 'response_format=json' flag,
    so we enforce via prompt and parse here.
    """
    # Trim whitespace and stray code fences if any slipped in.
    t = text.strip()
    if t.startswith("```"):
        # remove possible ```json ... ```
        t = t.strip("`")
        # after stripping backticks, sometimes there is "json\n{...}"
        if "\n" in t and t.split("\n", 1)[0].lower() == "json":
            t = t.split("\n", 1)[1]
        # ensure we only keep from first { to last }
        if "{" in t and "}" in t:
            t = t[t.find("{"): t.rfind("}") + 1]
    # Fallback: slice to the outermost JSON object if extra text exists
    if "{" in t and "}" in t:
        t = t[t.find("{"): t.rfind("}") + 1]
    return json.loads(t)


def classify_ticket(client: Groq, ticket_text: str, model: str = "llama-3.1-8b-instant") -> TicketClassification:
    """
    Calls the Groq Chat Completions API and parses into TicketClassification.
    """
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": ticket_text},
        ],
    )

    # Extract assistant text
    try:
        raw = completion.choices[0].message["content"]
    except Exception:
        # Some SDK versions expose .message.content as attribute
        raw = completion.choices[0].message.content

    try:
        data = _parse_json_strict(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {raw}") from e

    try:
        return TicketClassification(**data)
    except ValidationError as ve:
        raise ValueError(f"Model JSON failed schema validation:\n{ve}\n\nJSON was: {data}") from ve


# =========================
# 3) I/O helpers & UI
# =========================

@st.cache_data(show_spinner=False)
def load_tickets_from_file(path: str) -> List[dict]:
    """
    Accepts:
      - a list of ticket dicts: [{id, subject, body}, ...]
      - or a single ticket dict: {id, subject, body}
    Returns normalized list.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return obj
    raise ValueError("sample_ticket.json must be a JSON object or an array of objects")

def normalize_ticket_dict(d: dict, idx: int) -> dict:
    """
    Ensures we always have id/subject/body keys.
    """
    return {
        "id": d.get("id") or f"TICKET-{idx+1}",
        "subject": d.get("subject") or "(no subject)",
        "body": d.get("body") or "",
    }

def render_classification_card(ticket: dict, classification: TicketClassification):
    st.markdown("---")
    st.markdown(f"**Ticket ID:** `{ticket['id']}`")
    st.markdown(f"**Subject:** {ticket['subject']}")
    st.markdown("**Body:**")
    st.write(ticket["body"])

    st.markdown("### Classification")
    col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.1, 1.0])
    with col1:
        st.metric("Topic", TOPIC_LABELS[classification.topic.value])
    with col2:
        st.metric("Urgency", URGENCY_LABELS[classification.urgency.value])
    with col3:
        st.metric("Sentiment", SENTIMENT_LABELS[classification.sentiment.value])
    with col4:
        st.metric("Confidence", f"{classification.confidence:.2f}")

   
    

# =========================
# 4) Main app
# =========================

def main():
    st.set_page_config(page_title="Ticket Classification", page_icon="🧭", layout="wide")
    st.title("🎫 Ticket Classification (Groq API)")
    st.caption("Classify support tickets into Topic / Urgency / Sentiment with key facts.")

    # Sidebar: model + source
    st.sidebar.header("Settings")
    model = st.sidebar.selectbox(
        "Groq Model",
        [
            "llama-3.1-8b-instant",   # fast & free-tier friendly
            "llama-3.1-70b-versatile",
            "gemma2-9b-it",
        ],
        index=0,
    )
    st.sidebar.write("Tip: **llama-3.1-8b-instant** is very fast; larger models may be more accurate.")

    # Initialize Groq client
    try:
        client = get_groq_client()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # ---------- Input sources ----------
    st.subheader("Batch: Load from file")
    file_col, manual_col = st.columns([1, 1])

    with file_col:
        st.write("Place a file named `sample_ticket.json` in the working directory.")
        if st.button("🔄 Classify sample_ticket.json"):
            try:
                tickets_raw = load_tickets_from_file("sample_ticket.json")
                if not tickets_raw:
                    st.warning("No tickets found in sample_ticket.json")
                else:
                    for i, t in enumerate(tickets_raw):
                        tnorm = normalize_ticket_dict(t, i)
                        try:
                            cl = classify_ticket(client, tnorm["body"], model=model)
                            render_classification_card(tnorm, cl)
                        except Exception as e:
                            st.error(f"Classification failed for {tnorm['id']}: {e}")
                    st.markdown("---")
                    st.success("Done.")
            except FileNotFoundError:
                st.error("`sample_ticket.json` not found. Put it next to app.py.")
            except Exception as e:
                st.exception(e)

    with manual_col:
        st.write("Enter a single ticket below and classify it.")
        manual_subject = st.text_input("Subject", value="(manual ticket)")
        manual_body = st.text_area(
            "Body",
            height=180,
            placeholder="Paste the customer's ticket here...",
        )
        if st.button("✨ Classify manual ticket"):
            if not manual_body.strip():
                st.warning("Please paste the ticket body.")
            else:
                tnorm = {"id": "TICKET-MANUAL", "subject": manual_subject.strip(), "body": manual_body.strip()}
                try:
                    cl = classify_ticket(client, tnorm["body"], model=model)
                    render_classification_card(tnorm, cl)
                except Exception as e:
                    st.error(f"Classification failed: {e}")

    # Helpful info
    with st.expander("JSON format expected from the model"):
        st.code(
            json.dumps(
                {
                    "topic": "how_to",
                    "urgency": "low",
                    "sentiment": "neutral",
                    "confidence": 0.92,
                },
                indent=2,
            ),
            language="json",
        )

    st.caption(
        "Topics allowed: how_to, product, connector, lineage, api_sdk, sso, glossary, best_practices, sensitive_data, other. "
        "Urgency: low (P2), medium (P1), high (P0). Sentiment: angry, frustrated, neutral, curious."
    )

if __name__ == "__main__":
    main()
