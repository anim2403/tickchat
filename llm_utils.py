import json
from groq import Groq
from models import TicketClassification
from pydantic import ValidationError

TOPIC_TAG_OPTIONS = [
    "How-to", "Product", "Connector", "Lineage", "API/SDK", 
    "SSO", "Glossary", "Best practices", "Sensitive data"
]
PRIORITY_OPTIONS = ["P0", "P1", "P2"]
SENTIMENT_OPTIONS = ["Frustrated", "Curious", "Angry", "Neutral"]

SYSTEM_INSTRUCTIONS = f"""
You are an expert support ticket classifier.
Given a support ticket, classify it in this JSON schema:
{{
  "topic_tags": [list, choose relevant from {TOPIC_TAG_OPTIONS}],
  "core_problem": "short topic string",
  "priority": "P0, P1, or P2",
  "sentiment": "Frustrated, Curious, Angry, Neutral"
}}
Only output valid JSON. No explanation.
"""

def _parse_json_strict(raw: str):
    # Remove code block formatting if present
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    if raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return json.loads(raw)

def classify_ticket(client: Groq, subject: str, body: str, model: str = "llama-3.1-8b-instant") -> TicketClassification:
    ticket_text = f"Subject: {subject}\nBody: {body}"
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": ticket_text},
        ],
    )
    try:
        raw = completion.choices[0].message["content"]
    except Exception:
        raw = completion.choices[0].message.content

    try:
        data = _parse_json_strict(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {raw}") from e

    try:
        return TicketClassification(**data)
    except ValidationError as ve:
        raise ValueError(f"Model JSON failed schema validation:\n{ve}\n\nJSON was: {data}") from ve
