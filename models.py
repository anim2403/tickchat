from pydantic import BaseModel, Field, ValidationError
from typing import List

class TicketClassification(BaseModel):
    topic_tags: List[str] = Field(default_factory=list)
    core_problem: str = ""
    priority: str = ""
    sentiment: str = ""
