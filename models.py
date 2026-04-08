from pydantic import BaseModel
from typing import Optional, Literal, List


class Action(BaseModel):
    severity: Literal["low", "medium", "high"]
    action_type: Literal["reply", "escalate", "ignore"]
    reasoning: Optional[str] = None


class Observation(BaseModel):
    current_ticket: dict
    step: int
    tickets_remaining: int
    processed_ids: List[str]
    message: str


class Reward(BaseModel):
    value: float
    reason: str
    partial_credit: bool
