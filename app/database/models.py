"""
SentinelAI — Database Models
Defines the SQLite table schemas for state persistence.
"""

from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class SessionRecord(BaseModel):
    session_id: str
    candidate_id: str
    start_time: str
    end_time: Optional[str] = None
    violations_count: int = 0
    final_risk_score: int = 0
    status: str = "IN_PROGRESS"

class AlertRecord(BaseModel):
    id: Optional[int] = None
    session_id: str
    timestamp: str
    rule_name: str
    severity: str
    priority: str
    risk_score: int
    evidence_path: Optional[str] = None
