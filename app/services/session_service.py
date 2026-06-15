"""
SentinelAI — Session Service
Manages active candidate sessions, handles SQLite persistence, and compiles the final report.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any

from app.database.repository import DatabaseRepository
from app.database.models import SessionRecord

logger = logging.getLogger(__name__)

class SessionService:
    def __init__(self, repository: DatabaseRepository):
        self.repo = repository
        self.current_session: SessionRecord = None

    def start_session(self, candidate_id: str = "guest") -> str:
        session_id = str(uuid.uuid4())
        start_time = datetime.utcnow().isoformat() + "Z"
        
        self.current_session = SessionRecord(
            session_id=session_id,
            candidate_id=candidate_id,
            start_time=start_time,
            status="IN_PROGRESS"
        )
        self.repo.create_session(self.current_session)
        logger.info(f"Started session {session_id} for candidate {candidate_id}")
        return session_id

    def update_session(self, risk_score: int, violations_count: int) -> None:
        if not self.current_session:
            return
            
        self.current_session.final_risk_score = risk_score
        self.current_session.violations_count = violations_count
        self.repo.update_session(self.current_session)

    def end_session(self) -> Dict[str, Any]:
        if not self.current_session:
            return {}
            
        self.current_session.end_time = datetime.utcnow().isoformat() + "Z"
        
        # Determine final status
        if self.current_session.final_risk_score >= 80:
            self.current_session.status = "SUSPICIOUS"
        elif self.current_session.final_risk_score >= 50:
            self.current_session.status = "WARNING"
        else:
            self.current_session.status = "CLEARED"
            
        self.repo.update_session(self.current_session)
        
        alerts = self.repo.get_session_alerts(self.current_session.session_id)
        
        report = {
            "session_id": self.current_session.session_id,
            "candidate_id": self.current_session.candidate_id,
            "start_time": self.current_session.start_time,
            "end_time": self.current_session.end_time,
            "violations": self.current_session.violations_count,
            "session_risk_score": self.current_session.final_risk_score,
            "final_status": self.current_session.status,
            "timeline": [
                {
                    "time": a.timestamp,
                    "event": a.rule_name,
                    "priority": a.priority,
                    "severity": a.severity,
                    "risk_score": a.risk_score
                } for a in alerts
            ]
        }
        
        logger.info(f"Session {self.current_session.session_id} ended with status {self.current_session.status}")
        self.current_session = None
        return report
