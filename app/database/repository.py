"""
SentinelAI — Database Repository
Handles SQLite initialization and CRUD operations for sessions and alerts.
"""

import sqlite3
import logging
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from app.database.models import SessionRecord, AlertRecord
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

class DatabaseRepository:
    def __init__(self):
        self.db_path = get_settings().project_root / "sentinel.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    candidate_id TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    violations_count INTEGER,
                    final_risk_score INTEGER,
                    status TEXT
                )
            ''')
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT,
                    rule_name TEXT,
                    severity TEXT,
                    priority TEXT,
                    risk_score INTEGER,
                    evidence_path TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            ''')
            conn.commit()

    def create_session(self, session: SessionRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (session_id, candidate_id, start_time, end_time, violations_count, final_risk_score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session.session_id, session.candidate_id, session.start_time, session.end_time, session.violations_count, session.final_risk_score, session.status))
            conn.commit()

    def update_session(self, session: SessionRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions 
                SET end_time = ?, violations_count = ?, final_risk_score = ?, status = ?
                WHERE session_id = ?
            ''', (session.end_time, session.violations_count, session.final_risk_score, session.status, session.session_id))
            conn.commit()

    def save_alert(self, alert: AlertRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (session_id, timestamp, rule_name, severity, priority, risk_score, evidence_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (alert.session_id, alert.timestamp, alert.rule_name, alert.severity, alert.priority, alert.risk_score, alert.evidence_path))
            conn.commit()

    def get_session_alerts(self, session_id: str) -> List[AlertRecord]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM alerts WHERE session_id = ? ORDER BY timestamp ASC', (session_id,))
            rows = cursor.fetchall()
            return [AlertRecord(**dict(row)) for row in rows]
