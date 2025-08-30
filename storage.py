from dataclasses import dataclass
from typing import Optional, List, Tuple
import sqlite3
import os
from datetime import datetime

@dataclass
class FeedbackItem:
    row_idx: int
    label: int          # +1 for thumbs up, -1 for thumbs down
    comment: Optional[str] = None
    scenario: Optional[str] = None

class FeedbackStore:
    def __init__(self, path: str = "logs/feedback.sqlite"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS feedback(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    row_idx INTEGER NOT NULL,
                    label INTEGER NOT NULL,
                    comment TEXT,
                    scenario TEXT
                )
            """)
            con.commit()

    def add(self, item: FeedbackItem):
        with sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT INTO feedback(ts, row_idx, label, comment, scenario) VALUES(?,?,?,?,?)",
                (datetime.utcnow().isoformat(), item.row_idx, item.label, item.comment, item.scenario)
            )
            con.commit()

    def get_recent_avg_label(self, window: int = 200) -> Optional[float]:
        with sqlite3.connect(self.path) as con:
            cur = con.execute("SELECT label FROM feedback ORDER BY id DESC LIMIT ?", (window,))
            vals = [r[0] for r in cur.fetchall()]
        if not vals:
            return None
        return sum(vals) / float(len(vals))
