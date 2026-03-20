"""
ארכיטקטורת Event-Driven - הגדרות Events ו-State
שלב ב' של המטלה

Events מייצגים אירועים שקורים במערכת
State מייצג את מצב המערכת לאורך ה-Workflow
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


# ========================================
# Event Types - סוגי אירועים
# ========================================

class EventType(Enum):
    """סוגי אירועים שיכולים לקרות בזרימה"""
    # שלב קלט
    QUERY_RECEIVED = "query_received"
    QUERY_VALIDATED = "query_validated"
    QUERY_INVALID = "query_invalid"
    
    # שלב Embedding
    EMBEDDING_START = "embedding_start"
    EMBEDDING_SUCCESS = "embedding_success"
    EMBEDDING_FAILED = "embedding_failed"
    
    # שלב Retrieval
    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_SUCCESS = "retrieval_success"
    RETRIEVAL_NO_RESULTS = "retrieval_no_results"
    RETRIEVAL_LOW_CONFIDENCE = "retrieval_low_confidence"
    
    # שלב Synthesis
    SYNTHESIS_START = "synthesis_start"
    SYNTHESIS_SUCCESS = "synthesis_success"
    SYNTHESIS_FAILED = "synthesis_failed"
    
    # סיום
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"


@dataclass
class Event:
    """
    Event בסיסי - כל אירוע במערכת
    
    מאפיינים:
    - type: סוג האירוע
    - timestamp: מתי האירוע קרה
    - data: מידע נוסף על האירוע
    - step: באיזה שלב האירוע קרה
    """
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    step: Optional[str] = None
    
    def __repr__(self):
        return f"Event({self.type.value} at {self.timestamp.strftime('%H:%M:%S')})"


# ========================================
# State - מצב המערכת
# ========================================

@dataclass
class WorkflowState:
    """
    State של כל הזרימה
    
    מידע שנשמר לאורך כל ה-Workflow:
    - השאילתה המקורית
    - embeddings שנוצרו
    - תוצאות חיפוש
    - התשובה הסופית
    - אירועים שקרו
    - שגיאות
    """
    # קלט מקורי
    query: str = ""
    
    # שלב Embedding
    query_embedding: Optional[List[float]] = None
    embedding_time_ms: float = 0.0
    
    # שלב Retrieval
    retrieved_nodes: List[Any] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    retrieval_time_ms: float = 0.0
    
    # שלב Synthesis
    synthesized_response: str = ""
    synthesis_time_ms: float = 0.0
    
    # מטא-דאטה
    events: List[Event] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    current_step: str = "init"
    is_complete: bool = False
    confidence_score: float = 0.0
    
    # סטטיסטיקות
    total_time_ms: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def add_event(self, event: Event):
        """הוספת אירוע ל-State"""
        self.events.append(event)
        self.current_step = event.step or self.current_step
    
    def add_error(self, error: str):
        """הוספת שגיאה ל-State"""
        self.errors.append(error)
    
    def mark_complete(self):
        """סימון הזרימה כהושלמה"""
        self.is_complete = True
        self.end_time = datetime.now()
        if self.start_time:
            self.total_time_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def get_summary(self) -> Dict[str, Any]:
        """סיכום של ה-State"""
        return {
            "query": self.query,
            "current_step": self.current_step,
            "is_complete": self.is_complete,
            "confidence": self.confidence_score,
            "total_time_ms": self.total_time_ms,
            "num_events": len(self.events),
            "num_errors": len(self.errors),
            "num_results": len(self.retrieved_nodes)
        }


# ========================================
# Validation Results
# ========================================

@dataclass
class ValidationResult:
    """
    תוצאת בדיקת תקינות
    
    כל Step יכול להחזיר ValidationResult שקובע:
    - האם הקלט תקין
    - האם צריך להמשיך לשלב הבא
    - איזה אירוע לשגר
    """
    is_valid: bool
    message: str = ""
    next_event: Optional[EventType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
