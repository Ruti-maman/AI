"""
ארכיטקטורת Event-Driven - Steps (שלבים)
שלב ב' של המטלה

כל Step הוא יחידת עבודה עצמאית עם:
- קלט ברור
- פלט ברור
- בדיקות תקינות
- יכולת לשגר Events
"""

from typing import Optional, List
import time
from datetime import datetime

from workflow_events import (
    Event, EventType, WorkflowState, ValidationResult
)


# ========================================
# Step 1: Query Validation
# ========================================

class QueryValidationStep:
    """
    שלב 1: בדיקת תקינות השאילתה
    
    קלט: שאילתה (string)
    פלט: ValidationResult + Event
    
    בדיקות:
    - השאילתה לא ריקה
    - השאילתה לא קצרה מדי (< 2 תווים)
    - השאילתה לא ארוכה מדי (> 500 תווים)
    """
    
    MIN_QUERY_LENGTH = 2
    MAX_QUERY_LENGTH = 500
    
    def execute(self, state: WorkflowState) -> ValidationResult:
        """
        בדיקת תקינות השאילתה
        
        Args:
            state: מצב הזרימה הנוכחי
            
        Returns:
            ValidationResult עם התוצאה
        """
        query = state.query.strip()
        
        # בדיקה 1: ריק?
        if not query:
            event = Event(
                type=EventType.QUERY_INVALID,
                step="query_validation",
                data={"reason": "empty_query"}
            )
            state.add_event(event)
            state.add_error("השאילתה ריקה")
            
            return ValidationResult(
                is_valid=False,
                message="השאילתה ריקה. אנא הכניסי שאלה.",
                next_event=EventType.WORKFLOW_ERROR
            )
        
        # בדיקה 2: קצר מדי?
        if len(query) < self.MIN_QUERY_LENGTH:
            event = Event(
                type=EventType.QUERY_INVALID,
                step="query_validation",
                data={"reason": "too_short", "length": len(query)}
            )
            state.add_event(event)
            state.add_error(f"השאילתה קצרה מדי: {len(query)} תווים")
            
            return ValidationResult(
                is_valid=False,
                message=f"השאילתה קצרה מדי ({len(query)} תווים). אנא הכניסי לפחות {self.MIN_QUERY_LENGTH} תווים.",
                next_event=EventType.WORKFLOW_ERROR
            )
        
        # בדיקה 3: ארוך מדי?
        if len(query) > self.MAX_QUERY_LENGTH:
            event = Event(
                type=EventType.QUERY_INVALID,
                step="query_validation",
                data={"reason": "too_long", "length": len(query)}
            )
            state.add_event(event)
            state.add_error(f"השאילתה ארוכה מדי: {len(query)} תווים")
            
            return ValidationResult(
                is_valid=False,
                message=f"השאילתה ארוכה מדי ({len(query)} תווים). מקסימום: {self.MAX_QUERY_LENGTH} תווים.",
                next_event=EventType.WORKFLOW_ERROR
            )
        
        # הכל תקין!
        event = Event(
            type=EventType.QUERY_VALIDATED,
            step="query_validation",
            data={"query_length": len(query)}
        )
        state.add_event(event)
        
        return ValidationResult(
            is_valid=True,
            message="שאילתה תקינה",
            next_event=EventType.EMBEDDING_START,
            metadata={"query_length": len(query)}
        )


# ========================================
# Step 2: Embedding
# ========================================

class EmbeddingStep:
    """
    שלב 2: המרת השאילתה ל-Embedding
    
    קלט: שאילתה מתוקפת
    פלט: וקטור embedding + Event
    
    בדיקות:
    - ה-embedding נוצר בהצלחה
    - ה-embedding לא ריק
    - זמן ההמרה סביר (< 5 שניות)
    """
    
    MAX_EMBEDDING_TIME_MS = 5000
    
    def __init__(self, embed_model):
        self.embed_model = embed_model
    
    def execute(self, state: WorkflowState) -> ValidationResult:
        """
        המרת השאילתה לוקטור
        
        Args:
            state: מצב הזרימה הנוכחי
            
        Returns:
            ValidationResult עם התוצאה
        """
        # Event - התחלה
        start_event = Event(
            type=EventType.EMBEDDING_START,
            step="embedding",
            data={"query": state.query}
        )
        state.add_event(start_event)
        
        # ביצוע Embedding
        start_time = time.time()
        
        try:
            embedding = self.embed_model.get_text_embedding(state.query)
            
            # חישוב זמן
            elapsed_ms = (time.time() - start_time) * 1000
            state.embedding_time_ms = elapsed_ms
            
            # בדיקה: האם Embedding תקין?
            if not embedding or len(embedding) == 0:
                error_event = Event(
                    type=EventType.EMBEDDING_FAILED,
                    step="embedding",
                    data={"reason": "empty_embedding"}
                )
                state.add_event(error_event)
                state.add_error("Embedding ריק")
                
                return ValidationResult(
                    is_valid=False,
                    message="שגיאה ביצירת embedding",
                    next_event=EventType.WORKFLOW_ERROR
                )
            
            # בדיקה: זמן סביר?
            if elapsed_ms > self.MAX_EMBEDDING_TIME_MS:
                # אזהרה אבל ממשיכים
                state.add_error(f"Embedding איטי: {elapsed_ms:.0f}ms")
            
            # שמירה ב-State
            state.query_embedding = embedding
            
            # Event - הצלחה!
            success_event = Event(
                type=EventType.EMBEDDING_SUCCESS,
                step="embedding",
                data={
                    "embedding_dim": len(embedding),
                    "time_ms": elapsed_ms
                }
            )
            state.add_event(success_event)
            
            return ValidationResult(
                is_valid=True,
                message="Embedding נוצר בהצלחה",
                next_event=EventType.RETRIEVAL_START,
                metadata={
                    "embedding_dim": len(embedding),
                    "time_ms": elapsed_ms
                }
            )
            
        except Exception as e:
            # שגיאה
            elapsed_ms = (time.time() - start_time) * 1000
            state.embedding_time_ms = elapsed_ms
            
            error_event = Event(
                type=EventType.EMBEDDING_FAILED,
                step="embedding",
                data={"error": str(e)}
            )
            state.add_event(error_event)
            state.add_error(f"שגיאה ב-embedding: {str(e)}")
            
            return ValidationResult(
                is_valid=False,
                message=f"שגיאה ב-embedding: {str(e)}",
                next_event=EventType.WORKFLOW_ERROR
            )


# ========================================
# Step 3: Retrieval
# ========================================

class RetrievalStep:
    """
    שלב 3: חיפוש במאגר הוקטורים
    
    קלט: embedding של השאילתה
    פלט: רשימת nodes רלוונטיים + Event
    
    בדיקות:
    - נמצאו תוצאות
    - ציון ה-confidence גבוה מספיק
    - זמן החיפוש סביר
    """
    
    MIN_CONFIDENCE_SCORE = 0.3
    MIN_RESULTS = 1
    MAX_RETRIEVAL_TIME_MS = 2000
    
    def __init__(self, retriever, similarity_top_k=3):
        self.retriever = retriever
        self.similarity_top_k = similarity_top_k
    
    def execute(self, state: WorkflowState) -> ValidationResult:
        """
        חיפוש במאגר וקטורים
        
        Args:
            state: מצב הזרימה הנוכחי
            
        Returns:
            ValidationResult עם התוצאה
        """
        # Event - התחלה
        start_event = Event(
            type=EventType.RETRIEVAL_START,
            step="retrieval",
            data={"top_k": self.similarity_top_k}
        )
        state.add_event(start_event)
        
        # ביצוע חיפוש
        start_time = time.time()
        
        try:
            # שליפת nodes
            nodes = self.retriever.retrieve(state.query)
            
            # חישוב זמן
            elapsed_ms = (time.time() - start_time) * 1000
            state.retrieval_time_ms = elapsed_ms
            
            # בדיקה 1: יש תוצאות?
            if not nodes or len(nodes) < self.MIN_RESULTS:
                no_results_event = Event(
                    type=EventType.RETRIEVAL_NO_RESULTS,
                    step="retrieval",
                    data={
                        "num_results": len(nodes) if nodes else 0,
                        "time_ms": elapsed_ms
                    }
                )
                state.add_event(no_results_event)
                state.add_error("לא נמצאו תוצאות רלוונטיות")
                
                return ValidationResult(
                    is_valid=False,
                    message="לא נמצאו תוצאות רלוונטיות לשאילתה.",
                    next_event=EventType.WORKFLOW_COMPLETE,  # ממשיכים עם תשובה ריקה
                    metadata={"num_results": 0}
                )
            
            # חישוב ציון ממוצע
            scores = [node.score for node in nodes if hasattr(node, 'score') and node.score is not None]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            state.confidence_score = avg_score
            
            # בדיקה 2: ציון גבוה מספיק?
            if avg_score < self.MIN_CONFIDENCE_SCORE:
                low_conf_event = Event(
                    type=EventType.RETRIEVAL_LOW_CONFIDENCE,
                    step="retrieval",
                    data={
                        "avg_score": avg_score,
                        "num_results": len(nodes),
                        "time_ms": elapsed_ms
                    }
                )
                state.add_event(low_conf_event)
                # ממשיכים אבל עם אזהרה
                state.add_error(f"Confidence נמוך: {avg_score:.2f}")
            
            # שמירה ב-State
            state.retrieved_nodes = nodes
            state.retrieval_scores = scores
            
            # Event - הצלחה!
            success_event = Event(
                type=EventType.RETRIEVAL_SUCCESS,
                step="retrieval",
                data={
                    "num_results": len(nodes),
                    "avg_score": avg_score,
                    "time_ms": elapsed_ms
                }
            )
            state.add_event(success_event)
            
            return ValidationResult(
                is_valid=True,
                message=f"נמצאו {len(nodes)} תוצאות",
                next_event=EventType.SYNTHESIS_START,
                metadata={
                    "num_results": len(nodes),
                    "avg_score": avg_score,
                    "time_ms": elapsed_ms
                }
            )
            
        except Exception as e:
            # שגיאה
            elapsed_ms = (time.time() - start_time) * 1000
            state.retrieval_time_ms = elapsed_ms
            
            error_event = Event(
                type=EventType.RETRIEVAL_NO_RESULTS,
                step="retrieval",
                data={"error": str(e)}
            )
            state.add_event(error_event)
            state.add_error(f"שגיאה ב-retrieval: {str(e)}")
            
            return ValidationResult(
                is_valid=False,
                message=f"שגיאה בחיפוש: {str(e)}",
                next_event=EventType.WORKFLOW_ERROR
            )


# ========================================
# Step 4: Response Synthesis
# ========================================

class SynthesisStep:
    """
    שלב 4: בניית התשובה הסופית
    
    קלט: nodes שנמצאו בחיפוש
    פלט: תשובה טקסטואלית + Event
    
    בדיקות:
    - התשובה לא ריקה
    - התשובה בגודל סביר
    - זמן הבנייה סביר
    """
    
    MIN_RESPONSE_LENGTH = 10
    MAX_SYNTHESIS_TIME_MS = 10000
    
    def __init__(self, response_synthesizer):
        self.response_synthesizer = response_synthesizer
    
    def execute(self, state: WorkflowState) -> ValidationResult:
        """
        בניית תשובה סופית
        
        Args:
            state: מצב הזרימה הנוכחי
            
        Returns:
            ValidationResult עם התוצאה
        """
        # Event - התחלה
        start_event = Event(
            type=EventType.SYNTHESIS_START,
            step="synthesis",
            data={"num_nodes": len(state.retrieved_nodes)}
        )
        state.add_event(start_event)
        
        # ביצוע Synthesis
        start_time = time.time()
        
        try:
            # בניית התשובה
            response = self.response_synthesizer.synthesize(
                query=state.query,
                nodes=state.retrieved_nodes
            )
            
            response_text = str(response)
            
            # חישוב זמן
            elapsed_ms = (time.time() - start_time) * 1000
            state.synthesis_time_ms = elapsed_ms
            
            # בדיקה: תשובה לא ריקה?
            if not response_text or len(response_text.strip()) < self.MIN_RESPONSE_LENGTH:
                error_event = Event(
                    type=EventType.SYNTHESIS_FAILED,
                    step="synthesis",
                    data={"reason": "empty_response"}
                )
                state.add_event(error_event)
                state.add_error("התשובה ריקה")
                
                return ValidationResult(
                    is_valid=False,
                    message="לא הצלחתי לבנות תשובה",
                    next_event=EventType.WORKFLOW_ERROR
                )
            
            # שמירה ב-State
            state.synthesized_response = response_text
            
            # Event - הצלחה!
            success_event = Event(
                type=EventType.SYNTHESIS_SUCCESS,
                step="synthesis",
                data={
                    "response_length": len(response_text),
                    "time_ms": elapsed_ms
                }
            )
            state.add_event(success_event)
            
            # סיימנו!
            final_event = Event(
                type=EventType.WORKFLOW_COMPLETE,
                step="synthesis",
                data=state.get_summary()
            )
            state.add_event(final_event)
            state.mark_complete()
            
            return ValidationResult(
                is_valid=True,
                message="תשובה נבנתה בהצלחה",
                next_event=EventType.WORKFLOW_COMPLETE,
                metadata={
                    "response_length": len(response_text),
                    "time_ms": elapsed_ms
                }
            )
            
        except Exception as e:
            # שגיאה
            elapsed_ms = (time.time() - start_time) * 1000
            state.synthesis_time_ms = elapsed_ms
            
            error_event = Event(
                type=EventType.SYNTHESIS_FAILED,
                step="synthesis",
                data={"error": str(e)}
            )
            state.add_event(error_event)
            state.add_error(f"שגיאה ב-synthesis: {str(e)}")
            
            return ValidationResult(
                is_valid=False,
                message=f"שגיאה בבניית תשובה: {str(e)}",
                next_event=EventType.WORKFLOW_ERROR
            )
