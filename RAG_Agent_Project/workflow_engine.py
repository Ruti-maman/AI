"""
ארכיטקטורת Event-Driven - Workflow Engine
שלב ב' של המטלה

Engine שמריץ את ה-Workflow עם ניהול Events ו-State
"""

from typing import Optional, Callable
from datetime import datetime

from workflow_events import Event, EventType, WorkflowState, ValidationResult
from workflow_steps import (
    QueryValidationStep,
    EmbeddingStep,
    RetrievalStep,
    SynthesisStep
)


# ========================================
# Workflow Engine
# ========================================

class WorkflowEngine:
    """
    מנוע הזרימה - מריץ את כל ה-Steps עם ניהול Events
    
    האחראי על:
    - ניהול State
    - הפעלת Steps לפי סדר
    - טיפול ב-Events
    - הפעלת validations
    - טיפול בשגיאות
    """
    
    def __init__(
        self,
        embed_model,
        retriever,
        response_synthesizer,
        verbose: bool = True
    ):
        """
        יצירת Engine
        
        Args:
            embed_model: מודל ה-embedding
            retriever: מנוע החיפוש
            response_synthesizer: מנוע בניית התשובה
            verbose: האם להדפיס לוגים
        """
        self.embed_model = embed_model
        self.retriever = retriever
        self.response_synthesizer = response_synthesizer
        self.verbose = verbose
        
        # יצירת Steps
        self.query_validation_step = QueryValidationStep()
        self.embedding_step = EmbeddingStep(embed_model)
        self.retrieval_step = RetrievalStep(retriever, similarity_top_k=3)
        self.synthesis_step = SynthesisStep(response_synthesizer)
        
        # Event handlers (אפשר להוסיף callbacks)
        self.event_handlers = {}
    
    def log(self, message: str):
        """הדפסת לוג (אם verbose=True)"""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """
        רישום handler לאירוע מסוים
        
        Args:
            event_type: סוג האירוע
            handler: פונקציה שתופעל כשהאירוע קורה
        """
        self.event_handlers[event_type] = handler
    
    def fire_event(self, event: Event, state: WorkflowState):
        """
        שיגור אירוע והפעלת handler אם קיים
        
        Args:
            event: האירוע
            state: המצב הנוכחי
        """
        state.add_event(event)
        
        # הפעלת handler אם קיים
        if event.type in self.event_handlers:
            self.event_handlers[event.type](event, state)
    
    def execute(self, query: str) -> WorkflowState:
        """
        הרצת הזרימה המלאה
        
        Pipeline:
        1. Query Validation
        2. Embedding
        3. Retrieval
        4. Response Synthesis
        
        Args:
            query: השאילתה מהמשתמש
            
        Returns:
            WorkflowState עם כל המידע
        """
        # יצירת State חדש
        state = WorkflowState()
        state.query = query
        state.start_time = datetime.now()
        
        self.log("="*70)
        self.log("🚀 מתחיל Workflow Event-Driven")
        self.log("="*70)
        self.log(f"📝 שאילתה: {query}")
        self.log("")
        
        # Event - קבלת שאילתה
        received_event = Event(
            type=EventType.QUERY_RECEIVED,
            step="init",
            data={"query": query}
        )
        self.fire_event(received_event, state)
        
        # ========== STEP 1: VALIDATION ==========
        self.log("1️⃣ שלב: Query Validation")
        validation_result = self.query_validation_step.execute(state)
        
        if not validation_result.is_valid:
            self.log(f"   ❌ {validation_result.message}")
            state.mark_complete()
            return state
        
        self.log(f"   ✅ {validation_result.message}")
        self.log("")
        
        # ========== STEP 2: EMBEDDING ==========
        self.log("2️⃣ שלב: Embedding")
        embedding_result = self.embedding_step.execute(state)
        
        if not embedding_result.is_valid:
            self.log(f"   ❌ {embedding_result.message}")
            state.mark_complete()
            return state
        
        self.log(f"   ✅ {embedding_result.message}")
        self.log(f"   📊 Dimension: {embedding_result.metadata.get('embedding_dim')}")
        self.log(f"   ⏱️  Time: {embedding_result.metadata.get('time_ms'):.0f}ms")
        self.log("")
        
        # ========== STEP 3: RETRIEVAL ==========
        self.log("3️⃣ שלב: Retrieval")
        retrieval_result = self.retrieval_step.execute(state)
        
        if not retrieval_result.is_valid:
            self.log(f"   ❌ {retrieval_result.message}")
            # ממשיכים עם תשובה ריקה
            state.synthesized_response = "לא נמצאו מסמכים רלוונטיים לשאילתה."
            state.mark_complete()
            return state
        
        self.log(f"   ✅ {retrieval_result.message}")
        self.log(f"   📊 Avg Score: {retrieval_result.metadata.get('avg_score', 0):.2f}")
        self.log(f"   ⏱️  Time: {retrieval_result.metadata.get('time_ms', 0):.0f}ms")
        self.log("")
        
        # ========== STEP 4: SYNTHESIS ==========
        self.log("4️⃣ שלב: Response Synthesis")
        synthesis_result = self.synthesis_step.execute(state)
        
        if not synthesis_result.is_valid:
            self.log(f"   ❌ {synthesis_result.message}")
            state.mark_complete()
            return state
        
        self.log(f"   ✅ {synthesis_result.message}")
        self.log(f"   📊 Response Length: {synthesis_result.metadata.get('response_length')} chars")
        self.log(f"   ⏱️  Time: {synthesis_result.metadata.get('time_ms', 0):.0f}ms")
        self.log("")
        
        # ========== סיכום ==========
        self.log("="*70)
        self.log("🎉 Workflow הושלם!")
        self.log("="*70)
        self.log(f"⏱️  Total Time: {state.total_time_ms:.0f}ms")
        self.log(f"📊 Confidence: {state.confidence_score:.2f}")
        self.log(f"📝 Events: {len(state.events)}")
        self.log(f"⚠️  Errors: {len(state.errors)}")
        self.log("")
        
        if state.errors:
            self.log("⚠️  אזהרות:")
            for error in state.errors:
                self.log(f"   - {error}")
            self.log("")
        
        return state


# ========================================
# Workflow Manager
# ========================================

class WorkflowManager:
    """
    מנהל זרימות - שומר היסטוריה של Workflows
    
    יכול לשמש:
    - ניתוח ביצועים
    - Debug
    - מעקב אחר queries
    """
    
    def __init__(self):
        self.workflows = []
    
    def add_workflow(self, state: WorkflowState):
        """הוספת workflow להיסטוריה"""
        self.workflows.append(state)
    
    def get_statistics(self):
        """סטטיסטיקות על כל ה-workflows"""
        if not self.workflows:
            return {}
        
        total = len(self.workflows)
        successful = sum(1 for w in self.workflows if w.is_complete and not w.errors)
        failed = total - successful
        
        avg_time = sum(w.total_time_ms for w in self.workflows) / total
        avg_confidence = sum(w.confidence_score for w in self.workflows) / total
        
        return {
            "total_workflows": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "avg_time_ms": avg_time,
            "avg_confidence": avg_confidence
        }
    
    def get_recent_queries(self, n: int = 10):
        """שליפת n queries אחרונים"""
        return [w.query for w in self.workflows[-n:]]
    
    def get_slow_queries(self, threshold_ms: float = 1000):
        """שליפת queries איטיים"""
        return [
            (w.query, w.total_time_ms) 
            for w in self.workflows 
            if w.total_time_ms > threshold_ms
        ]
