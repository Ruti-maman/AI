"""
שלב ג' - אפליקציה מלאה: RAG + Workflow + Extraction + Router
משלב את כל 3 השלבים לאפליקציה אחת מקיפה

ארכיטקטורה:
1. שלב א': חיפוש סמנטי ב-VectorStoreIndex
2. שלב ב': Event-Driven Workflow עם validations
3. שלב ג': Router חכם + Data Extraction + שליפה מובנית

התהליך:
1. משתמש שולח שאילתה
2. Router מחליט: SEMANTIC / STRUCTURED / HYBRID
3. אם STRUCTURED → שליפה מה-extracted_data.json
4. אם SEMANTIC → חיפוש ב-VectorStoreIndex דרך Workflow
5. אם HYBRID → שילוב של שניהם
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dotenv import load_dotenv

import gradio as gr
from llama_index.core import (
    VectorStoreIndex,
    Settings
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from schema import ExtractedData, QueryIntent
from router import SmartQueryRouter
from workflow_engine import WorkflowEngine

# טעינת משתני סביבה
load_dotenv()

# הגדרת Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================
# תצורה
# ========================================

STORAGE_DIR = Path("storage")
CHROMA_DB_DIR = Path("chroma_db")
EXTRACTED_DATA_FILE = STORAGE_DIR / "extracted_data.json"


# ========================================
# טעינת רכיבים
# ========================================

class CompleteRAGSystem:
    """מערכת RAG מלאה עם כל 3 השלבים"""
    
    def __init__(self):
        logger.info("Initializing Complete RAG System...")
        
        # הגדרת LlamaIndex
        self._setup_llama_index()
        
        # טעינת VectorStoreIndex (שלב א')
        self.index = self._load_vector_index()
        
        # System Prompt - מגביל תשובות רק לנושאי RAG!
        from llama_index.core.prompts import PromptTemplate
        qa_prompt_tmpl = PromptTemplate(
            """אתה עוזר AI המתמחה **אך ורק** בנושא RAG (Retrieval-Augmented Generation).

⚠️ חשוב: ענה **רק** על השאלה הספציפית שנשאלה. אל תוסיף מידע נוסף שלא נשאל!

נושאים מותרים:
🚀 מה זה RAG?
🚀 למה RAG חשוב?
🚀 שביר מערכת RAG (אדריכלות, שלבים)
🚀 הבדל בין הטמעה לוקטוריזציה
🚀 טכנולוגיות נפוצות ב-RAG
🚀 יתרונות וחסרונות של RAG

כללים:
1. ענה רק על מה שנשאל - אל תוסיף הקדמות או הסברים נוספים
2. אם נשאל "למה RAG חשוב" - ענה רק על זה, אל תסביר גם "מה זה RAG"
3. היה תמציתי וישיר - 2-4 משפטים
4. השתמש רק במידע מהמסמכים שנמצאו
5. אם השאלה לא קשורה ל-RAG - השב: "אני עונה רק על שאלות על RAG."

---

מידע רלוונטי מהמסמכים:
{context_str}

שאלה: {query_str}

תשובה (ענה רק על השאלה, בקצרה וישירות):"""
        )
        
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact",
            text_qa_template=qa_prompt_tmpl
        )
        
        # טעינת Workflow Engine (שלב ב')
        self.workflow_engine = WorkflowEngine(self.index)
        
        # טעינת Structured Data + Router (שלב ג')
        self.extracted_data = self._load_extracted_data()
        self.router = SmartQueryRouter(
            extracted_data=self.extracted_data,
            use_llm_routing=True,
            llm=Settings.llm
        )
        
        logger.info("✓ Complete RAG System initialized")
    
    def _setup_llama_index(self):
        """הגדרת LlamaIndex Settings"""
        logger.info("Setting up LlamaIndex...")
        
        # בדיקת API Keys
        cohere_key = os.getenv("COHERE_API_KEY")
        if not cohere_key:
            raise ValueError("❌ חסר COHERE_API_KEY ב-.env")
        
        # Embeddings - Cohere
        embed_model = CohereEmbedding(
            api_key=cohere_key,
            model_name="embed-multilingual-v3.0",
            input_type="search_query"
        )
        
        # LLM - Cohere
        llm = Cohere(
            api_key=cohere_key,
            model="command-r-plus-08-2024",
            temperature=0.7
        )
        
        # הגדרות גלובליות
        Settings.embed_model = embed_model
        Settings.llm = llm
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        logger.info("✓ LlamaIndex configured (Cohere)")
    
    def _load_vector_index(self) -> VectorStoreIndex:
        """טעינת VectorStoreIndex"""
        logger.info("Loading VectorStoreIndex...")
        
        try:
            # בדיקת API Key
            pinecone_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX_NAME", "rag-agent-index")
            
            if not pinecone_key:
                raise ValueError("❌ חסר PINECONE_API_KEY ב-.env")
            
            # חיבור ל-Pinecone
            pc = Pinecone(api_key=pinecone_key)
            pinecone_index = pc.Index(index_name)
            
            # יצירת Vector Store
            vector_store = PineconeVectorStore(
                pinecone_index=pinecone_index,
                namespace="rag_agent_docs"
            )
            
            # טעינת אינדקס
            index = VectorStoreIndex.from_vector_store(vector_store)
            logger.info("✓ VectorStoreIndex loaded from Pinecone")
            return index
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise RuntimeError(
                "Failed to load VectorStoreIndex. "
                "Please run ingest_extraction.py first!"
            )
    
    def _load_extracted_data(self) -> ExtractedData:
        """טעינת המידע המובנה מ-JSON"""
        logger.info(f"Loading extracted data from: {EXTRACTED_DATA_FILE}")
        
        if not EXTRACTED_DATA_FILE.exists():
            logger.warning("No extracted data found. Running with empty data.")
            return ExtractedData()
        
        try:
            with open(EXTRACTED_DATA_FILE, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            
            # המרה ל-ExtractedData
            from schema import Decision, Rule, Warning, Dependency, Change, SourceInfo, Severity
            from datetime import datetime
            
            extracted = ExtractedData()
            
            # טעינת החלטות
            for dec_data in data_dict.get("items", {}).get("decisions", []):
                source = None
                if dec_data.get("source"):
                    source = SourceInfo(
                        tool=dec_data["source"]["tool"],
                        file=dec_data["source"]["file"],
                        anchor=dec_data["source"].get("anchor"),
                        line_range=dec_data["source"].get("line_range"),
                        observed_at=datetime.fromisoformat(dec_data["source"]["observed_at"])
                    )
                
                decision = Decision(
                    id=dec_data["id"],
                    title=dec_data["title"],
                    summary=dec_data["summary"],
                    tags=dec_data.get("tags", []),
                    source=source,
                    rationale=dec_data.get("rationale"),
                    alternatives=dec_data.get("alternatives")
                )
                extracted.add_decision(decision)
            
            # טעינת כללים
            for rule_data in data_dict.get("items", {}).get("rules", []):
                source = None
                if rule_data.get("source"):
                    source = SourceInfo(
                        tool=rule_data["source"]["tool"],
                        file=rule_data["source"]["file"],
                        anchor=rule_data["source"].get("anchor"),
                        line_range=rule_data["source"].get("line_range"),
                        observed_at=datetime.fromisoformat(rule_data["source"]["observed_at"])
                    )
                
                rule = Rule(
                    id=rule_data["id"],
                    rule=rule_data["rule"],
                    scope=rule_data["scope"],
                    notes=rule_data.get("notes"),
                    source=source,
                    exceptions=rule_data.get("exceptions")
                )
                extracted.add_rule(rule)
            
            # טעינת אזהרות
            for warn_data in data_dict.get("items", {}).get("warnings", []):
                source = None
                if warn_data.get("source"):
                    source = SourceInfo(
                        tool=warn_data["source"]["tool"],
                        file=warn_data["source"]["file"],
                        anchor=warn_data["source"].get("anchor"),
                        line_range=warn_data["source"].get("line_range"),
                        observed_at=datetime.fromisoformat(warn_data["source"]["observed_at"])
                    )
                
                severity = Severity[warn_data["severity"].upper()]
                warning = Warning(
                    id=warn_data["id"],
                    area=warn_data["area"],
                    message=warn_data["message"],
                    severity=severity,
                    source=source,
                    mitigation=warn_data.get("mitigation")
                )
                extracted.add_warning(warning)
            
            # טעינת תלויות
            for dep_data in data_dict.get("items", {}).get("dependencies", []):
                source = None
                if dep_data.get("source"):
                    source = SourceInfo(
                        tool=dep_data["source"]["tool"],
                        file=dep_data["source"]["file"],
                        anchor=dep_data["source"].get("anchor"),
                        line_range=dep_data["source"].get("line_range"),
                        observed_at=datetime.fromisoformat(dep_data["source"]["observed_at"])
                    )
                
                dependency = Dependency(
                    id=dep_data["id"],
                    name=dep_data["name"],
                    version=dep_data.get("version"),
                    purpose=dep_data["purpose"],
                    source=source,
                    required=dep_data.get("required", True)
                )
                extracted.add_dependency(dependency)
            
            # טעינת שינויים
            for change_data in data_dict.get("items", {}).get("changes", []):
                source = None
                if change_data.get("source"):
                    source = SourceInfo(
                        tool=change_data["source"]["tool"],
                        file=change_data["source"]["file"],
                        anchor=change_data["source"].get("anchor"),
                        line_range=change_data["source"].get("line_range"),
                        observed_at=datetime.fromisoformat(change_data["source"]["observed_at"])
                    )
                
                change = Change(
                    id=change_data["id"],
                    description=change_data["description"],
                    category=change_data["category"],
                    impact=change_data["impact"],
                    source=source,
                    migration_notes=change_data.get("migration_notes")
                )
                extracted.add_change(change)
            
            logger.info(f"✓ Loaded {len(extracted.get_all_items())} items from extracted data")
            return extracted
            
        except Exception as e:
            logger.error(f"Error loading extracted data: {e}")
            return ExtractedData()
    
    def query(self, query_text: str, use_workflow: bool = True) -> str:
        """
        ביצוע שאילתה מלא
        
        Args:
            query_text: טקסט השאילתה
            use_workflow: האם להשתמש ב-Workflow (שלב ב')
        
        Returns:
            תשובה טקסטואלית
        """
        logger.info(f"Processing query: {query_text}")
        
        try:
            # שלב 1: ניתוב (שלב ג')
            intent, structured_response = self.router.query(query_text)
            logger.info(f"Router decision: {intent.value}")
            
            # שלב 2: ביצוע לפי Intent
            if intent == QueryIntent.STRUCTURED:
                # שליפה מובנית בלבד
                return f"🔍 **שליפה מובנית:**\n\n{structured_response}"
            
            elif intent == QueryIntent.HYBRID:
                # שילוב של מובנה + סמנטי
                semantic_response = self._execute_semantic_query(query_text, use_workflow)
                
                combined = f"📊 **תוצאות מובנות:**\n{structured_response}\n\n"
                combined += f"🔍 **חיפוש סמנטי:**\n{semantic_response}"
                return combined
            
            else:  # SEMANTIC
                # חיפוש סמנטי
                return self._execute_semantic_query(query_text, use_workflow)
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ שגיאה: {str(e)}"
    
    def _execute_semantic_query(self, query_text: str, use_workflow: bool) -> str:
        """ביצוע חיפוש סמנטי"""
        if use_workflow:
            # שלב ב': Event-Driven Workflow
            logger.info("Using Event-Driven Workflow")
            state = self.workflow_engine.execute(query_text)
            
            if state.success and state.final_response:
                return state.final_response
            else:
                error_msg = state.error_message or "Unknown error"
                return f"⚠️ Workflow failed: {error_msg}"
        else:
            # שלב א': RAG פשוט
            logger.info("Using simple QueryEngine")
            response = self.query_engine.query(query_text)
            return str(response)


# ========================================
# Gradio Interface
# ========================================

def create_gradio_app(rag_system: CompleteRAGSystem):
    """יצירת Gradio UI"""
    
    def chat_fn(message, history, use_workflow):
        """פונקציית צ'אט"""
        try:
            response = rag_system.query(message, use_workflow=use_workflow)
            return response
        except Exception as e:
            return f"❌ שגיאה: {str(e)}"
    
    def get_statistics():
        """החזרת סטטיסטיקה על הנתונים"""
        stats = rag_system.extracted_data.to_dict()["statistics"]
        
        stats_text = f"""
📊 **סטטיסטיקת המערכת:**

**מידע מובנה:**
- 📋 החלטות: {stats['decisions']}
- 📏 כללים: {stats['rules']}
- ⚠️ אזהרות: {stats['warnings']}
- 📦 תלויות: {stats['dependencies']}
- 🔄 שינויים: {stats['changes']}
- **סה"כ:** {stats['total_items']} פריטים

**ארכיטקטורה:**
- ✅ שלב א': VectorStoreIndex + Pinecone
- ✅ שלב ב': Event-Driven Workflow
- ✅ שלב ג': Data Extraction + Smart Router

**רכיבים:**
- 🧠 Embeddings: Cohere embed-multilingual-v3.0
- 💬 LLM: Cohere command-r-plus-08-2024
- 🔍 Vector Store: Pinecone (serverless)
- 🎯 Router: Hybrid (Keyword + LLM)
        """
        return stats_text
    
    # יצירת ממשק
    with gr.Blocks(
        title="שלב ג' - Complete RAG System",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🎯 שלב ג' - מערכת RAG מלאה
        
        **אפליקציה מקיפה ששולבת את כל 3 השלבים:**
        - 📚 **שלב א':** RAG בסיסי עם VectorStoreIndex
        - ⚙️ **שלב ב':** Event-Driven Workflow עם Validations
        - 🧠 **שלב ג':** Data Extraction + Smart Router
        
        ---
        
        ## 🔍 איך זה עובד?
        
        1. **Router חכם** מנתח את השאילתה:
           - 📊 **STRUCTURED** → שליפה מידע מובנה (החלטות, כללים)
           - 🔍 **SEMANTIC** → חיפוש סמנטי במסמכים
           - 🎯 **HYBRID** → שילוב של שניהם
        
        2. **Workflow Engine** מבצע validations ו-error handling
        
        3. **Response Synthesis** מחזיר תשובה מקיפה
        
        ---
        """)
        
        with gr.Tab("💬 צ'אט"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="שיחה",
                        height=500,
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="שאילתה",
                            placeholder="הקלד שאילתה כאן...",
                            scale=4
                        )
                        use_workflow_checkbox = gr.Checkbox(
                            label="השתמש ב-Workflow (שלב ב')",
                            value=True,
                            scale=1
                        )
                    
                    with gr.Row():
                        submit_btn = gr.Button("שלח", variant="primary")
                        clear_btn = gr.ClearButton([msg, chatbot])
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 דוגמאות")
                    
                    example_queries = [
                        "מה ההחלטות שהתקבלו?",
                        "הצג לי את כל האזהרות",
                        "ספר לי על ה-API",
                        "מהן התלויות של הפרויקט?",
                        "איך עובד authentication?",
                        "כללים בנושא UI",
                    ]
                    
                    for example in example_queries:
                        gr.Button(example, size="sm").click(
                            lambda x=example: x,
                            outputs=msg
                        )
        
        with gr.Tab("📊 סטטיסטיקה"):
            stats_display = gr.Markdown(value=get_statistics())
            gr.Button("רענן סטטיסטיקה").click(
                get_statistics,
                outputs=stats_display
            )
        
        with gr.Tab("❓ עזרה"):
            gr.Markdown("""
            ## 🎓 מדריך שימוש
            
            ### סוגי שאילתות:
            
            #### 📊 שאילתות מובנות (STRUCTURED):
            - "הצג לי את כל ההחלטות"
            - "רשימת אזהרות"
            - "מה התלויות?"
            - "ספר לי את השינויים"
            - "כללים בנושא API"
            
            #### 🔍 שאילתות סמנטיות (SEMANTIC):
            - "איך עובד ה-authentication?"
            - "הסבר לי על ה-architecture"
            - "מה ההבדל בין Cursor ל-Claude?"
            - "למה נבחר TypeScript?"
            
            #### 🎯 שאילתות היברידיות (HYBRID):
            - "הצג החלטות והסבר למה"
            - "אילו אזהרות יש על API?"
            
            ---
            
            ### ⚙️ הגדרות:
            
            **Workflow Mode:**
            - ✅ **מופעל:** שימוש ב-Event-Driven Workflow (שלב ב')
              - Validations מלאות
              - Error handling
              - Event tracking
            - ❌ **כבוי:** RAG פשוט (שלב א')
              - מהיר יותר
              - פחות validations
            
            ---
            
            ### 📝 טיפים:
            
            1. לשליפה מהירה של רשימות - השתמש במילות מפתח: "הצג", "רשימה", "כל"
            2. לשאלות מעמיקות - נסח שאלות פתוחות: "איך", "למה", "הסבר"
            3. אם התשובה לא מספקת - נסה לנסח מחדש את השאילתה
            """)
        
        # חיבור אירועים
        msg.submit(
            chat_fn,
            inputs=[msg, chatbot, use_workflow_checkbox],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        submit_btn.click(
            chat_fn,
            inputs=[msg, chatbot, use_workflow_checkbox],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )
    
    return demo


# ========================================
# Main
# ========================================

def main():
    """הרצת האפליקציה"""
    
    print("\n" + "🚀" * 30)
    print("   שלב ג': Complete RAG System - Starting...")
    print("🚀" * 30 + "\n")
    
    try:
        # אתחול המערכת
        rag_system = CompleteRAGSystem()
        
        # יצירת Gradio UI
        demo = create_gradio_app(rag_system)
        
        # הרצה
        print("\n" + "✅" * 30)
        print("   המערכת מוכנה!")
        print("✅" * 30)
        print("""
🌐 פתח דפדפן ב: http://localhost:7860

📊 ארכיטקטורה:
   ✓ שלב א': VectorStoreIndex + ChromaDB
   ✓ שלב ב': Event-Driven Workflow
   ✓ שלב ג': Data Extraction + Router

🎯 נסה שאילתות כמו:
   • "הצג את כל ההחלטות"
   • "ספר לי על ה-API"
   • "מהן האזהרות על authentication?"
        """)
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
        
    except Exception as e:
        logger.error(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n⚠️  אם אתה רואה שגיאה, הרץ קודם:")
        print("   python ingest_extraction.py")


if __name__ == "__main__":
    main()
