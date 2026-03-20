"""
אפליקציה מושלמת - שלב ב': Event-Driven Workflow
✅ Workflow עם Steps מוגדרים
✅ Event Management
✅ State Tracking
✅ Validations

רכיבים:
1. VectorStoreIndex - טעינת האינדקס
2. WorkflowEngine - ניהול הזרימה
3. Events & State - מעקב אחר כל שלב
4. Validations - בדיקות בכל שלב
5. Gradio UI - ממשק משתמש
"""

import os
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from workflow_engine import WorkflowEngine, WorkflowManager
from workflow_events import EventType

# טעינת משתני סביבה
load_dotenv()

print("="*70)
print("🔧 מפעיל מערכת RAG - שלב ב': Event-Driven Workflow")
print("="*70)
print()

# ========================================
# בדיקת API Keys
# ========================================
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-agent-index")

if not COHERE_API_KEY:
    print("❌ חסר COHERE_API_KEY ב-.env")
    exit(1)
if not PINECONE_API_KEY:
    print("❌ חסר PINECONE_API_KEY ב-.env")
    exit(1)

# ========================================
# הגדרות
# ========================================
print("📥 טוען רכיבים...")

# Embedding model - Cohere (לפי דרישות המטלה)
print("   🧮 Embedding Model: Cohere multilingual...")
try:
    embed_model = CohereEmbedding(
        api_key=COHERE_API_KEY,
        model_name="embed-multilingual-v3.0",
        input_type="search_query"
    )
    Settings.embed_model = embed_model
    print("   ✅ Cohere Embeddings מוכנים")
except Exception as e:
    print(f"   ❌ שגיאה בטעינת Cohere embeddings: {e}")
    exit(1)

# LLM - Cohere (לפי דרישות המטלה)
print("   🤖 LLM: Cohere...")
try:
    llm = Cohere(
        api_key=COHERE_API_KEY,
        model="command-r-plus-08-2024",
        temperature=0.7
    )
    Settings.llm = llm
    print("   ✅ Cohere LLM מוכן")
except Exception as e:
    print(f"   ❌ שגיאה בחיבור ל-Cohere: {e}")
    exit(1)

print()

# ========================================
# טעינת VectorStoreIndex מ-Pinecone
# ========================================
print("📥 טוען VectorStoreIndex מ-Pinecone...")

try:
    # חיבור ל-Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    # יצירת Vector Store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace="rag_agent_docs"
    )
    
    # טעינת index
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    print(f"   ✅ VectorStoreIndex נטען מ-Pinecone ({PINECONE_INDEX_NAME})!")
    
except Exception as e:
    print(f"   ❌ שגיאה בטעינת Index מ-Pinecone: {e}")
    print("   💡 וודאי שהרצת את ingest_stage_a.py או ingest_stage_b.py קודם!")
    exit(1)

# ========================================
# יצירת Retriever ו-Response Synthesizer
# ========================================
print("\n🔧 מכין Retriever ו-Response Synthesizer...")

try:
    # Retriever
    retriever = index.as_retriever(similarity_top_k=3)
    
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
    
    # Response Synthesizer
    from llama_index.core.response_synthesizers import get_response_synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        llm=llm,
        text_qa_template=qa_prompt_tmpl
    )
    
    print("   ✅ Retriever מוכן")
    print("   ✅ Response Synthesizer מוכן (עם System Prompt)")
    
except Exception as e:
    print(f"   ❌ שגיאה: {e}")
    exit(1)

# ========================================
# יצירת Workflow Engine
# ========================================
print("\n🚀 יוצר Workflow Engine...")
print("   רכיבים:")
print("   1. QueryValidationStep - בדיקת תקינות שאילתה")
print("   2. EmbeddingStep - המרה לוקטור")
print("   3. RetrievalStep - חיפוש במאגר")
print("   4. SynthesisStep - בניית תשובה")
print("   5. Event Management - מעקב אחר אירועים")
print("   6. State Tracking - שמירת מצב")
print()

# יצירת Engine
engine = WorkflowEngine(
    embed_model=embed_model,
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    verbose=True  # לוגים מפורטים
)

# יצירת Manager להיסטוריה
manager = WorkflowManager()

print("   ✅ Workflow Engine מוכן!")
print("="*70)
print()

# ========================================
# פונקציית RAG עם Workflow
# ========================================
def rag_agent_workflow(message: str, history: list) -> str:
    """
    RAG עם Event-Driven Workflow - שלב ב' של המטלה!
    
    Pipeline:
    1. Query Validation → בדיקת תקינות
    2. Embedding → המרה לוקטור
    3. Retrieval → חיפוש במאגר
    4. Synthesis → בניית תשובה
    
    בכל שלב:
    - בדיקות תקינות
    - שיגור Events
    - עדכון State
    - טיפול בשגיאות
    """
    
    if not message or not message.strip():
        return "אנא הכנסי שאלה."
    
    # הרצת הזרימה
    state = engine.execute(message)
    
    # שמירה להיסטוריה
    manager.add_workflow(state)
    
    # החזרת התשובה
    if state.synthesized_response:
        # בניית תשובה עם מטא-דאטה
        response = state.synthesized_response
        
        # הוספת מידע נוסף (אופציונלי)
        if state.confidence_score < 0.5:
            response += f"\n\n⚠️ *הערה: רמת הוודאות נמוכה ({state.confidence_score:.2f})*"
        
        # הוספת סטטיסטיקות בסוף
        response += f"\n\n---\n"
        response += f"📊 **מטא-דאטה:**\n"
        response += f"- ⏱️ זמן כולל: {state.total_time_ms:.0f}ms\n"
        response += f"- 🎯 רמת ודאות: {state.confidence_score:.2f}\n"
        response += f"- 📝 מספר תוצאות: {len(state.retrieved_nodes)}\n"
        response += f"- 🔄 מספר שלבים: {len(state.events)}\n"
        
        if state.errors:
            response += f"\n⚠️ **אזהרות:** {', '.join(state.errors[:2])}"
        
        return response
    else:
        return "❌ לא הצלחתי לבנות תשובה. אנא נסי שאילתה אחרת."


# ========================================
# Gradio Interface
# ========================================
print("🌐 מפעיל ממשק Gradio...")
print()

demo = gr.ChatInterface(
    fn=rag_agent_workflow,
    title="🤖 RAG Agent - שלב ב': Event-Driven Workflow",
    description=(
        "## ✨ מערכת RAG עם ארכיטקטורה Event-Driven\n\n"
        "### 📋 שלב ב' - מה השתנה?\n\n"
        "| תכונה | שלב א' (MVP) | שלב ב' (Event-Driven) |\n"
        "|--------|-------------|----------------------|\n"
        "| **ארכיטקטורה** | Pipeline פשוט | Workflow עם Events |\n"
        "| **Validations** | ❌ אין | ✅ בכל שלב |\n"
        "| **State Management** | ❌ אין | ✅ מלא |\n"
        "| **Event Tracking** | ❌ אין | ✅ כל אירוע |\n"
        "| **Error Handling** | בסיסי | מתקדם |\n\n"
        "---\n\n"
        "### 🔄 Pipeline המלא:\n\n"
        "```\n"
        "Query → Validation → Embedding → Retrieval → Synthesis → Response\n"
        "   ↓         ↓           ↓            ↓           ↓          ↓\n"
        " Event    Event       Event        Event       Event     Event\n"
        "   ↓         ↓           ↓            ↓           ↓          ↓\n"
        " State    State       State        State       State     State\n"
        "```\n\n"
        "---\n\n"
        "### 📊 Validations:\n\n"
        "1. **Query Validation:** שאילתה לא ריקה, אורך תקין\n"
        "2. **Embedding:** וקטור תקין, זמן סביר\n"
        "3. **Retrieval:** יש תוצאות, confidence גבוה\n"
        "4. **Synthesis:** תשובה לא ריקה, גודל סביר\n\n"
        "---\n\n"
        "🎯 **מטרת שלב ב':** להפוך את המערכת ל-Workflow מסודר שמנוהל על ידי Events\n\n"
        "💻 **יתרון:** קל להוסיף שלבים, לעקוב אחר התהליך, ולנתח ביצועים"
    ),
    examples=[
        "מה הצבע של המערכת?",
        "איזה שפות נתמכות?",
        "ספרי לי על RTL",
        "מה השינויים במסד נתונים?",
        "מה גודל הקובץ המקסימלי?",
        "ספר לי על bug 142",
        "איך עובד ה-authentication ב-API?"
    ],
    retry_btn="🔄 נסה שוב",
    undo_btn="↩️ בטל",
    clear_btn="🗑️ נקה"
)

if __name__ == "__main__":
    print("="*70)
    print("✨ מערכת RAG - שלב ב' פועלת!")
    print("="*70)
    print()
    print("📊 ארכיטקטורה:")
    print("   Framework:     LlamaIndex")
    print("   Embeddings:    HuggingFace (paraphrase-multilingual)")
    print("   Vector Store:  ChromaDB")
    print("   LLM:           Ollama (Llama 3.2 - 3B)")
    print("   Workflow:      Event-Driven ⭐")
    print()
    print("🔍 Pipeline:")
    print("   Query → Validation → Embedding → Retrieval → Synthesis")
    print("   כל שלב: Event + Validation + State Update")
    print()
    print("📈 תכונות חדשות:")
    print("   ✅ Event Management")
    print("   ✅ State Tracking")
    print("   ✅ Validations בכל שלב")
    print("   ✅ Error Handling מתקדם")
    print("   ✅ Performance Metrics")
    print("   ✅ Workflow History")
    print()
    print("="*70)
    print()
    
    demo.launch()
