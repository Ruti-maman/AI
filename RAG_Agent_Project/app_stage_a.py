"""
שלב א' -אפליקציית RAG (Gradio UI)
====================================
ממשק צ'אט לתשאול קבצי MD של כלי Agentic Coding.

חיפוש סמנטי עם:
- Cohere Embeddings
- Pinecone Vector Store
- Cohere LLM
"""

# ============================================================
# AGGRESSIVE SSL BYPASS - MUST BE ABSOLUTELY FIRST
# ============================================================
import ssl
import os

# Environment variables to disable SSL verification everywhere
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['SSL_CERT_DIR'] = ''

# Create a reusable unverified SSL context
def _create_unverified_ssl_context():
    ctx = ssl._create_unverified_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

# Patch ssl.create_default_context to return unverified context
_original_create_default_context = ssl.create_default_context

def patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

ssl.create_default_context = patched_create_default_context
ssl._create_default_https_context = patched_create_default_context

# Patch httpx BEFORE any import uses it
try:
    import httpx._config
    
    def patched_httpx_ssl_context(verify=True, cert=None, trust_env=True, http2=False):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    
    httpx._config.create_ssl_context = patched_httpx_ssl_context
except Exception:
    pass

# urllib3 patches
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    from urllib3.util import ssl_
    original_ssl_wrap_socket = ssl_.ssl_wrap_socket
    
    def patched_ssl_wrap_socket(sock, keyfile=None, certfile=None, cert_reqs=None,
                                ca_certs=None, server_hostname=None,
                                ssl_version=None, ciphers=None, ssl_context=None,
                                ca_cert_dir=None, key_password=None, ca_cert_data=None, tls_in_tls=False):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return original_ssl_wrap_socket(
            sock, keyfile, certfile, ssl.CERT_NONE,
            ca_certs, server_hostname, ssl_version, ciphers, ctx,
            ca_cert_dir, key_password, ca_cert_data, tls_in_tls
        )
    
    ssl_.ssl_wrap_socket = patched_ssl_wrap_socket
except Exception:
    pass

# Windows certificate store (optional, for fallback)
try:
    import certifi_win32
    certifi_win32.wincerts.where()
except Exception:
    pass

# ============================================================
# END SSL BYPASS
# ============================================================

import logging
from dotenv import load_dotenv

import gradio as gr
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.cohere import Cohere
from pinecone import Pinecone

# הגדרת Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# טעינת משתני סביבה
load_dotenv()

# ========================================
# תצורה
# ========================================

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-agent-index")

if not COHERE_API_KEY:
    raise ValueError("❌ חסר COHERE_API_KEY ב-.env")
if not PINECONE_API_KEY:
    raise ValueError("❌ חסר PINECONE_API_KEY ב-.env")


# ========================================
# הגדרת RAG System
# ========================================

class RAGSystem:
    """מערכת RAG בסיסית - שלב א'"""
    
    def __init__(self):
        logger.info("🔧 מאתחל RAG System...")
        
        # הגדרת LlamaIndex
        self._setup_llama_index()
        
        # חיבור ל-Pinecone
        self.pinecone_index = self._connect_pinecone()
        
        # יצירת QueryEngine
        self.index = self._load_index()
        
        # System Prompt - מגביל תשובות רק לנושאי RAG!
        from llama_index.core.prompts import PromptTemplate
        qa_prompt_tmpl = PromptTemplate(
            """🚨 CRITICAL: Answer ONLY the specific question asked. Do NOT add introductions, context, or extra information!

You are an AI assistant specializing ONLY in RAG (Retrieval-Augmented Generation).

STRICT RULES:
1. If asked "למה RAG חשוב" (Why is RAG important) → Answer ONLY why it's important. DO NOT explain what RAG is!
2. If asked "מה זה RAG" (What is RAG) → Answer ONLY what it is
3. Maximum 2-3 sentences
4. Be direct and concise
5. Use only information from the documents below

Context from documents:
{context_str}

Question: {query_str}

Answer (ONLY answer the question, no extra information):"""
        )
        
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=2,  # קטינו ל-2 כדי לקבל פחות מידע
            response_mode="compact",
            text_qa_template=qa_prompt_tmpl
        )
        
        logger.info("✅ RAG System מוכן!\n")
    
    def _setup_llama_index(self):
        """הגדרת רכיבי LlamaIndex"""
        
        # Embeddings - Cohere
        embed_model = CohereEmbedding(
            api_key=COHERE_API_KEY,
            model_name="embed-multilingual-v3.0",
            input_type="search_query"  # לשאילתות
        )
        
        # LLM - Cohere
        llm = Cohere(
            api_key=COHERE_API_KEY,
            model="command-r-plus-08-2024",
            temperature=0.7
        )
        
        # הגדרות גלובליות
        Settings.embed_model = embed_model
        Settings.llm = llm
        
        logger.info("✓ Cohere Embeddings + LLM")
    
    def _connect_pinecone(self):
        """חיבור ל-Pinecone Cloud Index"""
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"✓ Pinecone Index: {PINECONE_INDEX_NAME}")
        return pinecone_index
    
    def _load_index(self):
        """טעינת VectorStoreIndex מ-Pinecone"""
        
        # יצירת Vector Store
        vector_store = PineconeVectorStore(
            pinecone_index=self.pinecone_index,
            namespace="rag_agent_docs"
        )
        
        # יצירת Storage Context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # יצירת Index מהVector Store
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
        logger.info("✓ VectorStoreIndex נטען מ-Pinecone")
        
        return index
    
    def query(self, query_text: str) -> str:
        """
        ביצוע שאילתה.
        
        התהליך:
        1. Retrieve - חיפוש וקטורי ב-Pinecone
        2. Synthesis - יצירת תשובה עם Cohere LLM
        """
        try:
            logger.info(f"🔍 שאילתה: {query_text}")
            
            # ביצוע חיפוש + synthesis
            response = self.query_engine.query(query_text)
            
            return str(response)
            
        except Exception as e:
            logger.error(f"❌ שגיאה: {e}")
            return f"⚠️ שגיאה בעיבוד השאילתה: {str(e)}"


# ========================================
# Gradio Interface
# ========================================

def create_gradio_app(rag_system: RAGSystem):
    """יצירת ממשק Gradio"""

    def chat_fn(message, history):
        """פונקציית צ'אט - מחזירה string בלבד"""
        if not message.strip():
            return ""
        return rag_system.query(message)

    with gr.Blocks(title="RAG Agent – Cohere + Pinecone") as demo:

        gr.HTML("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 28px 32px;
            color: white;
            margin-bottom: 16px;
            text-align: center;
        ">
            <h1 style="font-size:2rem; margin:0 0 6px 0;">🤖 RAG Agent</h1>
            <p style="font-size:1rem; margin:0; opacity:0.9;">חיפוש סמנטי חכם בתיעוד פרויקט &nbsp;·&nbsp; שלב א' MVP</p>
            <div style="display:flex; gap:10px; flex-wrap:wrap; justify-content:center; margin-top:14px;">
                <span style="background:rgba(255,255,255,0.22);border-radius:20px;padding:4px 14px;font-size:0.82rem;font-weight:600;">🧠 Cohere Embeddings</span>
                <span style="background:rgba(255,255,255,0.22);border-radius:20px;padding:4px 14px;font-size:0.82rem;font-weight:600;">📌 Pinecone Serverless</span>
                <span style="background:rgba(255,255,255,0.22);border-radius:20px;padding:4px 14px;font-size:0.82rem;font-weight:600;">💬 command-r-plus-08-2024</span>
                <span style="background:rgba(255,255,255,0.22);border-radius:20px;padding:4px 14px;font-size:0.82rem;font-weight:600;">⚡ LlamaIndex</span>
            </div>
        </div>
        """)

        gr.ChatInterface(
            fn=chat_fn,
            examples=[
                "מה זה RAG ולמה הוא חשוב?",
                "מה ההבדל בין embedding לבין vectorization?",
                "אילו טכנולוגיות נפוצות משמשות ל-RAG?",
                "מה ההבדל בין RAG ל-Fine-Tuning?",
                "מתי כדאי להשתמש ב-RAG ומתי לא?",
            ],
            chatbot=gr.Chatbot(height=420, label=""),
            textbox=gr.Textbox(placeholder="✏️  שאל אותי כל שאלה על RAG...", container=False),
        )

    return demo


# ========================================
# Main
# ========================================

def main():
    """הרצת האפליקציה"""
    
    print("\n" + "🚀" * 30)
    print("   שלב א' - RAG Agent (Cohere + Pinecone)")
    print("🚀" * 30 + "\n")
    
    try:
        # אתחול RAG System
        rag_system = RAGSystem()
        
        # יצירת Gradio UI
        demo = create_gradio_app(rag_system)
        
        # הרצה
        print("\n" + "✅" * 30)
        print("   האפליקציה מוכנה!")
        print("✅" * 30)
        print("""
🌐 פתח דפדפן ב: http://localhost:7860

📊 טכנולוגיות:
   ✓ Cohere Embeddings (embed-multilingual-v3.0)
   ✓ Pinecone Vector Store
   ✓ Cohere LLM (command-r-plus-08-2024)
   ✓ LlamaIndex Framework
   ✓ Gradio UI

🎯 נסה שאילתות כמו:
   • "מה הצבע העיקרי שנבחר?"
   • "האם יש הנחיה לגבי RTL?"
   • "אילו שינויים נעשו ב-DB?"
        """)
        
        demo.launch(
            server_name="0.0.0.0",
            share=False,
            css="""
            .gradio-container { max-width: 860px !important; margin: auto !important; }
            footer { display: none !important; }
            """
        )
        
    except Exception as e:
        logger.error(f"❌ שגיאה: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n⚠️  אם אתה רואה שגיאה:")
        print("   1. וודא שיש קובץ .env עם API keys")
        print("   2. הרץ קודם: python ingest_stage_a.py")


if __name__ == "__main__":
    main()
