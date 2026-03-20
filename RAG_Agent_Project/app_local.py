"""
RAG Agent - גרסה מקומית (ללא Pinecone)
=======================================
משתמש ב-Storage מקומי במקום Pinecone
"""

# ============================================================
# SSL BYPASS - MUST BE FIRST
# ============================================================
import ssl
import os

os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['SSL_CERT_DIR'] = ''

def patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

ssl.create_default_context = patched_create_default_context
ssl._create_default_https_context = patched_create_default_context

try:
    import httpx._config
    def patched_httpx_ssl_context(verify=True, cert=None, trust_env=True, http2=False):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    httpx._config.create_ssl_context = patched_httpx_ssl_context
except:
    pass

try:
    import urllib3
    urllib3.disable_warnings()
    from urllib3.util import ssl_
    _orig = ssl_.ssl_wrap_socket
    def patched_wrap(sock, keyfile=None, certfile=None, cert_reqs=None,
                     ca_certs=None, server_hostname=None, ssl_version=None,
                     ciphers=None, ssl_context=None, ca_cert_dir=None,
                     key_password=None, ca_cert_data=None, tls_in_tls=False):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return _orig(sock, keyfile, certfile, ssl.CERT_NONE, ca_certs,
                     server_hostname, ssl_version, ciphers, ctx,
                     ca_cert_dir, key_password, ca_cert_data, tls_in_tls)
    ssl_.ssl_wrap_socket = patched_wrap
except:
    pass

try:
    import certifi_win32
    certifi_win32.wincerts.where()
except:
    pass

# ============================================================
# IMPORTS
# ============================================================
import logging
from pathlib import Path
from dotenv import load_dotenv

import gradio as gr
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ========================================
# Config
# ========================================
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
STORAGE_DIR = "./storage_local"

if not COHERE_API_KEY:
    raise ValueError("❌ חסר COHERE_API_KEY ב-.env")


# ========================================
# RAG System
# ========================================
class RAGSystem:
    def __init__(self):
        logger.info("🔧 מאתחל RAG System (מקומי)...")
        
        # הגדרת Cohere
        self._setup_llama_index()
        
        # טעינת Index מקומי
        self.index = self._load_local_index()
        
        # Prompt
        from llama_index.core.prompts import PromptTemplate
        qa_prompt = PromptTemplate(
            """אתה עוזר AI מומחה ב-RAG (Retrieval-Augmented Generation).

ענה על השאלה בהתבסס על המסמכים הבאים:
{context_str}

שאלה: {query_str}

תשובה (בעברית, תמציתית ומדויקת):"""
        )
        
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact",
            text_qa_template=qa_prompt
        )
        
        logger.info("✅ RAG System מוכן!")
    
    def _setup_llama_index(self):
        embed_model = CohereEmbedding(
            api_key=COHERE_API_KEY,
            model_name="embed-multilingual-v3.0",
            input_type="search_query"
        )
        
        llm = Cohere(
            api_key=COHERE_API_KEY,
            model="command-r-plus-08-2024",
            temperature=0.7
        )
        
        Settings.embed_model = embed_model
        Settings.llm = llm
        logger.info("✓ Cohere Embeddings + LLM")
    
    def _load_local_index(self):
        if not Path(STORAGE_DIR).exists():
            raise ValueError(f"❌ לא נמצא {STORAGE_DIR}. הרץ קודם: python ingest_local.py")
        
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        logger.info(f"✓ Index נטען מ-{STORAGE_DIR}")
        return index
    
    def query(self, question: str) -> str:
        if not question.strip():
            return "❓ נא להזין שאלה"
        
        try:
            response = self.query_engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"⚠️ שגיאה: {e}"


# ========================================
# Gradio UI
# ========================================
def create_gradio_app(rag_system: RAGSystem):
    
    # Custom CSS - Soft pastel pink/lavender theme
    custom_css = """
    .gradio-container {
        max-width: 620px !important;
        margin: 50px auto !important;
        border-radius: 20px !important;
        box-shadow: 0 12px 40px rgba(180, 150, 180, 0.2) !important;
        background: #fefefe !important;
        overflow: hidden !important;
    }
    .main-header {
        background: linear-gradient(135deg, #e8d5e8 0%, #d4c4e8 50%, #c8d4e8 100%);
        padding: 26px 20px;
        text-align: center;
    }
    .main-header h1 {
        color: #5a4a6a;
        font-size: 1.9em;
        margin: 0 0 6px 0;
        font-weight: 600;
    }
    .main-header p {
        color: #6a5a7a;
        font-size: 0.92em;
        margin: 0 0 14px 0;
    }
    .badges {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255,255,255,0.6);
        padding: 5px 12px;
        border-radius: 14px;
        color: #5a4a6a;
        font-size: 0.76em;
        font-weight: 500;
    }
    """
    
    header_html = """
    <div class="main-header">
        <h1>🤖 RAG Agent</h1>
        <p>חיפוש סמנטי חכם בתיעוד פרויקט</p>
        <div class="badges">
            <span class="badge">🔮 Cohere</span>
            <span class="badge">💾 Local</span>
            <span class="badge">⚡ LlamaIndex</span>
        </div>
    </div>
    """
    
    def chat_fn(message, history):
        if not message.strip():
            return ""
        return rag_system.query(message)
    
    with gr.Blocks(title="RAG Agent") as demo:
        gr.HTML(f"<style>{custom_css}</style>")
        gr.HTML(header_html)
        
        gr.ChatInterface(
            fn=chat_fn,
            examples=["מה זה RAG?", "למה RAG חשוב?", "איך RAG עובד?"],
        )
    
    return demo


# ========================================
# Main
# ========================================
def main():
    print("\n" + "🚀"*30)
    print("   RAG Agent (גרסה מקומית - ללא Pinecone)")
    print("🚀"*30 + "\n")
    
    rag_system = RAGSystem()
    
    print("\n" + "✅"*30)
    print("   האפליקציה מוכנה!")
    print("✅"*30)
    print("\n🌐 פתח דפדפן ב: http://localhost:7860\n")
    print("📊 טכנולוגיות:")
    print("   ✓ Cohere Embeddings (embed-multilingual-v3.0)")
    print("   ✓ Local Vector Store (ללא Pinecone!)")
    print("   ✓ Cohere LLM (command-r-plus-08-2024)")
    print("   ✓ LlamaIndex + Gradio\n")
    
    demo = create_gradio_app(rag_system)
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
