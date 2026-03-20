"""
Ingest מקומי - ללא Pinecone
============================
שומר את הנתונים בקובץ מקומי (JSON)
"""

# SSL Bypass
import ssl
import os
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

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
except:
    pass

try:
    import certifi_win32
    certifi_win32.wincerts.where()
except:
    pass

# Imports
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.cohere import CohereEmbedding

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
STORAGE_DIR = "./storage_local"
DOCS_DIR = "./dummy_data"

def main():
    print("\n" + "="*50)
    print("   📥 Ingest מקומי (ללא Pinecone)")
    print("="*50 + "\n")
    
    # Cohere Embeddings
    print("🔧 מגדיר Cohere Embeddings...")
    embed_model = CohereEmbedding(
        api_key=COHERE_API_KEY,
        model_name="embed-multilingual-v3.0",
        input_type="search_document"
    )
    Settings.embed_model = embed_model
    print("✓ Cohere Embeddings מוגדר")
    
    # קריאת מסמכים
    print(f"\n📂 קורא מסמכים מ-{DOCS_DIR}...")
    reader = SimpleDirectoryReader(DOCS_DIR, recursive=True)
    documents = reader.load_data()
    print(f"✓ נמצאו {len(documents)} מסמכים")
    
    # חלוקה לחתיכות
    print("\n✂️ מחלק לחתיכות...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✓ נוצרו {len(nodes)} חתיכות")
    
    # יצירת Index ושמירה מקומית
    print("\n💾 יוצר Index ושומר מקומית...")
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # שמירה לתיקייה מקומית
    Path(STORAGE_DIR).mkdir(exist_ok=True)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    
    print(f"\n✅ הושלם! נשמר ב-{STORAGE_DIR}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
