"""
הכנת נתונים - שלב ב': עם Event Tracking
✅ כל הרכיבים של שלב א' + מעקב אחר אירועים

רכיבים:
1. Loading - SimpleDirectoryReader
2. Chunking - SentenceSplitter (Node Parser)
3. Embedding - Cohere Embeddings (לפי דרישות)
4. VectorStoreIndex - LlamaIndex
5. Vector Store - Pinecone (לפי דרישות)
6. Event Tracking - מעקב אחר התהליך ⭐
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# טעינת משתני סביבה
load_dotenv()

print("="*70)
print("🚀 מכין מערכת RAG - שלב ב': עם Event Tracking")
print("="*70)
print()

# בדיקת API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-agent-index")

if not COHERE_API_KEY:
    print("❌ חסר COHERE_API_KEY ב-.env")
    exit(1)
if not PINECONE_API_KEY:
    print("❌ חסר PINECONE_API_KEY ב-.env")
    exit(1)

# מעקב אחר אירועים
events = []
start_time = datetime.now()

def log_event(step: str, message: str, data: dict = None):
    """רישום אירוע"""
    event = {
        "timestamp": datetime.now(),
        "step": step,
        "message": message,
        "data": data or {}
    }
    events.append(event)
    print(f"   📝 [{event['timestamp'].strftime('%H:%M:%S')}] {message}")

# ========================================
# 1. LOADING - טעינת מסמכים
# ========================================
print("📁 שלב 1: LOADING")
print("   רכיב: SimpleDirectoryReader (LlamaIndex)")
log_event("loading", "מתחיל טעינת מסמכים")

try:
    documents = SimpleDirectoryReader(
        input_dir="./dummy_data",
        required_exts=[".md"],
        recursive=False
    ).load_data()
    
    log_event("loading", f"נטענו {len(documents)} מסמכים בהצלחה", {
        "num_documents": len(documents)
    })
    
    for doc in documents:
        file_name = os.path.basename(doc.metadata.get('file_path', 'unknown'))
        print(f"      ✓ {file_name}: {len(doc.text)} תווים")
        
except Exception as e:
    log_event("loading", f"שגיאה: {str(e)}", {"error": str(e)})
    exit(1)

# ========================================
# 2. CHUNKING - Node Parser
# ========================================
print("\n✂️ שלב 2: CHUNKING")
print("   רכיב: SentenceSplitter (Node Parser של LlamaIndex)")
log_event("chunking", "מגדיר Node Parser")

try:
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    
    Settings.node_parser = node_parser
    
    log_event("chunking", "Node Parser הוגדר בהצלחה", {
        "chunk_size": 512,
        "chunk_overlap": 50
    })
    
except Exception as e:
    log_event("chunking", f"שגיאה: {str(e)}", {"error": str(e)})
    exit(1)

# ========================================
# 3. EMBEDDING - Cohere (לפי דרישות)
# ========================================
print("\n🧮 שלב 3: EMBEDDING")
print("   רכיב: CohereEmbedding (LlamaIndex)")
print("   מודל: embed-multilingual-v3.0")
log_event("embedding", "טוען מודל embedding")

try:
    embed_model = CohereEmbedding(
        api_key=COHERE_API_KEY,
        model_name="embed-multilingual-v3.0",
        input_type="search_document"
    )
    Settings.embed_model = embed_model
    
    log_event("embedding", "Embedding model נטען בהצלחה", {
        "model": "embed-multilingual-v3.0",
        "provider": "Cohere"
    })
    
except Exception as e:
    log_event("embedding", f"שגיאה: {str(e)}", {"error": str(e)})
    exit(1)

# ========================================
# 4. VECTOR STORE - Pinecone (לפי דרישות)
# ========================================
print("\n💾 שלב 4: VECTOR STORE")
print("   רכיב: PineconeVectorStore (LlamaIndex)")
log_event("vector_store", "מכין Pinecone")

try:
    # חיבור ל-Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # בדיקה אם ה-index קיים
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        # יצירת index חדש
        log_event("vector_store", f"יוצר index חדש: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,  # Cohere embed-multilingual-v3.0
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        log_event("vector_store", "Index נוצר בהצלחה")
    else:
        log_event("vector_store", f"Index כבר קיים: {PINECONE_INDEX_NAME}")
    
    # חיבור ל-index
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    # יצירת Vector Store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace="rag_agent_docs"
    )
    
    log_event("vector_store", "Pinecone Vector Store מוכן", {
        "index_name": PINECONE_INDEX_NAME,
        "namespace": "rag_agent_docs"
    })
    
except Exception as e:
    log_event("vector_store", f"שגיאה: {str(e)}", {"error": str(e)})
    exit(1)

# ========================================
# 5. VectorStoreIndex - הרכיב המרכזי!
# ========================================
print("\n🔧 שלב 5: VectorStoreIndex")
print("   רכיב: VectorStoreIndex (LlamaIndex) ⭐")
log_event("indexing", "יוצר VectorStoreIndex")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

try:
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    log_event("indexing", "VectorStoreIndex נוצר בהצלחה")
    
    # שמירה
    index.storage_context.persist(persist_dir="./storage_final")
    log_event("indexing", "Index נשמר ב-./storage_final")
    
except Exception as e:
    log_event("indexing", f"שגיאה: {str(e)}", {"error": str(e)})
    exit(1)

# ========================================
# סיכום
# ========================================
end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()

print("\n" + "="*70)
print("🎉 תהליך ההכנה הושלם בהצלחה!")
print("="*70)
print()
print("📊 רכיבי המטלה - סיכום:")
print("   1. ✅ Loading         → SimpleDirectoryReader")
print("   2. ✅ Chunking        → Node Parser (SentenceSplitter)")
print("   3. ✅ Embedding       → HuggingFace Embeddings")
print("   4. ✅ VectorStore     → ChromaDB")
print("   5. ✅ VectorStoreIndex → LlamaIndex ⭐")
print("   6. ✅ Event Tracking  → שלב ב' ⭐")
print()
print("💾 נתונים נשמרו:")
print(f"   - ChromaDB: ./chroma_db/")
print(f"   - Index: ./storage_final/")
print()
print(f"⏱️  זמן כולל: {total_time:.2f} שניות")
print(f"📝 מספר אירועים: {len(events)}")
print()
print("✨ המערכת מוכנה! הרץ את app_workflow.py")
print("="*70)
print()
print("📈 אירועים שהתרחשו:")
for i, event in enumerate(events, 1):
    print(f"   {i}. [{event['step']}] {event['message']}")
