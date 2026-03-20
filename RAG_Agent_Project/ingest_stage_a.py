"""
שלב א' - הכנת נתונים (Data Preparation)
================================
מטעין קבצי MD מתיקיות של Cursor/Claude Code, מחלק לchunks, 
יוצר embeddings עם Cohere ושומר ב-Pinecone.

דרישות:
- Cohere API Key
- Pinecone API Key
"""

# Windows SSL Certificate Fix - MUST BE FIRST
import certifi_win32
certifi_win32.wincerts.where()

# SSL Fix for httpx (used by Cohere)
import httpx
import ssl

# Create unverified SSL context for httpx
_httpx_client = httpx.Client(verify=False)
_original_httpx_client = httpx.Client

class SSLClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)

httpx.Client = SSLClient

import os
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.cohere import Cohere

from pinecone import Pinecone, ServerlessSpec

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

# בדיקת API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-agent-index")

if not COHERE_API_KEY:
    raise ValueError("❌ חסר COHERE_API_KEY ב-.env")
if not PINECONE_API_KEY:
    raise ValueError("❌ חסר PINECONE_API_KEY ב-.env")

# תיקיות מקור - קבצי MD של כלי Agentic Coding
DATA_SOURCES = [
    Path(".cursor"),      # Cursor AI
    Path(".claude"),      # Claude Code
    Path(".kiro"),        # Kiro
    Path("dummy_data"),   # Fallback - אם אין תיקיות אמיתיות
]

# סינון לקבצי MD בלבד
MD_FILE_PATTERN = "**/*.md"


# ========================================
# הגדרת LlamaIndex Settings
# ========================================

def setup_llama_index():
    """הגדרת רכיבי LlamaIndex"""
    
    logger.info("🔧 מגדיר רכיבי LlamaIndex...")
    
    # 1. Embeddings - Cohere (לפי דרישה)
    embed_model = CohereEmbedding(
        api_key=COHERE_API_KEY,
        model_name="embed-multilingual-v3.0",  # תמיכה בעברית
        input_type="search_document"
    )
    logger.info(f"✓ Embeddings: Cohere (embed-multilingual-v3.0)")
    
    # 2. LLM - Cohere (לפי דרישה)
    llm = Cohere(
        api_key=COHERE_API_KEY,
        model="command-r-plus-08-2024",
        temperature=0.7
    )
    logger.info("✓ LLM: Cohere (command-r-plus-08-2024)")
    
    # 3. Node Parser - חיתוך לchunks
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    logger.info("✓ Node Parser: SentenceSplitter (chunk_size=512)")
    
    # הגדרות גלובליות
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.node_parser = node_parser
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    logger.info("✅ LlamaIndex מוכן!\n")
    
    return embed_model, llm


# ========================================
# שלב 1: Loading - טעינת קבצים
# ========================================

def load_documents() -> List[Document]:
    """
    טוען קבצי MD מתיקיות של כלי Agentic Coding.
    
    לפי הדרישות:
    - Cursor: .cursor/*.md
    - Claude Code: .claude/*.md
    - Kiro: .kiro/*.md
    - Fallback: dummy_data/*.md (אם אין תיקיות אמיתיות)
    """
    logger.info("=" * 60)
    logger.info("שלב 1: Loading - טעינת קבצים")
    logger.info("=" * 60)
    
    all_documents = []
    
    for source_dir in DATA_SOURCES:
        if not source_dir.exists():
            logger.warning(f"⚠️  תיקייה לא קיימת: {source_dir}")
            continue
        
        logger.info(f"📂 סורק תיקייה: {source_dir}")
        
        # טעינת קבצי MD
        try:
            documents = SimpleDirectoryReader(
                input_dir=str(source_dir),
                required_exts=[".md"],
                filename_as_id=True,
                recursive=True
            ).load_data()
            
            # הוספת metadata - מקור הכלי
            tool_name = source_dir.name.replace(".", "")
            for doc in documents:
                doc.metadata["tool"] = tool_name
                doc.metadata["source_dir"] = str(source_dir)
            
            all_documents.extend(documents)
            logger.info(f"   ✓ נטענו {len(documents)} קבצים מ-{source_dir}")
            
        except Exception as e:
            logger.error(f"   ❌ שגיאה בטעינה מ-{source_dir}: {e}")
    
    if not all_documents:
        raise ValueError("❌ לא נמצאו קבצי MD! וודא שיש תיקיות: .cursor, .claude או dummy_data")
    
    logger.info(f"\n✅ סה\"כ נטענו {len(all_documents)} מסמכים\n")
    return all_documents


# ========================================
# שלב 2: Pinecone Setup
# ========================================

def setup_pinecone():
    """חיבור/יצירה של Pinecone Index"""
    logger.info("=" * 60)
    logger.info("שלב 2: הגדרת Pinecone")
    logger.info("=" * 60)
    
    # חיבור ל-Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # בדיקה אם ה-index קיים
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        # יצירת index חדש
        logger.info(f"🔄 יוצר index חדש: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,  # Cohere embed-multilingual-v3.0
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info("✓ Index נוצר בהצלחה")
    else:
        logger.info(f"✓ Index כבר קיים: {PINECONE_INDEX_NAME}")
    
    # חיבור ל-index
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    logger.info(f"✓ Pinecone Index: {PINECONE_INDEX_NAME}")
    logger.info(f"✅ Pinecone מוכן\n")
    return pinecone_index


# ========================================
# שלב 3: Indexing - יצירת VectorStoreIndex
# ========================================

def build_index(documents: List[Document], pinecone_index):
    """
    בניית VectorStoreIndex עם Pinecone.
    
    התהליך:
    1. Chunking - חיתוך המסמכים לchunks
    2. Embedding - יצירת embeddings עם Cohere
    3. Storage - שמירה ב-Pinecone Cloud
    """
    logger.info("=" * 60)
    logger.info("שלב 3: Indexing - בניית אינדקס")
    logger.info("=" * 60)
    
    # יצירת PineconeVectorStore
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace="rag_agent_docs"
    )
    
    # יצירת StorageContext
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    # בניית האינדקס
    logger.info("🔄 בונה VectorStoreIndex...")
    logger.info("   (זה עלול לקחת מספר דקות...)")
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    logger.info("✅ אינדקס נבנה בהצלחה!\n")
    return index


# ========================================
# Main Execution
# ========================================

def main():
    """הרצת תהליך ההכנה המלא"""
    
    print("\n" + "🚀" * 30)
    print("   שלב א' - הכנת נתונים (Cohere + Pinecone)")
    print("🚀" * 30 + "\n")
    
    try:
        # הגדרת LlamaIndex
        embed_model, llm = setup_llama_index()
        
        # שלב 1: טעינת מסמכים
        documents = load_documents()
        
        # סטטיסטיקה
        total_chars = sum(len(doc.text) for doc in documents)
        logger.info(f"📊 סטטיסטיקה:")
        logger.info(f"   • מסמכים: {len(documents)}")
        logger.info(f"   • תווים: {total_chars:,}")
        logger.info(f"   • ממוצע תווים למסמך: {total_chars // len(documents):,}\n")
        
        # שלב 2: הגדרת Pinecone
        pinecone_index = setup_pinecone()
        
        # שלב 3: בניית אינדקס
        index = build_index(documents, pinecone_index)
        
        # סיכום
        print("\n" + "✅" * 30)
        print("   הכנת הנתונים הושלמה בהצלחה!")
        print("✅" * 30)
        print(f"""
📊 סיכום:
   • מסמכים: {len(documents)}
   • Embeddings: Cohere (embed-multilingual-v3.0)
   • Vector Store: Pinecone ({PINECONE_INDEX_NAME})
   • LLM: Cohere (command-r-plus-08-2024)

🎯 הצעד הבא:
   הרץ: python app_stage_a.py
        """)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ שגיאה: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
