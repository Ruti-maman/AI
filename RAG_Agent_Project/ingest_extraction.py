"""
שלב ג' - Data Extraction: הכנת נתונים
מריץ חילוץ מידע מובנה מכל קבצי ה-MD בתיקיית dummy_data

סקריפט זה:
1. טוען קבצי MD מ-dummy_data/
2. מחלץ מידע מובנה (החלטות, כללים, אזהרות, תלויות, שינויים)
3. שומר ל-JSON ב-storage/extracted_data.json
4. בונה גם VectorStoreIndex רגיל לחיפוש סמנטי (Pinecone)
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.cohere import Cohere
from pinecone import Pinecone, ServerlessSpec

from data_extractor import HybridExtractor
from schema import ExtractedData

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

DATA_DIR = Path("dummy_data")
STORAGE_DIR = Path("storage")
CHROMA_DB_DIR = Path("chroma_db")
EXTRACTED_DATA_FILE = STORAGE_DIR / "extracted_data.json"

# יצירת תיקיות אם לא קיימות
STORAGE_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)


# ========================================
# הגדרת LlamaIndex Settings
# ========================================

def setup_llama_index():
    """הגדרת הגדרות גלובליות של LlamaIndex"""
    
    logger.info("Setting up LlamaIndex components...")
    
    # בדיקת API Keys
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError("❌ חסר COHERE_API_KEY ב-.env")
    
    # Embeddings - Cohere עם תמיכה בעברית ורב-לשונית
    embed_model = CohereEmbedding(
        api_key=cohere_key,
        model_name="embed-multilingual-v3.0",
        input_type="search_document"
    )
    logger.info(f"✓ Embeddings: Cohere embed-multilingual-v3.0")
    
    # LLM - Cohere
    llm = Cohere(
        api_key=cohere_key,
        model="command-r-plus-08-2024",
        temperature=0.1
    )
    logger.info("✓ LLM: Cohere command-r-plus-08-2024")
    
    # Node Parser
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
    
    logger.info("✓ LlamaIndex setup complete")
    
    return llm


# ========================================
# שלב 1: חילוץ מידע מובנה
# ========================================

def extract_structured_data(use_llm: bool = True) -> ExtractedData:
    """
    חילוץ מידע מובנה מכל קבצי ה-MD
    
    Args:
        use_llm: האם להשתמש ב-LLM extraction (איטי אבל מדויק)
    
    Returns:
        ExtractedData עם כל המידע שחולץ
    """
    logger.info("=" * 60)
    logger.info("שלב 1: חילוץ מידע מובנה")
    logger.info("=" * 60)
    
    # יצירת Extractor
    llm = Cohere(
        api_key=os.getenv("COHERE_API_KEY"),
        model="command-r-plus-08-2024",
        temperature=0.1
    ) if use_llm else None
    extractor = HybridExtractor(use_llm=use_llm, llm=llm)
    
    # חילוץ מהתיקייה
    logger.info(f"Extracting from directory: {DATA_DIR}")
    extracted = extractor.extract_from_directory(DATA_DIR, pattern="*.md")
    
    # הדפסת סטטיסטיקה
    logger.info("\n" + "=" * 60)
    logger.info("📊 סטטיסטיקת חילוץ:")
    logger.info(f"   📋 החלטות: {len(extracted.decisions)}")
    logger.info(f"   📏 כללים: {len(extracted.rules)}")
    logger.info(f"   ⚠️  אזהרות: {len(extracted.warnings)}")
    logger.info(f"   📦 תלויות: {len(extracted.dependencies)}")
    logger.info(f"   🔄 שינויים: {len(extracted.changes)}")
    logger.info(f"   📊 סה\"כ: {len(extracted.get_all_items())} פריטים")
    logger.info("=" * 60 + "\n")
    
    return extracted


# ========================================
# שלב 2: שמירת המידע המובנה
# ========================================

def save_extracted_data(extracted: ExtractedData):
    """שמירת המידע המובנה ל-JSON"""
    logger.info("=" * 60)
    logger.info("שלב 2: שמירת מידע מובנה")
    logger.info("=" * 60)
    
    # המרה ל-dict
    data_dict = extracted.to_dict()
    
    # שמירה ל-JSON
    with open(EXTRACTED_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Saved to: {EXTRACTED_DATA_FILE}")
    logger.info(f"  File size: {EXTRACTED_DATA_FILE.stat().st_size / 1024:.1f} KB")
    logger.info("")


# ========================================
# שלב 3: בניית VectorStoreIndex (לחיפוש סמנטי)
# ========================================

def build_vector_index():
    """בניית VectorStoreIndex רגיל לחיפוש סמנטי ב-Pinecone"""
    logger.info("=" * 60)
    logger.info("שלב 3: בניית VectorStoreIndex")
    logger.info("=" * 60)
    
    # טעינת המסמכים
    logger.info(f"Loading documents from: {DATA_DIR}")
    documents = SimpleDirectoryReader(
        str(DATA_DIR),
        filename_as_id=True,
        recursive=True
    ).load_data()
    logger.info(f"✓ Loaded {len(documents)} documents")
    
    # חיבור ל-Pinecone
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-agent-index")
    
    if not pinecone_key:
        raise ValueError("❌ חסר PINECONE_API_KEY ב-.env")
    
    pc = Pinecone(api_key=pinecone_key)
    
    # בדיקה אם ה-index קיים, אם לא - יוצר חדש
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        logger.info(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1024,  # Cohere embed-multilingual-v3.0
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info("✓ Pinecone index created")
    else:
        logger.info(f"✓ Using existing Pinecone index: {index_name}")
    
    # חיבור ל-index
    pinecone_index = pc.Index(index_name)
    
    # יצירת Vector Store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace="rag_agent_docs"
    )
    
    # בניית האינדקס
    logger.info("Building VectorStoreIndex...")
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        show_progress=True
    )
    
    logger.info(f"✓ Index built and uploaded to Pinecone")
    logger.info("")
    
    return index


# ========================================
# Main Execution
# ========================================

def main():
    """הרצת תהליך ההכנה המלא"""
    
    print("\n" + "🚀" * 30)
    print("   שלב ג': הכנת נתונים - Extraction + Vector Index")
    print("🚀" * 30 + "\n")
    
    try:
        # הגדרת LlamaIndex
        llm = setup_llama_index()
        
        # שלב 1: חילוץ מידע מובנה
        # הערה: use_llm=False למהירות. שנה ל-True לדיוק מקסימלי (איטי יותר)
        extracted = extract_structured_data(use_llm=False)
        
        # שלב 2: שמירה
        save_extracted_data(extracted)
        
        # שלב 3: בניית Vector Index
        index = build_vector_index()
        
        # סיכום
        print("\n" + "✅" * 30)
        print("   הכנת הנתונים הושלמה בהצלחה!")
        print("✅" * 30)
        print(f"""
📁 קבצים שנוצרו:
   • {EXTRACTED_DATA_FILE} - מידע מובנה (JSON)
   • Pinecone Index: {os.getenv('PINECONE_INDEX_NAME', 'rag-agent-index')}

📊 סטטיסטיקה:
   • החלטות: {len(extracted.decisions)}
   • כללים: {len(extracted.rules)}
   • אזהרות: {len(extracted.warnings)}
   • תלויות: {len(extracted.dependencies)}
   • שינויים: {len(extracted.changes)}
   • סה"כ: {len(extracted.get_all_items())} פריטים

🎯 מוכן להרצת app_complete_full.py!
        """)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
