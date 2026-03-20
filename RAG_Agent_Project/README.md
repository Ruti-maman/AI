# 🤖 RAG Agent Project - מערכת RAG מתקדמת עם Cohere + Pinecone

## 📋 סיכום הפרויקט

מערכת RAG (Retrieval-Augmented Generation) מקצועית בשלושה שלבים, המשתמשת ב-**Cohere API + Pinecone** לפי דרישות המטלה.

### ✅ רכיבי המערכת

| רכיב               | דרישה         | מימוש                                      | סטטוס    |
| ------------------ | ------------- | ------------------------------------------ | -------- |
| **1. Loading**     | טעינת מסמכים  | `LlamaIndex SimpleDirectoryReader`         | ✅ מושלם |
| **2. Chunking**    | חלוקה לקטעים  | `LlamaIndex SentenceSplitter (512 tokens)` | ✅ מושלם |
| **3. Embedding**   | המרה לוקטורים | **`Cohere embed-multilingual-v3.0`**       | ✅ מושלם |
| **4. VectorStore** | מאגר וקטורים  | **`Pinecone (Serverless)`**                | ✅ מושלם |
| **5. Retrieval**   | חיפוש והשבה   | `Vector Similarity Search (cosine)`        | ✅ מושלם |
| **6. Synthesis**   | בניית תשובה   | **`Cohere command-r-plus`**                | ✅ מושלם |

---

## 🏗️ ארכיטקטורה

```
dummy_data/           ← מסמכי מקור (.md files)
    ├── spec.md
    ├── api_guide.md
    └── bugs.md

        ↓ [1. LOADING - SimpleDirectoryReader]

Documents             ← מסמכים מובנים

        ↓ [2. CHUNKING - SentenceSplitter]

Chunks                ← קטעי טקסט (512 tokens)

        ↓ [3. EMBEDDING - Cohere API]

Vectors (1024 dims)   ← ייצוג סמנטי רב-לשוני

        ↓ [4. VECTOR STORE - Pinecone]

Pinecone Cloud        ← מאגר וקטורים מנוהל
    └── rag-agent-index

        ↓ [5. RETRIEVAL - Similarity Search]

Relevant Chunks       ← תוצאות רלוונטיות (top-k)

        ↓ [6. SYNTHESIS - Cohere LLM]

Answer                ← תשובה חכמה למשתמש
```

---

## 🚀 הפעלת המערכת

### דרישות מוקדמות

1. **Python 3.10+**
2. **API Keys:**
   - Cohere API Key (https://cohere.com)
   - Pinecone API Key (https://www.pinecone.io)

### הגדרת API Keys

צור קובץ `.env`:

```bash
COHERE_API_KEY=your_cohere_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=rag-agent-index
```

### התקנת תלויות

```bash
pip install -r requirements_final.txt
```

### שלב 1: הכנת הנתונים (חד-פעמי)

**🔹 שלב א' - MVP RAG:**

```bash
python ingest_stage_a.py
```

**מה קורה?**

- 📁 טוען מסמכי MD מ-`dummy_data/`
- ✂️ מפצל לchunks (512 tokens, 50 overlap)
- 🧮 יוצר Cohere embeddings (1024 ממדים)
- 💾 מעלה ל-Pinecone index

**🔹 שלב ב' - Event-Driven Workflow:**

```bash
python ingest_workflow.py
```

**🔹 שלב ג' - Data Extraction + Router:**

```bash
python ingest_extraction.py
```

### שלב 2: הפעלת האפליקציה

**🔹 שלב א' - MVP RAG:**

```bash
python app_stage_a.py
```

**🔹 שלב ב' - Event-Driven Workflow:**

```bash
python app_workflow.py
```

**🔹 שלב ג' - מערכת מלאה (Extraction + Router):**

```bash
python app_complete_full.py
```

**גשי ל:** http://localhost:7860

---

## 🧠 כיצד זה עובד?

### תהליך מענה לשאילתה:

```python
# שאילתה: "מה הצבע של המערכת?"

1. EMBEDDING (Cohere)
   "מה הצבע של המערכת?"
   → Cohere embed-multilingual-v3.0
   → [0.0423, -0.1245, 0.3456, ..., 0.0891]  # וקטור 1024 ממדים

2. RETRIEVAL (Pinecone)
   Vector: [0.0423, -0.1245, ...]
   → Pinecone Similarity Search (cosine)
   → מחזיר top-3 הכי דומים מהאינדקס

3. SYNTHESIS (Cohere LLM)
   Context: [Top 3 chunks]
   Query: "מה הצבע של המערכת?"
   → Cohere command-r-plus
   → "הצבע הרשמי של המערכת הוא כחול כהה..."
```

---

## 🔬 הסבר טכני מעמיק

### למה Cohere + Pinecone?

**Cohere Embeddings**:

- ✅ מודלים מתקדמים לעיבוד שפה טבעית
- ✅ תמיכה מלאה בעברית ו-100+ ש embeddings סמנטיים איכוtiים (1024 ממדים)
- ✅ מותאם לחיפוש (input_type: search_document/search_query)
- ✅ API מהיר ויציב

**Pinecone Vector Store**:

- ✅ מאגר וקטורים מנוהל בענן
- ✅ חיפוש מהיר (milliseconds) על מיליוני וקטורים
- ✅ Serverless - ללא ניהול תשתית
- ✅ Scalable - מתרחב אוטומטית
- ✅ cosine similarity לחיפוש סמנטי מדויק

**Cohere LLM**:

- ✅ command-r-plus - מודל חזק ומתקדם
- ✅ הבנת הקשר עמוקה
- ✅ תשובות איכותיות ורלוונטיות
- ✅ תמיכה בעברית מצוינת

**Neural Embeddings (BERT, SentenceTransformers):**

- ❌ דורש הורדה מHuggingFace (חסומה ברשת הנוכחית)
- ❌ דורש GPU לביצועים טובים
- ⚠️ לא תמיד עדיף על TF-IDF בקורפוסים קטנים

### FAISS vs. אחרים

**FAISS (Facebook AI Similarity Search)**:

- ✅ פיתוח Meta AI - תקן תעשייתי
- ✅ מהיר מאוד (אופטימיזציות SIMD)

---

## 📊 שלבי הפרויקט

### שלב א' - MVP RAG:

- ✅ Loading + Chunking
- ✅ Cohere Embeddings
- ✅ Pinecone Vector Store
- ✅ חיפוש סמנטי בסיסי
- ✅ Cohere LLM לסינתזה

**קבצים:** `ingest_stage_a.py` + `app_stage_a.py`

### שלב ב' - Event-Driven Workflow:

- ✅ כל שלב א' +
- ✅ Workflow Engine עם 4 שלבים
- ✅ 15 סוגי Events למעקב
- ✅ מערכת Validation מתקדמת

**קבצים:** `ingest_workflow.py` + `app_workflow.py` + `workflow_*.py`

### שלב ג' - Complete System:

- ✅ כל שלבים א'+ב' +
- ✅ Data Extraction (החלטות, כללים, אזהרות)
- ✅ Smart Router (SEMANTIC/STRUCTURED/HYBRID)
- ✅ שליפה מובנית מ-JSON

**קבצים:** `ingest_extraction.py` + `app_complete_full.py` + `data_extractor.py` + `router.py`

---

## 🗂️ מבנה הקבצים

```
RAG_Agent_Project/
│
├── 📱 שלב א' - MVP RAG
│   ├── ingest_stage_a.py       # הכנת נתונים
│   └── app_stage_a.py          # אפליקציה
│
├── 🔄 שלב ב' - Workflow
│   ├── ingest_workflow.py      # הכנת נתונים עם Events
│   ├── app_workflow.py         # אפליקציה עם Workflow
│   ├── workflow_engine.py      # מנוע Workflow
│   ├── workflow_events.py      # הגדרות Events
│   └── workflow_steps.py       # 4 שלבי Workflow
│
├── 🎯 שלב ג' - Complete
│   ├── ingest_extraction.py    # חילוץ מידע מובנה
│   ├── app_complete_full.py    # אפליקציה מלאה
│   ├── data_extractor.py       # LLM + Pattern Extraction
│   ├── router.py               # Smart Query Router
│   └── schema.py               # Data Models
│
├── 📁 Data & Config
│   ├── dummy_data/             # מסמכי MD מקור
│   ├── .env                    # API Keys (לא ב-git)
│   ├── .env.example            # תבנית
│   └── requirements_final.txt  # תלויות
│
└── 📖 Documentation
    ├── README.md               # קובץ זה
    └── QUICKSTART.md           # מדריך מהיר
```

---

## 📦 תלויות (Dependencies)

```txt
# Core Framework
llama-index-core>=0.12.0

# Cohere Integration
llama-index-embeddings-cohere
llama-index-llms-cohere
cohere>=5.0.0

# Pinecone Vector Store
llama-index-vector-stores-pinecone
pinecone-client>=3.0.0

# UI & Utils
gradio>=5.9.1
python-dotenv
```

````

**התקנה:**
```bash
pip install -r requirements.txt
````

---

## 🎯 יתרונות המערכת

### 1. **עצמאות מלאה**

- ✅ לא דורש חיבור לאינטרנט
- ✅ לא דורש API keys

---

## ⚡ יתרונות המערכת

### 1. **טכנולוגיות מתקדמות**

- 🧠 Cohere Embeddings - state-of-the-art סמנטיים
- ☁️ Pinecone - vector store מנוהל ומהיר
- 🤖 Cohere LLM - תשובות איכותיות
- 🌍 תמיכה מלאה בעברית ו-100+ שפות

### 2. **ביצועים גבוהים**

- ⚡ חיפוש מהיר (milliseconds)
- ⚡ Scalable - מטפל במיליוני documents
- ⚡ אמין - תשתית מנוהלת

### 3. **גמישות**

- 🔧 3 שלבים: MVP → Workflow → Complete
- 🔧 Event-Driven architecture
- 🔧 Smart Router לסוגי שאילתות שונים
- 🔧 קוד מודולרי וברור

### 4. **מקצועי**

- 📚 עומד בדרישות המטלה במדויק
- 📚 משתמש בטכנולוגיות תעשייתיות
- 📚ניתן להרחבה וקנה מידה

---

## 🔧 הרחבת המערכת

### הוספת מסמכים חדשים:

1. הוסף קבצי `.md` חדשים ל-`dummy_data/`
2. הרץ מחדש את ההכנה:
   ```bash
   python ingest_stage_a.py     # או
   python ingest_workflow.py    # או
   python ingest_extraction.py
   ```
3. המערכת תדע לענות על שאלות מהמסמכים החדשים!

### דוגמה:

```bash
# יצירת מסמך חדש
echo "# פיצ'רים חדשים\n\nהמערכת כוללת dark mode." > dummy_data/features.md

# עיבוד מחדש
python ingest_stage_a.py

# הרץ אפליקציה
python app_stage_a.py

# עכשיו אפשר לשאול: "יש dark mode?"
```

---

## 🎓 הערות טכניות

### למה Cohere + Pinecone?

#### 1. **Cohere Embeddings**

**יתרונות:**

- 🧠 מודלים מתקדמים מבוססי Transformer
- 🌍 תמיכה ב-100+ שפות כולל עברית מצוינת
- 📊 1024 ממדים - ייצוג עשיר
- 🎯 מותאם לחיפוש (input_type parameters)
- ⚡ API מהיר ויציב

**מקורות:**

- Cohere.ai - "Embed v3: A New Frontier for Retrieval"
- תעשייתי - בשימוש ב-Notion, Oracle, Salesforce

#### 2. **Pinecone Vector Store**

**יתרונות:**

- ☁️ Fully managed - אין ניהול infrastructure
- ⚡ חיפוש מהיר (<100ms על מיליוני וקטורים)
- 📈 Scalable - גמיש לכל גודל
- 🔒 אמין - 99.9% uptime
- 💡 Serverless option - pay-per-use

**מקורות:**

- בשימוש: Gong, Hubspot, Replit
- מומלץ בתיעוד LlamaIndex הרשמי

#### 3. **שלבים מתקדמים**

**שלב ב' - Event-Driven:**

- מעקב מלא אחר כל שלב בתהליך
- 15 סוגי Events שונים
- Validation מתקדם

**שלב ג' - Hybrid System:**

- Data Extraction עם LLM
- Smart Router לזיהוי כוונות
- שילוב חיפוש סמנטי + structured data

🎯 **מתי TF-IDF מתאים?**

- קורפוס קטן-בינוני (< 10,000 מסמכים)
- חיפוש keyword-based
- צורך במהירות וקלות תחזוקה
- אין GPU זמין

#### 3. **FAISS הוא Vector Store מקצועי**

- 🏢 פותח ע"י Meta AI
- 📈 משמש ב-production במיליוני מערכות
- ⚡ מהיר יותר מפתרונות cloud רבים
- 💪 תומך ב-billions של vectors

#### 4. **LlamaIndex לטעינה ו-Chunking**

- ✅ ספרייה תקנית למטלות RAG
- ✅ כפי שנדרש במטלה
- ✅ אופטימיזציה אוטומטית של chunks

---

## 🆚 השוואה: app_rag.py (BM25) vs. app_complete.py (TF-IDF + FAISS)

| מאפיין           | app_rag.py (BM25) | app_complete.py (Vector) |
| ---------------- | ----------------- | ------------------------ |
| **Embedding**    | ❌ אין            | ✅ TF-IDF                |
| **VectorStore**  | ❌ JSON           | ✅ FAISS                 |
| **Retrieval**    | Keyword matching  | Vector similarity        |
| **מהירות**       | מהיר              | מהיר מאוד                |
| **דיוק**         | טוב               | מצוין                    |
| **סקלה**         | עד ~1000 docs     | עד מיליוני docs          |
| **עומד בדרישות** | ⚠️ חלקי           | ✅ מלא                   |

**המלצה להגשה:** השתמשי ב-`app_complete.py` - זו הגרסה המושלמת!

---

## 🧪 בדיקות איכות

### דוגמאות שאילתות שהמערכת עונה עליהן נכון:

```python
✅ "מה הצבע של המערכת?"
   → "הצבע הרשמי של המערכת הוא כחול כהה (#00008B)."

✅ "איזה שפות נתמכות?"
   → "המערכת תומכת בשלוש שפות: עברית, אנגלית וצרפתית."

✅ "ספרי על RTL"
   → "יש צורך ב-RTL support למסכי עברית כדי להציג את הטקסט מימין לשמאל."

✅ "איך עובד ה-authentication?"
   → "כל הבקשות דורשות X-API-Key בheader. קוד 401 מחזיר Unauthorized."

✅ "ספר על bug 142"
   → "Bug #142 - טעינה איטית בדף הבית כאשר יש מעל 1000 פריטים..."

✅ "מה ה-rate limit?"
   → "Rate limiting: 100 requests per minute per API key..."
```

---

## 💡 שאלות נפוצות

### ש: למה לא BERT/GPT embeddings?

**ת:** דורש הורדה מרשת (חסומה) + GPU. TF-IDF מצוין לקורפוס הנוכחי.

### ש: למה לא Pinecone?

**ת:** דורש חיבור רשת + תשלום. FAISS מקומי ומהיר יותר.

### ש: איפה ה-LLM לsynthesis?

**ת:** LLM לא נדרש במטלה. המערכת מחזירה קטעים רלוונטיים (RAG קלאסי). במערכת מלאה, זה היה עובר דרך GPT.

### ש: האם זה עומד בדרישות המטלה?

**ת:** כן! כל 6 הרכיבים מיושמים עם טכנולוגיות מקצועיות ומקובלות.

---

## 🎯 סיכום

הפרויקט מממש **מערכת RAG מושלמת** עם:

- ✅ כל 6 רכיבי המטלה
- ✅ טכנולוגיות אמינות ומקובלות (TF-IDF, FAISS)
- ✅ ביצועים מעולים
- ✅ קוד נקי ומתועד
- ✅ תמיכה מלאה בעברית
- ✅ עצמאית לחלוטין (ללא רשת)

**מוכן להגשה!** 🎓

---

## 👩‍💻 מחברת

**שם:** רותי
**קורס:** AI - שנה ב'
**תאריך:** מרץ 2026

---

## 📚 מקורות

1. **TF-IDF:**
   - Salton, G. & McGill, M. J. (1983). Introduction to Modern Information Retrieval.
   - Robertson, S. (2004). Understanding Inverse Document Frequency.

2. **FAISS:**
   - Johnson, J. et al. (2021). Billion-scale similarity search with GPUs. IEEE.
   - Meta AI FAISS Documentation: https://github.com/facebookresearch/faiss

3. **RAG Systems:**
   - Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
   - LlamaIndex Documentation: https://docs.llamaindex.ai/

---

**אם יש שאלות, כל הקוד מתועד ומוסבר! 📖**
