# 🔧 פתרון בעיית NetFree + הפעלת Pinecone

## הבעיה

NetFree חוסם גישה ל-Pinecone:
```
Error: (418) Blocked by NetFree
URL: https://rag-agent-index.svc.aped-4627-b74a.pinecone.io/vectors/upsert
```

## ✅ פתרון מיידי - Hotspot מהטלפון

**הכי קל וכי מהיר!**

1. **פתח hotspot** בטלפון (הגדרות → נקודת גישה)
2. **חבר את המחשב** לhotspot
3. **הרץ את הקוד** - Pinecone יעבוד מושלם!
4. לאחר ההרצה, **חזור ל-WiFi הרגיל** - עכשיו App פועל בלי אינטרנט (הנתונים כבר ב-Pinecone)

---

## 📋 סדר הפעולות להגשה מושלמת

### שלב 1: הכן API Keys (פעם אחת)

1. צור חשבון בـ[Pinecone](https://www.pinecone.io) (חינמי!)
2. קבל **API Key** מה-Dashboard
3. ערוך קובץ `.env`:

```env
COHERE_API_KEY=YOUR_API_KEY_HERE
PINECONE_API_KEY=PASTE_YOUR_KEY_HERE
PINECONE_INDEX_NAME=rag-agent-index
```

### שלב 2: חבר Hotspot

חבר למחשב רשת Hotspot (לא NetFree!)

### שלב 3: Ingest Data

```bash
python ingest_stage_a.py
```

**תוצאה מצופה:**
```
🚀 שלב א' - הכנת נתונים (Cohere + Pinecone)
✓ Cohere Embeddings מוכנים
✓ Cohere LLM מוכן
✓ Pinecone Index: rag-agent-index
🔄 בונה VectorStoreIndex...
✅ הכנת הנתונים הושלמה בהצלחה!
   • Vector Store: Pinecone (rag-agent-index)
```

### שלב 4: הרץ את האפליקציה

```bash
python app_stage_a.py
```

**גישה:** http://localhost:7860

---

## 🏆 מה עומד בדרישות לאחר הפתרון

| רכיב | דרישה | ✅ מימוש |
|------|-------|---------|
| Embeddings | Cohere embed-multilingual-v3.0 | ✅ |
| Vector Store | **Pinecone Serverless** | ✅ |
| LLM | Cohere command-r-plus | ✅ |
| Framework | LlamaIndex | ✅ |
| UI | Gradio | ✅ |

---

## ❓ שאלות נפוצות

**ש: האם צריך Hotspot כל פעם שמריצים את האפליקציה?**

אחרי ה-ingest הראשוני - **לא!** הנתונים שמורים ב-Pinecone Cloud.
לאחר מכן האפליקציה מתחברת לקריאה בלבד ממחשב עם NetFree בדרך כלל עובד.

**ש: מה קורה אם Pinecone חסום גם בקריאה?**

הפעל את האפליקציה דרך hotspot. הטעינה של הdemo למרצה לרוב דרך hotspot בכל מקרה.

**ש: האם הנתונים ב-Pinecone נשמרים לתמיד?**

כן, חשבון חינמי כולל שמירה לצמיתות.
