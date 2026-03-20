# 🚀 מדריך הפעלה מהיר

## הרצת המערכת המושלמת (מומלץ להגשה!)

### שלב 1: הכנת הנתונים
```bash
python ingest_complete.py
```
**פלט צפוי:**
```
✅ נטענו 3 מסמכים
✅ נוצרו 4 chunks  
✅ נוצרו 4 embeddings (609 ממדים)
✅ FAISS index מוכן
```

### שלב 2: הפעלת האפליקציה
```bash
python app_complete.py
```
**גשי ל:** http://127.0.0.1:7882

---

## 🎯 המערכת כוללת את כל 6 הרכיבים:

1. ✅ **Loading** - LlamaIndex SimpleDirectoryReader
2. ✅ **Chunking** - LlamaIndex SentenceSplitter  
3. ✅ **Embedding** - TF-IDF (scikit-learn)
4. ✅ **VectorStore** - FAISS
5. ✅ **Retrieval** - Vector Similarity Search
6. ✅ **Synthesis** - Response Generation

---

## 📊 נתונים:
- **3 מסמכים** (spec.md, api_guide.md, bugs.md)
- **4 chunks** מעובדים
- **609 ממדים** לכל embedding
- **תמיכה בעברית, אנגלית וצרפתית**

---

## 🧪 דוגמאות שאילתות:
- "מה הצבע של המערכת?"
- "איזה שפות נתמכות?"
- "ספר על bug 142"
- "איך עובד ה-authentication?"

---

## 📖 לפרטים מלאים: 
קראי את [README.md](README.md)

---

**מוכן להגשה!** 🎓
