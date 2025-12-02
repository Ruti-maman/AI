# 🤖 LLM CLI Generator

אפליקציה המשתמשת ב-AI כדי ליצור פקודות CLI (Command Line Interface) מהוראות בשפה טבעית.

## 📋 תיאור

האפליקציה מקבלת הוראות בעברית או אנגלית ומחזירה פקודות CLI מתאימות עבור Windows/Linux באמצעות מודל Llama 3.3 דרך Groq API.

## 🚀 התקנה

### דרישות מקדימות
- Python 3.8 ומעלה
- uv (כלי ניהול פרויקטים של Python)

### שלבי התקנה

1. **שכפול הפרויקט**
   ```bash
   git clone <repository-url>
   cd gradio_cli_app
   ```

2. **התקנת תלויות**
   ```bash
   uv sync
   ```

3. **הגדרת API Key**
   
   א. היכנסי ל-[console.groq.com](https://console.groq.com)
   
   ב. צרי חשבון חינמי והתחברי
   
   ג. עברי ל-"API Keys" וצרי מפתח חדש
   
   ד. העתיקי את המפתח ועדכני את קובץ `.env`:
   ```
   GROQ_API_KEY=gsk_your_actual_api_key_here
   ```

## ▶️ הרצה

הפעלת האפליקציה:
```bash
uv run main.py
```

לאחר מכן, פתחי דפדפן וגלשי ל: **http://127.0.0.1:7860**

## 💡 דוגמאות שימוש

הקלידי הוראות כמו:

- **"הצג את כל הקבצים בתיקייה הנוכחית"**
  ```
  ls
  ```

- **"צור תיקייה חדשה בשם projects"**
  ```
  mkdir projects
  ```

- **"מצא את כל הקבצים בסיומת .py"**
  ```
  Get-ChildItem -Filter *.py -Recurse
  ```

- **"הצג את תהליכי המערכת"**
  ```
  Get-Process
  ```

## 🛠️ טכנולוגיות

- **Gradio** - ממשק משתמש אינטראקטיבי
- **Groq API** - גישה למודל Llama 3.3-70B
- **Python-dotenv** - ניהול משתני סביבה
- **uv** - ניהול תלויות והרצת הפרויקט

## 📁 מבנה הפרויקט

```
gradio_cli_app/
├── main.py           # קובץ הראשי של האפליקציה
├── .env              # משתני סביבה (API keys)
├── pyproject.toml    # תלויות הפרויקט
├── README.md         # תיעוד
└── .venv/            # סביבה וירטואלית
```

## ⚠️ פתרון בעיות

### בעיית "Connection error"
אם מקבלת שגיאת חיבור, ייתכן שיש בעיית SSL. הקוד כבר מטפל בזה עם:
```python
http_client = httpx.Client(verify=False)
```

### API Key לא עובד
- ודאי שהמפתח מועתק במלואו ללא רווחים
- בדקי שקובץ `.env` נמצא בתיקיית הפרויקט
- ודאי שהמפתח לא פג תוקף (צרי חדש אם צריך)

### היישום לא עולה
```bash
# נקי את הסביבה והתקיני מחדש
rm -rf .venv
uv sync
uv run main.py
```

## 📝 רישיון

פרויקט לימודי - חופשי לשימוש ולמידה

## 🙏 תודות

- [Groq](https://groq.com) - על API מהיר וחינמי
- [Gradio](https://gradio.app) - על ממשק פשוט ונהדר
- [Meta](https://ai.meta.com/llama/) - על מודל Llama
