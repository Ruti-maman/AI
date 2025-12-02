import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ לא נמצא API key בקובץ .env")
else:
    print(f"✅ API key נמצא (מתחיל ב-{api_key[:20]}...)")
    
    try:
        client = OpenAI(api_key=api_key)
        # נסיון לשלוח בקשה פשוטה
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5
        )
        print("✅ החיבור ל-OpenAI עובד! יש לך קרדיט פעיל")
        print(f"תשובה: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ שגיאה: {e}")
        if "insufficient_quota" in str(e):
            print("   → אין לך מספיק קרדיט ב-OpenAI")
        elif "invalid_api_key" in str(e):
            print("   → המפתח לא תקין")
        else:
            print("   → בעיית חיבור או אימות")
