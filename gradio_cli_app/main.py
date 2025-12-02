import os
from dotenv import load_dotenv
import gradio as gr
from groq import Groq
import httpx

# -------------------------------
# 1️⃣ טעינת משתנים סודיים מה-.env
# -------------------------------
load_dotenv()  # טוען משתנים מהקובץ .env
api_key = os.getenv("GROQ_API_KEY")

if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
    print("⚠️ אזהרה: API key לא הוגדר! הכניסי מפתח תקין בקובץ .env")
    api_key = None

# יצירת HTTP client שמתעלם מ-SSL
http_client = httpx.Client(verify=False) if api_key else None
client = Groq(api_key=api_key, http_client=http_client) if api_key else None

# -------------------------------
# 2️⃣ פונקציה ליצירת פקודת CLI
# -------------------------------
def generate_cli_command(instruction: str) -> str:
    """
    מקבלת הוראה בשפה טבעית,
    שולחת ל-LLM ומחזירה פקודת CLI.
    """
    if not client:
        return "❌ שגיאה: API key לא הוגדר! היכנסי ל-console.groq.com, קבלי API key, ועדכני את קובץ .env"
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "אתה עוזר שמייצר פקודות CLI עבור לינוקס/ווינדוס. תן רק את הפקודה, ללא הסברים."},
                {"role": "user", "content": f"כתוב פקודת CLI עבור ההוראה הבאה:\n{instruction}"}
            ],
            max_tokens=100,
            temperature=0
        )
        command = response.choices[0].message.content.strip()
        return command or "לא התקבלה פקודה."
    except Exception as e:
        return f"שגיאה בחיבור ל-Groq: {e}"

# -------------------------------
# 3️⃣ יצירת ממשק Gradio
# -------------------------------
iface = gr.Interface(
    fn=generate_cli_command,
    inputs=gr.Textbox(lines=2, placeholder="הקלידי הוראה בשפה טבעית כאן..."),
    outputs="text",
    title="LLM CLI Generator",
    description="הקלידי הוראה בשפה טבעית, ותקבלי פקודת CLI מתאימה."
)

# -------------------------------
# 4️⃣ הפעלת האפליקציה
# -------------------------------
if __name__ == "__main__":
    iface.launch()


