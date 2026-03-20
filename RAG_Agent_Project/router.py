"""
שלב ג' - Router: מנגנון ניתוב חכם
מחליט מתי להשתמש בחיפוש סמנטי ומתי בשליפה מובנית

Router Types:
1. Keyword-Based Router - ניתוב על בסיס מילות מפתח
2. LLM-Based Router - ניתוב באמצעות LLM (מדויק יותר)
3. Hybrid Router - משלב את שניהם
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import re
from dotenv import load_dotenv

from llama_index.llms.cohere import Cohere

from schema import QueryIntent, ExtractedData

# טעינת משתני סביבה
load_dotenv()

# הגדרת Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================
# Router Result
# ========================================

@dataclass
class RouterResult:
    """תוצאת ניתוב"""
    intent: QueryIntent
    confidence: float  # 0.0 - 1.0
    reasoning: str
    suggested_query: Optional[str] = None  # שאילתה משופרת


# ========================================
# Keyword-Based Router (מהיר)
# ========================================

class KeywordRouter:
    """ניתוב על בסיס מילות מפתח"""
    
    # מילות מפתח לכל סוג שאילתה
    STRUCTURED_KEYWORDS = [
        # כללי
        "list", "show", "count", "all", "find", "what are",
        "רשימה", "הצג", "כמה", "כל", "מצא", "מה הם",
        
        # ספציפי לסוגים
        "decisions", "החלטות", "decision", "החלטה",
        "rules", "כללים", "rule", "כלל",
        "warnings", "אזהרות", "warning", "אזהרה",
        "dependencies", "תלויות", "dependency", "תלות",
        "changes", "שינויים", "change", "שינוי",
        
        # אופרטורים
        "where", "with", "by", "of", "בהם", "עם", "לפי", "של"
    ]
    
    SEMANTIC_KEYWORDS = [
        # שאלות פתוחות
        "how", "why", "explain", "describe", "what is", "when",
        "איך", "למה", "הסבר", "תאר", "מה זה", "מתי",
        
        # חיפוש תוכן
        "about", "related to", "similar to", "like",
        "על", "קשור ל", "דומה ל", "כמו",
        
        # יעוץ
        "recommend", "suggest", "best", "should i", "can i",
        "המלץ", "הצע", "הכי טוב", "האם כדאי", "האם אני יכול"
    ]
    
    def __init__(self):
        logger.info("Initialized KeywordRouter")
    
    def route(self, query: str) -> RouterResult:
        """ניתוב שאילתה"""
        query_lower = query.lower()
        
        # ספירת מילות מפתח
        structured_count = sum(1 for kw in self.STRUCTURED_KEYWORDS if kw in query_lower)
        semantic_count = sum(1 for kw in self.SEMANTIC_KEYWORDS if kw in query_lower)
        
        # החלטה
        if structured_count > semantic_count:
            confidence = min(0.5 + (structured_count * 0.1), 0.95)
            return RouterResult(
                intent=QueryIntent.STRUCTURED,
                confidence=confidence,
                reasoning=f"Found {structured_count} structured keywords vs {semantic_count} semantic"
            )
        elif semantic_count > structured_count:
            confidence = min(0.5 + (semantic_count * 0.1), 0.95)
            return RouterResult(
                intent=QueryIntent.SEMANTIC,
                confidence=confidence,
                reasoning=f"Found {semantic_count} semantic keywords vs {structured_count} structured"
            )
        else:
            # במקרה של שוויון - ברירת מחדל לסמנטי
            return RouterResult(
                intent=QueryIntent.SEMANTIC,
                confidence=0.5,
                reasoning="No clear keyword match - defaulting to semantic"
            )


# ========================================
# LLM-Based Router (מדויק)
# ========================================

class LLMRouter:
    """ניתוב באמצעות LLM"""
    
    ROUTING_PROMPT_TEMPLATE = """
You are a query routing expert. Your job is to classify user queries.

Query types:
1. STRUCTURED - Query asking for specific structured data:
   - Lists (e.g., "show all decisions", "list warnings")
   - Counts (e.g., "how many rules", "count dependencies")
   - Filters (e.g., "find high-severity warnings", "decisions about API")
   - Specific data retrieval

2. SEMANTIC - Query asking for conceptual or open-ended information:
   - Explanations (e.g., "how does authentication work")
   - Advice (e.g., "what should I do for error handling")
   - General knowledge (e.g., "explain the architecture")
   - Open questions

3. HYBRID - Query that needs both:
   - Lists with context
   - Structured data with explanations

User query: "{query}"

Respond in JSON format:
{{
    "intent": "STRUCTURED" or "SEMANTIC" or "HYBRID",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation",
    "suggested_query": "improved query (optional)"
}}

Return ONLY the JSON, no additional text.
"""
    
    def __init__(self, llm: Optional[Cohere] = None):
        """
        Args:
            llm: אובייקט LLM מותאם אישית
        """
        self.llm = llm or Cohere(
            api_key=os.getenv("COHERE_API_KEY"),
            model="command-r-plus-08-2024",
            temperature=0.1
        )
        logger.info("Initialized LLMRouter with Cohere")
    
    def route(self, query: str) -> RouterResult:
        """ניתוב שאילתה באמצעות LLM"""
        try:
            # יצירת prompt
            prompt = self.ROUTING_PROMPT_TEMPLATE.format(query=query)
            
            # שליחה ל-LLM
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            
            # פרסור תשובה
            import json
            
            # ניקוי markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response_text)
            
            # המרה ל-QueryIntent
            intent_str = data.get("intent", "SEMANTIC").upper()
            if intent_str == "STRUCTURED":
                intent = QueryIntent.STRUCTURED
            elif intent_str == "HYBRID":
                intent = QueryIntent.HYBRID
            else:
                intent = QueryIntent.SEMANTIC
            
            return RouterResult(
                intent=intent,
                confidence=data.get("confidence", 0.7),
                reasoning=data.get("reasoning", "LLM routing"),
                suggested_query=data.get("suggested_query")
            )
            
        except Exception as e:
            logger.error(f"Error in LLM routing: {e}")
            # Fallback לסמנטי
            return RouterResult(
                intent=QueryIntent.SEMANTIC,
                confidence=0.5,
                reasoning=f"Fallback due to error: {e}"
            )


# ========================================
# Hybrid Router (החכם ביותר)
# ========================================

class HybridRouter:
    """משלב Keyword + LLM routing"""
    
    def __init__(self, use_llm: bool = True, llm: Optional[Cohere] = None):
        """
        Args:
            use_llm: האם להשתמש ב-LLM (איטי יותר אבל מדויק יותר)
            llm: אובייקט LLM מותאם אישית
        """
        self.keyword_router = KeywordRouter()
        self.llm_router = LLMRouter(llm) if use_llm else None
        self.use_llm = use_llm
        logger.info(f"Initialized HybridRouter (LLM: {use_llm})")
    
    def route(self, query: str) -> RouterResult:
        """ניתוב חכם משולב"""
        
        # שלב 1: Keyword routing (מהיר)
        keyword_result = self.keyword_router.route(query)
        logger.info(f"Keyword routing: {keyword_result.intent.value} (confidence: {keyword_result.confidence:.2f})")
        
        # אם confidence גבוה מספיק, לא צריך LLM
        if keyword_result.confidence > 0.8:
            return keyword_result
        
        # שלב 2: LLM routing (מדויק)
        if self.use_llm and self.llm_router:
            llm_result = self.llm_router.route(query)
            logger.info(f"LLM routing: {llm_result.intent.value} (confidence: {llm_result.confidence:.2f})")
            
            # אם LLM בטוח, לקחת את התשובה שלו
            if llm_result.confidence > 0.7:
                return llm_result
            
            # אחרת, לשלב בין השניים (ממוצע משוקלל)
            # LLM מקבל משקל גבוה יותר כי הוא מדויק יותר
            if keyword_result.intent == llm_result.intent:
                # שניהם מסכימים - confidence גבוה
                combined_confidence = (keyword_result.confidence * 0.3 + llm_result.confidence * 0.7)
                return RouterResult(
                    intent=llm_result.intent,
                    confidence=min(combined_confidence * 1.2, 0.95),  # בונוס להסכמה
                    reasoning=f"Agreement: {keyword_result.reasoning} + {llm_result.reasoning}",
                    suggested_query=llm_result.suggested_query
                )
            else:
                # אי-הסכמה - לקחת את ה-LLM אבל עם confidence נמוך
                return RouterResult(
                    intent=llm_result.intent,
                    confidence=min(llm_result.confidence * 0.8, 0.7),
                    reasoning=f"Disagreement (keyword: {keyword_result.intent.value}, llm: {llm_result.intent.value}) - using LLM",
                    suggested_query=llm_result.suggested_query
                )
        
        # אם אין LLM, להשתמש ב-keyword
        return keyword_result


# ========================================
# Query Executor (מבצע שאילתות)
# ========================================

class QueryExecutor:
    """מבצע שאילתות על בסיס ה-routing"""
    
    def __init__(self, extracted_data: ExtractedData):
        """
        Args:
            extracted_data: המידע המובנה שחולץ
        """
        self.data = extracted_data
        logger.info(f"Initialized QueryExecutor with {len(self.data.get_all_items())} items")
    
    def execute_structured(self, query: str) -> List[Any]:
        """ביצוע שאילתה מובנית"""
        query_lower = query.lower()
        results = []
        
        # זיהוי סוג הפריט המבוקש
        if any(kw in query_lower for kw in ["decision", "החלטה", "החלטות", "decisions"]):
            results.extend(self.data.decisions)
        
        if any(kw in query_lower for kw in ["rule", "כלל", "כללים", "rules"]):
            results.extend(self.data.rules)
        
        if any(kw in query_lower for kw in ["warning", "אזהרה", "אזהרות", "warnings"]):
            results.extend(self.data.warnings)
        
        if any(kw in query_lower for kw in ["dependency", "תלות", "תלויות", "dependencies"]):
            results.extend(self.data.dependencies)
        
        if any(kw in query_lower for kw in ["change", "שינוי", "שינויים", "changes"]):
            results.extend(self.data.changes)
        
        # אם לא צוין סוג, להחזיר הכל
        if not results:
            results = self.data.get_all_items()
        
        # סינון לפי מילות מפתח נוספות בשאילתה
        filtered_results = []
        for item in results:
            item_text = self._item_to_text(item).lower()
            # בדיקה אם השאילתה מופיעה בפריט
            if any(word in item_text for word in query_lower.split() if len(word) > 3):
                filtered_results.append(item)
        
        # אם הסינון ריק מדי, להחזיר את כל התוצאות
        if not filtered_results and results:
            return results[:10]  # להגביל ל-10 ראשונים
        
        return filtered_results[:10]  # להגביל ל-10 ראשונים
    
    def _item_to_text(self, item: Any) -> str:
        """המרת פריט לטקסט לחיפוש"""
        from schema import Decision, Rule, Warning, Dependency, Change
        
        if isinstance(item, Decision):
            return f"{item.title} {item.summary} {item.rationale or ''}"
        elif isinstance(item, Rule):
            return f"{item.rule} {item.scope} {item.notes or ''}"
        elif isinstance(item, Warning):
            return f"{item.area} {item.message} {item.mitigation or ''}"
        elif isinstance(item, Dependency):
            return f"{item.name} {item.purpose}"
        elif isinstance(item, Change):
            return f"{item.description} {item.category}"
        
        return str(item)
    
    def format_results(self, items: List[Any]) -> str:
        """עיצוב תוצאות למשתמש"""
        if not items:
            return "לא נמצאו תוצאות."
        
        from schema import Decision, Rule, Warning, Dependency, Change
        
        output_lines = [f"נמצאו {len(items)} תוצאות:\n"]
        
        for i, item in enumerate(items, 1):
            if isinstance(item, Decision):
                output_lines.append(f"{i}. 📋 **החלטה**: {item.title}")
                output_lines.append(f"   תיאור: {item.summary}")
                if item.rationale:
                    output_lines.append(f"   נימוק: {item.rationale}")
            
            elif isinstance(item, Rule):
                output_lines.append(f"{i}. 📏 **כלל**: {item.rule}")
                output_lines.append(f"   תחום: {item.scope}")
                if item.notes:
                    output_lines.append(f"   הערות: {item.notes}")
            
            elif isinstance(item, Warning):
                severity_emoji = {
                    "low": "ℹ️",
                    "medium": "⚠️",
                    "high": "🚨",
                    "critical": "🔴"
                }
                emoji = severity_emoji.get(item.severity.value, "⚠️")
                output_lines.append(f"{i}. {emoji} **אזהרה** ({item.severity.value}): {item.message}")
                output_lines.append(f"   תחום: {item.area}")
                if item.mitigation:
                    output_lines.append(f"   פתרון: {item.mitigation}")
            
            elif isinstance(item, Dependency):
                output_lines.append(f"{i}. 📦 **תלות**: {item.name}")
                if item.version:
                    output_lines.append(f"   גרסה: {item.version}")
                output_lines.append(f"   מטרה: {item.purpose}")
            
            elif isinstance(item, Change):
                output_lines.append(f"{i}. 🔄 **שינוי** ({item.category}): {item.description}")
                output_lines.append(f"   השפעה: {item.impact}")
            
            output_lines.append("")  # שורה ריקה בין פריטים
        
        return "\n".join(output_lines)


# ========================================
# Complete Routing System
# ========================================

class SmartQueryRouter:
    """מערכת ניתוב מלאה ששילבה routing + execution"""
    
    def __init__(
        self,
        extracted_data: ExtractedData,
        use_llm_routing: bool = True,
        llm: Optional[Cohere] = None
    ):
        """
        Args:
            extracted_data: המידע המובנה שחולץ
            use_llm_routing: האם להשתמש ב-LLM לניתוב
            llm: אובייקט LLM מותאם אישית
        """
        self.router = HybridRouter(use_llm=use_llm_routing, llm=llm)
        self.executor = QueryExecutor(extracted_data)
        logger.info("Initialized SmartQueryRouter")
    
    def query(self, query: str) -> Tuple[QueryIntent, str]:
        """
        ביצוע שאילתה מלא
        
        Returns:
            (intent, response_text)
        """
        # שלב 1: ניתוב
        route_result = self.router.route(query)
        logger.info(f"Routed to: {route_result.intent.value} (confidence: {route_result.confidence:.2f})")
        logger.info(f"Reasoning: {route_result.reasoning}")
        
        # שלב 2: ביצוע
        if route_result.intent == QueryIntent.STRUCTURED:
            # שליפה מובנית
            items = self.executor.execute_structured(query)
            response = self.executor.format_results(items)
            return QueryIntent.STRUCTURED, response
        
        elif route_result.intent == QueryIntent.HYBRID:
            # שילוב - גם מובנה וגם סמנטי
            items = self.executor.execute_structured(query)
            structured_response = self.executor.format_results(items)
            
            response = f"**תוצאות מובנות:**\n{structured_response}\n\n"
            response += "**הערה:** לחיפוש סמנטי מעמיק יותר, נסה לנסח את השאילתה כשאלה פתוחה."
            
            return QueryIntent.HYBRID, response
        
        else:  # SEMANTIC
            # יש לבצע חיפוש סמנטי ב-vector store
            # זה יטופל על ידי query_engine הרגיל
            return QueryIntent.SEMANTIC, ""
