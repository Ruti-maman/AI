"""
שלב ג' - Data Extraction: חילוץ מידע מובנה
מחלץ החלטות, כללים, אזהרות, תלויות ושינויים מקבצי MD

שיטות חילוץ:
1. Pattern Matching - חיפוש תבניות (מהיר, דטרמיניסטי)
2. LLM Extraction - שימוש ב-LLM (איכותי יותר, רב-לשוני)
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

from llama_index.core import Document
from llama_index.llms.cohere import Cohere

from schema import (
    Decision, Rule, Warning, Dependency, Change,
    ExtractedData, SourceInfo, Severity, ItemType
)

# טעינת משתני סביבה
load_dotenv()

# הגדרת Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================
# Pattern-Based Extractor (דטרמיניסטי)
# ========================================

class PatternExtractor:
    """מחלץ מידע על בסיס תבניות רגקס"""
    
    # תבניות חיפוש
    DECISION_PATTERNS = [
        r"(?:decision|החלטה|החלטנו)[\s:]+(.+?)(?:\n|$)",
        r"(?:decided to|החלטנו ש)[\s:]+(.+?)(?:\n|$)",
        r"(?:we chose|בחרנו ב)[\s:]+(.+?)(?:\n|$)"
    ]
    
    RULE_PATTERNS = [
        r"(?:rule|כלל|הנחיה)[\s:]+(.+?)(?:\n|$)",
        r"(?:must|חייב|צריך)[\s:]+(.+?)(?:\n|$)",
        r"(?:never|לעולם לא|אסור)[\s:]+(.+?)(?:\n|$)"
    ]
    
    WARNING_PATTERNS = [
        r"(?:warning|אזהרה|שים לב)[\s:]+(.+?)(?:\n|$)",
        r"(?:important|חשוב|קריטי)[\s:]+(.+?)(?:\n|$)",
        r"(?:careful|זהירות|תזהר)[\s:]+(.+?)(?:\n|$)"
    ]
    
    DEPENDENCY_PATTERNS = [
        r"(?:dependency|תלות|צריך את)[\s:]+([a-zA-Z0-9_-]+)",
        r"(?:requires|דורש|תלוי ב)[\s:]+([a-zA-Z0-9_-]+)",
        r"(?:install|התקן)[\s:]+([a-zA-Z0-9_-]+)"
    ]
    
    def __init__(self):
        self.counter = {"decision": 0, "rule": 0, "warning": 0, "dependency": 0, "change": 0}
    
    def extract_from_text(self, text: str, source_file: str) -> ExtractedData:
        """חילוץ מידע מטקסט"""
        extracted = ExtractedData()
        
        # חילוץ החלטות
        for pattern in self.DECISION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                self.counter["decision"] += 1
                decision = Decision(
                    id=f"dec_{self.counter['decision']}",
                    title=match.group(1).strip()[:100],
                    summary=match.group(1).strip(),
                    source=SourceInfo(tool="pattern_match", file=source_file)
                )
                extracted.add_decision(decision)
        
        # חילוץ כללים
        for pattern in self.RULE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                self.counter["rule"] += 1
                rule = Rule(
                    id=f"rule_{self.counter['rule']}",
                    rule=match.group(1).strip(),
                    scope="all",
                    source=SourceInfo(tool="pattern_match", file=source_file)
                )
                extracted.add_rule(rule)
        
        # חילוץ אזהרות
        for pattern in self.WARNING_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                self.counter["warning"] += 1
                warning = Warning(
                    id=f"warn_{self.counter['warning']}",
                    area="general",
                    message=match.group(1).strip(),
                    severity=Severity.MEDIUM,
                    source=SourceInfo(tool="pattern_match", file=source_file)
                )
                extracted.add_warning(warning)
        
        # חילוץ תלויות
        for pattern in self.DEPENDENCY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                self.counter["dependency"] += 1
                dependency = Dependency(
                    id=f"dep_{self.counter['dependency']}",
                    name=match.group(1).strip(),
                    purpose="Extracted from documentation",
                    source=SourceInfo(tool="pattern_match", file=source_file)
                )
                extracted.add_dependency(dependency)
        
        return extracted


# ========================================
# LLM-Based Extractor (רב-ערוצי)
# ========================================

class LLMExtractor:
    """מחלץ מידע באמצעות LLM"""
    
    EXTRACTION_PROMPT_TEMPLATE = """
You are an expert at extracting structured information from technical documentation.

Given the following text, extract:
1. Decisions - important technical decisions made
2. Rules - guidelines and rules to follow
3. Warnings - important warnings and sensitivities
4. Dependencies - technical dependencies and requirements
5. Changes - significant changes made

Text:
{text}

Return the information in the following JSON format:
{{
    "decisions": [
        {{"title": "...", "summary": "...", "rationale": "..."}}
    ],
    "rules": [
        {{"rule": "...", "scope": "...", "notes": "..."}}
    ],
    "warnings": [
        {{"area": "...", "message": "...", "severity": "low|medium|high|critical"}}
    ],
    "dependencies": [
        {{"name": "...", "version": "...", "purpose": "..."}}
    ],
    "changes": [
        {{"description": "...", "category": "feature|bugfix|refactor|breaking", "impact": "high|medium|low"}}
    ]
}}

Return ONLY the JSON, no additional text.
"""
    
    def __init__(self, llm: Optional[Cohere] = None):
        """
        Args:
            llm: אובייקט LLM. אם None, ייווצר אוטומטית
        """
        self.llm = llm or Cohere(
            api_key=os.getenv("COHERE_API_KEY"),
            model="command-r-plus-08-2024",
            temperature=0.1  # נמוך לדטרמיניזם
        )
        self.counter = {"decision": 0, "rule": 0, "warning": 0, "dependency": 0, "change": 0}
        logger.info(f"Initialized LLMExtractor with model: command-r-plus-08-2024 (Cohere)")
    
    def extract_from_text(self, text: str, source_file: str) -> ExtractedData:
        """חילוץ מידע מטקסט באמצעות LLM"""
        extracted = ExtractedData()
        
        # חלק טקסט ארוך לחלקים קטנים יותר
        chunks = self._chunk_text(text, max_chars=3000)
        logger.info(f"Splitting text into {len(chunks)} chunks for extraction")
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                chunk_data = self._extract_chunk(chunk, source_file)
                
                # מיזוג תוצאות
                for decision in chunk_data.decisions:
                    extracted.add_decision(decision)
                for rule in chunk_data.rules:
                    extracted.add_rule(rule)
                for warning in chunk_data.warnings:
                    extracted.add_warning(warning)
                for dependency in chunk_data.dependencies:
                    extracted.add_dependency(dependency)
                for change in chunk_data.changes:
                    extracted.add_change(change)
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                continue
        
        logger.info(f"Extraction complete: {len(extracted.get_all_items())} items")
        return extracted
    
    def _chunk_text(self, text: str, max_chars: int = 3000) -> List[str]:
        """חלוקת טקסט לחלקים קטנים"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            if current_length + line_length > max_chars and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _extract_chunk(self, chunk: str, source_file: str) -> ExtractedData:
        """חילוץ מידע מחלק בודד"""
        extracted = ExtractedData()
        
        # יצירת prompt
        prompt = self.EXTRACTION_PROMPT_TEMPLATE.format(text=chunk)
        
        # שליחה ל-LLM
        response = self.llm.complete(prompt)
        response_text = response.text.strip()
        
        # ניסיון לפרסר JSON
        try:
            # ניקוי של markdown code blocks אם קיימים
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response_text)
            
            # המרה ל-dataclasses
            for dec_data in data.get("decisions", []):
                self.counter["decision"] += 1
                decision = Decision(
                    id=f"dec_{self.counter['decision']}",
                    title=dec_data.get("title", "")[:100],
                    summary=dec_data.get("summary", ""),
                    rationale=dec_data.get("rationale"),
                    source=SourceInfo(tool="llm_extraction", file=source_file)
                )
                extracted.add_decision(decision)
            
            for rule_data in data.get("rules", []):
                self.counter["rule"] += 1
                rule = Rule(
                    id=f"rule_{self.counter['rule']}",
                    rule=rule_data.get("rule", ""),
                    scope=rule_data.get("scope", "all"),
                    notes=rule_data.get("notes"),
                    source=SourceInfo(tool="llm_extraction", file=source_file)
                )
                extracted.add_rule(rule)
            
            for warn_data in data.get("warnings", []):
                self.counter["warning"] += 1
                severity_str = warn_data.get("severity", "medium")
                severity = Severity.MEDIUM
                if severity_str == "low":
                    severity = Severity.LOW
                elif severity_str == "high":
                    severity = Severity.HIGH
                elif severity_str == "critical":
                    severity = Severity.CRITICAL
                
                warning = Warning(
                    id=f"warn_{self.counter['warning']}",
                    area=warn_data.get("area", "general"),
                    message=warn_data.get("message", ""),
                    severity=severity,
                    source=SourceInfo(tool="llm_extraction", file=source_file)
                )
                extracted.add_warning(warning)
            
            for dep_data in data.get("dependencies", []):
                self.counter["dependency"] += 1
                dependency = Dependency(
                    id=f"dep_{self.counter['dependency']}",
                    name=dep_data.get("name", ""),
                    version=dep_data.get("version"),
                    purpose=dep_data.get("purpose", ""),
                    source=SourceInfo(tool="llm_extraction", file=source_file)
                )
                extracted.add_dependency(dependency)
            
            for change_data in data.get("changes", []):
                self.counter["change"] += 1
                change = Change(
                    id=f"change_{self.counter['change']}",
                    description=change_data.get("description", ""),
                    category=change_data.get("category", "feature"),
                    impact=change_data.get("impact", "medium"),
                    source=SourceInfo(tool="llm_extraction", file=source_file)
                )
                extracted.add_change(change)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response was: {response_text[:500]}")
        except Exception as e:
            logger.error(f"Error in _extract_chunk: {e}")
        
        return extracted


# ========================================
# Hybrid Extractor (משלב שתי השיטות)
# ========================================

class HybridExtractor:
    """משלב Pattern Extraction + LLM Extraction"""
    
    def __init__(self, use_llm: bool = True, llm: Optional[Cohere] = None):
        """
        Args:
            use_llm: האם להשתמש ב-LLM extraction (איטי יותר אבל מדויק יותר)
            llm: אובייקט LLM מותאם אישית
        """
        self.pattern_extractor = PatternExtractor()
        self.llm_extractor = LLMExtractor(llm) if use_llm else None
        self.use_llm = use_llm
        logger.info(f"Initialized HybridExtractor (LLM: {use_llm})")
    
    def extract_from_file(self, file_path: Path) -> ExtractedData:
        """חילוץ מידע מקובץ"""
        logger.info(f"Extracting from file: {file_path}")
        
        # קריאת קובץ
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.extract_from_text(text, str(file_path))
    
    def extract_from_text(self, text: str, source_file: str) -> ExtractedData:
        """חילוץ מידע מטקסט"""
        # שלב 1: Pattern extraction (מהיר)
        pattern_data = self.pattern_extractor.extract_from_text(text, source_file)
        logger.info(f"Pattern extraction: {len(pattern_data.get_all_items())} items")
        
        # שלב 2: LLM extraction (איכותי)
        if self.use_llm and self.llm_extractor:
            llm_data = self.llm_extractor.extract_from_text(text, source_file)
            logger.info(f"LLM extraction: {len(llm_data.get_all_items())} items")
            
            # מיזוג תוצאות (LLM לוקח עדיפות)
            combined = llm_data
            
            # הוספת פריטים ממיצוי תבניות שלא נמצאו ב-LLM
            for item in pattern_data.get_all_items():
                combined_items = combined.get_all_items()
                # בדיקה פשוטה לדופליקטים
                if not any(self._is_similar(item, existing) for existing in combined_items):
                    if isinstance(item, Decision):
                        combined.add_decision(item)
                    elif isinstance(item, Rule):
                        combined.add_rule(item)
                    elif isinstance(item, Warning):
                        combined.add_warning(item)
                    elif isinstance(item, Dependency):
                        combined.add_dependency(item)
                    elif isinstance(item, Change):
                        combined.add_change(item)
            
            return combined
        else:
            return pattern_data
    
    def _is_similar(self, item1: Any, item2: Any) -> bool:
        """בדיקה אם שני פריטים דומים (למניעת כפילויות)"""
        if type(item1) != type(item2):
            return False
        
        if isinstance(item1, Decision):
            return item1.title.lower() == item2.title.lower()
        elif isinstance(item1, Rule):
            return item1.rule.lower() == item2.rule.lower()
        elif isinstance(item1, Warning):
            return item1.message.lower() == item2.message.lower()
        elif isinstance(item1, Dependency):
            return item1.name.lower() == item2.name.lower()
        elif isinstance(item1, Change):
            return item1.description.lower() == item2.description.lower()
        
        return False
    
    def extract_from_directory(self, directory: Path, pattern: str = "*.md") -> ExtractedData:
        """חילוץ מידע מכל הקבצים בתיקייה"""
        logger.info(f"Extracting from directory: {directory}")
        
        combined = ExtractedData()
        files = list(directory.glob(pattern))
        logger.info(f"Found {len(files)} files matching {pattern}")
        
        for file_path in files:
            try:
                file_data = self.extract_from_file(file_path)
                
                # מיזוג
                for item in file_data.get_all_items():
                    if isinstance(item, Decision):
                        combined.add_decision(item)
                    elif isinstance(item, Rule):
                        combined.add_rule(item)
                    elif isinstance(item, Warning):
                        combined.add_warning(item)
                    elif isinstance(item, Dependency):
                        combined.add_dependency(item)
                    elif isinstance(item, Change):
                        combined.add_change(item)
                        
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Total extracted: {len(combined.get_all_items())} items")
        return combined
