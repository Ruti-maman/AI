"""
שלב ג' - Data Extraction: Schema Definition
הגדרת Schema למידע מובנה שנחלץ מקבצי ה-MD

סכמה זו מגדירה את סוגי הפריטים שנחלץ:
- decisions (החלטות)
- rules (כללים/הנחיות)
- warnings (אזהרות/רגישויות)
- dependencies (תלויות)
- changes (שינויים חשובים)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ========================================
# Enums - סוגי פריטים
# ========================================

class ItemType(Enum):
    """סוגי פריטים שניתן לחלץ"""
    DECISION = "decision"
    RULE = "rule"
    WARNING = "warning"
    DEPENDENCY = "dependency"
    CHANGE = "change"


class Severity(Enum):
    """רמת חומרה"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ========================================
# Data Classes - מבני נתונים
# ========================================

@dataclass
class SourceInfo:
    """מידע על מקור הנתון"""
    tool: str  # cursor / claude_code / kiro
    file: str  # נתיב קובץ
    anchor: Optional[str] = None  # סימון בקובץ (כותרת, line range)
    line_range: Optional[List[int]] = None  # [start, end]
    observed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "file": self.file,
            "anchor": self.anchor,
            "line_range": self.line_range,
            "observed_at": self.observed_at.isoformat()
        }


@dataclass
class Decision:
    """החלטה טכנית"""
    id: str
    title: str
    summary: str
    tags: List[str] = field(default_factory=list)
    source: Optional[SourceInfo] = None
    rationale: Optional[str] = None  # הסבר למה ההחלטה התקבלה
    alternatives: Optional[str] = None  # אלטרנטיבות שנשקלו
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": ItemType.DECISION.value,
            "title": self.title,
            "summary": self.summary,
            "tags": self.tags,
            "source": self.source.to_dict() if self.source else None,
            "rationale": self.rationale,
            "alternatives": self.alternatives
        }


@dataclass
class Rule:
    """כלל/הנחיה"""
    id: str
    rule: str
    scope: str  # ui / backend / db / api / all
    notes: Optional[str] = None
    source: Optional[SourceInfo] = None
    exceptions: Optional[str] = None  # חריגים לכלל
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": ItemType.RULE.value,
            "rule": self.rule,
            "scope": self.scope,
            "notes": self.notes,
            "source": self.source.to_dict() if self.source else None,
            "exceptions": self.exceptions
        }


@dataclass
class Warning:
    """אזהרה/רגישות"""
    id: str
    area: str  # auth / db / api / ui
    message: str
    severity: Severity
    source: Optional[SourceInfo] = None
    mitigation: Optional[str] = None  # איך להימנע מהבעיה
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": ItemType.WARNING.value,
            "area": self.area,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source.to_dict() if self.source else None,
            "mitigation": self.mitigation
        }


@dataclass
class Dependency:
    """תלות טכנית"""
    id: str
    name: str
    version: Optional[str] = None
    purpose: str = ""  # למה צריך את התלות הזו
    source: Optional[SourceInfo] = None
    required: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": ItemType.DEPENDENCY.value,
            "name": self.name,
            "version": self.version,
            "purpose": self.purpose,
            "source": self.source.to_dict() if self.source else None,
            "required": self.required
        }


@dataclass
class Change:
    """שינוי חשוב"""
    id: str
    description: str
    category: str  # feature / bugfix / refactor / breaking
    impact: str  # high / medium / low
    source: Optional[SourceInfo] = None
    migration_notes: Optional[str] = None  # הערות מעבר
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": ItemType.CHANGE.value,
            "description": self.description,
            "category": self.category,
            "impact": self.impact,
            "source": self.source.to_dict() if self.source else None,
            "migration_notes": self.migration_notes
        }


# ========================================
# Extracted Data Container
# ========================================

@dataclass
class ExtractedData:
    """מכיל את כל המידע המחולץ"""
    schema_version: str = "1.0"
    generated_at: datetime = field(default_factory=datetime.now)
    
    decisions: List[Decision] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    warnings: List[Warning] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    changes: List[Change] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """המרה ל-JSON"""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at.isoformat(),
            "items": {
                "decisions": [d.to_dict() for d in self.decisions],
                "rules": [r.to_dict() for r in self.rules],
                "warnings": [w.to_dict() for w in self.warnings],
                "dependencies": [d.to_dict() for d in self.dependencies],
                "changes": [c.to_dict() for c in self.changes]
            },
            "statistics": {
                "total_items": len(self.decisions) + len(self.rules) + len(self.warnings) + len(self.dependencies) + len(self.changes),
                "decisions": len(self.decisions),
                "rules": len(self.rules),
                "warnings": len(self.warnings),
                "dependencies": len(self.dependencies),
                "changes": len(self.changes)
            }
        }
    
    def add_decision(self, decision: Decision):
        """הוספת החלטה"""
        self.decisions.append(decision)
    
    def add_rule(self, rule: Rule):
        """הוספת כלל"""
        self.rules.append(rule)
    
    def add_warning(self, warning: Warning):
        """הוספת אזהרה"""
        self.warnings.append(warning)
    
    def add_dependency(self, dependency: Dependency):
        """הוספת תלות"""
        self.dependencies.append(dependency)
    
    def add_change(self, change: Change):
        """הוספת שינוי"""
        self.changes.append(change)
    
    def get_all_items(self) -> List[Any]:
        """החזרת כל הפריטים"""
        return (
            self.decisions + 
            self.rules + 
            self.warnings + 
            self.dependencies + 
            self.changes
        )
    
    def search(self, query: str) -> List[Any]:
        """חיפוש בכל הפריטים"""
        query_lower = query.lower()
        results = []
        
        # חיפוש בהחלטות
        for decision in self.decisions:
            if query_lower in decision.title.lower() or query_lower in decision.summary.lower():
                results.append(decision)
        
        # חיפוש בכללים
        for rule in self.rules:
            if query_lower in rule.rule.lower() or query_lower in rule.scope.lower():
                results.append(rule)
        
        # חיפוש באזהרות
        for warning in self.warnings:
            if query_lower in warning.area.lower() or query_lower in warning.message.lower():
                results.append(warning)
        
        # חיפוש בתלויות
        for dependency in self.dependencies:
            if query_lower in dependency.name.lower():
                results.append(dependency)
        
        # חיפוש בשינויים
        for change in self.changes:
            if query_lower in change.description.lower():
                results.append(change)
        
        return results


# ========================================
# Query Types - סוגי שאילתות
# ========================================

class QueryIntent(Enum):
    """כוונת השאילתה"""
    SEMANTIC = "semantic"  # חיפוש סמנטי במסמכים
    STRUCTURED = "structured"  # שליפה מובנית מה-schema
    HYBRID = "hybrid"  # שילוב של שניהם
