"""
Carbon Protocol SDK - Skill Database (skill_db)

This module implements the SkillDB class that provides O(L) pattern matching 
for skill retrieval using Trie/Aho-Corasick, replacing traditional vector 
database lookups.

Reference: Carbon Protocol Research Paper
    "Self-Optimizing Skill Registry" - Section III.C

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────┐
    │                      SKILL DATABASE                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
    │  │   Exact     │  │    Aho-     │  │   Regex     │         │
    │  │   Hash      │  │  Corasick   │  │  Patterns   │         │
    │  │   O(1)      │  │   O(L)      │  │   O(L*p)    │         │
    │  └─────────────┘  └─────────────┘  └─────────────┘         │
    │         │                │                │                  │
    │         └────────────────┴────────────────┘                  │
    │                          │                                   │
    │                   ┌──────▼──────┐                           │
    │                   │  SkillMatch │                           │
    │                   │  or MISS    │                           │
    │                   └─────────────┘                           │
    └─────────────────────────────────────────────────────────────┘
    
    MISS → OP_GEN (Generative Path)
    HIT  → OP_MACRO (Deterministic Path)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .pattern_matcher import PatternMatcher, MatchResult, MatcherConfig
from .macro_store import MacroStore, GenerativeTrace, PromotionCriteria
from ..c_isa.instruction import MacroDefinition, CarbonInstruction, InstructionSequence
from ..c_isa.opcodes import CarbonOpCode


@dataclass(frozen=True)
class SkillDBConfig:
    """
    Configuration for SkillDB.
    
    Attributes:
        auto_promote: Whether to automatically promote traces to macros.
        track_misses: Whether to track registry misses for analysis.
        max_skills: Maximum number of skills to store.
        persist_path: Optional path for persistence.
    """
    auto_promote: bool = True
    track_misses: bool = True
    max_skills: int = 10000
    persist_path: Path | None = None


# Alias for backward compatibility
RegistryConfig = SkillDBConfig
DEFAULT_REGISTRY_CONFIG = SkillDBConfig()


@dataclass
class SkillEntry:
    """
    An entry in the skill database.
    
    Represents a single skill that can be matched and executed.
    
    Attributes:
        skill_id: Unique identifier for this skill.
        name: Human-readable name.
        pattern: Trigger pattern (exact, substring, or regex).
        pattern_type: Type of pattern ("exact", "substring", "regex").
        instruction: CarbonInstruction to execute when matched.
        hit_count: Number of times this skill has been invoked.
        last_hit: Timestamp of last invocation.
        metadata: Additional metadata.
    """
    skill_id: str
    name: str
    pattern: str
    pattern_type: str  # "exact", "substring", "regex"
    instruction: CarbonInstruction
    hit_count: int = 0
    last_hit: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def record_hit(self) -> None:
        """Record a skill invocation."""
        self.hit_count += 1
        self.last_hit = time.time()
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "pattern": self.pattern,
            "pattern_type": self.pattern_type,
            "instruction": self.instruction.to_string(),
            "hit_count": self.hit_count,
            "last_hit": self.last_hit,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillEntry:
        """Deserialize from dictionary."""
        return cls(
            skill_id=data["skill_id"],
            name=data["name"],
            pattern=data["pattern"],
            pattern_type=data["pattern_type"],
            instruction=CarbonInstruction.from_string(data["instruction"]),
            hit_count=data.get("hit_count", 0),
            last_hit=data.get("last_hit", 0.0),
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SkillMatch:
    """
    Result of a skill database lookup.
    
    Attributes:
        found: Whether a skill was matched.
        skill: The matched SkillEntry (if found).
        match_type: Type of match ("exact", "substring", "regex").
        score: Match confidence score (0.0-1.0).
        instruction: The instruction to execute.
    """
    found: bool
    skill: SkillEntry | None = None
    match_type: str | None = None
    score: float = 0.0
    instruction: CarbonInstruction | None = None
    
    @property
    def opcode(self) -> CarbonOpCode:
        """Get the opcode for this match result."""
        if self.found and self.instruction:
            return self.instruction.opcode
        return CarbonOpCode.GEN  # Fallback to generative


class SkillDB:
    """
    Skill Database with O(L) Trie/Aho-Corasick lookup.
    
    The SkillDB is the core component of the Carbon Protocol's
    deterministic execution model. It maps input patterns to skills
    (pre-compiled instruction sequences) using hash and trie-based
    matching instead of vector similarity.
    
    Key Features:
        - O(1) exact match via hash table
        - O(L) substring match via Aho-Corasick
        - O(L) regex match for complex patterns
        - Self-optimization: traces → macros promotion
        - Persistence for skill retention
    
    Reference: Carbon Protocol Research Paper
        "Deterministic Skill Lookup" - Section III.C
    
    Example:
        >>> db = SkillDB()
        >>> 
        >>> # Add a skill
        >>> db.add_skill(
        ...     name="python_hello",
        ...     pattern="hello world python",
        ...     instruction=CarbonInstruction(
        ...         opcode=CarbonOpCode.MACRO,
        ...         args={"template": "print('Hello, World!')"}
        ...     )
        ... )
        >>> 
        >>> # Lookup
        >>> match = db.lookup("Write hello world in python")
        >>> if match.found:
        ...     print(match.instruction)  # OP:MACRO --template=...
    """
    
    def __init__(
        self,
        config: SkillDBConfig | None = None,
        matcher_config: MatcherConfig | None = None,
        promotion_criteria: PromotionCriteria | None = None,
    ) -> None:
        """
        Initialize the SkillDB.
        
        Args:
            config: Registry configuration.
            matcher_config: Pattern matcher configuration.
            promotion_criteria: Criteria for trace-to-macro promotion.
        """
        self._config = config or DEFAULT_REGISTRY_CONFIG
        
        # Core components
        self._matcher = PatternMatcher(matcher_config)
        self._macro_store = MacroStore(
            criteria=promotion_criteria,
            persist_path=self._config.persist_path,
        )
        
        # Skill entries by ID
        self._skills: dict[str, SkillEntry] = {}
        
        # Miss tracking for analysis
        self._misses: list[tuple[str, float]] = []  # (input_hash, timestamp)
        
        # Load persisted data if available
        if self._config.persist_path:
            self._load_from_disk()
    
    @property
    def config(self) -> SkillDBConfig:
        """Get database configuration."""
        return self._config
    
    @property
    def skill_count(self) -> int:
        """Number of registered skills."""
        return len(self._skills)
    
    @property
    def macro_count(self) -> int:
        """Number of promoted macros."""
        return self._macro_store.macro_count
    
    @property
    def matcher(self) -> PatternMatcher:
        """Get the pattern matcher."""
        return self._matcher
    
    @property
    def macro_store(self) -> MacroStore:
        """Get the macro store."""
        return self._macro_store
    
    def _generate_skill_id(self, pattern: str) -> str:
        """Generate a unique skill ID from pattern."""
        hash_val = hashlib.sha256(pattern.lower().encode('utf-8')).hexdigest()[:12]
        return f"skill_{hash_val}"
    
    def add_skill(
        self,
        name: str,
        pattern: str,
        instruction: CarbonInstruction,
        pattern_type: str = "substring",
        skill_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SkillEntry:
        """
        Add a skill to the database.
        
        Args:
            name: Human-readable skill name.
            pattern: Trigger pattern.
            instruction: Instruction to execute when matched.
            pattern_type: "exact", "substring", or "regex".
            skill_id: Optional custom ID. Auto-generated if None.
            metadata: Optional metadata.
            
        Returns:
            The created SkillEntry.
            
        Raises:
            ValueError: If pattern_type is invalid or skill limit reached.
        """
        if len(self._skills) >= self._config.max_skills:
            raise ValueError(f"Skill limit reached ({self._config.max_skills})")
        
        if pattern_type not in ("exact", "substring", "regex"):
            raise ValueError(f"Invalid pattern_type: {pattern_type}")
        
        # Generate ID if not provided
        if skill_id is None:
            skill_id = self._generate_skill_id(pattern)
        
        # Create entry
        entry = SkillEntry(
            skill_id=skill_id,
            name=name,
            pattern=pattern,
            pattern_type=pattern_type,
            instruction=instruction,
            metadata=metadata or {},
        )
        
        # Add to matcher
        if pattern_type == "exact":
            self._matcher.add_exact(pattern, skill_id)
        elif pattern_type == "substring":
            self._matcher.add_substring(pattern, skill_id)
        else:  # regex
            self._matcher.add_regex(pattern, skill_id)
        
        # Store entry
        self._skills[skill_id] = entry
        
        # Rebuild matcher
        self._matcher.build()
        
        # Persist if configured
        self._save_to_disk()
        
        return entry
    
    def add_macro_skill(
        self,
        name: str,
        pattern: str,
        template: str,
        pattern_type: str = "substring",
    ) -> SkillEntry:
        """
        Add a macro skill (shorthand for common case).
        
        Args:
            name: Skill name.
            pattern: Trigger pattern.
            template: Output template text.
            pattern_type: Pattern type.
            
        Returns:
            The created SkillEntry.
        """
        instruction = CarbonInstruction(
            opcode=CarbonOpCode.MACRO,
            args={"template": template},
        )
        return self.add_skill(name, pattern, instruction, pattern_type)
    
    def add_scaffold_skill(
        self,
        name: str,
        pattern: str,
        lang: str,
        arch: str = "default",
        db: str | None = None,
    ) -> SkillEntry:
        """
        Add a scaffolding skill.
        
        Args:
            name: Skill name.
            pattern: Trigger pattern.
            lang: Programming language.
            arch: Architecture pattern.
            db: Optional database.
            
        Returns:
            The created SkillEntry.
        """
        args: dict[str, Any] = {"lang": lang, "arch": arch}
        if db:
            args["db"] = db
        
        instruction = CarbonInstruction(
            opcode=CarbonOpCode.SCAFFOLD,
            args=args,
        )
        return self.add_skill(name, pattern, instruction)
    
    def remove_skill(self, skill_id: str) -> bool:
        """
        Remove a skill from the database.
        
        Args:
            skill_id: The skill ID to remove.
            
        Returns:
            True if removed, False if not found.
        """
        if skill_id not in self._skills:
            return False
        
        del self._skills[skill_id]
        self._matcher.remove_pattern(skill_id)
        self._matcher.build()
        self._save_to_disk()
        
        return True
    
    def get_skill(self, skill_id: str) -> SkillEntry | None:
        """Get a skill by ID."""
        return self._skills.get(skill_id)
    
    def lookup(self, text: str) -> SkillMatch:
        """
        Look up a skill matching the input text.
        
        This is the main entry point for skill retrieval. It searches
        the pattern matcher in order of efficiency:
        1. Exact hash match (O(1))
        2. Aho-Corasick substring (O(L))
        3. Regex patterns (O(L) per pattern)
        
        Args:
            text: Input text to match.
            
        Returns:
            SkillMatch indicating whether a skill was found.
        """
        # First check macro store (promoted traces)
        input_hash = hashlib.sha256(text.lower().strip().encode('utf-8')).hexdigest()[:16]
        macro = self._macro_store.get_macro(input_hash)
        
        if macro is not None:
            macro.increment_hit()
            return SkillMatch(
                found=True,
                skill=None,  # Macro, not skill
                match_type="macro",
                score=1.0,
                instruction=macro.to_instruction(),
            )
        
        # Then check pattern matcher
        result = self._matcher.match(text)
        
        if result.matched and result.pattern_id:
            skill = self._skills.get(result.pattern_id)
            if skill:
                skill.record_hit()
                return SkillMatch(
                    found=True,
                    skill=skill,
                    match_type=result.match_type,
                    score=result.score,
                    instruction=skill.instruction,
                )
        
        # No match - track miss
        if self._config.track_misses:
            self._misses.append((input_hash, time.time()))
            # Keep only recent misses
            if len(self._misses) > 1000:
                self._misses = self._misses[-500:]
        
        return SkillMatch(found=False)
    
    def record_generative_trace(
        self,
        input_text: str,
        output_text: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> GenerativeTrace:
        """
        Record a generative (LLM) interaction for potential promotion.
        
        Args:
            input_text: Original user input.
            output_text: LLM response.
            tokens_in: Input tokens.
            tokens_out: Output tokens.
            latency_ms: Response latency.
            success: Whether successful.
            
        Returns:
            The recorded GenerativeTrace.
        """
        trace = GenerativeTrace.create(
            input_text=input_text,
            output_text=output_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            success=success,
        )
        
        self._macro_store.record_trace(trace)
        
        # Auto-promote if configured
        if self._config.auto_promote:
            self._check_promotions()
        
        return trace
    
    def _check_promotions(self) -> list[MacroDefinition]:
        """Check for and execute promotions."""
        candidates = self._macro_store.get_promotion_candidates()
        promoted = []
        
        for input_hash, traces in candidates:
            macro = self._macro_store.promote_to_macro(traces)
            promoted.append(macro)
        
        return promoted
    
    def promote_to_macro(self, generative_trace: GenerativeTrace) -> MacroDefinition:
        """
        Manually promote a single trace to a macro.
        
        This bypasses the normal promotion criteria.
        
        Args:
            generative_trace: The trace to promote.
            
        Returns:
            The created MacroDefinition.
        """
        return self._macro_store.promote_to_macro([generative_trace])
    
    def list_skills(self) -> list[SkillEntry]:
        """Get all registered skills."""
        return list(self._skills.values())
    
    def list_hot_skills(self, min_hits: int = 10) -> list[SkillEntry]:
        """Get frequently-used skills."""
        return [s for s in self._skills.values() if s.hit_count >= min_hits]
    
    def get_miss_patterns(self, min_count: int = 3) -> list[tuple[str, int]]:
        """
        Get frequently-missed patterns for analysis.
        
        Args:
            min_count: Minimum miss count to include.
            
        Returns:
            List of (input_hash, count) tuples.
        """
        from collections import Counter
        miss_counts = Counter(h for h, _ in self._misses)
        return [(h, c) for h, c in miss_counts.most_common() if c >= min_count]
    
    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        total_hits = sum(s.hit_count for s in self._skills.values())
        
        return {
            "skill_count": self.skill_count,
            "macro_count": self.macro_count,
            "total_hits": total_hits,
            "miss_count": len(self._misses),
            "matcher": self._matcher.get_stats(),
            "macro_store": self._macro_store.get_stats(),
        }
    
    def _save_to_disk(self) -> None:
        """Save state to disk."""
        if self._config.persist_path is None:
            return
        
        data = {
            "skills": {
                skill_id: entry.to_dict()
                for skill_id, entry in self._skills.items()
            },
        }
        
        skills_path = self._config.persist_path.parent / "skills.json"
        skills_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(skills_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _load_from_disk(self) -> None:
        """Load state from disk."""
        if self._config.persist_path is None:
            return
        
        skills_path = self._config.persist_path.parent / "skills.json"
        if not skills_path.exists():
            return
        
        with open(skills_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for skill_id, skill_data in data.get("skills", {}).items():
            entry = SkillEntry.from_dict(skill_data)
            self._skills[skill_id] = entry
            
            # Re-add to matcher
            if entry.pattern_type == "exact":
                self._matcher.add_exact(entry.pattern, skill_id)
            elif entry.pattern_type == "substring":
                self._matcher.add_substring(entry.pattern, skill_id)
            else:
                self._matcher.add_regex(entry.pattern, skill_id)
        
        self._matcher.build()
    
    def __repr__(self) -> str:
        return f"SkillDB(skills={self.skill_count}, macros={self.macro_count})"


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# SkillRegistry is an alias for SkillDB for backward compatibility
SkillRegistry = SkillDB
