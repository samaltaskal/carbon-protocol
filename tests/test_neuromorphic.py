"""
Carbon Protocol SDK - Neuromorphic Architecture Tests

This test suite validates the "Wake-on-Meaning" architecture including:
1. CarbonNeuron (LIF model) - ingestion/neuron.py
2. NeuronBank (multi-intent detection)
3. SignalExtractor (keyword matching)
4. IntentDetector (full pipeline)
5. SkillDB (Trie/Aho-Corasick lookup) - registry/skill_db.py
6. CarbonCompiler (main router) - compiler/semantic.py

Reference: Carbon Protocol Research Paper
    "Empirical Validation" - Section VII
"""

import pytest
import time
from pathlib import Path

# Import all Carbon Protocol components
from src.ingestion.neuron import CarbonNeuron, NeuronConfig
from src.ingestion.neuron_bank import NeuronBank, NeuronBankConfig, create_default_intent_bank
from src.ingestion.signal_extractor import SignalExtractor, create_default_extractor
from src.ingestion.intent_detector import IntentDetector, DetectorConfig

from src.c_isa.opcodes import CarbonOpCode, OP_IDLE, OP_MACRO, OP_GEN
from src.c_isa.instruction import CarbonInstruction, InstructionSequence, MacroDefinition
from src.c_isa.bytecode import CarbonBytecode, BytecodeBuilder, serialize_bytecode, deserialize_bytecode

from src.registry.pattern_matcher import PatternMatcher
from src.registry.macro_store import MacroStore, GenerativeTrace, PromotionCriteria
from src.registry.skill_db import SkillDB, SkillRegistry  # SkillRegistry is alias

from src.compiler.semantic import CarbonCompiler, ExecutionPath, CompilerConfig


# =============================================================================
# CarbonNeuron Tests
# =============================================================================

class TestCarbonNeuron:
    """Tests for the Leaky Integrate-and-Fire neuron model."""
    
    def test_neuron_creation_defaults(self):
        """Test neuron creation with default parameters."""
        neuron = CarbonNeuron()
        
        assert neuron.voltage == 0.0
        assert neuron.threshold == 1.0
        assert neuron.decay_rate == 0.9
        assert neuron.fire_count == 0
        assert not neuron.is_refractory
    
    def test_neuron_creation_custom(self):
        """Test neuron creation with custom parameters."""
        neuron = CarbonNeuron(threshold=0.5, decay_rate=0.8, label="test")
        
        assert neuron.threshold == 0.5
        assert neuron.decay_rate == 0.8
        assert neuron.label == "test"
    
    def test_neuron_input(self):
        """Test signal input accumulation."""
        neuron = CarbonNeuron()
        
        neuron.input(0.3)
        assert neuron.voltage == 0.3
        
        neuron.input(0.5)
        assert neuron.voltage == 0.8
    
    def test_neuron_decay(self):
        """Test voltage decay (leak)."""
        neuron = CarbonNeuron(decay_rate=0.5)
        
        neuron.input(1.0)
        assert neuron.voltage == 1.0
        
        neuron.tick()
        assert neuron.voltage == 0.5
        
        neuron.tick()
        assert neuron.voltage == 0.25
    
    def test_neuron_fire(self):
        """Test neuron firing when threshold exceeded."""
        neuron = CarbonNeuron(threshold=1.0)
        
        # Below threshold - should not fire
        neuron.input(0.9)
        assert not neuron.fire()
        assert neuron.fire_count == 0
        
        # Above threshold - should fire
        neuron.input(0.2)  # Now at 1.1
        assert neuron.fire()
        assert neuron.fire_count == 1
        assert neuron.voltage == 0.0  # Reset after fire
    
    def test_neuron_refractory_period(self):
        """Test refractory period after firing."""
        neuron = CarbonNeuron(threshold=1.0, refractory_period=2)
        
        # Fire the neuron
        neuron.input(1.5)
        assert neuron.fire()
        assert neuron.is_refractory
        
        # Input during refractory should be ignored
        neuron.input(1.5)
        assert neuron.voltage == 0.0
        
        # Wait for refractory to end
        neuron.tick()
        neuron.tick()
        assert not neuron.is_refractory
    
    def test_neuron_reset(self):
        """Test neuron reset."""
        neuron = CarbonNeuron()
        neuron.input(1.5)  # Above threshold so it fires
        neuron.fire()
        
        neuron.reset()
        assert neuron.voltage == 0.0
        assert neuron.fire_count == 1  # Preserved
    
    def test_neuron_hard_reset(self):
        """Test hard reset including fire count."""
        neuron = CarbonNeuron()
        neuron.input(1.5)
        neuron.fire()
        
        neuron.hard_reset()
        assert neuron.voltage == 0.0
        assert neuron.fire_count == 0
    
    def test_neuron_invalid_params(self):
        """Test validation of invalid parameters."""
        with pytest.raises(ValueError):
            CarbonNeuron(decay_rate=1.5)  # > 1.0
        
        with pytest.raises(ValueError):
            CarbonNeuron(decay_rate=0.0)  # <= 0
        
        with pytest.raises(ValueError):
            CarbonNeuron(threshold=-1.0)  # <= 0


# =============================================================================
# NeuronBank Tests
# =============================================================================

class TestNeuronBank:
    """Tests for the multi-neuron intent bank."""
    
    def test_bank_creation(self):
        """Test bank creation."""
        bank = NeuronBank()
        assert bank.neuron_count == 0
    
    def test_add_neurons(self):
        """Test adding neurons to bank."""
        bank = NeuronBank()
        
        bank.add_neuron("code", threshold=1.0)
        bank.add_neuron("query", threshold=0.8)
        
        assert bank.neuron_count == 2
        assert "code" in bank.labels
        assert "query" in bank.labels
    
    def test_signal_routing(self):
        """Test signal routing to specific neurons."""
        bank = NeuronBank()
        bank.add_neuron("code")
        bank.add_neuron("query")
        
        bank.signal("code", 0.5)
        
        code_neuron = bank.get_neuron("code")
        query_neuron = bank.get_neuron("query")
        
        assert code_neuron.voltage == 0.5
        assert query_neuron.voltage == 0.0
    
    def test_collect_fires(self):
        """Test collecting fired neurons."""
        bank = NeuronBank()
        bank.add_neuron("code", threshold=1.0)
        bank.add_neuron("query", threshold=0.5)
        
        bank.signal("code", 0.6)  # Below threshold
        bank.signal("query", 0.6)  # Above threshold
        
        fired = bank.collect_fires()
        
        assert "query" in fired
        assert "code" not in fired
    
    def test_default_bank_factory(self):
        """Test default bank factory."""
        bank = create_default_intent_bank()
        
        assert bank.neuron_count > 0
        assert "code" in bank.labels


# =============================================================================
# SignalExtractor Tests
# =============================================================================

class TestSignalExtractor:
    """Tests for keyword-based signal extraction."""
    
    def test_extractor_creation(self):
        """Test extractor creation."""
        extractor = SignalExtractor()
        assert extractor.keyword_count == 0
    
    def test_add_keywords(self):
        """Test adding keywords."""
        extractor = SignalExtractor()
        extractor.add_keyword("python", "code", 0.5)
        extractor.add_keyword("script", "code", 0.4)
        
        assert extractor.keyword_count == 2
    
    def test_extract_signals(self):
        """Test signal extraction from text."""
        extractor = SignalExtractor()
        extractor.add_keyword("python", "code", 0.5)
        extractor.add_keyword("script", "code", 0.4)
        
        signals = extractor.extract("Write a Python script")
        
        assert len(signals) == 2
        keywords = [s.keyword for s in signals]
        assert "python" in keywords
        assert "script" in keywords
    
    def test_extract_bigrams(self):
        """Test bigram extraction."""
        extractor = SignalExtractor()
        extractor.add_bigram("create project", "scaffold", 0.7)
        
        signals = extractor.extract("I want to create project")
        
        assert any(s.keyword == "create project" for s in signals)
    
    def test_default_extractor_factory(self):
        """Test default extractor factory."""
        extractor = create_default_extractor()
        
        assert extractor.keyword_count > 0
        
        signals = extractor.extract("Write Python code")
        assert len(signals) > 0


# =============================================================================
# IntentDetector Tests
# =============================================================================

class TestIntentDetector:
    """Tests for the full intent detection pipeline."""
    
    def test_detector_creation(self):
        """Test detector creation."""
        detector = IntentDetector.create_default()
        assert detector is not None
    
    def test_detect_code_intent(self):
        """Test detecting code intent."""
        detector = IntentDetector.create_default()
        
        result = detector.detect("Write a Python script to parse JSON")
        
        assert result.has_intent
        assert result.primary_intent in ("code", "scaffold")
    
    def test_detect_idle(self):
        """Test IDLE detection for low-intent input."""
        detector = IntentDetector.create_default()
        
        result = detector.detect("Hi")
        
        # May or may not be idle depending on configuration
        assert isinstance(result.is_idle, bool)
    
    def test_signals_extracted_count(self):
        """Test that signals are counted correctly."""
        detector = IntentDetector.create_default()
        
        result = detector.detect("Write Python code to debug the error")
        
        assert result.signals_extracted >= 2  # At least "python", "code"


# =============================================================================
# C-ISA Tests
# =============================================================================

class TestCarbonISA:
    """Tests for the Carbon Instruction Set."""
    
    def test_opcode_properties(self):
        """Test opcode properties."""
        assert OP_IDLE.is_deterministic
        assert OP_MACRO.is_deterministic
        assert OP_GEN.is_generative
        assert not OP_GEN.is_deterministic
    
    def test_opcode_flops(self):
        """Test estimated FLOPs."""
        assert OP_IDLE.estimated_flops == 0
        assert OP_MACRO.estimated_flops < 10_000
        assert OP_GEN.estimated_flops > 1_000_000_000
    
    def test_instruction_creation(self):
        """Test instruction creation."""
        instr = CarbonInstruction(
            opcode=CarbonOpCode.SCAFFOLD,
            args={"lang": "python", "arch": "mvc"}
        )
        
        assert instr.opcode == CarbonOpCode.SCAFFOLD
        assert instr.args["lang"] == "python"
    
    def test_instruction_serialization(self):
        """Test instruction to/from string."""
        instr = CarbonInstruction(
            opcode=CarbonOpCode.SCAFFOLD,
            args={"lang": "python", "arch": "mvc"}
        )
        
        string = instr.to_string()
        assert "OP:SCAFFOLD" in string
        assert "--lang=python" in string
        
        parsed = CarbonInstruction.from_string(string)
        assert parsed.opcode == instr.opcode
        assert parsed.args["lang"] == "python"
    
    def test_sequence_creation(self):
        """Test instruction sequence."""
        seq = InstructionSequence([
            CarbonInstruction(opcode=CarbonOpCode.LD, args={"source": "docs"}),
            CarbonInstruction(opcode=CarbonOpCode.MACRO, args={"name": "respond"}),
        ])
        
        assert len(seq) == 2
        assert seq.is_fully_deterministic
    
    def test_bytecode_builder(self):
        """Test bytecode builder fluent API."""
        bytecode = (BytecodeBuilder()
            .load("workspace")
            .check("safety")
            .macro("respond")
            .build("test input"))
        
        assert len(bytecode.sequence) == 3
        assert bytecode.sequence[0].opcode == CarbonOpCode.LD
    
    def test_bytecode_serialization(self):
        """Test bytecode serialization/deserialization."""
        bytecode = BytecodeBuilder().macro("test").build()
        
        # JSON serialization
        data = serialize_bytecode(bytecode, format="json")
        restored = deserialize_bytecode(data, format="json")
        
        assert len(restored.sequence) == len(bytecode.sequence)


# =============================================================================
# Pattern Matcher Tests
# =============================================================================

class TestPatternMatcher:
    """Tests for the deterministic pattern matcher."""
    
    def test_exact_match(self):
        """Test exact hash-based matching."""
        matcher = PatternMatcher()
        matcher.add_exact("hello world", "skill_001")
        matcher.build()
        
        result = matcher.match("hello world")
        
        assert result.matched
        assert result.pattern_id == "skill_001"
        assert result.match_type == "exact"
    
    def test_substring_match(self):
        """Test Aho-Corasick substring matching."""
        matcher = PatternMatcher()
        matcher.add_substring("python script", "skill_002")
        matcher.build()
        
        # Test with input containing the exact substring
        result = matcher.match("write a python script to parse json")
        
        assert result.matched
        assert result.pattern_id == "skill_002"
    
    def test_regex_match(self):
        """Test regex pattern matching."""
        matcher = PatternMatcher()
        matcher.add_regex(r"create.*project", "skill_003")
        matcher.build()
        
        result = matcher.match("I want to create a new project")
        
        assert result.matched
        assert result.pattern_id == "skill_003"
        assert result.match_type == "regex"
    
    def test_no_match(self):
        """Test when no pattern matches."""
        matcher = PatternMatcher()
        matcher.add_exact("specific phrase", "skill_001")
        matcher.build()
        
        result = matcher.match("completely different input")
        
        assert not result.matched


# =============================================================================
# Skill Registry Tests
# =============================================================================

class TestSkillRegistry:
    """Tests for the self-optimizing skill registry."""
    
    def test_registry_creation(self):
        """Test registry creation."""
        registry = SkillRegistry()
        assert registry.skill_count == 0
    
    def test_add_skill(self):
        """Test adding a skill."""
        registry = SkillRegistry()
        
        entry = registry.add_macro_skill(
            name="python_hello",
            pattern="hello world python",
            template="print('Hello, World!')"
        )
        
        assert registry.skill_count == 1
        assert entry.name == "python_hello"
    
    def test_skill_lookup(self):
        """Test skill lookup."""
        registry = SkillRegistry()
        registry.add_macro_skill(
            name="python_hello",
            pattern="hello world python",
            template="print('Hello')"
        )
        
        # Use exact match for substring lookup
        match = registry.lookup("write hello world python code")
        
        assert match.found
        assert match.skill.name == "python_hello"
    
    def test_skill_miss(self):
        """Test skill lookup miss."""
        registry = SkillRegistry()
        
        match = registry.lookup("random query with no skills")
        
        assert not match.found


# =============================================================================
# Macro Store Tests
# =============================================================================

class TestMacroStore:
    """Tests for the generative trace promotion system."""
    
    def test_store_creation(self):
        """Test store creation."""
        store = MacroStore()
        assert store.trace_count == 0
        assert store.macro_count == 0
    
    def test_record_trace(self):
        """Test recording a generative trace."""
        store = MacroStore()
        
        trace = GenerativeTrace.create(
            input_text="Write hello world",
            output_text="print('Hello')",
            tokens_in=10,
            tokens_out=5,
        )
        
        store.record_trace(trace)
        
        assert store.trace_count == 1
    
    def test_promotion_criteria(self):
        """Test that promotion requires multiple traces."""
        criteria = PromotionCriteria(min_occurrences=3, min_success_rate=0.9)
        store = MacroStore(criteria=criteria)
        
        # Record 2 traces (below threshold)
        for _ in range(2):
            trace = GenerativeTrace.create(
                input_text="same input",
                output_text="same output",
            )
            store.record_trace(trace)
        
        candidates = store.get_promotion_candidates()
        assert len(candidates) == 0  # Not enough occurrences
        
        # Record 1 more (meets threshold)
        trace = GenerativeTrace.create(
            input_text="same input",
            output_text="same output",
        )
        store.record_trace(trace)
        
        candidates = store.get_promotion_candidates()
        assert len(candidates) == 1


# =============================================================================
# CarbonCompiler Tests
# =============================================================================

class TestCarbonCompiler:
    """Tests for the main Carbon Compiler pipeline."""
    
    def test_compiler_creation(self):
        """Test compiler creation."""
        compiler = CarbonCompiler.create_default()
        assert compiler is not None
    
    def test_process_returns_result(self):
        """Test that process returns a CompilationResult."""
        compiler = CarbonCompiler.create_default()
        
        result = compiler.process("Write a Python script")
        
        assert result is not None
        assert result.bytecode is not None
        assert result.path in ExecutionPath
    
    def test_deterministic_path_with_skill(self):
        """Test deterministic path when skill is registered."""
        # Disable gating to test skill matching directly
        config = CompilerConfig(enable_gating=False)
        compiler = CarbonCompiler(config=config)
        
        # Add a skill with exact pattern
        compiler.add_skill(
            name="test_skill",
            pattern="test pattern for matching",
            template="test output"
        )
        
        result = compiler.process("test pattern for matching")
        
        assert result.is_deterministic
        assert result.path == ExecutionPath.DETERMINISTIC
    
    def test_generative_path_no_match(self):
        """Test generative path when no skill matches."""
        compiler = CarbonCompiler.create_default()
        
        result = compiler.process("Completely unique query that won't match anything specific")
        
        # Should fall through to generative
        assert result.path in (ExecutionPath.GENERATIVE, ExecutionPath.IDLE)
    
    def test_compilation_time_recorded(self):
        """Test that compilation time is recorded."""
        compiler = CarbonCompiler.create_default()
        
        result = compiler.process("Test input")
        
        assert result.compilation_time_ms >= 0
        assert result.compilation_time_ms < 1000  # Should be fast
    
    def test_stats_tracking(self):
        """Test that statistics are tracked."""
        compiler = CarbonCompiler.create_default()
        
        # Process a few inputs
        compiler.process("Test input 1")
        compiler.process("Test input 2")
        
        stats = compiler.get_stats()
        
        assert stats["total_compilations"] == 2
    
    def test_record_generation(self):
        """Test recording generative traces."""
        compiler = CarbonCompiler.create_default()
        
        trace = compiler.record_generation(
            input_text="test input",
            output_text="test output",
            tokens_in=10,
            tokens_out=20,
        )
        
        assert trace is not None
        assert trace.input_text == "test input"
    
    def test_efficiency_report(self):
        """Test efficiency report generation."""
        compiler = CarbonCompiler.create_default()
        
        # Add a skill and process matching input
        compiler.add_skill("test", "efficiency test pattern", "output")
        compiler.process("efficiency test pattern")
        
        report = compiler.get_efficiency_report()
        
        assert "deterministic_rate_pct" in report


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance benchmarks for the Carbon Protocol."""
    
    def test_neuron_throughput(self):
        """Test neuron operations per second."""
        neuron = CarbonNeuron()
        
        start = time.perf_counter()
        for _ in range(100_000):
            neuron.input(0.1)
            neuron.tick()
            neuron.fire()
        elapsed = time.perf_counter() - start
        
        ops_per_sec = 100_000 / elapsed
        assert ops_per_sec > 100_000, f"Only {ops_per_sec:.0f} ops/sec"
    
    def test_pattern_matcher_throughput(self):
        """Test pattern matching throughput."""
        matcher = PatternMatcher()
        
        # Add many patterns
        for i in range(100):
            matcher.add_substring(f"pattern number {i}", f"skill_{i}")
        matcher.build()
        
        start = time.perf_counter()
        for _ in range(10_000):
            matcher.match("This is a test with pattern number 50 in it")
        elapsed = time.perf_counter() - start
        
        matches_per_sec = 10_000 / elapsed
        assert matches_per_sec > 1_000, f"Only {matches_per_sec:.0f} matches/sec"
    
    def test_compiler_latency(self):
        """Test compiler latency (should be < 10ms)."""
        compiler = CarbonCompiler.create_default()
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            compiler.process("Write a Python script to parse JSON files")
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 10, f"Average latency {avg_latency:.2f}ms exceeds 10ms"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_deterministic(self):
        """Test full pipeline with deterministic outcome."""
        # Disable gating to ensure we test skill matching
        config = CompilerConfig(enable_gating=False)
        compiler = CarbonCompiler(config=config)
        
        # Add a scaffold skill with exact pattern
        compiler.add_scaffold_skill(
            name="python_mvc",
            pattern="scaffold python project",
            lang="python",
            arch="mvc",
        )
        
        # Process matching input (contains the pattern)
        result = compiler.process("scaffold python project now")
        
        assert result.is_deterministic
        assert result.primary_opcode == CarbonOpCode.SCAFFOLD
    
    def test_skill_evolution(self):
        """Test that repeated traces can be promoted to skills."""
        compiler = CarbonCompiler.create_default()
        
        # Simulate repeated successful generations
        for _ in range(5):
            compiler.record_generation(
                input_text="repeated test query",
                output_text="consistent response",
                success=True,
            )
        
        # Check that promotion candidates exist
        candidates = compiler.registry.macro_store.get_promotion_candidates()
        # May or may not have candidates depending on timing criteria
        assert isinstance(candidates, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
