"""
Unit Tests for Compiler Class.

IEEE 829 Test Specification:
    Test Suite ID: TS-CMP-001
    Test Suite Name: Compiler Unit Tests
    Component Under Test: src.compiler.Compiler
    
Test Coverage:
    - CMP-001: Compiler initialization
    - CMP-002: Basic compression functionality
    - CMP-003: Pattern removal (empty output)
    - CMP-004: Longest match first semantics
    - CMP-005: Non-overlapping match selection
    - CMP-006: Compression result object
    - CMP-007: Batch compression
"""

import pytest

from src import Registry, Compiler, CompressionResult


@pytest.mark.unit
class TestCompilerInitialization:
    """
    IEEE Test Case ID: CMP-001
    Test Case Name: Compiler Initialization
    Objective: Verify Compiler initializes correctly with Registry
    """
    
    def test_init_with_registry(self, registry: Registry):
        """
        Test ID: CMP-001-01
        Description: Compiler should initialize with provided Registry
        Pre-conditions: Registry with patterns loaded
        Expected Result: Compiler created with registry reference
        """
        compiler = Compiler(registry)
        
        assert compiler.registry is registry
        assert compiler.preserve_whitespace is True
    
    def test_init_preserve_whitespace_false(self, registry: Registry):
        """
        Test ID: CMP-001-02
        Description: Compiler should accept preserve_whitespace parameter
        Pre-conditions: Registry with patterns loaded
        Expected Result: Compiler created with custom whitespace setting
        """
        compiler = Compiler(registry, preserve_whitespace=False)
        
        assert compiler.preserve_whitespace is False


@pytest.mark.unit
class TestBasicCompression:
    """
    IEEE Test Case ID: CMP-002
    Test Case Name: Basic Compression Functionality
    Objective: Verify patterns are correctly replaced with tokens
    """
    
    def test_compress_single_pattern(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-002-01
        Description: Single pattern should be replaced with its token
        Pre-conditions: Compiler with 'hello world' -> '@TEST:HELLO' pattern
        Expected Result: 'hello world' replaced with '@TEST:HELLO'
        """
        result = minimal_compiler.compress("hello world")
        
        assert result.compressed == "@TEST:HELLO"
        assert result.matches_found == 1
    
    def test_compress_multiple_patterns(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-002-02
        Description: Multiple non-overlapping patterns should all be replaced
        Pre-conditions: Compiler with multiple patterns
        Expected Result: All patterns replaced in single pass
        """
        result = minimal_compiler.compress("hello world and foo bar")
        
        assert "@TEST:HELLO" in result.compressed
        assert "@TEST:FOOBAR" in result.compressed
        assert result.matches_found == 2
    
    def test_compress_empty_string(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-002-03
        Description: Empty string should return empty result
        Pre-conditions: Valid compiler
        Expected Result: Empty compressed string, zero matches
        """
        result = minimal_compiler.compress("")
        
        assert result.compressed == ""
        assert result.matches_found == 0
    
    def test_compress_no_matches(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-002-04
        Description: Text with no matching patterns should remain unchanged
        Pre-conditions: Valid compiler
        Expected Result: Original text returned, zero matches
        """
        result = minimal_compiler.compress("no patterns here xyz")
        
        assert result.compressed == "no patterns here xyz"
        assert result.matches_found == 0
    
    def test_compress_case_insensitive(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-002-05
        Description: Pattern matching should be case-insensitive
        Pre-conditions: Pattern 'hello world' defined
        Expected Result: 'HELLO WORLD' matches pattern
        """
        result = minimal_compiler.compress("HELLO WORLD")
        
        assert result.compressed == "@TEST:HELLO"
        assert result.matches_found == 1


@pytest.mark.unit
class TestPatternRemoval:
    """
    IEEE Test Case ID: CMP-003
    Test Case Name: Pattern Removal (Empty Output)
    Objective: Verify patterns with empty output are removed from text
    """
    
    def test_removal_pattern(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-003-01
        Description: Pattern with empty output should be stripped
        Pre-conditions: 'remove me' -> '' pattern defined
        Expected Result: Pattern removed from text
        """
        result = minimal_compiler.compress("please remove me now")
        
        assert "remove me" not in result.compressed
        assert result.matches_found == 1
    
    def test_removal_preserves_surrounding_text(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-003-02
        Description: Text around removed pattern should be preserved
        Pre-conditions: 'remove me' -> '' pattern defined
        Expected Result: Surrounding text intact
        """
        result = minimal_compiler.compress("start remove me end")
        
        assert "start" in result.compressed
        assert "end" in result.compressed


@pytest.mark.unit
class TestLongestMatchFirst:
    """
    IEEE Test Case ID: CMP-004
    Test Case Name: Longest Match First Semantics
    Objective: Verify longer patterns are preferred over shorter overlapping ones
    """
    
    def test_longest_match_preferred(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-004-01
        Description: 'foo bar baz' should match before 'foo bar'
        Pre-conditions: Both 'foo bar' and 'foo bar baz' patterns defined
        Expected Result: Longer pattern matched
        """
        result = minimal_compiler.compress("foo bar baz")
        
        # Should match 'foo bar baz' -> @TEST:FOOBARBAZ, not 'foo bar' -> @TEST:FOOBAR
        assert result.compressed == "@TEST:FOOBARBAZ"
        assert result.matches_found == 1
    
    def test_shorter_match_when_longer_not_present(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-004-02
        Description: Shorter pattern should match when longer not present
        Pre-conditions: Both patterns defined, only shorter present in text
        Expected Result: Shorter pattern matched
        """
        result = minimal_compiler.compress("foo bar only")
        
        assert "@TEST:FOOBAR" in result.compressed
        assert result.matches_found == 1
    
    def test_visual_studio_example(self, compiler: Compiler):
        """
        Test ID: CMP-004-03
        Description: 'visual studio code' should match before 'visual studio'
        Pre-conditions: Core domain loaded with both patterns
        Expected Result: Correct token for each phrase
        """
        result = compiler.compress("I use visual studio code and visual studio")
        
        assert "@TOOL:VSCODE" in result.compressed
        assert "@TOOL:VS" in result.compressed


@pytest.mark.unit
class TestNonOverlappingMatches:
    """
    IEEE Test Case ID: CMP-005
    Test Case Name: Non-Overlapping Match Selection
    Objective: Verify overlapping matches are correctly resolved
    """
    
    def test_left_to_right_greedy(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-005-01
        Description: Earlier matches should be preferred (left-to-right greedy)
        Pre-conditions: Valid compiler
        Expected Result: Leftmost non-overlapping matches selected
        """
        result = minimal_compiler.compress("hello world here foo bar there")
        
        assert "@TEST:HELLO" in result.compressed
        assert "@TEST:FOOBAR" in result.compressed
        assert result.matches_found == 2


@pytest.mark.unit
class TestCompressionResult:
    """
    IEEE Test Case ID: CMP-006
    Test Case Name: Compression Result Object
    Objective: Verify CompressionResult provides accurate metrics
    """
    
    def test_compression_ratio(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-006-01
        Description: Compression ratio should be calculated correctly
        Pre-conditions: Valid compression performed
        Expected Result: Ratio = len(compressed) / len(original)
        """
        result = minimal_compiler.compress("hello world")  # 11 chars -> ~12 chars
        
        expected_ratio = len(result.compressed) / len(result.original)
        assert abs(result.compression_ratio - expected_ratio) < 0.001
    
    def test_bytes_saved(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-006-02
        Description: bytes_saved should reflect actual UTF-8 byte difference
        Pre-conditions: Valid compression performed
        Expected Result: Correct byte difference calculated
        """
        result = minimal_compiler.compress("hello world")
        
        expected_bytes = (
            len(result.original.encode('utf-8')) - 
            len(result.compressed.encode('utf-8'))
        )
        assert result.bytes_saved == expected_bytes
    
    def test_empty_input_ratio(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-006-03
        Description: Empty input should return ratio of 1.0
        Pre-conditions: Empty string input
        Expected Result: compression_ratio = 1.0 (no division by zero)
        """
        result = minimal_compiler.compress("")
        
        assert result.compression_ratio == 1.0


@pytest.mark.unit
class TestBatchCompression:
    """
    IEEE Test Case ID: CMP-007
    Test Case Name: Batch Compression
    Objective: Verify multiple texts can be compressed in batch
    """
    
    def test_compress_batch(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-007-01
        Description: compress_batch() should process multiple texts
        Pre-conditions: Valid compiler
        Expected Result: List of CompressionResult objects
        """
        texts = ["hello world", "foo bar", "no match"]
        results = minimal_compiler.compress_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, CompressionResult) for r in results)
        assert results[0].matches_found == 1
        assert results[1].matches_found == 1
        assert results[2].matches_found == 0
    
    def test_compress_batch_empty_list(self, minimal_compiler: Compiler):
        """
        Test ID: CMP-007-02
        Description: Empty batch should return empty list
        Pre-conditions: Valid compiler
        Expected Result: Empty list returned
        """
        results = minimal_compiler.compress_batch([])
        
        assert results == []


@pytest.mark.unit
class TestCompilerStatistics:
    """
    IEEE Test Case ID: CMP-008
    Test Case Name: Compiler Statistics
    Objective: Verify compiler statistics are accurate
    """
    
    def test_get_stats(self, compiler: Compiler):
        """
        Test ID: CMP-008-01
        Description: get_stats() should return accurate configuration
        Pre-conditions: Compiler with registry
        Expected Result: Stats reflect registry state
        """
        stats = compiler.get_stats()
        
        assert stats["registry_patterns"] > 0
        assert stats["automaton_ready"] is True
        assert "preserve_whitespace" in stats
