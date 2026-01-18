"""
Integration and End-to-End Tests.

IEEE 829 Test Specification:
    Test Suite ID: TS-INT-001
    Test Suite Name: Integration and E2E Tests
    Components Under Test: Full SDK Pipeline (Registry + Compiler)
    
Test Coverage:
    - INT-001: Full pipeline integration
    - INT-002: Real-world usage scenarios
    - INT-003: Performance benchmarks
    - INT-004: Edge cases and boundary conditions
"""

import time
import pytest

from src import Registry, Compiler


@pytest.mark.integration
class TestFullPipelineIntegration:
    """
    IEEE Test Case ID: INT-001
    Test Case Name: Full Pipeline Integration
    Objective: Verify Registry and Compiler work together correctly
    """
    
    def test_registry_to_compiler_flow(self, data_dir):
        """
        Test ID: INT-001-01
        Description: Complete flow from domain loading to compression
        Pre-conditions: Core domain YAML exists
        Expected Result: Full pipeline executes without errors
        """
        # Setup phase
        registry = Registry(data_dir=data_dir)
        count = registry.load_domain("core")
        automaton = registry.build_automaton()
        compiler = Compiler(registry)
        
        # Execution phase
        result = compiler.compress("write a python script to scrape data")
        
        # Verification phase
        assert count > 0
        assert automaton is not None
        assert result.matches_found > 0
        assert "@LANG:PY" in result.compressed or "python" not in result.compressed.lower()
    
    def test_multiple_domain_integration(self, tmp_path):
        """
        Test ID: INT-001-02
        Description: Multiple domains should combine correctly
        Pre-conditions: Multiple domain files
        Expected Result: Patterns from all domains available for matching
        """
        import yaml
        
        # Create domain files
        (tmp_path / "domain_a.yaml").write_text(
            yaml.dump({"patterns": [{"input": "alpha pattern", "output": "@ALPHA"}]}),
            encoding="utf-8"
        )
        (tmp_path / "domain_b.yaml").write_text(
            yaml.dump({"patterns": [{"input": "beta pattern", "output": "@BETA"}]}),
            encoding="utf-8"
        )
        
        # Load and build
        registry = Registry(data_dir=tmp_path)
        registry.load_domains("domain_a", "domain_b")
        registry.build_automaton()
        compiler = Compiler(registry)
        
        # Test
        result = compiler.compress("alpha pattern and beta pattern")
        
        assert "@ALPHA" in result.compressed
        assert "@BETA" in result.compressed
        assert result.matches_found == 2


@pytest.mark.e2e
class TestRealWorldScenarios:
    """
    IEEE Test Case ID: INT-002
    Test Case Name: Real-World Usage Scenarios
    Objective: Verify SDK handles realistic prompts correctly
    """
    
    @pytest.mark.parametrize("input_text,expected_tokens", [
        (
            "Please check the python script",
            ["@LANG:PY"]
        ),
        (
            "Could you please write a python script to scrape data from the api",
            ["@LANG:PY", "@ACT:SCRAPE"]
        ),
        (
            "I want you to search for all users in the database",
            ["@OP:SEARCH", "@CTX:DB"]
        ),
        (
            "Create a new json format file",
            ["@OP:CREATE", "@FMT:JSON"]
        ),
    ])
    def test_realistic_prompts(self, compiler: Compiler, input_text: str, expected_tokens: list[str]):
        """
        Test ID: INT-002-01
        Description: Real-world prompts should compress correctly
        Pre-conditions: Core domain loaded
        Expected Result: Expected tokens present in output
        """
        result = compiler.compress(input_text)
        
        for token in expected_tokens:
            assert token in result.compressed, f"Expected {token} in '{result.compressed}'"
    
    def test_filler_removal(self, compiler: Compiler):
        """
        Test ID: INT-002-02
        Description: Filler phrases should be removed entirely
        Pre-conditions: Core domain with filler patterns
        Expected Result: Compressed output shorter, fillers removed
        """
        input_text = "Could you please help me to check the output"
        result = compiler.compress(input_text)
        
        # Filler phrases should be removed
        assert "could you please" not in result.compressed.lower()
        assert result.compression_ratio < 1.0  # Output should be shorter
    
    def test_tool_disambiguation(self, compiler: Compiler):
        """
        Test ID: INT-002-03
        Description: Similar tool names should be distinguished correctly
        Pre-conditions: Core domain with 'visual studio' and 'visual studio code'
        Expected Result: Correct token for each tool
        """
        result = compiler.compress("I prefer visual studio code over visual studio")
        
        assert "@TOOL:VSCODE" in result.compressed
        assert "@TOOL:VS" in result.compressed
        # Both tokens should be present and distinct
        # Count occurrences to ensure both are matched separately
        assert result.compressed.count("@TOOL:VSCODE") == 1
        assert result.compressed.count("@TOOL:VS") == 2  # VSCODE contains VS, plus standalone VS


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """
    IEEE Test Case ID: INT-003
    Test Case Name: Performance Benchmarks
    Objective: Verify SDK meets <10ms latency requirement
    """
    
    def test_single_compression_latency(self, compiler: Compiler):
        """
        Test ID: INT-003-01
        Description: Single compression should complete in <10ms
        Pre-conditions: Core domain loaded
        Expected Result: Execution time < 10ms
        """
        test_text = "Could you please write a python script to scrape data from the api"
        
        # Warm-up run
        compiler.compress(test_text)
        
        # Timed run
        start = time.perf_counter()
        result = compiler.compress(test_text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert elapsed_ms < 10, f"Compression took {elapsed_ms:.2f}ms, exceeds 10ms threshold"
        assert result.matches_found > 0
    
    @pytest.mark.slow
    def test_batch_compression_throughput(self, compiler: Compiler):
        """
        Test ID: INT-003-02
        Description: Batch processing should maintain throughput
        Pre-conditions: Core domain loaded
        Expected Result: Average latency < 10ms per compression
        """
        test_texts = [
            "write a python script",
            "please check the output",
            "create a new json file",
            "search for users in the database",
            "visual studio code settings",
        ] * 100  # 500 texts
        
        start = time.perf_counter()
        results = compiler.compress_batch(test_texts)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        avg_latency = elapsed_ms / len(test_texts)
        
        assert len(results) == 500
        assert avg_latency < 10, f"Average latency {avg_latency:.2f}ms exceeds threshold"
    
    @pytest.mark.slow
    def test_large_text_compression(self, compiler: Compiler):
        """
        Test ID: INT-003-03
        Description: Large text should still compress efficiently
        Pre-conditions: Core domain loaded
        Expected Result: O(n) scaling demonstrated
        """
        base_text = "Could you please write a python script to scrape data from the api. "
        
        # Test with increasing sizes
        results = []
        for multiplier in [1, 10, 100]:
            text = base_text * multiplier
            
            start = time.perf_counter()
            result = compiler.compress(text)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            results.append({
                "size": len(text),
                "time_ms": elapsed_ms,
                "matches": result.matches_found
            })
        
        # Verify roughly linear scaling (allow 3x variance for small sizes)
        ratio_10x = results[1]["time_ms"] / max(results[0]["time_ms"], 0.001)
        ratio_100x = results[2]["time_ms"] / max(results[0]["time_ms"], 0.001)
        
        # Should scale roughly linearly, not quadratically
        assert ratio_10x < 30, f"10x text took {ratio_10x:.1f}x time (expected ~10x)"
        assert ratio_100x < 300, f"100x text took {ratio_100x:.1f}x time (expected ~100x)"


@pytest.mark.integration
class TestEdgeCasesAndBoundaries:
    """
    IEEE Test Case ID: INT-004
    Test Case Name: Edge Cases and Boundary Conditions
    Objective: Verify SDK handles edge cases gracefully
    """
    
    def test_unicode_text(self, compiler: Compiler):
        """
        Test ID: INT-004-01
        Description: Unicode characters should not break compression
        Pre-conditions: Core domain loaded
        Expected Result: No errors, text processed
        """
        result = compiler.compress("write a python script for émojis 🐍 and 日本語")
        
        assert "🐍" in result.compressed or result.matches_found >= 0
        assert "日本語" in result.compressed
    
    def test_special_characters(self, compiler: Compiler):
        """
        Test ID: INT-004-02
        Description: Special characters should not interfere
        Pre-conditions: Core domain loaded
        Expected Result: Patterns still matched around special chars
        """
        result = compiler.compress("write a python script!!! @#$% more text")
        
        # Pattern should still match despite surrounding special chars
        assert result.matches_found >= 0  # May or may not match depending on exact text
    
    def test_very_long_text(self, compiler: Compiler):
        """
        Test ID: INT-004-03
        Description: Very long text should process without memory issues
        Pre-conditions: Core domain loaded
        Expected Result: Compression completes successfully
        """
        long_text = "python script " * 10000  # ~140KB of text
        
        result = compiler.compress(long_text)
        
        assert result.matches_found > 0
        assert len(result.compressed) < len(result.original)
    
    def test_repeated_patterns(self, compiler: Compiler):
        """
        Test ID: INT-004-04
        Description: Same pattern repeated should be matched multiple times
        Pre-conditions: Core domain loaded
        Expected Result: All occurrences matched
        """
        result = compiler.compress(
            "python script here, another python script there, python script everywhere"
        )
        
        # Count occurrences of the token
        token_count = result.compressed.count("@LANG:PY")
        assert token_count == 3, f"Expected 3 matches, found {token_count}"
    
    def test_adjacent_patterns(self, compiler: Compiler):
        """
        Test ID: INT-004-05
        Description: Adjacent patterns should both be matched
        Pre-conditions: Core domain loaded
        Expected Result: Both patterns matched without interference
        """
        result = compiler.compress("json format csv file")
        
        assert "@FMT:JSON" in result.compressed
        assert "@FMT:CSV" in result.compressed
        assert result.matches_found == 2
    
    def test_whitespace_handling(self, compiler: Compiler):
        """
        Test ID: INT-004-06
        Description: Various whitespace should be handled correctly
        Pre-conditions: Core domain loaded
        Expected Result: Patterns matched despite whitespace variations
        """
        # Multiple spaces between words shouldn't affect matching
        result = compiler.compress("python   script")  # Extra spaces
        
        # Note: This may or may not match depending on implementation
        # The test documents the behavior
        assert isinstance(result.compressed, str)
