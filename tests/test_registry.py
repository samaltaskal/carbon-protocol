"""
Unit Tests for Registry Class.

IEEE 829 Test Specification:
    Test Suite ID: TS-REG-001
    Test Suite Name: Registry Unit Tests
    Component Under Test: src.registry.Registry
    
Test Coverage:
    - REG-001: Registry initialization
    - REG-002: Domain loading from YAML
    - REG-003: Multiple domain loading
    - REG-004: Automaton building
    - REG-005: Error handling
    - REG-006: Registry statistics
"""

import pytest
from pathlib import Path

from src import Registry


@pytest.mark.unit
class TestRegistryInitialization:
    """
    IEEE Test Case ID: REG-001
    Test Case Name: Registry Initialization
    Objective: Verify Registry initializes with correct default state
    """
    
    def test_init_default_data_dir(self):
        """
        Test ID: REG-001-01
        Description: Registry should initialize with default data directory
        Pre-conditions: None
        Expected Result: Registry created with data_dir pointing to src/data
        """
        registry = Registry()
        
        assert registry.patterns == {}
        assert registry.automaton is None
        assert registry.loaded_domains == set()
        assert registry.data_dir.name == "data"
    
    def test_init_custom_data_dir(self, tmp_path: Path):
        """
        Test ID: REG-001-02
        Description: Registry should accept custom data directory
        Pre-conditions: Temporary directory exists
        Expected Result: Registry created with custom data_dir
        """
        registry = Registry(data_dir=tmp_path)
        
        assert registry.data_dir == tmp_path
    
    def test_init_string_data_dir(self, tmp_path: Path):
        """
        Test ID: REG-001-03
        Description: Registry should accept string path for data directory
        Pre-conditions: Temporary directory exists
        Expected Result: Registry converts string to Path object
        """
        registry = Registry(data_dir=str(tmp_path))
        
        assert registry.data_dir == tmp_path


@pytest.mark.unit
class TestDomainLoading:
    """
    IEEE Test Case ID: REG-002
    Test Case Name: Domain Loading
    Objective: Verify YAML domain files are loaded correctly
    """
    
    def test_load_core_domain(self, empty_registry: Registry):
        """
        Test ID: REG-002-01
        Description: Loading 'core' domain should populate patterns
        Pre-conditions: core.yaml exists in data directory
        Expected Result: Patterns loaded, domain tracked, count > 0
        """
        count = empty_registry.load_domain("core")
        
        assert count > 0
        assert "core" in empty_registry.loaded_domains
        assert len(empty_registry.patterns) == count
    
    def test_load_domain_invalidates_automaton(self, registry: Registry):
        """
        Test ID: REG-002-02
        Description: Loading new domain should invalidate existing automaton
        Pre-conditions: Registry has automaton built
        Expected Result: Automaton becomes None after loading new domain
        """
        assert registry.automaton is not None
        
        # Create a new domain file dynamically
        import yaml
        new_domain = registry.data_dir / "temp_test.yaml"
        new_domain.write_text(
            yaml.dump({"patterns": [{"input": "temp", "output": "@TEMP"}]}),
            encoding="utf-8"
        )
        
        try:
            registry.load_domain("temp_test")
            assert registry.automaton is None
        finally:
            new_domain.unlink()  # Cleanup
    
    def test_load_nonexistent_domain(self, empty_registry: Registry):
        """
        Test ID: REG-002-03
        Description: Loading non-existent domain should raise FileNotFoundError
        Pre-conditions: Domain file does not exist
        Expected Result: FileNotFoundError raised with helpful message
        """
        with pytest.raises(FileNotFoundError) as exc_info:
            empty_registry.load_domain("nonexistent_domain_xyz")
        
        assert "nonexistent_domain_xyz" in str(exc_info.value)
    
    def test_load_domain_normalizes_input(self, tmp_path: Path):
        """
        Test ID: REG-002-04
        Description: Pattern inputs should be normalized to lowercase
        Pre-conditions: Domain file with mixed-case inputs
        Expected Result: All pattern keys are lowercase
        """
        import yaml
        
        test_yaml = tmp_path / "mixed_case.yaml"
        test_yaml.write_text(
            yaml.dump({
                "patterns": [
                    {"input": "HELLO World", "output": "@TEST"},
                    {"input": "  spaced  ", "output": "@SPACED"},
                ]
            }),
            encoding="utf-8"
        )
        
        registry = Registry(data_dir=tmp_path)
        registry.load_domain("mixed_case")
        
        assert "hello world" in registry.patterns
        assert "spaced" in registry.patterns


@pytest.mark.unit
class TestMultipleDomainLoading:
    """
    IEEE Test Case ID: REG-003
    Test Case Name: Multiple Domain Loading
    Objective: Verify multiple domains can be loaded and combined
    """
    
    def test_load_domains_method(self, tmp_path: Path):
        """
        Test ID: REG-003-01
        Description: load_domains() should load multiple domains at once
        Pre-conditions: Multiple domain files exist
        Expected Result: All domains loaded, total count returned
        """
        import yaml
        
        # Create multiple domain files
        for name, pattern in [("d1", "alpha"), ("d2", "beta"), ("d3", "gamma")]:
            (tmp_path / f"{name}.yaml").write_text(
                yaml.dump({"patterns": [{"input": pattern, "output": f"@{name.upper()}"}]}),
                encoding="utf-8"
            )
        
        registry = Registry(data_dir=tmp_path)
        total = registry.load_domains("d1", "d2", "d3")
        
        assert total == 3
        assert registry.loaded_domains == {"d1", "d2", "d3"}
        assert len(registry.patterns) == 3


@pytest.mark.unit
class TestAutomatonBuilding:
    """
    IEEE Test Case ID: REG-004
    Test Case Name: Automaton Building
    Objective: Verify Aho-Corasick automaton is built correctly
    """
    
    def test_build_automaton_success(self, empty_registry: Registry):
        """
        Test ID: REG-004-01
        Description: build_automaton() should create valid automaton
        Pre-conditions: Patterns loaded
        Expected Result: Automaton object created and ready for matching
        """
        empty_registry.load_domain("core")
        automaton = empty_registry.build_automaton()
        
        assert automaton is not None
        assert empty_registry.automaton is automaton
    
    def test_build_automaton_no_patterns(self, empty_registry: Registry):
        """
        Test ID: REG-004-02
        Description: build_automaton() should fail if no patterns loaded
        Pre-conditions: No patterns loaded
        Expected Result: ValueError raised
        """
        with pytest.raises(ValueError) as exc_info:
            empty_registry.build_automaton()
        
        assert "No patterns loaded" in str(exc_info.value)
    
    def test_get_automaton_lazy_build(self, empty_registry: Registry):
        """
        Test ID: REG-004-03
        Description: get_automaton() should build if not already built
        Pre-conditions: Patterns loaded, automaton not built
        Expected Result: Automaton built on first access
        """
        empty_registry.load_domain("core")
        assert empty_registry.automaton is None
        
        automaton = empty_registry.get_automaton()
        
        assert automaton is not None
        assert empty_registry.automaton is automaton


@pytest.mark.unit
class TestErrorHandling:
    """
    IEEE Test Case ID: REG-005
    Test Case Name: Error Handling
    Objective: Verify proper error handling for edge cases
    """
    
    def test_empty_yaml_file(self, tmp_path: Path):
        """
        Test ID: REG-005-01
        Description: Empty YAML file should raise ValueError
        Pre-conditions: Empty YAML file exists
        Expected Result: ValueError raised
        """
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")
        
        registry = Registry(data_dir=tmp_path)
        
        with pytest.raises(ValueError) as exc_info:
            registry.load_domain("empty")
        
        assert "Empty YAML" in str(exc_info.value)
    
    def test_invalid_yaml_structure(self, tmp_path: Path):
        """
        Test ID: REG-005-02
        Description: YAML with invalid structure should raise ValueError
        Pre-conditions: YAML file with 'patterns' as string instead of list
        Expected Result: ValueError raised with helpful message
        """
        import yaml
        
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text(
            yaml.dump({"patterns": "not a list"}),
            encoding="utf-8"
        )
        
        registry = Registry(data_dir=tmp_path)
        
        with pytest.raises(ValueError) as exc_info:
            registry.load_domain("invalid")
        
        assert "must be a list" in str(exc_info.value)


@pytest.mark.unit
class TestRegistryStatistics:
    """
    IEEE Test Case ID: REG-006
    Test Case Name: Registry Statistics
    Objective: Verify statistics reporting is accurate
    """
    
    def test_get_stats_empty(self, empty_registry: Registry):
        """
        Test ID: REG-006-01
        Description: Stats for empty registry should show zeros
        Pre-conditions: No domains loaded
        Expected Result: Empty domains, zero patterns, automaton not built
        """
        stats = empty_registry.get_stats()
        
        assert stats["loaded_domains"] == []
        assert stats["total_patterns"] == 0
        assert stats["automaton_built"] is False
    
    def test_get_stats_after_load(self, registry: Registry):
        """
        Test ID: REG-006-02
        Description: Stats after loading should reflect current state
        Pre-conditions: Core domain loaded and automaton built
        Expected Result: Stats show loaded domain, pattern count, automaton ready
        """
        stats = registry.get_stats()
        
        assert "core" in stats["loaded_domains"]
        assert stats["total_patterns"] > 0
        assert stats["automaton_built"] is True
    
    def test_clear_registry(self, registry: Registry):
        """
        Test ID: REG-006-03
        Description: clear() should reset registry to initial state
        Pre-conditions: Registry with loaded domain
        Expected Result: All state cleared
        """
        registry.clear()
        
        assert registry.patterns == {}
        assert registry.automaton is None
        assert registry.loaded_domains == set()
