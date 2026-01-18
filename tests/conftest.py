"""
Pytest Configuration and Shared Fixtures.

This module provides shared test fixtures and configuration for the
Carbon Protocol SDK test suite, following pytest best practices.

Fixtures:
    - registry: Pre-configured Registry instance with core domain loaded
    - compiler: Pre-configured Compiler instance ready for compression
    - sample_patterns: Dictionary of test patterns for validation
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import Registry, Compiler


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent.parent / "src" / "data"


@pytest.fixture(scope="session")
def sample_patterns() -> dict[str, str]:
    """
    Provide a minimal set of test patterns for unit testing.
    
    These patterns are used for isolated unit tests that don't
    need the full core.yaml domain loaded.
    """
    return {
        "hello world": "@TEST:HELLO",
        "foo bar": "@TEST:FOOBAR",
        "foo bar baz": "@TEST:FOOBARBAZ",  # Longer pattern for overlap tests
        "remove me": "",  # Removal pattern
        "python script": "@LANG:PY",
    }


@pytest.fixture(scope="function")
def empty_registry(data_dir: Path) -> Registry:
    """Provide an empty Registry instance (no domains loaded)."""
    return Registry(data_dir=data_dir)


@pytest.fixture(scope="function")
def registry(data_dir: Path) -> Registry:
    """
    Provide a Registry with core domain loaded and automaton built.
    
    This fixture is function-scoped to ensure test isolation.
    """
    reg = Registry(data_dir=data_dir)
    reg.load_domain("core")
    reg.build_automaton()
    return reg


@pytest.fixture(scope="function")
def compiler(registry: Registry) -> Compiler:
    """Provide a Compiler instance configured with the core registry."""
    return Compiler(registry)


@pytest.fixture(scope="function")
def minimal_registry(sample_patterns: dict[str, str], tmp_path: Path) -> Registry:
    """
    Provide a Registry with minimal test patterns for isolated unit tests.
    
    Creates a temporary YAML file with sample patterns to avoid
    dependencies on production data files.
    """
    import yaml
    
    # Create temporary domain file
    patterns_list = [
        {"input": k, "output": v} for k, v in sample_patterns.items()
    ]
    
    test_yaml = tmp_path / "test.yaml"
    test_yaml.write_text(
        yaml.dump({"patterns": patterns_list}),
        encoding="utf-8"
    )
    
    reg = Registry(data_dir=tmp_path)
    reg.load_domain("test")
    reg.build_automaton()
    return reg


@pytest.fixture(scope="function")
def minimal_compiler(minimal_registry: Registry) -> Compiler:
    """Provide a Compiler with minimal test patterns."""
    return Compiler(minimal_registry)


# =============================================================================
# Test Markers Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests across components"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests simulating real usage"
    )
    config.addinivalue_line(
        "markers", "performance: Performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to execute"
    )
