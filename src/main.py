#!/usr/bin/env python3
"""
Carbon Protocol SDK - Main Entry Point

This module provides the main entry point for the Carbon Protocol SDK,
demonstrating the "Wake-on-Meaning" neuromorphic architecture.

Usage:
    python -m src.main
    python src/main.py
    
Reference: Carbon Protocol Research Paper
    "Wake-on-Meaning Architecture for Sustainable AI"
"""

from __future__ import annotations

import sys
from typing import Optional

# Import main components from the SDK
from .compiler import CarbonCompiler, CompilationResult, ExecutionPath
from .c_isa import CarbonOpCode, OP_IDLE, OP_MACRO, OP_GEN
from .ingestion import CarbonNeuron, IntentDetector
from .registry import SkillDB


def demo_basic_pipeline() -> None:
    """Demonstrate the basic Carbon Protocol pipeline."""
    print("=" * 60)
    print("Carbon Protocol SDK - Wake-on-Meaning Demo")
    print("=" * 60)
    
    # Create the compiler (main router)
    compiler = CarbonCompiler.create_default()
    
    # Add some example skills
    compiler.add_skill(
        name="python_hello",
        pattern="hello world python",
        template="print('Hello, World!')",
    )
    compiler.add_skill(
        name="python_json",
        pattern="parse json python",
        template="import json\ndata = json.loads(text)",
    )
    
    # Test inputs
    test_inputs = [
        "Write hello world in python",  # Should match skill
        "Parse JSON in python",          # Should match skill
        "What is the meaning of life?",  # No match → generative
        "",                              # IDLE
    ]
    
    print("\nProcessing test inputs:\n")
    
    for user_input in test_inputs:
        result = compiler.process(user_input)
        
        print(f"Input: '{user_input}'")
        print(f"  Path: {result.path.name}")
        print(f"  OpCode: {result.primary_opcode.name}")
        print(f"  Time: {result.compilation_time_ms:.3f}ms")
        print()
    
    # Print efficiency report
    print("\nEfficiency Report:")
    report = compiler.get_efficiency_report()
    for key, value in report.items():
        if "pct" in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value}")


def demo_neuron_behavior() -> None:
    """Demonstrate LIF neuron behavior."""
    print("\n" + "=" * 60)
    print("LIF Neuron Demo")
    print("=" * 60)
    
    neuron = CarbonNeuron(threshold=1.0, decay_rate=0.9, label="demo")
    
    print("\nSimulating neuron with threshold=1.0, decay=0.9:")
    print()
    
    signals = [0.3, 0.4, 0.5, 0.3, 0.1]
    
    for i, signal in enumerate(signals):
        neuron.input(signal)
        fired = neuron.fire()
        print(f"  Step {i+1}: Input={signal:.1f}, Voltage={neuron.voltage:.3f}, Fired={fired}")
        neuron.tick()  # Apply decay


def demo_skill_db() -> None:
    """Demonstrate skill database with Aho-Corasick."""
    print("\n" + "=" * 60)
    print("Skill Database (Trie/Aho-Corasick) Demo")
    print("=" * 60)
    
    from .c_isa import CarbonInstruction
    
    db = SkillDB()
    
    # Add skills with different pattern types
    db.add_skill(
        name="exact_match",
        pattern="hello world",
        instruction=CarbonInstruction(CarbonOpCode.MACRO, {"template": "Hi!"}),
        pattern_type="exact",
    )
    
    db.add_skill(
        name="substring_match",
        pattern="python script",
        instruction=CarbonInstruction(CarbonOpCode.SCAFFOLD, {"lang": "python"}),
        pattern_type="substring",
    )
    
    print("\nRegistered skills:")
    for skill in db.list_skills():
        print(f"  - {skill.name}: '{skill.pattern}' ({skill.pattern_type})")
    
    # Test lookups
    print("\nLookup tests:")
    test_queries = [
        "hello world",                    # Exact match
        "Write a python script for me",   # Substring match
        "Do something random",            # No match
    ]
    
    for query in test_queries:
        match = db.lookup(query)
        if match.found:
            print(f"  '{query}' -> MATCH ({match.match_type}): {match.skill.name if match.skill else 'macro'}")
        else:
            print(f"  '{query}' -> NO MATCH (would use OP_GEN)")


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point.
    
    Args:
        args: Command line arguments (optional).
        
    Returns:
        Exit code (0 for success).
    """
    if args is None:
        args = sys.argv[1:]
    
    print(f"\n{'='*60}")
    print("Carbon Protocol SDK v2.0.0")
    print("Wake-on-Meaning Neuromorphic Architecture")
    print(f"{'='*60}\n")
    
    # Run demos
    demo_basic_pipeline()
    demo_neuron_behavior()
    demo_skill_db()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
