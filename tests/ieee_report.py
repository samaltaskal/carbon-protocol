"""
IEEE 829 Compliant Test Report Generator.

This module provides a pytest plugin that generates test reports in
IEEE 829 Standard for Software and System Test Documentation format.

IEEE 829 Sections Generated:
    - Test Summary Report (Section 8)
    - Test Case Results with full traceability
    - Test Execution Statistics
    - Defect/Failure Analysis
    
Usage:
    pytest --ieee-report
    pytest --ieee-report --ieee-output=report.txt
"""

import datetime
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pytest


@dataclass
class IEEETestCase:
    """IEEE 829 compliant test case result structure."""
    test_id: str
    test_name: str
    test_module: str
    test_class: str | None
    description: str
    status: str  # PASS, FAIL, ERROR, SKIP
    execution_time_ms: float
    timestamp: str
    preconditions: str = ""
    expected_result: str = ""
    actual_result: str = ""
    failure_reason: str = ""
    markers: list[str] = field(default_factory=list)


@dataclass 
class IEEETestSummary:
    """IEEE 829 Test Summary Report structure."""
    report_id: str
    report_title: str
    project_name: str
    test_date: str
    test_environment: str
    total_tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    total_execution_time_ms: float
    pass_rate: float
    test_cases: list[IEEETestCase] = field(default_factory=list)


class IEEEReportPlugin:
    """Pytest plugin for generating IEEE 829 compliant test reports."""
    
    def __init__(self):
        self.test_cases: list[IEEETestCase] = []
        self.start_time: float = 0
        self.end_time: float = 0
    
    def pytest_sessionstart(self, session):
        """Record session start time."""
        import time
        self.start_time = time.time()
    
    def pytest_runtest_logreport(self, report):
        """Capture test results."""
        if report.when == "call" or (report.when == "setup" and report.failed):
            # Extract test information
            test_id = report.nodeid
            parts = test_id.split("::")
            
            module = parts[0] if len(parts) > 0 else ""
            test_class = parts[1] if len(parts) > 2 else None
            test_name = parts[-1] if parts else ""
            
            # Determine status
            if report.passed:
                status = "PASS"
            elif report.failed:
                status = "FAIL" if report.when == "call" else "ERROR"
            elif report.skipped:
                status = "SKIP"
            else:
                status = "UNKNOWN"
            
            # Extract docstring for description
            description = ""
            if hasattr(report, "item") and report.item:
                func = getattr(report.item, "function", None)
                if func and func.__doc__:
                    description = func.__doc__.strip().split("\n")[0]
            
            # Extract failure information
            failure_reason = ""
            if report.failed and report.longrepr:
                failure_reason = str(report.longrepr)[:500]  # Truncate
            
            # Extract markers
            markers = []
            if hasattr(report, "item") and report.item:
                for marker in report.item.iter_markers():
                    markers.append(marker.name)
            
            test_case = IEEETestCase(
                test_id=test_id,
                test_name=test_name,
                test_module=module,
                test_class=test_class,
                description=description,
                status=status,
                execution_time_ms=report.duration * 1000,
                timestamp=datetime.datetime.now().isoformat(),
                markers=markers,
                failure_reason=failure_reason,
            )
            
            self.test_cases.append(test_case)
    
    def pytest_sessionfinish(self, session, exitstatus):
        """Generate IEEE report at session end."""
        import time
        self.end_time = time.time()
        
        # Calculate statistics
        passed = sum(1 for tc in self.test_cases if tc.status == "PASS")
        failed = sum(1 for tc in self.test_cases if tc.status == "FAIL")
        errors = sum(1 for tc in self.test_cases if tc.status == "ERROR")
        skipped = sum(1 for tc in self.test_cases if tc.status == "SKIP")
        total = len(self.test_cases)
        
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        summary = IEEETestSummary(
            report_id=f"TSR-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
            report_title="Carbon Protocol SDK - Test Summary Report",
            project_name="Carbon Protocol SDK",
            test_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            test_environment=f"Python {self._get_python_version()}",
            total_tests=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            total_execution_time_ms=(self.end_time - self.start_time) * 1000,
            pass_rate=pass_rate,
            test_cases=self.test_cases,
        )
        
        # Store for report generation
        self.summary = summary
    
    def _get_python_version(self) -> str:
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def generate_text_report(self) -> str:
        """Generate IEEE 829 formatted text report."""
        s = self.summary
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("IEEE 829 TEST SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Section 1: Report Identification
        lines.append("1. REPORT IDENTIFICATION")
        lines.append("-" * 40)
        lines.append(f"   Report ID:      {s.report_id}")
        lines.append(f"   Report Title:   {s.report_title}")
        lines.append(f"   Project:        {s.project_name}")
        lines.append(f"   Test Date:      {s.test_date}")
        lines.append(f"   Environment:    {s.test_environment}")
        lines.append("")
        
        # Section 2: Test Summary
        lines.append("2. TEST SUMMARY")
        lines.append("-" * 40)
        lines.append(f"   Total Tests:        {s.total_tests}")
        lines.append(f"   Passed:             {s.passed}")
        lines.append(f"   Failed:             {s.failed}")
        lines.append(f"   Errors:             {s.errors}")
        lines.append(f"   Skipped:            {s.skipped}")
        lines.append(f"   Pass Rate:          {s.pass_rate:.1f}%")
        lines.append(f"   Execution Time:     {s.total_execution_time_ms:.2f}ms")
        lines.append("")
        
        # Section 3: Test Results by Category
        lines.append("3. TEST RESULTS BY CATEGORY")
        lines.append("-" * 40)
        
        categories = {}
        for tc in s.test_cases:
            for marker in tc.markers:
                if marker not in categories:
                    categories[marker] = {"pass": 0, "fail": 0, "skip": 0}
                if tc.status == "PASS":
                    categories[marker]["pass"] += 1
                elif tc.status in ("FAIL", "ERROR"):
                    categories[marker]["fail"] += 1
                else:
                    categories[marker]["skip"] += 1
        
        for cat, counts in sorted(categories.items()):
            total = sum(counts.values())
            rate = counts["pass"] / total * 100 if total > 0 else 0
            lines.append(f"   [{cat.upper()}] Pass: {counts['pass']}, Fail: {counts['fail']}, Skip: {counts['skip']} ({rate:.0f}%)")
        lines.append("")
        
        # Section 4: Detailed Test Results
        lines.append("4. DETAILED TEST RESULTS")
        lines.append("-" * 40)
        
        for i, tc in enumerate(s.test_cases, 1):
            status_icon = {"PASS": "✓", "FAIL": "✗", "ERROR": "!", "SKIP": "○"}.get(tc.status, "?")
            lines.append(f"   {i:3d}. [{status_icon}] {tc.test_name}")
            lines.append(f"        Module: {tc.test_module}")
            if tc.test_class:
                lines.append(f"        Class:  {tc.test_class}")
            lines.append(f"        Time:   {tc.execution_time_ms:.2f}ms")
            if tc.description:
                lines.append(f"        Desc:   {tc.description[:60]}...")
            if tc.failure_reason:
                lines.append(f"        FAILURE: {tc.failure_reason[:100]}...")
            lines.append("")
        
        # Section 5: Failed Tests Summary
        failed_tests = [tc for tc in s.test_cases if tc.status in ("FAIL", "ERROR")]
        if failed_tests:
            lines.append("5. FAILED TESTS SUMMARY")
            lines.append("-" * 40)
            for tc in failed_tests:
                lines.append(f"   • {tc.test_id}")
                lines.append(f"     Reason: {tc.failure_reason[:200]}")
                lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def generate_json_report(self) -> str:
        """Generate JSON format report for machine processing."""
        return json.dumps(asdict(self.summary), indent=2, default=str)


# Global plugin instance
_ieee_plugin: IEEEReportPlugin | None = None


def pytest_addoption(parser):
    """Add IEEE report options to pytest."""
    group = parser.getgroup("ieee", "IEEE 829 Test Report")
    group.addoption(
        "--ieee-report",
        action="store_true",
        default=False,
        help="Generate IEEE 829 compliant test report",
    )
    group.addoption(
        "--ieee-output",
        action="store",
        default="ieee_test_report.txt",
        help="Output file for IEEE report (default: ieee_test_report.txt)",
    )
    group.addoption(
        "--ieee-json",
        action="store_true",
        default=False,
        help="Also generate JSON format report",
    )


def pytest_configure(config):
    """Register IEEE plugin if enabled."""
    global _ieee_plugin
    if config.getoption("--ieee-report", default=False):
        _ieee_plugin = IEEEReportPlugin()
        config.pluginmanager.register(_ieee_plugin, "ieee_report_plugin")


def _get_results_directory() -> Path:
    """
    Get or create the results directory with date-based subfolder.
    
    Structure: tests/results/YYYY-MM-DD/
    """
    tests_dir = Path(__file__).parent
    results_dir = tests_dir / "results"
    date_folder = results_dir / datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Create directories if they don't exist
    date_folder.mkdir(parents=True, exist_ok=True)
    
    return date_folder


def pytest_unconfigure(config):
    """Generate and save report after tests complete."""
    global _ieee_plugin
    if _ieee_plugin and hasattr(_ieee_plugin, "summary"):
        # Generate text report
        report_text = _ieee_plugin.generate_text_report()
        
        # Get results directory with date-based subfolder
        results_dir = _get_results_directory()
        report_id = _ieee_plugin.summary.report_id
        
        # Save with report ID as filename
        output_file = results_dir / f"{report_id}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        print(f"\n\nIEEE 829 Report saved to: {output_file}")
        
        # Optionally generate JSON
        if config.getoption("--ieee-json", default=False):
            json_file = results_dir / f"{report_id}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                f.write(_ieee_plugin.generate_json_report())
            print(f"IEEE 829 JSON Report saved to: {json_file}")
        
        # Print summary to console (ASCII-safe for Windows compatibility)
        try:
            print("\n" + report_text)
        except UnicodeEncodeError:
            # Fall back to ASCII-safe output
            ascii_report = report_text.replace("✓", "PASS").replace("✗", "FAIL").replace("!", "ERR").replace("○", "SKIP")
            print("\n" + ascii_report)
