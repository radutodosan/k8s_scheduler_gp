"""Root conftest — automatic timestamped test result archiving.

Creates a timestamped directory under ``tmp/tests_results/`` for each
pytest invocation.  If any test fails, a separate ``.txt`` error
file is written with the full traceback, named after the test.
"""

from datetime import datetime
from pathlib import Path

import pytest

# Module-level so both hooks share the same directory.
_results_dir: Path = Path(".")


def pytest_configure(config):
    """Register timestamped JUnit XML output and prepare the results dir."""
    global _results_dir
    ts = datetime.now().strftime("%Y_%m_%d_%H%M")
    _results_dir = Path("tmp") / "tests_results" / f"{ts}_tests_results"
    _results_dir.mkdir(parents=True, exist_ok=True)

    # Only set if the user hasn't explicitly passed --junitxml
    if not config.option.xmlpath:
        config.option.xmlpath = str(_results_dir / "results.xml")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """After each test phase, write an error file if the test failed."""
    outcome = yield
    report = outcome.get_result()

    # Only care about the "call" phase (not setup/teardown)
    if report.when != "call" or not report.failed:
        return

    # Build a human-readable filename from the test node id
    # e.g. tests/test_dynamics.py::TestNodeAvailability::test_node_starts_available
    #   → tests__test_dynamics__TestNodeAvailability__test_node_starts_available.txt
    safe_name = report.nodeid.replace("/", "__").replace("\\", "__").replace("::", "__")
    error_file = _results_dir / f"FAIL_{safe_name}.txt"

    lines = [
        f"TEST FAILED: {report.nodeid}",
        f"Duration:    {report.duration:.3f}s",
        "",
        "─" * 60,
        "Traceback / Error:",
        "─" * 60,
        report.longreprtext,
    ]
    error_file.write_text("\n".join(lines), encoding="utf-8")
