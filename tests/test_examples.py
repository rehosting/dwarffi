import subprocess
import sys
import os
from pathlib import Path
import pytest

# Find all .py files in the examples directory
EXAMPLE_DIR = Path(__file__).parent.parent / "examples"
EXAMPLES = list(EXAMPLE_DIR.glob("*.py"))

@pytest.mark.parametrize("example_path", EXAMPLES)
def test_example_execution(example_path):
    """Dynamically runs each example file as a test case."""
    res = subprocess.run([sys.executable, str(example_path)], capture_output=True, text=True)
    
    # Assert return code is 0 (Success)
    assert res.returncode == 0, f"Example {example_path.name} failed!\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"