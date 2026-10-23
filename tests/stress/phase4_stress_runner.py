"""Compatibility entry point: real assertions and a nonzero exit status on failure."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    raise SystemExit(subprocess.call([
        sys.executable, '-m', 'pytest', '-q',
        'tests/stress/test_media_stress.py', 'tests/unit/test_media.py',
        'tests/integration/test_pipelines_integration.py',
    ], cwd=ROOT))
