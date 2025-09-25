import unittest
from pathlib import Path

# Ensure project root is the first entry on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

loader = unittest.defaultTestLoader
suite = loader.discover(start_dir=str(Path(__file__).resolve().parent), pattern="test_*.py", top_level_dir=str(PROJECT_ROOT))

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
