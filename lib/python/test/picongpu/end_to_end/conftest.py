import sys
from pathlib import Path

# Ensure the parent directory is in sys.path so relative imports work
test_dir = Path(__file__).parent
parent_dir = test_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
