import sys
from pathlib import Path


DZ1_DIR = Path(__file__).resolve().parents[1]
if str(DZ1_DIR) not in sys.path:
    sys.path.insert(0, str(DZ1_DIR))
