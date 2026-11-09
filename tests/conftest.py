"""Enable project-local imports when pytest is launched from any directory."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "python"), str(PROJECT)]
