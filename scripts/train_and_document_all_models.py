"""Forwarding wrapper for training/train_and_document_all_models.py."""
import sys
from pathlib import Path

# Add project root to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Forward to categorized script
import runpy

if __name__ == '__main__':
    target = Path(__file__).resolve().parent / 'training' / 'train_and_document_all_models.py'
    runpy.run_path(str(target), run_name='__main__')
