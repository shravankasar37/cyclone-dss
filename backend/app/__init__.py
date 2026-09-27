import os
import sys

# make the repo's shared `ml` package importable (repo root locally, /app in Docker)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
