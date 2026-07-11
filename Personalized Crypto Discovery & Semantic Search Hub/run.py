"""Start the Crypto Discovery Hub API server."""
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import uvicorn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import API_HOST, API_PORT

if __name__ == "__main__":
    print("Starting Company X Crypto Discovery Hub...")
    print(f"Open http://localhost:{API_PORT} in your browser")
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=False)
