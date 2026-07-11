"""Restore files after extracting Gmail-safe package."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    # Restore frontend HTML from .txt
    txt = ROOT / "frontend" / "index.html.txt"
    html = ROOT / "frontend" / "index.html"
    if txt.exists() and not html.exists():
        txt.rename(html)
        print(f"Restored: {html}")
    elif txt.exists() and html.exists():
        print(f"Already exists: {html} (delete index.html first to restore from .txt)")

    print("\nNext steps:")
    print("  python setup.py    # installs deps + trains models")
    print("  python run.py      # start server on http://localhost:8081")

if __name__ == "__main__":
    main()
