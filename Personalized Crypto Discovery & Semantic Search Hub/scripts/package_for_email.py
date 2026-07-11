"""
Create a Gmail-safe ZIP for sharing this project.

Gmail blocks ZIPs that contain:
  - Script extensions (.ps1, .bat, .js, .sh, etc.)
  - HTML files with embedded JavaScript (frontend/index.html)
  - Binary ML files often flagged by scanners (.pt, .faiss, .lgb, .joblib, .gensim)

This script packages ONLY safe files. Recipients run: python setup.py
"""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ZIP = ROOT / "Project-2-GmailSafe.zip"
STAGING = ROOT / "_email_staging"

# Extensions safe for Gmail attachments
SAFE_EXTENSIONS = {
    ".py", ".json", ".md", ".txt", ".png", ".gitignore",
}

# Directories to include (artifacts excluded — regenerate via setup.py)
INCLUDE_DIRS = ["src", "data", "scripts", "frontend", "outputs"]

# Individual root files
INCLUDE_ROOT_FILES = [
    "config.py", "run.py", "setup.py", "requirements.txt",
    "README.md", ".gitignore", "EMAIL_INSTRUCTIONS.txt",
]

# Skip these inside included dirs
SKIP_NAMES = {"__pycache__", ".venv", "venv", "env", "artifacts", "_email_staging"}
SKIP_EXTENSIONS = {
    ".html", ".htm", ".pt", ".pth", ".faiss", ".lgb", ".joblib",
    ".gensim", ".npz", ".npy", ".pyc", ".pyo", ".exe", ".bat",
    ".cmd", ".ps1", ".sh", ".js", ".vbs", ".msi", ".dll",
    ".id2word", ".state",
}


def should_include(path: Path) -> bool:
    if any(part in SKIP_NAMES for part in path.parts):
        return False
    ext = path.suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return False
    # Skip large optional JSON
    if path.name == "demo_results.json":
        return False
    if ext in SAFE_EXTENSIONS or path.name == ".gitignore":
        return True
    return False


def stage_frontend_html(staging: Path):
    """Ship frontend as .txt so Gmail does not flag JS inside HTML."""
    src = ROOT / "frontend" / "index.html"
    if src.exists():
        dst_dir = staging / "frontend"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "index.html.txt")


def build_zip():
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir()

    # Root files
    for name in INCLUDE_ROOT_FILES:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, STAGING / name)

    # Directory files
    for dir_name in INCLUDE_DIRS:
        src_dir = ROOT / dir_name
        if not src_dir.exists():
            continue
        for path in src_dir.rglob("*"):
            if path.is_file() and should_include(path.relative_to(ROOT)):
                rel = path.relative_to(ROOT)
                dest = STAGING / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)

    stage_frontend_html(STAGING)

    # Restore script inside package
    restore = ROOT / "scripts" / "restore_after_email.py"
    if restore.exists():
        shutil.copy2(restore, STAGING / "scripts" / "restore_after_email.py")

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in STAGING.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(STAGING))

    try:
        shutil.rmtree(STAGING)
    except PermissionError:
        pass

    size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Created: {OUT_ZIP}")
    print(f"Size:    {size_mb:.2f} MB")
    print("\nSafe contents: .py .json .md .txt .png only")
    print("Excluded: artifacts/, *.html, *.pt, *.faiss, *.lgb, *.joblib, etc.")
    print("\nRecipient steps:")
    print("  1. Unzip")
    print("  2. python scripts/restore_after_email.py")
    print("  3. python setup.py")
    print("  4. python run.py")


if __name__ == "__main__":
    build_zip()
