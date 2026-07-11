"""Company X Crypto Discovery Hub - Setup Script (cross-platform, Gmail-safe)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


def run(cmd, step):
    print(f"\n[{step}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    print("=" * 40)
    print(" Company X Crypto Discovery Hub - Setup")
    print("=" * 40)

    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], "1/4")
    run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], "2/4")
    run([sys.executable, "scripts/train_pipeline.py"], "3/4")

    print("\n[4/4] Setup complete!")
    print("Start the app with:  python run.py")
    print("Then open:           http://localhost:8081")


if __name__ == "__main__":
    main()
