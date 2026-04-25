#!/usr/bin/env python
"""Quick setup check and dependency installer."""

import subprocess
import sys
from pathlib import Path


def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def run_cmd(cmd):
    """Run a command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def main():
    LOG_FILE.write_text("")  # Clear log

    log("=== Setup Check ===")
    log(f"Python: {sys.version}")
    log(f"Executable: {sys.executable}")

    # Check data
    log("\n=== Checking Data ===")
    try:
        import pandas as pd
        for split in ["train", "val", "test"]:
            path = Path(f"data/annotations/{split}.parquet")
            if path.exists():
                df = pd.read_parquet(path)
                log(f"  {split}: {len(df):,} samples")
            else:
                log(f"  {split}: MISSING")
    except Exception as e:
        log(f"  Error: {e}")

    # Install missing packages
    log("\n=== Installing Missing Packages ===")
    packages = ["pytorch-crf", "seqeval"]
    for pkg in packages:
        log(f"  Installing {pkg}...")
        out = run_cmd(f"{sys.executable} -m pip install {pkg}")
        if "Successfully installed" in out:
            log(f"    Installed {pkg}")
        elif "already satisfied" in out.lower():
            log(f"    Already installed: {pkg}")
        else:
            log(f"    Status: {out[:100]}")

    # Verify imports
    log("\n=== Verifying Imports ===")
    imports = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("seqeval", "seqeval"),
        ("pandas", "pandas"),
    ]
    for module, pkg in imports:
        try:
            __import__(module)
            log(f"  ✓ {module}")
        except ImportError as e:
            log(f"  ✗ {module}: {e}")

    # Special check for CRF (can be torchcrf or TorchCRF)
    try:
        from torchcrf import CRF
        log(f"  ✓ torchcrf (CRF)")
    except ImportError:
        try:
            from TorchCRF import CRF
            log(f"  ✓ TorchCRF (CRF)")
        except ImportError as e:
            log(f"  ✗ CRF module: {e}")

    log("\n=== Done ===")
    log(f"See {LOG_FILE} for full log")

if __name__ == "__main__":
    main()


