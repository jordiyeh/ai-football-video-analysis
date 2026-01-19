#!/usr/bin/env python3
"""
Launch the Veo Soccer Analysis web UI.

Usage:
    python run_ui.py [runs_directory]
    python run_ui.py runs/
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    from src.ui.server import main

    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    main(runs_dir=runs_dir)
