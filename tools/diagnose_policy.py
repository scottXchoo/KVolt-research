#!/usr/bin/env python3
"""Run the CPU policy diagnostic directly from an uninstalled source checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kvolt_runtime.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['diagnose', *sys.argv[1:]]))
