"""Discover optional local measured LUT data for tests only."""

import os
from pathlib import Path


if "KVOLT_LUT_BUNDLE" not in os.environ:
    workspace = Path(__file__).resolve().parents[2]
    for source in (workspace / "KVolt-support/KVolt-lut-source",
                   workspace / "KVolt-lut-source"):
        bundle = source / "data/luts/agx2-repeat5-20261103"
        if bundle.is_dir():
            os.environ["KVOLT_LUT_BUNDLE"] = str(bundle)
            break
