#!/usr/bin/env python3
"""Attach optional INA3221 energy collection to a Python experiment."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

# Allow running this source-checkout example without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvolt_runtime.telemetry import EnergyMeter, PowerProfile


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError('duration must be finite and greater than zero')
    return seconds


def run_experiment(duration_s):
    """Replace this harmless sleep with the work whose energy you need."""
    time.sleep(duration_s)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--power-profile', type=Path,
                        help='explicit PowerProfile JSON; otherwise discover INA3221 labels')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='result directory; existing result files are never overwritten')
    parser.add_argument('--no-energy', action='store_true',
                        help='run the same workload without sensors, collector, or synchronization')
    parser.add_argument('--duration-s', type=positive_seconds, default=1.0,
                        help='sleep demo duration in seconds (default: 1.0)')
    args = parser.parse_args(argv)

    enabled = not args.no_energy
    profile = PowerProfile.from_json(args.power_profile) if enabled and args.power_profile else None
    # For asynchronous GPU work, pass its completion callback here, for example
    # synchronize=torch.cuda.synchronize after importing torch in your experiment.
    synchronize = None
    with EnergyMeter(profile, enabled=enabled, synchronize=synchronize) as meter:
        run_experiment(args.duration_s)
    meter.save(args.output_dir)
    print(json.dumps(meter.result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if not enabled or meter.result['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
