"""Run the pipeline in order.

    python run_all.py                 every implemented step
    python run_all.py explore clean   only the named steps, in pipeline order
"""
from __future__ import annotations

import sys
import time

from src import clean, events, explore, runs, score_wells, sizing, train_eval

STEPS = {
    "explore": explore.main,
    "clean": clean.main,
    "events": events.main,
    "sizing": sizing.main,
    "runs": runs.main,
    "train_eval": train_eval.main,
    "score_wells": score_wells.main,
}


def main(argv: list[str]) -> None:
    unknown = [a for a in argv if a not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s): {', '.join(unknown)}. Choose from: {', '.join(STEPS)}")
    # Steps always run in pipeline order, whatever order they were typed in,
    # because each one reads the files written by the step before it.
    chosen = [s for s in STEPS if not argv or s in argv]
    t_all = time.time()
    for name in chosen:
        print(f"\n===== {name} =====")
        t0 = time.time()
        STEPS[name]()
        print(f"===== {name} done in {time.time() - t0:.1f}s =====")
    print(f"\nrun_all: {len(chosen)} step(s) in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
