"""Temporary benchmark harness: representative subset of the same-start sweep.

Runs the same ProcessPoolExecutor pattern as run_sweep.main() but with a fixed
representative cell subset and no DB persistence. MAX_WORKERS comes from argv.

Usage: python bench/bench_workers.py <max_workers>
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from coverage_planner.experiments.run_sweep import run_sweep_cell

STEPS = 8
NUM_AGENTS = 7

def main():
    max_workers = int(sys.argv[1])
    cells = [
        (STEPS, n, chunk)
        for n in range(1, NUM_AGENTS + 1)
        for chunk in range(1, STEPS + 1)
    ]
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_sweep_cell, steps, n, chunk): (steps, n, chunk)
            for steps, n, chunk in cells
        }
        for fut in as_completed(futures):
            fut.result()
            done += 1
    elapsed = time.perf_counter() - t0
    print(f"BENCH_RESULT workers={max_workers} cells={done} wall_s={elapsed:.2f}")

if __name__ == "__main__":
    main()
