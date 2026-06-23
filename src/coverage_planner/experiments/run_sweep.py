import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from coverage_planner.experiments.config import (
    NUM_AGENTS,
    MAX_SIZE,
    SOLVE_OPTIMAL,
    STEP_SIZES,
    SWEEP_NAME,
    MAX_WORKERS,
)
from coverage_planner.env import Agent, EmptyResult
from coverage_planner.experiments import storage
from coverage_planner.experiments.run_single import run_single_compare
from coverage_planner.metrics import CompareSweep
from coverage_planner.policies.greedy import full_horizon_greedy_solve, rolling_horizon_greedy_solve

from tqdm.auto import tqdm

#=====All agents start at same location======
def run_all_start_positions(num_agents, size, steps, chunksize, run_optimal=True):
    full_results = []
    per_start = []
    for row in range(size):
        for c in range(size):
            Agent._id_counter = 0
            agents = [Agent(row, c) for _ in range(num_agents)]
            agent_order = [agent.id for agent in agents]
            single_result = run_single_compare(agents, agent_order, size,
                            steps, chunksize, run_optimal)
            full_results += single_result
            per_start.append((row, c, single_result))

    return CompareSweep(full_results), per_start


def run_sweep_cell(steps, n, chunk):
    '''Run a single sweep cell over all start positions and return plain,
    picklable result data.

    Returns:
        new_row (dict): aggregate ratios for this (steps, agents, chunksize) cell.
        raw_rows_cell (list[dict]): per-method, per-start-cell raw scores/runtimes.
    '''
    results, per_start = run_all_start_positions(
        n,
        MAX_SIZE,
        steps,
        chunksize=chunk,
        run_optimal=SOLVE_OPTIMAL,
    )

    raw_rows_cell = []
    for start_row, start_col, single in per_start:
        for res in single:
            if isinstance(res, EmptyResult):
                continue
            method_name = (
                res.method.__name__
                if callable(res.method) else str(res.method)
            )
            raw_rows_cell.append({
                "agents": n,
                "steps": steps,
                "chunksize": chunk,
                "start_row": start_row,
                "start_col": start_col,
                "method": method_name,
                "score": float(res.score),
                "runtime": float(res.runtime),
            })

    summary = results.summary(reference_method=full_horizon_greedy_solve.__name__,
                              include_baseline=SOLVE_OPTIMAL)

    split_ratio_to_seq_min_score = summary.loc[
        rolling_horizon_greedy_solve.__name__,
        ("score", "min_ratio_to_reference"),
    ]
    split_ratio_to_seq_mean_score = summary.loc[
        rolling_horizon_greedy_solve.__name__,
        ("score", "mean_ratio_to_reference"),
    ]

    split_ratio_to_seq_max_runtime = summary.loc[
        rolling_horizon_greedy_solve.__name__,
        ("runtime", "max_ratio_to_reference"),
    ]
    split_ratio_to_seq_mean_runtime = summary.loc[
        rolling_horizon_greedy_solve.__name__,
        ("runtime", "mean_ratio_to_reference"),
    ]

    new_row = {
        "agents": n,
        "steps": steps,
        "chunksize": chunk,
        "split_ratio_to_seq_min": split_ratio_to_seq_min_score,
        "split_ratio_to_seq_mean": split_ratio_to_seq_mean_score,
        "split_ratio_to_seq_max_runtime": split_ratio_to_seq_max_runtime,
        "split_ratio_to_seq_mean_runtime": split_ratio_to_seq_mean_runtime,
    }

    if SOLVE_OPTIMAL:
        seq_ratio_to_baseline_max_runtime = summary.loc[
            full_horizon_greedy_solve.__name__,
            ("runtime", "max_ratio_to_baseline"),
        ]

        seq_ratio_to_baseline_min_score = summary.loc[
            full_horizon_greedy_solve.__name__,
            ("score", "min_ratio_to_baseline"),
        ]
        split_ratio_to_baseline_max_runtime = summary.loc[
            rolling_horizon_greedy_solve.__name__,
            ("runtime", "max_ratio_to_baseline"),
        ]
        split_ratio_to_baseline_min_score = summary.loc[
            rolling_horizon_greedy_solve.__name__,
            ("score", "min_ratio_to_baseline"),
        ]
        new_row |= {
            "split_ratio_to_baseline": split_ratio_to_baseline_min_score,
            "seq_ratio_to_baseline": seq_ratio_to_baseline_min_score,
            "split_ratio_to_baseline_runtime": split_ratio_to_baseline_max_runtime,
            "seq_ratio_to_baseline_runtime": seq_ratio_to_baseline_max_runtime,
        }

    return new_row, raw_rows_cell


def main():
    rows = []
    raw_rows = []

    total_runs = sum(
        NUM_AGENTS * steps
        for steps in STEP_SIZES
    )

    cells = [
        (steps, n, chunk)
        for steps in STEP_SIZES
        for n in range(1, NUM_AGENTS + 1)
        for chunk in range(1, steps + 1)
    ]

    max_workers = MAX_WORKERS or os.cpu_count()

    with tqdm(total=total_runs, desc="Running same-start sweep") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_sweep_cell, steps, n, chunk): (steps, n, chunk)
                for steps, n, chunk in cells
            }
            for fut in as_completed(futures):
                new_row, raw_rows_cell = fut.result()
                rows.append(new_row)
                raw_rows.extend(raw_rows_cell)

                steps, n, chunk = futures[fut]
                pbar.set_postfix({
                    "agents": n,
                    "steps": steps,
                    "chunk": chunk,
                })
                pbar.update(1)

    # Deterministic ordering for DB inserts (completion order is nondeterministic).
    rows.sort(key=lambda r: (r["steps"], r["agents"], r["chunksize"]))
    raw_rows.sort(key=lambda r: (
        r["steps"], r["agents"], r["chunksize"],
        r["start_row"], r["start_col"], r["method"],
    ))

    print(
        f"Persisting {len(rows)} aggregate rows and "
        f"{len(raw_rows)} raw result rows to {storage.DEFAULT_DB_PATH}..."
    )

    with storage.connect() as conn:
        storage.init_schema(conn)
        sweep_id = storage.create_sweep(
            conn,
            name=SWEEP_NAME,
            grid_size=MAX_SIZE,
            max_agents=NUM_AGENTS,
            max_steps=MAX_SIZE,
            solve_optimal=SOLVE_OPTIMAL,
        )
        storage.insert_sweep_rows(conn, sweep_id, rows)
        storage.insert_sweep_results(conn, sweep_id, raw_rows)

    print(f"Done. Persisted sweep_id={sweep_id} to {storage.DEFAULT_DB_PATH}")


if __name__ == "__main__":
    main()
