"""CLI dispatcher for rendering plots from a persisted sweep.

Examples:
    python -m coverage_planner.experiments.plot_sweep heatmap
    python -m coverage_planner.experiments.plot_sweep scatter --series-by method
    python -m coverage_planner.experiments.plot_sweep scatter --series-by chunksize \\
        --sweep-id 3
    python -m coverage_planner.experiments.plot_sweep efficiency --x-axis chunksize \\
        --agents 3 --steps 8
    python -m coverage_planner.experiments.plot_sweep efficiency \\
        --db-path results/same_start/sweeps.db
    python -m coverage_planner.experiments.plot_sweep efficiency --x-axis chunksize \\
        --steps 8 --animate-over agents --frame-duration 0.5
    python -m coverage_planner.experiments.plot_sweep pareto
    python -m coverage_planner.experiments.plot_sweep pareto --steps 8 \\
        --series-by agents
"""

from __future__ import annotations

import argparse

from coverage_planner.experiments import plotting, storage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render plots from a persisted sweep in results/sweeps.db.",
    )
    parser.add_argument(
        "kind",
        choices=["heatmap", "scatter", "efficiency", "pareto"],
        help="Plot kind to render.",
    )
    parser.add_argument(
        "--sweep-id",
        type=int,
        default=None,
        help="Specific sweep_id to plot. Defaults to the latest matching --name.",
    )
    parser.add_argument(
        "--name",
        default="same_start",
        help="Sweep name to look up when --sweep-id is not given.",
    )
    parser.add_argument(
        "--db-path",
        default=storage.DEFAULT_DB_PATH,
        help="SQLite database path to read, resolved from the project root.",
    )
    parser.add_argument(
        "--series-by",
        default=None,
        choices=list(plotting.PLOT_DIMENSION_COLUMNS),
        help=(
            "Column controlling color/series in scatter, efficiency, or "
            "pareto plots. Defaults to 'method' (scatter/efficiency) or "
            "'agents' (pareto)."
        ),
    )
    parser.add_argument(
        "--x-axis",
        default="agents",
        choices=list(plotting.PLOT_DIMENSION_COLUMNS),
        help="Column to use as the x-axis for efficiency plots.",
    )
    parser.add_argument(
        "--reference-method",
        default="full_horizon_greedy_solve",
        help="Reference method used as denominator for ratio plots.",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        help="Filter raw-result plots to one agent count.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Filter raw-result plots to one planning horizon.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Filter raw-result plots to one chunksize.",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Filter raw-result plots to one method name.",
    )
    parser.add_argument(
        "--animate-over",
        default=None,
        choices=list(plotting.PLOT_DIMENSION_COLUMNS),
        help=(
            "Render a looping GIF with one frame per value of this column "
            "(values taken from the data). For heatmaps only 'agents' is valid."
        ),
    )
    parser.add_argument(
        "--frame-duration",
        type=float,
        default=plotting.DEFAULT_FRAME_DURATION,
        help="Seconds each GIF frame is shown (only used with --animate-over).",
    )

    args = parser.parse_args()

    series_by = args.series_by
    if series_by is None:
        series_by = "agents" if args.kind == "pareto" else "method"

    if args.kind == "pareto":
        if args.animate_over is not None:
            parser.error("pareto plots do not support --animate-over.")
        if series_by == "chunksize":
            parser.error(
                "pareto plots trace chunksize along each curve; "
                "--series-by chunksize is not supported."
            )

    if args.animate_over is not None:
        if args.kind == "heatmap":
            if args.animate_over != "agents":
                parser.error(
                    "heatmap GIFs can only animate over 'agents' "
                    f"(got --animate-over {args.animate_over})."
                )
        else:
            if args.animate_over == args.x_axis and args.kind == "efficiency":
                parser.error(
                    "--animate-over must differ from --x-axis "
                    f"(both are {args.animate_over!r})."
                )
            if args.animate_over == series_by:
                parser.error(
                    "--animate-over must differ from --series-by "
                    f"(both are {args.animate_over!r})."
                )

    with storage.connect(args.db_path) as conn:
        if args.kind == "heatmap":
            sweep_id, meta, df = storage.load_sweep_df(
                conn, sweep_id=args.sweep_id, name=args.name,
            )
            print(
                f"Loaded sweep_id={sweep_id} (name={meta['name']}, "
                f"grid={meta['grid_size']}x{meta['grid_size']}, "
                f"rows={len(df)})"
            )
            if args.animate_over is not None:
                print(
                    "Rendering heatmap GIFs animated over agents "
                    f"(frame_duration={args.frame_duration}s)..."
                )
                out_dir = plotting.render_heatmap_gif(
                    df, meta, frame_duration=args.frame_duration,
                )
            else:
                print("Rendering heatmaps...")
                out_dir = plotting.render_heatmaps(df, meta)
        else:
            sweep_id, meta, raw_df = storage.load_sweep_raw_df(
                conn, sweep_id=args.sweep_id, name=args.name,
            )
            print(
                f"Loaded sweep_id={sweep_id} (name={meta['name']}, "
                f"grid={meta['grid_size']}x{meta['grid_size']}, "
                f"raw_rows={len(raw_df)})"
            )
            filters = {
                key: value
                for key, value in {
                    "agents": args.agents,
                    "steps": args.steps,
                    "chunksize": args.chunksize,
                    "method": args.method,
                }.items()
                if value is not None
            }

            if args.animate_over is not None and args.animate_over in filters:
                print(
                    f"Ignoring --{args.animate_over} filter; it is driven "
                    "per-frame by --animate-over."
                )
                filters = {
                    k: v for k, v in filters.items() if k != args.animate_over
                }

            if args.kind == "scatter":
                if args.animate_over is not None:
                    print(
                        f"Rendering scatter GIF (series_by={series_by}, "
                        f"animate_over={args.animate_over}, "
                        f"frame_duration={args.frame_duration}s)..."
                    )
                    out_dir = plotting.render_scatter_gif(
                        raw_df,
                        meta,
                        animate_over=args.animate_over,
                        series_by=series_by,
                        reference_method=args.reference_method,
                        filters=filters,
                        frame_duration=args.frame_duration,
                    )
                else:
                    print(f"Rendering scatter (series_by={series_by})...")
                    out_dir = plotting.render_scatter(
                        raw_df,
                        meta,
                        series_by=series_by,
                        reference_method=args.reference_method,
                        filters=filters,
                    )
            elif args.kind == "pareto":
                print(f"Rendering pareto (series_by={series_by})...")
                out_dir = plotting.render_pareto(
                    raw_df,
                    meta,
                    series_by=series_by,
                    reference_method=args.reference_method,
                    filters=filters,
                )
            else:
                if args.animate_over is not None:
                    print(
                        "Rendering efficiency GIF "
                        f"(x_axis={args.x_axis}, series_by={series_by}, "
                        f"animate_over={args.animate_over}, "
                        f"frame_duration={args.frame_duration}s)..."
                    )
                    out_dir = plotting.render_efficiency_gif(
                        raw_df,
                        meta,
                        animate_over=args.animate_over,
                        x_axis=args.x_axis,
                        series_by=series_by,
                        reference_method=args.reference_method,
                        filters=filters,
                        frame_duration=args.frame_duration,
                    )
                else:
                    print(
                        "Rendering efficiency "
                        f"(x_axis={args.x_axis}, series_by={series_by})..."
                    )
                    out_dir = plotting.render_efficiency_lines(
                        raw_df,
                        meta,
                        x_axis=args.x_axis,
                        series_by=series_by,
                        reference_method=args.reference_method,
                        filters=filters,
                    )

    print(f"Done. Wrote plots for sweep_id={sweep_id} to {out_dir}")


if __name__ == "__main__":
    main()
