"""Rendering helpers for persisted sweep data.

Top-level entry points:
    - render_heatmaps(df, meta, output_root)  -> per-agent ratio heatmaps.
    - render_scatter(raw_df, meta, ...)       -> score-vs-runtime scatter
                                                 of methods relative to a
                                                 reference method.
    - render_efficiency_lines(raw_df, meta, ...)
                                               -> score/runtime ratio lines
                                                  relative to a reference
                                                  method.
    - render_pareto(raw_df, meta, ...)        -> score-vs-runtime Pareto
                                                 tradeoff curve traced over
                                                 chunksize, relative to a
                                                 reference method.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import FuncNorm, TwoSlopeNorm
from matplotlib.figure import Figure
from PIL import Image


SCATTER_SERIES_COLUMNS: tuple[str, ...] = (
    "method",
    "chunksize",
    "agents",
    "steps",
)

PLOT_DIMENSION_COLUMNS = SCATTER_SERIES_COLUMNS

DEFAULT_FRAME_DURATION = 0.5

DEFAULT_PARETO_METHOD = "rolling_horizon_greedy_solve"


def _figure_to_png_bytes(fig: Figure, *, dpi: int = 200) -> bytes:
    """Render a figure to PNG bytes (used to build GIF frames)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _save_gif(
    frames: list[bytes],
    out_path: Path,
    frame_duration: float = DEFAULT_FRAME_DURATION,
) -> None:
    """Encode a list of PNG byte buffers into a looping GIF.

    Frames are padded to a common canvas size so that minor per-frame
    bounding-box differences (e.g. tick-label widths) do not make the
    animation jitter.
    """
    if not frames:
        raise ValueError("No frames to encode into a GIF.")

    images = [Image.open(io.BytesIO(buf)).convert("RGBA") for buf in frames]

    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)

    padded: list[Image.Image] = []
    for im in images:
        if im.size == (max_w, max_h):
            padded.append(im)
            continue
        canvas = Image.new("RGBA", (max_w, max_h), (255, 255, 255, 255))
        offset = ((max_w - im.width) // 2, (max_h - im.height) // 2)
        canvas.paste(im, offset, im)
        padded.append(canvas)

    rgb_frames = [im.convert("P", palette=Image.ADAPTIVE) for im in padded]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames[0].save(
        out_path,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=int(frame_duration * 1000),
        loop=0,
        disposal=2,
    )


def centered_ratio_norm(series_list, center=1.0, min_dev=0.05):
    """
    Build a TwoSlopeNorm centered at 1.0.

    min_dev prevents vmin == vcenter == vmax when all ratios are exactly 1.
    """
    values = pd.concat(
        [s.dropna() for s in series_list],
        ignore_index=True,
    )

    if values.empty:
        max_dev = min_dev
    else:
        max_dev = np.nanmax(np.abs(values.to_numpy() - center))
        max_dev = max(max_dev, min_dev)

    return TwoSlopeNorm(
        vmin=max(0.5, center - max_dev),
        vcenter=center,
        vmax=min(3, center + max_dev),
    )


def centered_ratio_log_norm(series_list, center=1.0, min_factor=1.05, max_factor=3.0):
    """
    Build a logarithmic norm centered at `center`.

    Equal multiplicative changes around center get equal color distance:
        0.5 and 2.0 are equally far from 1.0.

    All plotted values must be positive.
    """
    if center <= 0:
        raise ValueError("center must be positive for logarithmic normalization")

    values = pd.concat(
        [s.dropna() for s in series_list],
        ignore_index=True,
    )

    values = values[np.isfinite(values) & (values > 0)]

    if values.empty:
        max_log_dev = np.log(min_factor)
    else:
        log_devs = np.abs(np.log(values.to_numpy() / center))
        max_log_dev = np.nanmax(log_devs)
        max_log_dev = max(max_log_dev, np.log(min_factor))

    max_log_dev = min(max_log_dev, np.log(max_factor))

    vmin = center * np.exp(-max_log_dev)
    vmax = center * np.exp(max_log_dev)

    return FuncNorm(
        (
            lambda x: np.log(x / center),
            lambda y: center * np.exp(y),
        ),
        vmin=vmin,
        vmax=vmax,
    )


def make_scale_note(norm):
    return (
        f"Colors use a log-centered scale at 1.00; "
        f"displayed range clipped to [{norm.vmin:.2f}, {norm.vmax:.2f}]."
    )


def add_centered_colorbar(fig, ax, im, norm, label):
    cbar = fig.colorbar(im, ax=ax, label=label)

    ticks = [norm.vmin, 1.0, norm.vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([
        f"{norm.vmin:.2f}",
        "1.00",
        f"{norm.vmax:.2f}",
    ])

    return cbar


def _draw_ratio_heatmap(
    fig,
    ax,
    df,
    *,
    agents,
    value_col,
    title,
    colorbar_label,
    cmap,
    norm,
    index_values=None,
    column_values=None,
):
    """Draw a single ratio heatmap onto a provided fig/ax.

    ``index_values``/``column_values`` pin the steps/chunksize axes so that
    every frame of an animation shares identical ticks even when a given
    agent count is missing some cells.
    """
    df_n = df[df["agents"] == agents]

    heat = df_n.pivot(
        index="steps",
        columns="chunksize",
        values=value_col,
    )

    if index_values is not None:
        heat = heat.reindex(index=index_values)
    if column_values is not None:
        heat = heat.reindex(columns=column_values)

    masked_heat = np.ma.masked_invalid(heat.to_numpy(dtype=float))

    im = ax.imshow(
        masked_heat,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=norm,
    )

    ax.set_xlabel("chunksize")
    ax.set_ylabel("steps")
    ax.set_title(title)

    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns)

    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)

    add_centered_colorbar(fig, ax, im, norm, colorbar_label)

    fig.text(
        0.5,
        0.01,
        make_scale_note(norm),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])


def plot_ratio_heatmap(
    df,
    *,
    agents,
    value_col,
    title,
    colorbar_label,
    filename,
    cmap,
    norm,
    save_dir,
):
    fig, ax = plt.subplots()

    _draw_ratio_heatmap(
        fig,
        ax,
        df,
        agents=agents,
        value_col=value_col,
        title=title,
        colorbar_label=colorbar_label,
        cmap=cmap,
        norm=norm,
    )

    plt.savefig(
        save_dir / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def _heatmap_specs(df: pd.DataFrame) -> list[dict]:
    """Build the shared config for the standard 4-heatmap family.

    Each spec carries the value column, color map/norm, labels, and the
    filename stub used for both the static PNGs and the animated GIFs.
    """
    score_norm = centered_ratio_log_norm([
        df["split_ratio_to_seq_min"],
        df["split_ratio_to_seq_mean"],
    ])

    runtime_norm = centered_ratio_log_norm([
        df["split_ratio_to_seq_max_runtime"],
        df["split_ratio_to_seq_mean_runtime"],
    ])

    score_cmap = plt.get_cmap("RdYlGn").copy()
    score_cmap.set_bad(color="lightgray")

    runtime_cmap = plt.get_cmap("RdYlGn_r").copy()
    runtime_cmap.set_bad(color="lightgray")

    score_label = "Score ratio (split / greedy)\n1 = equal, >1 split better, <1 split worse"
    runtime_label = "Runtime ratio (split / greedy)\n1 = equal, >1 split slower, <1 split faster"

    return [
        {
            "value_col": "split_ratio_to_seq_min",
            "title_prefix": "Split vs Greedy Min Performance by Chunksize",
            "colorbar_label": score_label,
            "filename_stub": "split_min_perf_heatmap",
            "cmap": score_cmap,
            "norm": score_norm,
        },
        {
            "value_col": "split_ratio_to_seq_mean",
            "title_prefix": "Split vs Greedy Mean Performance by Chunksize",
            "colorbar_label": score_label,
            "filename_stub": "split_mean_perf_heatmap",
            "cmap": score_cmap,
            "norm": score_norm,
        },
        {
            "value_col": "split_ratio_to_seq_max_runtime",
            "title_prefix": "Split vs Greedy Max Runtime by Chunksize",
            "colorbar_label": runtime_label,
            "filename_stub": "split_max_runtime_heatmap",
            "cmap": runtime_cmap,
            "norm": runtime_norm,
        },
        {
            "value_col": "split_ratio_to_seq_mean_runtime",
            "title_prefix": "Split vs Greedy Mean Runtime by Chunksize",
            "colorbar_label": runtime_label,
            "filename_stub": "split_mean_runtime_heatmap",
            "cmap": runtime_cmap,
            "norm": runtime_norm,
        },
    ]


def render_heatmaps(
    df: pd.DataFrame,
    meta: dict,
    output_root: Path = Path("results"),
) -> Path:
    """Render the standard 4-heatmap family for every agent count in the sweep."""
    grid_size = int(meta["grid_size"])
    max_agents = int(meta["max_agents"])
    name = meta["name"]

    grid_dir = output_root / name / f"grid_{grid_size}x{grid_size}"
    grid_dir.mkdir(parents=True, exist_ok=True)

    specs = _heatmap_specs(df)

    for n in range(1, max_agents + 1):
        for spec in specs:
            plot_ratio_heatmap(
                df,
                agents=n,
                value_col=spec["value_col"],
                title=f"{spec['title_prefix']} — agents={n}",
                colorbar_label=spec["colorbar_label"],
                filename=(
                    f"{name}__{spec['filename_stub']}"
                    f"__agents_{n}"
                    f"__grid_{grid_size}x{grid_size}.png"
                ),
                cmap=spec["cmap"],
                norm=spec["norm"],
                save_dir=grid_dir,
            )

    return grid_dir


def render_heatmap_gif(
    df: pd.DataFrame,
    meta: dict,
    *,
    output_root: Path = Path("results"),
    frame_duration: float = DEFAULT_FRAME_DURATION,
) -> Path:
    """Render the 4-heatmap family as GIFs animated over agent count.

    Produces one GIF per heatmap variant (4 total). The steps/chunksize axes
    and color norms are shared across frames so the animation stays aligned.
    """
    grid_size = int(meta["grid_size"])
    max_agents = int(meta["max_agents"])
    name = meta["name"]

    grid_dir = output_root / name / f"grid_{grid_size}x{grid_size}"
    grid_dir.mkdir(parents=True, exist_ok=True)

    index_values = _sorted_unique(df["steps"])
    column_values = _sorted_unique(df["chunksize"])
    agent_values = list(range(1, max_agents + 1))

    for spec in _heatmap_specs(df):
        frames: list[bytes] = []
        for n in agent_values:
            fig, ax = plt.subplots()
            _draw_ratio_heatmap(
                fig,
                ax,
                df,
                agents=n,
                value_col=spec["value_col"],
                title=f"{spec['title_prefix']} — agents={n}",
                colorbar_label=spec["colorbar_label"],
                cmap=spec["cmap"],
                norm=spec["norm"],
                index_values=index_values,
                column_values=column_values,
            )
            frames.append(_figure_to_png_bytes(fig))

        filename = (
            f"{name}__{spec['filename_stub']}"
            f"__anim_agents"
            f"__grid_{grid_size}x{grid_size}.gif"
        )
        _save_gif(frames, grid_dir / filename, frame_duration=frame_duration)

    return grid_dir


def _filters_suffix(filters: dict | None) -> str:
    if not filters:
        return ""
    parts = [f"{k}_{v}" for k, v in sorted(filters.items())]
    return "__" + "_".join(parts)


def _apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, value in filters.items():
        if col not in df.columns:
            raise ValueError(
                f"Cannot filter on unknown column {col!r}. "
                f"Available columns: {list(df.columns)}"
            )
        mask &= df[col] == value
    return df[mask]


def _split_method_filter(filters: dict | None) -> tuple[dict | None, str | None]:
    if not filters or "method" not in filters:
        return filters, None

    non_method_filters = {
        key: value for key, value in filters.items() if key != "method"
    }
    return non_method_filters or None, filters["method"]


def _ratio_axis_bounds(values: np.ndarray, *, max_factor: float = 10.0) -> tuple[float, float]:
    """Symmetric log bounds around 1.0, clipped to [1/max_factor, max_factor]."""
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return (1.0 / max_factor, max_factor)

    log_dev = np.max(np.abs(np.log(finite)))
    log_dev = min(log_dev, np.log(max_factor))
    factor = float(np.exp(log_dev)) if log_dev > 0 else 1.05
    return (1.0 / factor, factor)


def _sorted_unique(series: pd.Series) -> list:
    values = list(series.dropna().unique())
    if pd.api.types.is_numeric_dtype(series):
        return sorted(values)
    return sorted(values, key=lambda v: str(v))


def _efficiency_ratios(
    raw_df: pd.DataFrame,
    *,
    x_axis: str = "agents",
    series_by: str = "method",
    reference_method: str = "full_horizon_greedy_solve",
    filters: dict | None = None,
) -> pd.DataFrame:
    if x_axis not in PLOT_DIMENSION_COLUMNS:
        raise ValueError(
            f"x_axis={x_axis!r} not supported. "
            f"Choose one of {PLOT_DIMENSION_COLUMNS}."
        )
    if series_by not in PLOT_DIMENSION_COLUMNS:
        raise ValueError(
            f"series_by={series_by!r} not supported. "
            f"Choose one of {PLOT_DIMENSION_COLUMNS}."
        )
    if x_axis == series_by:
        raise ValueError("x_axis and series_by must be different.")
    if raw_df.empty:
        raise ValueError("raw_df is empty; nothing to plot.")

    non_method_filters, method_filter = _split_method_filter(filters)
    df = _apply_filters(raw_df, non_method_filters).copy()
    if df.empty:
        raise ValueError(f"No rows left after applying filters={filters!r}.")

    methods_present = set(df["method"].unique())
    if reference_method not in methods_present:
        raise ValueError(
            f"reference_method={reference_method!r} not found in raw data. "
            f"Available methods: {sorted(methods_present)}"
        )

    df = df[
        np.isfinite(df["score"])
        & np.isfinite(df["runtime"])
        & (df["score"] > 0)
        & (df["runtime"] > 0)
    ].copy()

    join_keys = ["agents", "steps", "chunksize", "start_row", "start_col"]
    reference = (
        df[df["method"] == reference_method]
        .rename(columns={"score": "ref_score", "runtime": "ref_runtime"})
        [join_keys + ["ref_score", "ref_runtime"]]
    )

    merged = df.merge(reference, on=join_keys, how="inner")
    if method_filter is not None:
        merged = merged[merged["method"] == method_filter].copy()

    merged["efficiency_ratio"] = (
        (merged["score"] / merged["runtime"])
        / (merged["ref_score"] / merged["ref_runtime"])
    )

    merged = merged[
        np.isfinite(merged["efficiency_ratio"])
        & (merged["efficiency_ratio"] > 0)
    ].copy()

    if merged.empty:
        raise ValueError(
            "No finite, positive efficiency ratios left to plot (check for "
            "zero or missing scores/runtimes in the plotted or reference "
            "method rows)."
        )

    return merged


def _efficiency_summary(
    raw_df: pd.DataFrame,
    *,
    x_axis: str = "agents",
    series_by: str = "method",
    reference_method: str = "full_horizon_greedy_solve",
    filters: dict | None = None,
) -> pd.DataFrame:
    ratios = _efficiency_ratios(
        raw_df,
        x_axis=x_axis,
        series_by=series_by,
        reference_method=reference_method,
        filters=filters,
    )

    summary = (
        ratios.groupby([series_by, x_axis])["efficiency_ratio"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    return summary


def _efficiency_title(
    meta: dict,
    *,
    x_axis: str,
    series_by: str,
    reference_method: str,
    filters: dict | None,
) -> str:
    title_bits = [
        f"Score/Runtime Efficiency ({meta['name']}, "
        f"grid {meta['grid_size']}x{meta['grid_size']})",
        f"x={x_axis}, series={series_by}, reference={reference_method}",
    ]
    if filters:
        filt_str = ", ".join(f"{k}={v}" for k, v in sorted(filters.items()))
        title_bits.append(f"filters: {filt_str}")
    return "\n".join(title_bits)


def _draw_efficiency_lines(
    ax,
    summary: pd.DataFrame,
    *,
    x_axis: str,
    series_by: str,
    title: str,
    x_values=None,
    series_values=None,
    x_is_numeric: bool | None = None,
    y_bounds: tuple[float, float] | None = None,
) -> None:
    """Draw efficiency lines onto ``ax``.

    Passing ``x_values``/``series_values``/``y_bounds`` explicitly pins the
    axes, x-tick categories, and per-series colors so that every frame of an
    animation stays visually aligned even when a frame is missing some series.
    """
    cmap = plt.get_cmap("tab10")

    if x_values is None:
        x_values = _sorted_unique(summary[x_axis])
    if series_values is None:
        series_values = _sorted_unique(summary[series_by])
    if x_is_numeric is None:
        x_is_numeric = pd.api.types.is_numeric_dtype(summary[x_axis])

    if x_is_numeric:
        x_positions = {value: value for value in x_values}
    else:
        x_positions = {value: i for i, value in enumerate(x_values)}

    for i, value in enumerate(series_values):
        subset = summary[summary[series_by] == value].copy()
        subset["_x_pos"] = subset[x_axis].map(x_positions)
        subset = subset.sort_values("_x_pos")

        x = subset["_x_pos"].to_numpy(dtype=float)
        mean = subset["mean"].to_numpy(dtype=float)
        min_y = subset["min"].to_numpy(dtype=float)
        max_y = subset["max"].to_numpy(dtype=float)
        color = cmap(i % cmap.N)

        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=1.6,
            label=f"{series_by}={value}",
            color=color,
        )
        ax.fill_between(x, min_y, max_y, color=color, alpha=0.16, linewidth=0)

    ax.axhline(1.0, linestyle="--", linewidth=0.8, color="gray")
    ax.set_yscale("log")

    if y_bounds is None:
        y_bounds = _ratio_axis_bounds(
            summary[["mean", "min", "max"]].to_numpy().ravel()
        )
    ax.set_ylim(*y_bounds)

    if not x_is_numeric:
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([str(value) for value in x_values], rotation=30, ha="right")

    ax.set_xlabel(x_axis)
    ax.set_ylabel("Efficiency ratio ((score/runtime) / reference)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)


def render_efficiency_lines(
    raw_df: pd.DataFrame,
    meta: dict,
    *,
    x_axis: str = "agents",
    series_by: str = "method",
    reference_method: str = "full_horizon_greedy_solve",
    output_root: Path = Path("results"),
    filters: dict | None = None,
) -> Path:
    """Render normalized score/runtime efficiency lines across sweep axes."""
    summary = _efficiency_summary(
        raw_df,
        x_axis=x_axis,
        series_by=series_by,
        reference_method=reference_method,
        filters=filters,
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _draw_efficiency_lines(
        ax,
        summary,
        x_axis=x_axis,
        series_by=series_by,
        title=_efficiency_title(
            meta,
            x_axis=x_axis,
            series_by=series_by,
            reference_method=reference_method,
            filters=filters,
        ),
    )

    grid_size = int(meta["grid_size"])
    grid_dir = output_root / meta["name"] / f"grid_{grid_size}x{grid_size}"
    grid_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{meta['name']}__efficiency__x_{x_axis}"
        f"__seriesby_{series_by}"
        f"{_filters_suffix(filters)}"
        f"__grid_{grid_size}x{grid_size}.png"
    )

    fig.tight_layout()
    fig.savefig(grid_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return grid_dir


def render_efficiency_gif(
    raw_df: pd.DataFrame,
    meta: dict,
    *,
    animate_over: str,
    x_axis: str = "agents",
    series_by: str = "method",
    reference_method: str = "full_horizon_greedy_solve",
    output_root: Path = Path("results"),
    filters: dict | None = None,
    frame_duration: float = DEFAULT_FRAME_DURATION,
) -> Path:
    """Render an efficiency-line GIF, one frame per value of ``animate_over``.

    Axis bounds, x-tick categories, and per-series colors are computed once
    over the full (filtered) data so the animation does not jitter.
    """
    if animate_over not in PLOT_DIMENSION_COLUMNS:
        raise ValueError(
            f"animate_over={animate_over!r} not supported. "
            f"Choose one of {PLOT_DIMENSION_COLUMNS}."
        )
    if animate_over in (x_axis, series_by):
        raise ValueError(
            "animate_over must differ from both x_axis and series_by "
            f"(got animate_over={animate_over!r}, x_axis={x_axis!r}, "
            f"series_by={series_by!r})."
        )

    base_filters = {
        key: value
        for key, value in (filters or {}).items()
        if key != animate_over
    }

    global_summary = _efficiency_summary(
        raw_df,
        x_axis=x_axis,
        series_by=series_by,
        reference_method=reference_method,
        filters=base_filters or None,
    )

    ratios = _efficiency_ratios(
        raw_df,
        x_axis=x_axis,
        series_by=series_by,
        reference_method=reference_method,
        filters=base_filters or None,
    )
    frame_values = _sorted_unique(ratios[animate_over])
    if not frame_values:
        raise ValueError(
            f"No values of {animate_over!r} available to animate over."
        )

    x_values = _sorted_unique(global_summary[x_axis])
    series_values = _sorted_unique(global_summary[series_by])
    x_is_numeric = pd.api.types.is_numeric_dtype(global_summary[x_axis])
    y_bounds = _ratio_axis_bounds(
        global_summary[["mean", "min", "max"]].to_numpy().ravel()
    )

    frames: list[bytes] = []
    for value in frame_values:
        frame_filters = {**base_filters, animate_over: value}
        try:
            summary = _efficiency_summary(
                raw_df,
                x_axis=x_axis,
                series_by=series_by,
                reference_method=reference_method,
                filters=frame_filters,
            )
        except ValueError:
            summary = global_summary.iloc[0:0]

        fig, ax = plt.subplots(figsize=(8, 5.5))
        _draw_efficiency_lines(
            ax,
            summary,
            x_axis=x_axis,
            series_by=series_by,
            title=_efficiency_title(
                meta,
                x_axis=x_axis,
                series_by=series_by,
                reference_method=reference_method,
                filters=frame_filters,
            ),
            x_values=x_values,
            series_values=series_values,
            x_is_numeric=x_is_numeric,
            y_bounds=y_bounds,
        )
        fig.tight_layout()
        frames.append(_figure_to_png_bytes(fig))

    grid_size = int(meta["grid_size"])
    grid_dir = output_root / meta["name"] / f"grid_{grid_size}x{grid_size}"

    filename = (
        f"{meta['name']}__efficiency__x_{x_axis}"
        f"__seriesby_{series_by}"
        f"__anim_{animate_over}"
        f"{_filters_suffix(base_filters or None)}"
        f"__grid_{grid_size}x{grid_size}.gif"
    )

    _save_gif(frames, grid_dir / filename, frame_duration=frame_duration)
    return grid_dir


def _scatter_ratios(
    raw_df: pd.DataFrame,
    *,
    series_by: str,
    reference_method: str,
    filters: dict | None,
) -> pd.DataFrame:
    """Compute per-result runtime/score ratios against the reference method."""
    if raw_df.empty:
        raise ValueError("raw_df is empty; nothing to plot.")

    non_method_filters, method_filter = _split_method_filter(filters)
    df = _apply_filters(raw_df, non_method_filters)
    if df.empty:
        raise ValueError(f"No rows left after applying filters={filters!r}.")

    methods_present = set(df["method"].unique())
    if reference_method not in methods_present:
        raise ValueError(
            f"reference_method={reference_method!r} not found in raw data. "
            f"Available methods: {sorted(methods_present)}"
        )

    join_keys = ["agents", "steps", "chunksize", "start_row", "start_col"]

    reference = (
        df[df["method"] == reference_method]
        .rename(columns={"score": "ref_score", "runtime": "ref_runtime"})
        [join_keys + ["ref_score", "ref_runtime"]]
    )

    if method_filter is None:
        non_ref = df[df["method"] != reference_method].copy()
    else:
        non_ref = df[df["method"] == method_filter].copy()

    if non_ref.empty:
        raise ValueError(
            f"After filtering, only the reference method "
            f"({reference_method!r}) is present; nothing to plot."
        )

    merged = non_ref.merge(reference, on=join_keys, how="inner")

    merged["runtime_ratio"] = merged["runtime"] / merged["ref_runtime"]
    merged["score_ratio"] = merged["score"] / merged["ref_score"]

    merged = merged[
        np.isfinite(merged["runtime_ratio"])
        & np.isfinite(merged["score_ratio"])
        & (merged["runtime_ratio"] > 0)
        & (merged["score_ratio"] > 0)
    ]

    if merged.empty:
        raise ValueError(
            "No finite, positive ratios left to plot (check for zero "
            "scores or runtimes in the reference method)."
        )

    return merged


def _scatter_title(meta: dict, *, series_by: str, filters: dict | None) -> str:
    title_bits = [
        f"Score vs Runtime ({meta['name']}, "
        f"grid {meta['grid_size']}x{meta['grid_size']})",
        f"colored by {series_by}",
    ]
    if filters:
        filt_str = ", ".join(f"{k}={v}" for k, v in sorted(filters.items()))
        title_bits.append(f"filters: {filt_str}")
    return "\n".join(title_bits)


def _draw_scatter(
    ax,
    merged: pd.DataFrame,
    *,
    series_by: str,
    reference_method: str,
    title: str,
    series_values=None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
) -> None:
    """Draw a ratio scatter onto ``ax``.

    Passing ``series_values``/``x_bounds``/``y_bounds`` pins colors and axis
    extents so animation frames stay aligned even when a frame is empty.
    """
    cmap = plt.get_cmap("tab10")

    if series_values is None:
        series_values = sorted(
            merged[series_by].unique(), key=lambda v: (str(type(v)), v)
        )

    for i, value in enumerate(series_values):
        subset = merged[merged[series_by] == value]
        ax.scatter(
            subset["runtime_ratio"],
            subset["score_ratio"],
            label=f"{series_by}={value}",
            color=cmap(i % cmap.N),
            alpha=0.7,
            edgecolors="none",
            s=28,
        )

    ax.axhline(1.0, linestyle="--", linewidth=0.8, color="gray")
    ax.axvline(1.0, linestyle="--", linewidth=0.8, color="gray")
    ax.scatter(
        [1.0], [1.0],
        marker="x", color="black", s=60, zorder=5,
        label=f"{reference_method} (reference)",
    )

    if x_bounds is None:
        x_bounds = _ratio_axis_bounds(merged["runtime_ratio"].to_numpy())
    if y_bounds is None:
        y_bounds = _ratio_axis_bounds(merged["score_ratio"].to_numpy())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)

    ax.set_xlabel(f"Runtime ratio (method / {reference_method})")
    ax.set_ylabel(f"Score ratio (method / {reference_method})")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)


def render_scatter(
    raw_df: pd.DataFrame,
    meta: dict,
    *,
    series_by: str = "method",
    reference_method: str = "full_horizon_greedy_solve",
    output_root: Path = Path("results"),
    filters: dict | None = None,
) -> Path:
    """Render a score-vs-runtime scatter with each non-reference result
    plotted as a ratio against the reference method run on the same
    `(agents, steps, chunksize, start_row, start_col)` cell.

    Args:
        raw_df: per-method, per-start results (from `storage.load_sweep_raw_df`).
        meta: the sweep header dict (used for output path and titles).
        series_by: column controlling color/marker. One of
            {"method", "chunksize", "agents", "steps"}.
        reference_method: method used as the denominator for ratios.
        output_root: parent directory under which the plot is saved.
        filters: optional dict of column->value to slice `raw_df` before plotting
            (e.g. `{"agents": 3}`).

    Returns:
        The directory the plot was written to.
    """
    if series_by not in SCATTER_SERIES_COLUMNS:
        raise ValueError(
            f"series_by={series_by!r} not supported. "
            f"Choose one of {SCATTER_SERIES_COLUMNS}."
        )

    merged = _scatter_ratios(
        raw_df,
        series_by=series_by,
        reference_method=reference_method,
        filters=filters,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_scatter(
        ax,
        merged,
        series_by=series_by,
        reference_method=reference_method,
        title=_scatter_title(meta, series_by=series_by, filters=filters),
    )

    grid_size = int(meta["grid_size"])
    grid_dir = output_root / meta["name"] / f"grid_{grid_size}x{grid_size}"
    grid_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{meta['name']}__scatter__seriesby_{series_by}"
        f"{_filters_suffix(filters)}"
        f"__grid_{grid_size}x{grid_size}.png"
    )

    fig.tight_layout()
    fig.savefig(grid_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return grid_dir


def render_scatter_gif(
    raw_df: pd.DataFrame,
    meta: dict,
    *,
    animate_over: str,
    series_by: str = "method",
    reference_method: str = "full_horizon_greedy_solve",
    output_root: Path = Path("results"),
    filters: dict | None = None,
    frame_duration: float = DEFAULT_FRAME_DURATION,
) -> Path:
    """Render a score-vs-runtime scatter GIF, one frame per ``animate_over`` value."""
    if series_by not in SCATTER_SERIES_COLUMNS:
        raise ValueError(
            f"series_by={series_by!r} not supported. "
            f"Choose one of {SCATTER_SERIES_COLUMNS}."
        )
    if animate_over not in SCATTER_SERIES_COLUMNS:
        raise ValueError(
            f"animate_over={animate_over!r} not supported. "
            f"Choose one of {SCATTER_SERIES_COLUMNS}."
        )
    if animate_over == series_by:
        raise ValueError(
            "animate_over must differ from series_by "
            f"(got animate_over={animate_over!r}, series_by={series_by!r})."
        )

    base_filters = {
        key: value
        for key, value in (filters or {}).items()
        if key != animate_over
    }

    global_merged = _scatter_ratios(
        raw_df,
        series_by=series_by,
        reference_method=reference_method,
        filters=base_filters or None,
    )

    frame_values = _sorted_unique(global_merged[animate_over])
    if not frame_values:
        raise ValueError(
            f"No values of {animate_over!r} available to animate over."
        )

    series_values = sorted(
        global_merged[series_by].unique(), key=lambda v: (str(type(v)), v)
    )
    x_bounds = _ratio_axis_bounds(global_merged["runtime_ratio"].to_numpy())
    y_bounds = _ratio_axis_bounds(global_merged["score_ratio"].to_numpy())

    frames: list[bytes] = []
    for value in frame_values:
        frame_filters = {**base_filters, animate_over: value}
        try:
            merged = _scatter_ratios(
                raw_df,
                series_by=series_by,
                reference_method=reference_method,
                filters=frame_filters,
            )
        except ValueError:
            merged = global_merged.iloc[0:0]

        fig, ax = plt.subplots(figsize=(7, 6))
        _draw_scatter(
            ax,
            merged,
            series_by=series_by,
            reference_method=reference_method,
            title=_scatter_title(meta, series_by=series_by, filters=frame_filters),
            series_values=series_values,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        fig.tight_layout()
        frames.append(_figure_to_png_bytes(fig))

    grid_size = int(meta["grid_size"])
    grid_dir = output_root / meta["name"] / f"grid_{grid_size}x{grid_size}"

    filename = (
        f"{meta['name']}__scatter__seriesby_{series_by}"
        f"__anim_{animate_over}"
        f"{_filters_suffix(base_filters or None)}"
        f"__grid_{grid_size}x{grid_size}.gif"
    )

    _save_gif(frames, grid_dir / filename, frame_duration=frame_duration)
    return grid_dir


def _pareto_summary(
    raw_df: pd.DataFrame,
    *,
    series_by: str = "agents",
    reference_method: str = "full_horizon_greedy_solve",
    filters: dict | None = None,
) -> pd.DataFrame:
    """Aggregate per-result ratios into one Pareto point per chunksize.

    Each row is a `(series_by value, chunksize)` pair carrying the mean
    runtime ratio plus the mean and min score ratio against the reference
    method, averaged over all matched sweep cells.
    """
    if series_by not in PLOT_DIMENSION_COLUMNS:
        raise ValueError(
            f"series_by={series_by!r} not supported. "
            f"Choose one of {PLOT_DIMENSION_COLUMNS}."
        )
    if series_by == "chunksize":
        raise ValueError(
            "series_by='chunksize' is not supported for pareto plots; "
            "chunksize is the dimension traced along each curve."
        )

    merged = _scatter_ratios(
        raw_df,
        series_by=series_by,
        reference_method=reference_method,
        filters=filters,
    )

    summary = (
        merged.groupby([series_by, "chunksize"])
        .agg(
            runtime_ratio_mean=("runtime_ratio", "mean"),
            score_ratio_mean=("score_ratio", "mean"),
            score_ratio_min=("score_ratio", "min"),
        )
        .reset_index()
    )
    return summary


def _pareto_title(
    meta: dict,
    *,
    series_by: str,
    reference_method: str,
    filters: dict | None,
) -> str:
    title_bits = [
        f"Score vs Runtime Tradeoff by Chunksize ({meta['name']}, "
        f"grid {meta['grid_size']}x{meta['grid_size']})",
        f"series={series_by}, reference={reference_method}",
    ]
    if filters:
        filt_str = ", ".join(f"{k}={v}" for k, v in sorted(filters.items()))
        title_bits.append(f"filters: {filt_str}")
    return "\n".join(title_bits)


def _draw_pareto(
    ax,
    summary: pd.DataFrame,
    *,
    series_by: str,
    reference_method: str,
    title: str,
) -> None:
    """Draw the chunksize-traced Pareto tradeoff curve onto ``ax``.

    Each series is a line through `(mean runtime ratio, mean score ratio)`
    points ordered by chunksize, with a shaded band down to the min score
    ratio (worst start cell). Point labels are the chunksize values.
    """
    cmap = plt.get_cmap("tab10")

    series_values = _sorted_unique(summary[series_by])

    for i, value in enumerate(series_values):
        subset = summary[summary[series_by] == value].sort_values("chunksize")

        x = subset["runtime_ratio_mean"].to_numpy(dtype=float)
        mean_y = subset["score_ratio_mean"].to_numpy(dtype=float)
        min_y = subset["score_ratio_min"].to_numpy(dtype=float)
        color = cmap(i % cmap.N)

        ax.plot(
            x,
            mean_y,
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=f"{series_by}={value}",
            color=color,
        )
        ax.fill_between(x, min_y, mean_y, color=color, alpha=0.14, linewidth=0)

        for xi, yi, chunk in zip(x, mean_y, subset["chunksize"]):
            ax.annotate(
                str(chunk),
                (xi, yi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                color=color,
            )

    ax.axhline(1.0, linestyle="--", linewidth=0.8, color="gray")
    ax.axvline(1.0, linestyle="--", linewidth=0.8, color="gray")
    ax.scatter(
        [1.0], [1.0],
        marker="x", color="black", s=60, zorder=5,
        label=f"{reference_method} (reference)",
    )

    ax.set_xscale("log")

    ax.set_xlabel(f"Mean runtime ratio (method / {reference_method}, log scale)")
    ax.set_ylabel(f"Score ratio (method / {reference_method})")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)


def render_pareto(
    raw_df: pd.DataFrame,
    meta: dict,
    *,
    series_by: str = "agents",
    reference_method: str = "full_horizon_greedy_solve",
    output_root: Path = Path("results"),
    filters: dict | None = None,
) -> Path:
    """Render the score-vs-runtime Pareto tradeoff curve traced over chunksize.

    Each point is one chunksize, placed at the mean runtime ratio (x, log
    scale) and mean score ratio (y) against the reference method over all
    matched `(agents, steps, chunksize, start_row, start_col)` cells. The
    shaded band drops to the min score ratio (worst start cell), and lines
    are split by ``series_by``.

    If ``filters`` does not pin a method, the curve defaults to
    ``DEFAULT_PARETO_METHOD`` so a single method's frontier is traced.

    Args:
        raw_df: per-method, per-start results (from `storage.load_sweep_raw_df`).
        meta: the sweep header dict (used for output path and titles).
        series_by: column controlling line color. One of
            {"method", "agents", "steps"} ("chunksize" is the traced axis).
        reference_method: method used as the denominator for ratios.
        output_root: parent directory under which the plot is saved.
        filters: optional dict of column->value to slice `raw_df` before
            plotting (e.g. `{"steps": 8}`).

    Returns:
        The directory the plot was written to.
    """
    filters = dict(filters or {})
    if "method" not in filters and series_by != "method":
        filters["method"] = DEFAULT_PARETO_METHOD

    summary = _pareto_summary(
        raw_df,
        series_by=series_by,
        reference_method=reference_method,
        filters=filters,
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _draw_pareto(
        ax,
        summary,
        series_by=series_by,
        reference_method=reference_method,
        title=_pareto_title(
            meta,
            series_by=series_by,
            reference_method=reference_method,
            filters=filters,
        ),
    )

    fig.text(
        0.5,
        0.01,
        "Points are labeled with chunksize; shaded band spans mean to "
        "worst-case (min) score ratio.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    grid_size = int(meta["grid_size"])
    grid_dir = output_root / meta["name"] / f"grid_{grid_size}x{grid_size}"
    grid_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{meta['name']}__pareto__seriesby_{series_by}"
        f"{_filters_suffix(filters or None)}"
        f"__grid_{grid_size}x{grid_size}.png"
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(grid_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return grid_dir
