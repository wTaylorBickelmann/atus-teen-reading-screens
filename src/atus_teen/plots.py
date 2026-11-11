"""Figures for the teen reading and screen-time analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from atus_teen.models import Analysis, specification_table
from atus_teen.paths import figures_dir

BLUE = "#1f4e79"
CLAY = "#b4532a"
GREEN = "#2f6f4e"
GOLD = "#c48a2a"
SLATE = "#4b5563"
GRID = "#e5e7eb"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    path = figures_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, pad_inches=0.25)
    return path


def plot_correlation(analysis: Analysis) -> plt.Figure:
    apply_style()
    corr = analysis.correlations
    fig, ax = plt.subplots(figsize=(10.2, 8.4))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-0.6,
        vmax=0.6,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Weighted correlation", "shrink": 0.8},
        square=True,
    )
    ax.set_title("Weighted correlations, ATUS ages 15–18, 2003–2024")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    _save(fig, "corr_heatmap.png")
    return fig


def plot_bins(analysis: Analysis) -> plt.Figure:
    apply_style()
    bins = analysis.bins
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(
        bins["screen_hours"],
        bins["reading_min"],
        color=CLAY,
        marker="o",
        linewidth=2,
        markersize=7,
        label="Reading minutes",
    )
    ax.set_xlabel("Total screen time (hours on the diary day)")
    ax.set_ylabel("Reading minutes", color=CLAY)
    ax.tick_params(axis="y", colors=CLAY)
    ax.set_ylim(0, max(12, bins["reading_min"].max() * 1.35))
    twin = ax.twinx()
    twin.plot(
        bins["screen_hours"],
        100 * bins["any_reading"],
        color=BLUE,
        marker="s",
        linewidth=1.8,
        markersize=5.5,
        label="Any reading",
    )
    twin.set_ylabel("Any reading (percent of diary days)", color=BLUE)
    twin.tick_params(axis="y", colors=BLUE)
    twin.spines["right"].set_visible(True)
    twin.grid(False)
    twin.set_ylim(0, max(20, 100 * bins["any_reading"].max() * 1.45))
    ax.set_title("Reading by screen-time decile")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "reading_vs_screen_bins.png")
    return fig


def plot_trends(analysis: Analysis) -> plt.Figure:
    apply_style()
    trends = analysis.trends
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    ax = axes[0]
    minutes_line = ax.plot(
        trends["year"], trends["reading_min"], color=CLAY, marker="o", linewidth=2, label="Minutes per day"
    )
    ax.set_ylabel("Reading minutes per day", color=CLAY)
    ax.tick_params(axis="y", colors=CLAY)
    ax.set_ylim(0, max(16, trends["reading_min"].max() * 1.45))
    twin = ax.twinx()
    share_line = twin.plot(
        trends["year"], 100 * trends["any_reading"], color=BLUE, marker="s", linewidth=1.6, label="Any reading (percent)"
    )
    twin.set_ylabel("Any reading (percent)", color=BLUE)
    twin.tick_params(axis="y", colors=BLUE)
    twin.spines["right"].set_visible(True)
    twin.grid(False)
    twin.set_ylim(0, max(25, 100 * trends["any_reading"].max() * 1.45))
    ax.legend(
        minutes_line + share_line,
        [line.get_label() for line in minutes_line + share_line],
        frameon=False,
        fontsize=8,
        loc="lower left",
    )
    ax.set_title("Reading")
    ax.axvspan(2019.5, 2020.5, color="#f3e6c8", zorder=0)

    ax2 = axes[1]
    ax2.plot(trends["year"], trends["screen_hours"], color=BLUE, marker="o", linewidth=2.2, label="Total screen")
    ax2.plot(trends["year"], trends["tv_hours"], color=SLATE, marker=".", linewidth=1.4, label="TV and movies")
    ax2.plot(trends["year"], trends["games_hours"], color=GREEN, marker=".", linewidth=1.4, label="Games")
    ax2.plot(trends["year"], trends["computer_hours"], color=GOLD, marker=".", linewidth=1.4, label="Leisure computer")
    ax2.set_title("Screen time")
    ax2.set_ylabel("Hours per diary day")
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.axvspan(2019.5, 2020.5, color="#f3e6c8", zorder=0)
    for axis in axes:
        axis.set_xlabel("Diary year")
        axis.set_xticks(list(range(2003, 2025, 3)))
    fig.suptitle("Ages 15–18, weighted. Shaded year is 2020 (partial-year weights).", y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, "trends_by_year.png")
    return fig


def plot_spec_forest(analysis: Analysis) -> plt.Figure:
    apply_style()
    table = specification_table(analysis)
    minutes = table[table["unit"].str.startswith("Minutes") | table["unit"].str.startswith("Latent")].copy()
    # Draw the main screen-hour specs, then the three components, with a gap before components.
    # Last category is drawn at the top of the forest plot.
    main_order = ["tobit", "intensive", "winsor", "no_2020", "bivariate", "diary", "full"]
    component_order = ["computer_hours", "games_hours", "tv_hours"]
    main = minutes[minutes["spec"].isin(main_order)].copy()
    main["spec"] = pd_categorical(main["spec"], main_order)
    main = main.sort_values("spec")
    components = minutes[minutes["spec"].isin(component_order)].copy()
    components["spec"] = pd_categorical(components["spec"], component_order)
    components = components.sort_values("spec")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 5.6),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    _forest(axes[0], main, "Minutes of reading per extra hour", "Screen time, across specifications")
    _forest(axes[1], components, "Minutes of reading per extra hour of that activity", "Which screen activity?")
    fig.tight_layout()
    _save(fig, "screen_coef_forest.png")
    return fig


def plot_control_forest(analysis: Analysis) -> plt.Figure:
    apply_style()
    table = analysis.models["full"].table()
    skip = table["term"].str.startswith("C(year") | table["term"].eq("Intercept")
    # Three teens have an unknown enrollment code; the interval is too wide to plot.
    skip = skip | table["term"].str.contains("unknown")
    shown = table.loc[~skip].copy()
    # Screen coefficient first, then the rest in model order.
    shown = pd_move_first(shown, "screen_hours")
    fig, ax = plt.subplots(figsize=(8.6, 9.2))
    _forest(ax, shown.iloc[::-1], "Coefficient (minutes of reading)", "Full weighted OLS, controls other than year")
    fig.tight_layout()
    _save(fig, "full_model_coefs.png")
    return fig


def _forest(ax: plt.Axes, table, xlabel: str, title: str) -> None:
    y = np.arange(len(table))
    ax.axvline(0, color="#9ca3af", linewidth=1)
    ax.errorbar(
        table["coef"] if "coef" in table.columns else table["estimate"],
        y,
        xerr=np.vstack(
            [
                (table["coef"] if "coef" in table.columns else table["estimate"]) - table["ci_low"],
                table["ci_high"] - (table["coef"] if "coef" in table.columns else table["estimate"]),
            ]
        ),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        elinewidth=1.4,
        capsize=3,
        markersize=5.5,
    )
    labels = table["label"] if "label" in table.columns else table["label"]
    # Matplotlib treats $...$ as mathtext, which swallows income-bracket labels.
    ax.set_yticks(y)
    ax.set_yticklabels([str(label).replace("$", r"\$") for label in labels])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)


def pd_categorical(series, order: list[str]):
    import pandas as pd

    return pd.Categorical(series, categories=[item for item in order if item in set(series)], ordered=True)


def pd_move_first(frame, term: str):
    top = frame[frame["term"] == term]
    rest = frame[frame["term"] != term]
    return __import__("pandas").concat([top, rest], ignore_index=True)


def save_all(analysis: Analysis) -> list[Path]:
    paths = []
    for plot in (plot_correlation, plot_bins, plot_trends, plot_spec_forest, plot_control_forest):
        fig = plot(analysis)
        paths.append(figures_dir() / {
            plot_correlation: "corr_heatmap.png",
            plot_bins: "reading_vs_screen_bins.png",
            plot_trends: "trends_by_year.png",
            plot_spec_forest: "screen_coef_forest.png",
            plot_control_forest: "full_model_coefs.png",
        }[plot])
        plt.close(fig)
    return paths
