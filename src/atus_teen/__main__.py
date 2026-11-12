"""Build the teen extract, fit the models, and write figures and tables."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")

from atus_teen.models import run_analysis, specification_table
from atus_teen.paths import results_dir
from atus_teen.plots import save_all
from atus_teen.prepare import load_analysis_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ATUS teen reading analysis")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Download BLS files if needed and rebuild the teen extract",
    )
    args = parser.parse_args()
    frame, info = load_analysis_frame(rebuild=args.rebuild)
    print(
        f"Sample: n={info['n']:,} diary days, ages 15–18, "
        f"{info['year_min']}–{info['year_max']}"
    )
    analysis = run_analysis(frame)
    results = results_dir()
    results.mkdir(parents=True, exist_ok=True)
    analysis.descriptives.to_csv(results / "descriptives.csv", index=False)
    analysis.correlations.to_csv(results / "correlations.csv")
    analysis.bins.to_csv(results / "screen_deciles.csv", index=False)
    analysis.trends.to_csv(results / "trends.csv", index=False)
    analysis.models["full"].table().to_csv(results / "coefficients_full.csv", index=False)
    analysis.models["components"].table().to_csv(results / "coefficients_components.csv", index=False)
    specs = specification_table(analysis)
    specs.to_csv(results / "specifications.csv", index=False)
    (results / "key_numbers.json").write_text(json.dumps(analysis.keys, indent=2))
    save_all(analysis)
    keys = analysis.keys
    print(
        f"Weighted correlation, screen and reading: {keys['corr_screen_reading']:.3f}\n"
        f"OLS screen coefficient: {keys['ols_coef']:.3f} minutes per hour "
        f"(95% CI {keys['ols_ci_low']:.3f} to {keys['ols_ci_high']:.3f})\n"
        f"Weighted R²: {keys['ols_r2']:.4f}; partial R² of screen time: {keys['partial_r2']:.4f}"
    )


if __name__ == "__main__":
    main()
