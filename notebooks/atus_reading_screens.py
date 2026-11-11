# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.25",
#     "pandas>=2.2",
#     "pyarrow>=15",
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "statsmodels>=0.14",
#     "matplotlib>=3.8",
#     "seaborn>=0.13",
#     "patsy>=0.5",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Screen time and reading among 15–18 year olds

        **Question.** In the American Time Use Survey, does screen time explain
        reading for personal interest among teenagers?

        Each ATUS respondent describes one diary day. This notebook keeps
        respondents aged 15–18 in the 2003–2024 multi-year files, builds a
        screen-time total from the published activity codes, and estimates
        weighted associations with reading. The estimates are descriptive.
        A diary day is one draw from a person's week, and teens choose both
        activities.
        """
    )
    return


@app.cell
def _():
    import os

    import pandas as pd

    from atus_teen.models import FORMULA_COMPONENTS, FORMULA_FULL, run_analysis, specification_table
    from atus_teen.plots import (
        plot_bins,
        plot_control_forest,
        plot_correlation,
        plot_spec_forest,
        plot_trends,
    )
    from atus_teen.prepare import ACTIVITY_CODES, load_analysis_frame

    frame, info = load_analysis_frame(rebuild=os.environ.get("ATUS_REBUILD") == "1")
    analysis = run_analysis(frame)
    specs = specification_table(analysis)
    keys = analysis.keys
    return (
        ACTIVITY_CODES,
        FORMULA_COMPONENTS,
        FORMULA_FULL,
        analysis,
        pd,
        frame,
        info,
        keys,
        plot_bins,
        plot_control_forest,
        plot_correlation,
        plot_spec_forest,
        plot_trends,
        specs,
    )


@app.cell
def _(frame, info, keys, mo):
    mo.md(
        f"""
        ## Sample

        **{info["n"]:,}** diary days, one per respondent, ages 15–18, diary years
        **{info["year_min"]}–{info["year_max"]}**. The extract is
        `{info["extract"]}` (rebuilt this run: {info["rebuilt"]}).

        Set `ATUS_REBUILD=1` to re-download the BLS zips and rebuild the extract.
        The download script tries the official URLs on
        [the BLS multi-year page](https://www.bls.gov/tus/data/datafiles-0324.htm)
        and, if that host rejects the client, uses the Internet Archive capture
        of the same file. Raw zips stay in `data/raw/` and are gitignored.

        Survey weight: `TUFNWGTP` in every year except 2020, and `TU20FWGT` in
        2020, when `TUFNWGTP` is zero. BLS constructed the 2020 weight for the
        days that were fielded (January 1–March 17 and May 10–December 31).
        Weighted estimates pool years in proportion to each year's teen population.
        The weighted weekend share is **{keys["weekend_pct"]:.1f}%**, in line with
        two weekend days out of seven. The unweighted file is about half weekend
        diaries because of the ATUS sample design.
        """
    )
    return


@app.cell
def _(ACTIVITY_CODES, mo):
    rows = "\n".join(
        f"| `{code}` | {label} |" for code, label in ACTIVITY_CODES.items()
    )
    mo.md(
        f"""
        ## Activity codes

        Minutes come from the Activity Summary file. Column names are the code
        with a leading `t`. Labels below follow the BLS read-in programs and the
        2003–2024 coding lexicon.

        | Code | Definition used here |
        | --- | --- |
        {rows}

        **Outcome.** `reading_min` is `t120312`. `any_reading` is 1 when that
        minute total is positive. The lexicon places print, e-readers, and
        audiobooks in this code.

        **Screen time.** `screen_min` = television (`t120303` + `t120304`) +
        games (`t120307`) + leisure computer use (`t120308`). Religious
        television is part of the requested TV total; in this age group it
        averages a few seconds. Games include video and computer games and also
        board games, cards, and puzzles. Leisure computer use includes browsing
        and, in later lexicon years, checking social-media sites, and it
        excludes games. ATUS records the primary activity, and it does not
        identify the device, so phone use is inside these codes only when the
        respondent described that activity. Scrolling while doing something else
        is not counted. Telephone calls and in-person socializing have their
        own codes and are left out of this screen total.

        **Controls.** Age, sex, diary year, weekend (Saturday or Sunday) versus
        weekday, holiday, summer (June–August), school enrollment (`TESCHENR`),
        labor-force status (`TELFS`), metropolitan status (`GTMETSTA`),
        Hispanic origin, a coarse race group from `PTDTRACE`, CPS family-income
        bracket, and the highest education among coresident parents (roster
        relationship code 24 linked to CPS `PEEDUCA`). Income uses edited
        `HEFAMINC` when that field is coded (2010–2024) and unedited `HUFAMINC`
        in 2003–2009. Brackets are nominal dollars.
        """
    )
    return


@app.cell
def _(analysis, mo):
    pretty = analysis.descriptives.copy()
    pretty["estimate"] = [
        f"{value:,.0f}" if unit == "count" else f"{value:,.2f}"
        for value, unit in zip(pretty["estimate"], pretty["unit"])
    ]
    mo.vstack(
        [
            mo.md("## Weighted descriptives"),
            pretty,
        ]
    )
    return


@app.cell
def _(analysis, mo, plot_correlation):
    mo.vstack(
        [
            mo.md(
                """
                ## Correlations

                Pearson correlations use the survey weights. Income bracket and
                parent education are ordinal codes; rows with a missing code are
                dropped for that pair only. Total screen time is the sum of TV,
                games, and computer use, so those cells are large by construction.
                """
            ),
            plot_correlation(analysis),
        ]
    )
    return


@app.cell
def _(analysis, mo, plot_bins, plot_trends):
    mo.vstack(
        [
            mo.md(
                """
                ## Reading against screen time, and the year path

                The left-hand points are weighted deciles of total screen hours.
                The year chart uses the same weights. The shaded year is 2020.
                Teen samples are much smaller after about 2015 (a few hundred
                diaries a year), so single-year spikes are noisy.
                """
            ),
            plot_bins(analysis),
            plot_trends(analysis),
        ]
    )
    return


@app.cell
def _(FORMULA_COMPONENTS, FORMULA_FULL, mo):
    mo.md(
        f"""
        ## Models

        The main estimate is weighted least squares of reading minutes on screen
        hours and the controls. Weights are scaled to sum to the sample size.
        The covariance is an HC1 probability-weight sandwich (a robust
        weighted regression). It is not the BLS replicate-weight variance:
        the standard replicate file omits 2020, and this repository does not
        ship that 285 MB file. Intervals are approximate.

        ```
        {FORMULA_FULL}
        ```

        A second OLS replaces total screen hours with the three components:

        ```
        {FORMULA_COMPONENTS}
        ```

        Reading is zero on about nine diary days in ten. Two checks take that
        seriously. A weighted logit models any reading, and the reported number
        is the average change in probability for one extra screen hour. A
        second OLS uses only diary days with some reading (the intensive
        margin). A normal Tobit censored at zero does not settle on a stable
        estimate here: the zero mass is large and the positive tail is long, so
        the two-part description above is the one to read. Other OLS rows drop
        2020 or cap reading and screen time at their weighted 99th percentiles.
        """
    )
    return


@app.cell
def _(mo, specs):
    spec_table = specs.copy()
    for spec_column in ("estimate", "se", "ci_low", "ci_high"):
        spec_table[spec_column] = spec_table[spec_column].map(lambda value: f"{value:.3f}")
    spec_table["fit"] = spec_table["fit"].map(lambda value: "" if value != value else f"{value:.4f}")
    spec_table = spec_table[["label", "estimate", "se", "ci_low", "ci_high", "n", "fit", "unit"]]
    mo.vstack(
        [
            mo.md(
                """
                ## Screen-time coefficient across specifications

                Minute-unit rows are minutes of reading per extra hour. The logit
                row is a change in the probability of any reading. `fit` is the
                weighted R², or McFadden's pseudo-R² for the logit.
                """
            ),
            spec_table,
        ]
    )
    return


@app.cell
def _(analysis, mo, plot_control_forest, plot_spec_forest):
    mo.vstack(
        [
            mo.md(
                """
                ## Coefficient plots

                Intervals are 95% t intervals from the robust weighted covariance.
                Year indicators are in the full model and omitted from the
                control plot so the rest of the coefficients stay readable.
                Three respondents with an unknown enrollment code are in the
                model and left off that plot.
                """
            ),
            plot_spec_forest(analysis),
            plot_control_forest(analysis),
        ]
    )
    return


@app.cell
def _(analysis, mo, pd):
    table = analysis.models["full"].table()
    screen = table[table["term"] == "screen_hours"]
    others = table[~table["term"].isin(["Intercept", "screen_hours"]) & ~table["term"].str.startswith("C(year")]
    coef_table = pd.concat([screen, others], ignore_index=True)[
        ["label", "coef", "se", "ci_low", "ci_high", "pvalue"]
    ].copy()
    for coef_column in ("coef", "se", "ci_low", "ci_high"):
        coef_table[coef_column] = coef_table[coef_column].map(lambda value: f"{value:.3f}")
    coef_table["pvalue"] = coef_table["pvalue"].map(lambda value: f"{float(value):.3g}")
    mo.vstack(
        [
            mo.md(
                """
                ## Full-model coefficients, excluding year indicators

                The outcome is minutes of reading on the diary day. Screen time
                is in hours, so its coefficient is minutes per extra hour.
                """
            ),
            coef_table,
        ]
    )
    return


@app.cell
def _(keys, mo):
    k = keys
    mo.md(
        f"""
        ## What this says in plain language

        Teenagers in this pooled ATUS sample average **{k["reading_min"]:.1f} minutes**
        of reading for personal interest and **{k["screen_hours"]:.2f} hours** of
        the screen activities defined above. Any reading shows up on
        **{k["any_reading_pct"]:.1f}%** of diary days. Television is most of the
        screen total ({k["tv_min"]:.0f} minutes), then games ({k["games_min"]:.0f})
        and leisure computer use ({k["computer_min"]:.0f}). Religious television
        is {k["religious_tv_min"]:.2f} minutes.

        The weighted correlation between total screen minutes and reading
        minutes is **{k["corr_screen_reading"]:.3f}**.

        In the full weighted regression, an extra hour of screen time is
        associated with **{k["ols_coef"]:.2f} minutes** of reading
        (95% CI {k["ols_ci_low"]:.2f} to {k["ols_ci_high"]:.2f}).
        The weighted R² is **{k["ols_r2"]:.3f}**. Adding screen time to the
        controls raises the explained share by a partial R² of
        **{k["partial_r2"]:.4f}**, about {100 * k["partial_r2"]:.2f}% of the
        variation the controls had left over. A gap from the 10th percentile of
        screen time ({k["screen_p10_hours"]:.1f} hours) to the 90th
        ({k["screen_p90_hours"]:.1f} hours) lines up with
        **{k["implied_p90_p10_reading_min"]:.1f} minutes** of reading.

        The decile plot tells the same story with less smoothing. Through most
        of the distribution, average reading stays in a narrow band. The top
        decile, around 10 hours of screen time, is where both minutes and the
        share who read are lower (about 6 minutes, and about 6% of days with
        any reading, against about 10 minutes and 11% in the lowest deciles).

        On the extensive margin, one extra screen hour is associated with a
        **{100 * k["logit_ame"]:.2f} percentage-point** change in the probability
        of any reading (95% CI {100 * (k["logit_ame"] - 1.96 * k["logit_ame_se"]):.2f}
        to {100 * (k["logit_ame"] + 1.96 * k["logit_ame_se"]):.2f}), from a base
        of {k["any_reading_pct"]:.1f}%. McFadden's pseudo-R² for that logit is
        {k["logit_mcfadden"]:.3f}. Among the {k["intensive_n"]:,} diary days that
        include some reading, the screen coefficient is {k["intensive_coef"]:.2f}
        minutes per hour (95% CI {k["intensive_ci_low"]:.2f} to
        {k["intensive_ci_high"]:.2f}).

        TV ({k["component_tv_hours"]:.2f} minutes per hour, CI
        {k["component_tv_hours_ci_low"]:.2f} to {k["component_tv_hours_ci_high"]:.2f}),
        games ({k["component_games_hours"]:.2f}, CI
        {k["component_games_hours_ci_low"]:.2f} to {k["component_games_hours_ci_high"]:.2f}),
        and leisure computer use ({k["component_computer_hours"]:.2f}, CI
        {k["component_computer_hours_ci_low"]:.2f} to
        {k["component_computer_hours_ci_high"]:.2f}) have the same sign and
        overlapping intervals. Games are the noisiest of the three, which fits
        a code that mixes video games with tabletop games.

        School enrollment (about {k["enrolled_pct"]:.0f}% of weighted teens) and
        employment (about {k["employed_pct"]:.0f}%) have larger reading
        coefficients than a single screen hour, on the order of 5 fewer
        minutes. Summer diaries run a few minutes higher. Those contrasts still
        sit inside a model that leaves about 98% of diary-day reading
        unaccounted for. Across years, average reading wanders between roughly
        7 and 14 minutes with no steady decline, while total screen time stays
        near 3 hours except in 2020, when it rises above 4 hours and reading
        does not fall. Game time is the piece that drifts up, from about half
        an hour early in the period toward about an hour in the 2020s.

        **Reading the result.** Screen time has a small negative association
        with teen reading, visible because the sample is large. It accounts for
        a fraction of a percent of the variation in a diary day. That is a weak
        statistical explanation of who reads, and of how long.

        **Limits of the design.** One person, one day: a diary that happens to
        contain a long game or a long book is a poor measure of either habit,
        which is why R² stays near zero even with a long list of controls.
        Both activities are chosen, and a long screen spell leaves fewer
        minutes in a 1,440-minute day, so the coefficient mixes preference with
        arithmetic. The activity codes miss a lot of phone use and mix
        non-screen games into the games total. Income brackets are nominal.
        The intervals use a robust weighted covariance. A replicate-weight
        variance would also reflect the ATUS strata and the way the weights
        were built; the partial R² is unaffected by that choice. The 2020
        sample covers the months BLS fielded, and dropping that year leaves
        the screen coefficient at about the same value.
        """
    )
    return


if __name__ == "__main__":
    app.run()
