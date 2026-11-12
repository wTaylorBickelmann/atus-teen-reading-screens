"""Weighted descriptives and regressions for teen reading and screen time.

Point estimates use ATUS final weights (TUFNWGTP, or TU20FWGT in 2020),
scaled to sum to the sample size. OLS covariance is the probability-weight
sandwich with an HC1 finite-sample factor, the same idea as a weighted
regression with robust standard errors. It is not the BLS replicate-weight
variance: replicate weights are a separate file, and 2020 respondents are
missing from the standard 2003–2024 replicate file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import patsy
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM

CONTROLS = (
    "C(age, Treatment(reference=15))"
    " + female + weekend + holiday + summer"
    " + C(enrolled, Treatment(reference='not_enrolled'))"
    " + C(labor_force, Treatment(reference='nilf'))"
    " + C(metro, Treatment(reference='metro'))"
    " + hispanic"
    " + C(race, Treatment(reference='white'))"
    " + C(income_group, Treatment(reference='50_75'))"
    " + C(parent_educ_group, Treatment(reference='hs'))"
    " + C(year, Treatment(reference=2003))"
)

FORMULA_FULL = f"reading_min ~ screen_hours + {CONTROLS}"
FORMULA_COMPONENTS = f"reading_min ~ tv_hours + games_hours + computer_hours + {CONTROLS}"
FORMULA_BIVARIATE = "reading_min ~ screen_hours"
FORMULA_DIARY = "reading_min ~ screen_hours + weekend + holiday + summer + C(year, Treatment(reference=2003))"
FORMULA_ANY = f"any_reading ~ screen_hours + {CONTROLS}"

TERM_LABELS = {
    "Intercept": "Intercept",
    "screen_hours": "Total screen (hours)",
    "tv_hours": "TV and movies (hours)",
    "games_hours": "Games (hours)",
    "computer_hours": "Leisure computer use (hours)",
    "female": "Female",
    "weekend": "Weekend diary",
    "holiday": "Holiday diary",
    "summer": "Summer month (June–August)",
    "hispanic": "Hispanic",
}

LEVEL_LABELS = {
    "16": "Age 16",
    "17": "Age 17",
    "18": "Age 18",
    "enrolled": "Enrolled in school",
    "unknown": "Enrollment unknown",
    "employed": "Employed",
    "unemployed": "Unemployed",
    "nonmetro": "Nonmetropolitan",
    "not_identified": "Metro status not identified",
    "black": "Black only",
    "asian": "Asian only",
    "other": "Another race or multiracial",
    "lt25": "Family income under $25,000",
    "25_50": "Family income $25,000 to $49,999",
    "75_100": "Family income $75,000 to $99,999",
    "100plus": "Family income $100,000 or more",
    "missing": "Family income or parent education missing",
    "lt_hs": "Parent education: less than high school",
    "some_college": "Parent education: some college or associate",
    "ba": "Parent education: bachelor's degree",
    "grad": "Parent education: graduate or professional",
}


def pretty_term(name: str) -> str:
    if name in TERM_LABELS:
        return TERM_LABELS[name]
    if "[T." in name and name.endswith("]"):
        level = name.split("[T.", 1)[1][:-1]
        if name.startswith("C(year"):
            return f"Year {level}"
        if name.startswith("C(age"):
            return f"Age {level}"
        if "parent_educ_group" in name and level == "missing":
            return "Parent education missing"
        if "income_group" in name and level == "missing":
            return "Family income missing"
        return LEVEL_LABELS.get(level, level.replace("_", " "))
    return name


def _scale_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if np.any(weights <= 0):
        raise ValueError("Weights must be positive")
    return weights * (len(weights) / weights.sum())


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights * values) / np.sum(weights))


def weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    mx = np.sum(weights * x)
    my = np.sum(weights * y)
    cov = np.sum(weights * (x - mx) * (y - my))
    sx = np.sqrt(np.sum(weights * (x - mx) ** 2))
    sy = np.sqrt(np.sum(weights * (y - my) ** 2))
    if sx == 0 or sy == 0:
        return float("nan")
    return float(cov / (sx * sy))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= cdf[-1]
    return float(np.interp(quantile, cdf, values))


def weighted_corr_matrix(frame: pd.DataFrame, columns: list[str], weight: str = "weight") -> pd.DataFrame:
    matrix = np.eye(len(columns))
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            right = columns[j]
            subset = frame[[left, right, weight]].dropna()
            corr = weighted_corr(subset[left].to_numpy(), subset[right].to_numpy(), subset[weight].to_numpy())
            matrix[i, j] = matrix[j, i] = corr
    return pd.DataFrame(matrix, index=columns, columns=columns)


@dataclass
class RegressionResult:
    name: str
    formula: str
    terms: list[str]
    coef: np.ndarray
    se: np.ndarray
    n: int
    k: int
    r2: float | None
    ssr: float | None
    tss: float | None
    extra: dict = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        crit = stats.t.ppf(0.975, max(self.n - self.k, 1))
        zstat = self.coef / self.se
        pvalues = 2 * stats.t.sf(np.abs(zstat), max(self.n - self.k, 1))
        out = pd.DataFrame(
            {
                "term": self.terms,
                "label": [pretty_term(term) for term in self.terms],
                "coef": self.coef,
                "se": self.se,
                "ci_low": self.coef - crit * self.se,
                "ci_high": self.coef + crit * self.se,
                "pvalue": pvalues,
            }
        )
        return out

    def term(self, name: str) -> pd.Series:
        table = self.table()
        rows = table.loc[table["term"] == name]
        if rows.empty:
            raise KeyError(name)
        return rows.iloc[0]


def fit_wls(formula: str, data: pd.DataFrame, name: str) -> RegressionResult:
    """Survey-weighted OLS with an HC1 probability-weight sandwich."""
    response, design = patsy.dmatrices(formula, data, return_type="dataframe")
    y = np.asarray(response, dtype=float).ravel()
    x = np.asarray(design, dtype=float)
    weights = _scale_weights(data.loc[design.index, "weight"].to_numpy())
    n, k = x.shape
    beta = np.linalg.lstsq(x * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)[0]
    resid = y - x @ beta
    xtwx = x.T @ (x * weights[:, None])
    bread = np.linalg.inv(xtwx)
    score = x * (weights * resid)[:, None]
    meat = score.T @ score
    cov = bread @ meat @ bread * (n / (n - k))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ybar = np.average(y, weights=weights)
    tss = float(np.sum(weights * (y - ybar) ** 2))
    ssr = float(np.sum(weights * resid**2))
    return RegressionResult(
        name=name,
        formula=formula,
        terms=list(design.columns),
        coef=beta,
        se=se,
        n=n,
        k=k,
        r2=1 - ssr / tss if tss else None,
        ssr=ssr,
        tss=tss,
    )


def _logit_sandwich(result, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """HC1 sandwich for a frequency-weighted binomial GLM."""
    mu = np.asarray(result.fittedvalues, dtype=float)
    mu = np.clip(mu, 1e-8, 1 - 1e-8)
    resid = np.asarray(result.model.endog, dtype=float).ravel() - mu
    n, k = x.shape
    # Score contribution of a weighted Bernoulli observation.
    score = x * (weights * resid)[:, None]
    xtwx = x.T @ (x * (weights * mu * (1 - mu))[:, None])
    bread = np.linalg.inv(xtwx)
    meat = score.T @ score
    return bread @ meat @ bread * (n / (n - k))


def fit_logit(formula: str, data: pd.DataFrame, name: str) -> RegressionResult:
    response, design = patsy.dmatrices(formula, data, return_type="dataframe")
    y = np.asarray(response, dtype=float).ravel()
    x = np.asarray(design, dtype=float)
    index = design.index
    weights = _scale_weights(data.loc[index, "weight"].to_numpy())
    model = GLM(y, x, family=Binomial(), freq_weights=weights)
    fitted = model.fit(maxiter=100, disp=0)
    cov = _logit_sandwich(fitted, x, weights)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    # Average marginal effect of +1 hour of screen, weighted.
    screen_idx = list(design.columns).index("screen_hours")
    beta = np.asarray(fitted.params, dtype=float)
    # Rebuild predictions at screen + 1 using the linear predictor.
    eta = x @ beta
    eta_up = eta + beta[screen_idx]
    ame = float(np.average(1 / (1 + np.exp(-eta_up)) - 1 / (1 + np.exp(-eta)), weights=weights))
    # Delta-method SE treating other covariates as fixed at observed values.
    dens = np.exp(-eta) / (1 + np.exp(-eta)) ** 2
    dens_up = np.exp(-eta_up) / (1 + np.exp(-eta_up)) ** 2
    # d(p_up - p)/d beta = dens_up * (x + e_s) - dens * x
    eye = np.zeros(x.shape[1])
    eye[screen_idx] = 1.0
    grad = (dens_up[:, None] * (x + eye)) - (dens[:, None] * x)
    gbar = np.average(grad, axis=0, weights=weights)
    ame_se = float(np.sqrt(gbar @ cov @ gbar))
    null = GLM(y, np.ones((len(y), 1)), family=Binomial(), freq_weights=weights).fit(disp=0)
    mcfadden = float(1 - fitted.llf / null.llf) if null.llf != 0 else None
    return RegressionResult(
        name=name,
        formula=formula,
        terms=list(design.columns),
        coef=beta,
        se=se,
        n=int(len(y)),
        k=int(x.shape[1]),
        r2=mcfadden,
        ssr=None,
        tss=None,
        extra={"ame_screen": ame, "ame_se": ame_se, "mcfadden_r2": mcfadden},
    )


def fit_tobit(formula: str, data: pd.DataFrame, name: str) -> RegressionResult | None:
    """Left-censored Tobit at zero reading minutes, weighted."""
    response, design = patsy.dmatrices(formula, data, return_type="dataframe")
    y = np.asarray(response, dtype=float).ravel()
    x = np.asarray(design, dtype=float)
    weights = _scale_weights(data.loc[design.index, "weight"].to_numpy())
    n, k = x.shape
    beta0 = np.linalg.lstsq(x * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)[0]
    sigma0 = float(np.sqrt(np.average((y - x @ beta0) ** 2, weights=weights)))
    sigma0 = max(sigma0, 1.0)
    start = np.r_[beta0, np.log(sigma0)]

    def nll(theta: np.ndarray) -> float:
        beta = theta[:-1]
        sigma = float(np.exp(theta[-1]))
        if not np.isfinite(sigma) or sigma < 1e-4:
            return 1e12
        xb = x @ beta
        cens = y <= 0
        ll = np.empty(n)
        ll[cens] = norm.logcdf((0 - xb[cens]) / sigma)
        ll[~cens] = norm.logpdf((y[~cens] - xb[~cens]) / sigma) - np.log(sigma)
        if not np.isfinite(ll).all():
            return 1e12
        return float(-np.sum(weights * ll))

    # A normal Tobit is a poor fit when about 90% of days are exact zeros and the
    # positive tail is long. Cap the attempt so a failed fit stays cheap.
    opt = minimize(nll, start, method="BFGS", options={"maxiter": 40})
    if not opt.success:
        return None
    hess_inv = opt.hess_inv
    if hasattr(hess_inv, "todense"):
        hess_inv = hess_inv.todense()
    hess_inv = np.asarray(hess_inv, dtype=float)
    se = np.sqrt(np.clip(np.diag(hess_inv)[:-1], 0, None))
    beta = opt.x[:-1]
    return RegressionResult(
        name=name,
        formula=formula,
        terms=list(design.columns),
        coef=beta,
        se=se,
        n=n,
        k=k,
        r2=None,
        ssr=None,
        tss=None,
        extra={"sigma": float(np.exp(opt.x[-1])), "converged": True},
    )


def partial_r2(full: RegressionResult, reduced: RegressionResult) -> float:
    if not full.ssr or not reduced.ssr:
        return float("nan")
    return float((reduced.ssr - full.ssr) / reduced.ssr)


def weighted_decile(values: np.ndarray, weights: np.ndarray, bins: int = 10) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    cdf = np.cumsum(weights[order])
    cdf = cdf / cdf[-1]
    assigned = np.minimum((cdf * bins).astype(int), bins - 1)
    # The heaviest observation can land on the boundary; keep the top bin non-empty.
    assigned[-1] = bins - 1
    out = np.empty(len(values), dtype=int)
    out[order] = assigned
    return out


def screen_bins(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    decile = weighted_decile(frame["screen_hours"].to_numpy(), frame["weight"].to_numpy(), bins)
    work = frame.copy()
    work["decile"] = decile + 1
    rows = []
    for label, part in work.groupby("decile"):
        w = part["weight"].to_numpy()
        rows.append(
            {
                "decile": int(label),
                "n": int(len(part)),
                "screen_hours": weighted_mean(part["screen_hours"].to_numpy(), w),
                "reading_min": weighted_mean(part["reading_min"].to_numpy(), w),
                "any_reading": weighted_mean(part["any_reading"].to_numpy(), w),
                "tv_hours": weighted_mean(part["tv_hours"].to_numpy(), w),
                "games_hours": weighted_mean(part["games_hours"].to_numpy(), w),
                "computer_hours": weighted_mean(part["computer_hours"].to_numpy(), w),
            }
        )
    return pd.DataFrame(rows)


def yearly_trends(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, part in frame.groupby("year"):
        w = part["weight"].to_numpy()
        rows.append(
            {
                "year": int(year),
                "n": int(len(part)),
                "reading_min": weighted_mean(part["reading_min"].to_numpy(), w),
                "any_reading": weighted_mean(part["any_reading"].to_numpy(), w),
                "screen_hours": weighted_mean(part["screen_hours"].to_numpy(), w),
                "tv_hours": weighted_mean(part["tv_hours"].to_numpy(), w),
                "games_hours": weighted_mean(part["games_hours"].to_numpy(), w),
                "computer_hours": weighted_mean(part["computer_hours"].to_numpy(), w),
            }
        )
    return pd.DataFrame(rows)


CORR_COLUMNS = [
    ("reading_min", "Reading (min)"),
    ("any_reading", "Any reading"),
    ("tv_min", "TV (min)"),
    ("games_min", "Games (min)"),
    ("computer_min", "Computer (min)"),
    ("screen_min", "Total screen (min)"),
    ("age", "Age"),
    ("female", "Female"),
    ("weekend", "Weekend"),
    ("employed", "Employed"),
    ("summer", "Summer"),
    ("income_code", "Income bracket"),
    ("parent_educ", "Parent education"),
]


def winsorize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("reading_min", "screen_hours"):
        cap = weighted_quantile(out[column].to_numpy(), out["weight"].to_numpy(), 0.99)
        out[column] = np.minimum(out[column], cap)
    return out


@dataclass
class Analysis:
    descriptives: pd.DataFrame
    correlations: pd.DataFrame
    bins: pd.DataFrame
    trends: pd.DataFrame
    models: dict[str, RegressionResult]
    keys: dict


def _describe(frame: pd.DataFrame) -> pd.DataFrame:
    w = frame["weight"].to_numpy()

    def row(measure: str, value: float, unit: str) -> dict:
        return {"measure": measure, "estimate": value, "unit": unit}

    enrolled = frame["enrolled"].eq("enrolled").astype(float)
    rows = [
        row("Respondents (one diary day each)", float(len(frame)), "count"),
        row("Reading for personal interest", weighted_mean(frame["reading_min"].to_numpy(), w), "minutes per day"),
        row("Any reading for personal interest", 100 * weighted_mean(frame["any_reading"].to_numpy(), w), "percent of days"),
        row("Television and movies", weighted_mean(frame["tv_min"].to_numpy(), w), "minutes per day"),
        row("Religious television", weighted_mean(frame["tv_religious_min"].to_numpy(), w), "minutes per day"),
        row("Playing games", weighted_mean(frame["games_min"].to_numpy(), w), "minutes per day"),
        row("Computer use for leisure", weighted_mean(frame["computer_min"].to_numpy(), w), "minutes per day"),
        row("Total screen", weighted_mean(frame["screen_min"].to_numpy(), w), "minutes per day"),
        row("Total screen", weighted_mean(frame["screen_hours"].to_numpy(), w), "hours per day"),
        row("Female", 100 * weighted_mean(frame["female"].to_numpy(), w), "percent"),
        row("Weekend diary", 100 * weighted_mean(frame["weekend"].to_numpy(), w), "percent"),
        row("Enrolled in school", 100 * weighted_mean(enrolled.to_numpy(), w), "percent"),
        row("Employed", 100 * weighted_mean(frame["employed"].to_numpy(), w), "percent"),
        row("Summer diary month", 100 * weighted_mean(frame["summer"].to_numpy(), w), "percent"),
        row("Parent education observed", 100 * frame["parent_educ"].notna().mean(), "percent of sample, unweighted"),
        row("Family income observed", 100 * frame["income_code"].notna().mean(), "percent of sample, unweighted"),
    ]
    return pd.DataFrame(rows)


def run_analysis(frame: pd.DataFrame) -> Analysis:
    descriptives = _describe(frame)
    corr_keys = [key for key, _label in CORR_COLUMNS]
    correlations = weighted_corr_matrix(frame, corr_keys)
    correlations.index = [label for _key, label in CORR_COLUMNS]
    correlations.columns = correlations.index
    bins = screen_bins(frame)
    trends = yearly_trends(frame)

    models: dict[str, RegressionResult] = {}
    models["bivariate"] = fit_wls(FORMULA_BIVARIATE, frame, "Bivariate weighted OLS")
    models["diary"] = fit_wls(FORMULA_DIARY, frame, "OLS with diary-day controls")
    models["full"] = fit_wls(FORMULA_FULL, frame, "OLS with full controls")
    models["reduced"] = fit_wls(f"reading_min ~ {CONTROLS}", frame, "OLS controls only")
    models["components"] = fit_wls(FORMULA_COMPONENTS, frame, "OLS screen components")
    models["no_2020"] = fit_wls(FORMULA_FULL, frame[frame["year"] != 2020].copy(), "OLS excluding 2020")
    models["winsor"] = fit_wls(FORMULA_FULL, winsorize_frame(frame), "OLS winsorized at weighted 99th percentile")
    readers = frame[frame["any_reading"] == 1].copy()
    models["intensive"] = fit_wls(FORMULA_FULL, readers, "OLS among diary days with any reading")
    models["logit"] = fit_logit(FORMULA_ANY, frame, "Weighted logit for any reading")
    tobit = fit_tobit(FORMULA_FULL, frame, "Weighted Tobit")
    if tobit is not None:
        models["tobit"] = tobit

    full = models["full"]
    screen = full.term("screen_hours")
    corr = weighted_corr(frame["screen_min"].to_numpy(), frame["reading_min"].to_numpy(), frame["weight"].to_numpy())
    p10 = weighted_quantile(frame["screen_hours"].to_numpy(), frame["weight"].to_numpy(), 0.10)
    p90 = weighted_quantile(frame["screen_hours"].to_numpy(), frame["weight"].to_numpy(), 0.90)
    keys = {
        "n": int(len(frame)),
        "year_min": int(frame["year"].min()),
        "year_max": int(frame["year"].max()),
        "reading_min": float(descriptives.loc[descriptives["measure"] == "Reading for personal interest", "estimate"].iloc[0]),
        "any_reading_pct": float(descriptives.loc[descriptives["measure"] == "Any reading for personal interest", "estimate"].iloc[0]),
        "screen_min": float(descriptives.loc[descriptives["measure"] == "Total screen", "estimate"].iloc[0]),
        "screen_hours": float(descriptives.loc[descriptives["unit"] == "hours per day", "estimate"].iloc[0]),
        "tv_min": float(descriptives.loc[descriptives["measure"] == "Television and movies", "estimate"].iloc[0]),
        "games_min": float(descriptives.loc[descriptives["measure"] == "Playing games", "estimate"].iloc[0]),
        "computer_min": float(descriptives.loc[descriptives["measure"] == "Computer use for leisure", "estimate"].iloc[0]),
        "religious_tv_min": float(descriptives.loc[descriptives["measure"] == "Religious television", "estimate"].iloc[0]),
        "weekend_pct": float(descriptives.loc[descriptives["measure"] == "Weekend diary", "estimate"].iloc[0]),
        "enrolled_pct": float(descriptives.loc[descriptives["measure"] == "Enrolled in school", "estimate"].iloc[0]),
        "employed_pct": float(descriptives.loc[descriptives["measure"] == "Employed", "estimate"].iloc[0]),
        "female_pct": float(descriptives.loc[descriptives["measure"] == "Female", "estimate"].iloc[0]),
        "corr_screen_reading": corr,
        "ols_coef": float(screen["coef"]),
        "ols_se": float(screen["se"]),
        "ols_ci_low": float(screen["ci_low"]),
        "ols_ci_high": float(screen["ci_high"]),
        "ols_pvalue": float(screen["pvalue"]),
        "ols_r2": float(full.r2),
        "reduced_r2": float(models["reduced"].r2),
        "partial_r2": partial_r2(full, models["reduced"]),
        "ols_n": int(full.n),
        "screen_p10_hours": p10,
        "screen_p90_hours": p90,
        "implied_p90_p10_reading_min": float(screen["coef"] * (p90 - p10)),
        "logit_ame": float(models["logit"].extra["ame_screen"]),
        "logit_ame_se": float(models["logit"].extra["ame_se"]),
        "logit_mcfadden": float(models["logit"].extra["mcfadden_r2"]),
        "intensive_coef": float(models["intensive"].term("screen_hours")["coef"]),
        "intensive_ci_low": float(models["intensive"].term("screen_hours")["ci_low"]),
        "intensive_ci_high": float(models["intensive"].term("screen_hours")["ci_high"]),
        "intensive_n": int(models["intensive"].n),
        "bivariate_coef": float(models["bivariate"].term("screen_hours")["coef"]),
        "bivariate_r2": float(models["bivariate"].r2),
    }
    for piece in ("tv_hours", "games_hours", "computer_hours"):
        row = models["components"].term(piece)
        keys[f"component_{piece}"] = float(row["coef"])
        keys[f"component_{piece}_ci_low"] = float(row["ci_low"])
        keys[f"component_{piece}_ci_high"] = float(row["ci_high"])
    if "tobit" in models:
        row = models["tobit"].term("screen_hours")
        keys["tobit_coef"] = float(row["coef"])
        keys["tobit_ci_low"] = float(row["ci_low"])
        keys["tobit_ci_high"] = float(row["ci_high"])
    else:
        keys["tobit_coef"] = None
    return Analysis(descriptives, correlations, bins, trends, models, keys)


def specification_table(analysis: Analysis) -> pd.DataFrame:
    """Screen-time coefficient across specifications, in comparable units where possible."""
    rows = []

    def add(key: str, label: str, unit: str, coef: float, se: float, n: int, fit: float | None, note: str) -> None:
        crit = stats.t.ppf(0.975, max(n - 2, 1))
        rows.append(
            {
                "spec": key,
                "label": label,
                "estimate": coef,
                "se": se,
                "ci_low": coef - crit * se,
                "ci_high": coef + crit * se,
                "n": n,
                "fit": fit,
                "unit": unit,
                "note": note,
            }
        )

    order = [
        ("bivariate", "Bivariate OLS", "Minutes of reading per extra screen hour"),
        ("diary", "OLS + weekend, holiday, summer, year", "Minutes of reading per extra screen hour"),
        ("full", "OLS + full controls", "Minutes of reading per extra screen hour"),
        ("no_2020", "OLS + full controls, excluding 2020", "Minutes of reading per extra screen hour"),
        ("winsor", "OLS + full controls, winsorized", "Minutes of reading per extra screen hour"),
        ("intensive", "OLS among days with any reading", "Minutes of reading per extra screen hour, readers only"),
    ]
    for key, label, unit in order:
        result = analysis.models[key]
        row = result.term("screen_hours")
        add(key, label, unit, float(row["coef"]), float(row["se"]), result.n, result.r2, "Weighted R²" if result.r2 is not None else "")
    if "tobit" in analysis.models:
        result = analysis.models["tobit"]
        row = result.term("screen_hours")
        add(
            "tobit",
            "Tobit, censored at zero",
            "Latent minutes of reading per extra screen hour",
            float(row["coef"]),
            float(row["se"]),
            result.n,
            None,
            "Model-based SE from the weighted likelihood",
        )
    logit = analysis.models["logit"]
    ame = float(logit.extra["ame_screen"])
    ame_se = float(logit.extra["ame_se"])
    add(
        "logit",
        "Logit, any reading",
        "Change in the probability of any reading per extra screen hour",
        ame,
        ame_se,
        logit.n,
        logit.r2,
        "Average marginal effect; fit is McFadden pseudo-R²",
    )
    for piece, label in (
        ("tv_hours", "TV and movies, from the component OLS"),
        ("games_hours", "Games, from the component OLS"),
        ("computer_hours", "Leisure computer use, from the component OLS"),
    ):
        result = analysis.models["components"]
        row = result.term(piece)
        add(
            piece,
            label,
            "Minutes of reading per extra hour of this component",
            float(row["coef"]),
            float(row["se"]),
            result.n,
            result.r2,
            "Other screen components and the full controls are in the model",
        )
    return pd.DataFrame(rows)
