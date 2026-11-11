"""Build the age 15–18 analysis file from ATUS microdata.

Activity minutes come from the Activity Summary file. Diary month comes from
the Respondent file. Family income and parental education come from the
ATUS-CPS file, linked to parents identified on the Roster file.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from atus_teen.download import ensure_files
from atus_teen.paths import extract_path, manifest_path, raw_dir, repo_root

# Activity-summary minutes. Codes are the 2003–2024 harmonized lexicon.
# Labels are the BLS variable labels shipped in atussum_0324.sas.
ACTIVITY_CODES = {
    "t120303": "Television and movies (not religious)",
    "t120304": "Television (religious)",
    "t120307": "Playing games (includes video and computer games, and also board games, cards, and puzzles)",
    "t120308": "Computer use for leisure, excluding games (includes leisure browsing and social-media sites; device is not identified)",
    "t120312": "Reading for personal interest (includes print, e-readers, and audiobooks)",
}

SUMMARY_COLS = [
    "TUCASEID",
    "GTMETSTA",
    "PEHSPNON",
    "PTDTRACE",
    "TEAGE",
    "TELFS",
    "TESCHENR",
    "TESCHLVL",
    "TESEX",
    "TRHOLIDAY",
    "TUDIARYDAY",
    "TUFNWGTP",
    "TUYEAR",
    "TU20FWGT",
    *ACTIVITY_CODES.keys(),
]

INCOME_LABELS = {
    "lt25": "Under $25,000",
    "25_50": "$25,000 to $49,999",
    "50_75": "$50,000 to $74,999",
    "75_100": "$75,000 to $99,999",
    "100plus": "$100,000 or more",
    "missing": "Missing",
}

PARENT_ED_LABELS = {
    "lt_hs": "Less than high school",
    "hs": "High school diploma",
    "some_college": "Some college or associate degree",
    "ba": "Bachelor's degree",
    "grad": "Graduate or professional degree",
    "missing": "Missing",
}


def _read_dat(zip_path: Path, inner: str, usecols: list[str], chunk_ids: set[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(inner) as handle:
            reader = pd.read_csv(
                handle,
                usecols=usecols,
                dtype={"TUCASEID": str},
                chunksize=250_000,
            )
            for chunk in reader:
                if chunk_ids is not None:
                    chunk = chunk[chunk["TUCASEID"].isin(chunk_ids)]
                if len(chunk):
                    frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def income_group(code: float | int | None) -> str:
    """Collapse CPS family-income codes 1–16 into nominal dollar bands.

    Codes 1–7 are under $25,000, 8–11 are $25,000–$49,999, 12–13 are
    $50,000–$74,999, 14 is $75,000–$99,999, and 15–16 are $100,000 or more.
    These brackets are not inflation-adjusted.
    """
    if code is None or (isinstance(code, float) and np.isnan(code)) or int(code) <= 0:
        return "missing"
    code = int(code)
    if code <= 7:
        return "lt25"
    if code <= 11:
        return "25_50"
    if code <= 13:
        return "50_75"
    if code == 14:
        return "75_100"
    if code <= 16:
        return "100plus"
    return "missing"


def parent_educ_group(code: float | int | None) -> str:
    """Collapse PEEDUCA of the highest-educated coresident parent."""
    if code is None or (isinstance(code, float) and np.isnan(code)) or int(code) < 31:
        return "missing"
    code = int(code)
    if code <= 38:
        return "lt_hs"
    if code == 39:
        return "hs"
    if code <= 42:
        return "some_college"
    if code == 43:
        return "ba"
    if code <= 46:
        return "grad"
    return "missing"


def _positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values > 0)


def _household_income(cps: pd.DataFrame) -> pd.DataFrame:
    """One income code per household.

    HEFAMINC (edited) is populated from 2010 onward. HUFAMINC (unedited) is
    the populated family-income field in 2003–2009. Both use the 1–16 bracket.
    """
    work = cps[["TUCASEID", "HEFAMINC", "HUFAMINC"]].copy()
    work["he"] = _positive(work["HEFAMINC"])
    work["hu"] = _positive(work["HUFAMINC"])
    grouped = work.groupby("TUCASEID", as_index=False).agg(he=("he", "max"), hu=("hu", "max"))
    use_edited = grouped["he"].notna()
    grouped["income_code"] = np.where(use_edited, grouped["he"], grouped["hu"])
    grouped["income_source"] = np.where(
        use_edited,
        "HEFAMINC",
        np.where(grouped["hu"].notna(), "HUFAMINC", "missing"),
    )
    grouped["income_group"] = [
        income_group(code) for code in grouped["income_code"]
    ]
    return grouped[["TUCASEID", "income_code", "income_source", "income_group"]]


def _parent_education(cps: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Highest PEEDUCA among roster members coded as the respondent's parent.

    Roster TERRP = 24 is 'Parent'. Education is the CPS value on that line.
    CPS education was collected two to five months before the diary day.
    """
    parents = roster.loc[roster["TERRP"] == 24, ["TUCASEID", "TULINENO"]].copy()
    educ = cps[["TUCASEID", "TULINENO", "PEEDUCA"]].copy()
    linked = parents.merge(educ, on=["TUCASEID", "TULINENO"], how="left")
    linked["PEEDUCA"] = pd.to_numeric(linked["PEEDUCA"], errors="coerce")
    linked.loc[linked["PEEDUCA"] < 31, "PEEDUCA"] = np.nan
    counts = linked.groupby("TUCASEID").size().rename("n_parents_on_roster")
    highest = linked.groupby("TUCASEID")["PEEDUCA"].max().rename("parent_educ")
    out = pd.concat([highest, counts], axis=1).reset_index()
    out["parent_educ_group"] = [parent_educ_group(code) for code in out["parent_educ"]]
    return out


def _race_group(code: int) -> str:
    # PTDTRACE: 1 white only, 2 black only, 4 Asian only. Those three codes are
    # stable across 2003–2024. Remaining codes, including multiracial codes
    # that were expanded over time, are grouped together.
    if code == 1:
        return "white"
    if code == 2:
        return "black"
    if code == 4:
        return "asian"
    return "other"


def _labor_force(code: int) -> str:
    # TELFS: 1 employed at work, 2 employed absent, 3 unemployed on layoff,
    # 4 unemployed looking, 5 not in labor force.
    if code in (1, 2):
        return "employed"
    if code in (3, 4):
        return "unemployed"
    return "nilf"


def _enrolled(code: int) -> str:
    # TESCHENR universe is ages 15–49. 1 yes, 2 no. -3 is refused (rare).
    if code == 1:
        return "enrolled"
    if code == 2:
        return "not_enrolled"
    return "unknown"


def _metro(code: int) -> str:
    # GTMETSTA: 1 metropolitan, 2 nonmetropolitan, 3 not identified, -1 blank.
    if code == 1:
        return "metro"
    if code == 2:
        return "nonmetro"
    return "not_identified"


def build_extract(force_download: bool = False) -> pd.DataFrame:
    """Download any missing BLS files and write the teen extract."""
    file_report = ensure_files(force=force_download)
    raw = raw_dir()

    summary = _read_dat(raw / "atussum-0324.zip", "atussum_0324.dat", SUMMARY_COLS)
    summary["TEAGE"] = pd.to_numeric(summary["TEAGE"], errors="coerce")
    teens = summary[summary["TEAGE"].between(15, 18)].copy()
    if teens["TUCASEID"].duplicated().any():
        raise RuntimeError("Duplicate TUCASEID in the Activity Summary file")
    teen_ids = set(teens["TUCASEID"])

    respondent = _read_dat(
        raw / "atusresp-0324.zip",
        "atusresp_0324.dat",
        ["TUCASEID", "TUMONTH", "TUDIARYDATE"],
        chunk_ids=teen_ids,
    )
    roster = _read_dat(
        raw / "atusrost-0324.zip",
        "atusrost_0324.dat",
        ["TUCASEID", "TULINENO", "TERRP", "TEAGE", "TESEX"],
        chunk_ids=teen_ids,
    )
    cps = _read_dat(
        raw / "atuscps-0324.zip",
        "atuscps_0324.dat",
        ["TUCASEID", "TULINENO", "HEFAMINC", "HUFAMINC", "PEEDUCA"],
        chunk_ids=teen_ids,
    )

    # Roster line 1 is the respondent. Age and sex must match the summary.
    respondent_roster = roster.loc[roster["TULINENO"] == 1, ["TUCASEID", "TEAGE", "TESEX"]]
    check = teens.merge(respondent_roster, on="TUCASEID", suffixes=("", "_roster"))
    if not np.allclose(check["TEAGE"], check["TEAGE_roster"]):
        raise RuntimeError("Respondent age disagrees between summary and roster")
    if not np.allclose(check["TESEX"], check["TESEX_roster"]):
        raise RuntimeError("Respondent sex disagrees between summary and roster")

    income = _household_income(cps)
    parents = _parent_education(cps, roster)
    frame = (
        teens.merge(respondent, on="TUCASEID", how="left", validate="one_to_one")
        .merge(income, on="TUCASEID", how="left", validate="one_to_one")
        .merge(parents, on="TUCASEID", how="left", validate="one_to_one")
    )
    analysis = _assemble(frame)
    _assert_weights(analysis)

    out = extract_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_parquet(out, index=False)
    manifest = {
        "n": int(len(analysis)),
        "years": [int(analysis["year"].min()), int(analysis["year"].max())],
        "extract": "data/extract/teens_15_18.parquet",
        "activity_codes": ACTIVITY_CODES,
        "files": file_report,
        "notes": [
            "Age filter is TEAGE 15 through 18 inclusive.",
            "Survey weight is TU20FWGT in 2020 and TUFNWGTP in every other year.",
            "2020 weights represent January 1–March 17 and May 10–December 31 only.",
            "Family income is HEFAMINC when that edited field is coded, otherwise HUFAMINC.",
            "Parent education is the highest PEEDUCA among roster persons with TERRP 24.",
        ],
    }
    manifest_path().write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out} ({len(analysis):,} teens)")
    return analysis


def _assemble(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["tucaseid"] = frame["TUCASEID"].astype(str)
    out["year"] = pd.to_numeric(frame["TUYEAR"], errors="coerce").astype(int)
    out["month"] = pd.to_numeric(frame["TUMONTH"], errors="coerce").astype(int)
    out["diary_date"] = pd.to_numeric(frame["TUDIARYDATE"], errors="coerce").astype(int)
    out["diary_day"] = pd.to_numeric(frame["TUDIARYDAY"], errors="coerce").astype(int)
    out["age"] = pd.to_numeric(frame["TEAGE"], errors="coerce").astype(int)
    out["sex"] = pd.to_numeric(frame["TESEX"], errors="coerce").astype(int)
    out["female"] = (out["sex"] == 2).astype(int)

    tufn = pd.to_numeric(frame["TUFNWGTP"], errors="coerce")
    tu20 = pd.to_numeric(frame["TU20FWGT"], errors="coerce")
    out["weight"] = np.where(out["year"].eq(2020), tu20, tufn)
    out["weight_var"] = np.where(out["year"].eq(2020), "TU20FWGT", "TUFNWGTP")

    for column in ACTIVITY_CODES:
        out[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    out["tv_nonreligious_min"] = out["t120303"]
    out["tv_religious_min"] = out["t120304"]
    out["tv_min"] = out["t120303"] + out["t120304"]
    out["games_min"] = out["t120307"]
    out["computer_min"] = out["t120308"]
    out["reading_min"] = out["t120312"]
    out["screen_min"] = out["tv_min"] + out["games_min"] + out["computer_min"]
    out["any_reading"] = (out["reading_min"] > 0).astype(int)
    for stem in ("tv", "games", "computer", "screen", "reading"):
        out[f"{stem}_hours"] = out[f"{stem}_min"] / 60.0

    enrolled_code = pd.to_numeric(frame["TESCHENR"], errors="coerce")
    school_level = pd.to_numeric(frame["TESCHLVL"], errors="coerce")
    out["enrolled"] = [_enrolled(int(code)) for code in enrolled_code.fillna(-1)]
    out["in_college"] = (school_level == 2).astype(int)
    labor = pd.to_numeric(frame["TELFS"], errors="coerce")
    out["labor_force"] = [_labor_force(int(code)) for code in labor.fillna(5)]
    out["employed"] = (out["labor_force"] == "employed").astype(int)
    out["weekend"] = out["diary_day"].isin([1, 7]).astype(int)
    out["holiday"] = pd.to_numeric(frame["TRHOLIDAY"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out["summer"] = out["month"].isin([6, 7, 8]).astype(int)
    metro = pd.to_numeric(frame["GTMETSTA"], errors="coerce")
    out["metro"] = [_metro(int(code)) for code in metro.fillna(-1)]
    hispanic = pd.to_numeric(frame["PEHSPNON"], errors="coerce")
    out["hispanic"] = (hispanic == 1).astype(int)
    race = pd.to_numeric(frame["PTDTRACE"], errors="coerce").fillna(-1)
    out["race"] = [_race_group(int(code)) for code in race]
    out["income_code"] = pd.to_numeric(frame["income_code"], errors="coerce")
    out["income_source"] = frame["income_source"].fillna("missing")
    out["income_group"] = frame["income_group"].fillna("missing")
    out["parent_educ"] = pd.to_numeric(frame["parent_educ"], errors="coerce")
    out["parent_educ_group"] = frame["parent_educ_group"].fillna("missing")
    out["n_parents_on_roster"] = pd.to_numeric(frame["n_parents_on_roster"], errors="coerce").fillna(0).astype(int)

    keep = out["weight"].gt(0) & out["age"].between(15, 18)
    out = out.loc[keep].reset_index(drop=True)
    if out[["reading_min", "screen_min"]].lt(0).any().any():
        raise RuntimeError("Negative activity minutes")
    if (out["reading_min"] + out["screen_min"] > 1440 + 1e-6).any():
        raise RuntimeError("Reading plus screen exceeds the 1,440-minute diary day")
    return out


def _assert_weights(df: pd.DataFrame) -> None:
    """Weekend diaries are half the sample by design; weights should pull the share near 2/7."""
    if df.empty:
        raise RuntimeError("No teen respondents with a positive weight")
    weekend = np.average(df["weekend"], weights=df["weight"])
    if not 0.22 <= weekend <= 0.36:
        raise RuntimeError(
            f"Weighted weekend share is {weekend:.3f}; expected about 2/7. "
            "The 2020 weight switch may be wrong."
        )


def load_analysis_frame(rebuild: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load the committed extract, or rebuild it from the BLS zips."""
    path = extract_path()
    missing = not path.exists()
    if rebuild or missing:
        frame = build_extract()
    else:
        frame = pd.read_parquet(path)
    try:
        shown = str(path.relative_to(repo_root()))
    except ValueError:
        shown = str(path)
    info = {
        "n": int(len(frame)),
        "year_min": int(frame["year"].min()),
        "year_max": int(frame["year"].max()),
        "extract": shown,
        "rebuilt": bool(rebuild or missing),
    }
    if manifest_path().exists():
        info["manifest"] = json.loads(manifest_path().read_text())
    return frame, info
