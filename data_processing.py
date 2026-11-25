"""
data_processing.py
==================
Loads the raw football injury dataset, checks its quality, cleans it and
creates the calculated columns used everywhere else in the project.

The ORIGINAL file is never modified. Cleaning always happens on a copy.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

# Path is built relative to this file, so it works locally AND on Streamlit Cloud.
_BASE = Path(__file__).parent
_CANDIDATES = [_BASE / "data" / "dataset.csv",
               _BASE / "dataset.csv",                      # flat repo layout
               _BASE / "data" / "player_injuries_impact.csv",
               _BASE / "player_injuries_impact.csv"]
DATA_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])

PHASES = ["before", "during", "after"]          # before injury / missed matches / after return
POINTS = {"win": 3, "draw": 1, "lose": 0}         # football points system

# Words found in the raw 'Injury' column -> broader category used for filtering.
# The first keyword that matches wins, so order matters.
INJURY_CATEGORIES = [
    ("Hamstring", ["hamstring"]),
    ("Knee", ["knee", "cruciate", "acl", "meniscus"]),
    ("Ankle", ["ankle"]),
    ("Calf", ["calf"]),
    ("Groin", ["groin", "hernia"]),
    ("Thigh", ["thigh", "quad"]),
    ("Foot", ["foot", "toe", "metatarsal", "heel", "achilles"]),
    ("Back", ["back", "spine"]),
    ("Shoulder / Arm", ["shoulder", "arm", "hand", "wrist", "elbow", "finger"]),
    ("Head / Concussion", ["head", "concussion", "facial", "nose", "eye", "jaw"]),
    ("Illness / Covid", ["ill", "coronavirus", "covid", "virus", "infection", "flu", "fever"]),
    ("Muscle (general)", ["muscle", "strain", "fatigue", "tear"]),
    ("Knock / Bruise", ["knock", "bruise", "dead leg", "contusion"]),
    ("Unknown", ["unknown", "fitness", "personal", "rest"]),
]


# --------------------------------------------------------------------------
# 1. LOADING
# --------------------------------------------------------------------------
def load_raw_data(path=DATA_PATH):
    """Read the CSV exactly as provided. 'N.A.' and blanks become NaN (missing)."""
    return pd.read_csv(path, na_values=["N.A.", "N.A", "NA", ""], keep_default_na=False)


# --------------------------------------------------------------------------
# 2. DATA QUALITY REPORT (shown on the Data Information page)
# --------------------------------------------------------------------------
def data_quality_report(raw):
    """Summarise shape, missing values and duplicates of the ORIGINAL data."""
    missing = raw.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return {
        "rows": raw.shape[0],
        "columns": raw.shape[1],
        "duplicate_rows": int(raw.duplicated().sum()),
        "total_missing_cells": int(raw.isna().sum().sum()),
        "missing_by_column": missing.rename("Missing values").to_frame(),
    }


# --------------------------------------------------------------------------
# 3. CLEANING HELPERS
# --------------------------------------------------------------------------
def parse_dates(series):
    """
    The raw dates come in mixed styles: 'Nov 9, 2019', 'Dec 6,2019' (no space),
    'January 4, 2023', and the word 'Present'. We normalise the spacing and
    try each format. Anything that still fails becomes NaT (not guessed).
    """
    s = series.astype("string").str.strip().str.replace(r",\s*", ", ", regex=True)
    parsed = pd.to_datetime(s, format="%b %d, %Y", errors="coerce")
    # second attempt for full month names (e.g. 'January 4, 2023')
    retry = pd.to_datetime(s[parsed.isna()], format="%B %d, %Y", errors="coerce")
    parsed = parsed.fillna(retry)
    return parsed


def categorise_injury(text):
    """Map a free-text injury description to a broad category."""
    if pd.isna(text):
        return "Unknown"
    t = str(text).lower()
    for category, keywords in INJURY_CATEGORIES:
        for k in keywords:
            # 'ill' must be a whole word, otherwise it matches 'chill', 'illness' is fine
            if k == "ill":
                if re.search(r"\bill(ness)?\b", t):
                    return category
            elif k in t:
                return category
    return "Other"


def _rename_match_columns(df):
    """'Match1_before_injury_Player_rating' -> 'before1_rating', and so on."""
    mapping = {
        "Name": "player", "Team Name": "club", "Position": "position", "Age": "age",
        "Season": "season", "FIFA rating": "fifa_rating", "Injury": "injury_raw",
        "Date of Injury": "injury_date_raw", "Date of return": "return_date_raw",
    }
    phase_names = {"before_injury": "before", "missed_match": "during", "after_injury": "after"}
    stat_names = {"Result": "result", "Opposition": "opponent", "GD": "gd", "Player_rating": "rating"}
    for i in (1, 2, 3):
        for raw_phase, phase in phase_names.items():
            for raw_stat, stat in stat_names.items():
                mapping[f"Match{i}_{raw_phase}_{raw_stat}"] = f"{phase}{i}_{stat}"
    return df.rename(columns=mapping)


# --------------------------------------------------------------------------
# 4. MAIN CLEANING + FEATURE ENGINEERING
# --------------------------------------------------------------------------
def clean_data(raw):
    """
    Returns a cleaned copy of the dataset with calculated columns.

    Missing values are NOT deleted or filled with invented numbers.
    A missing match simply does not count towards an average
    (pandas skips NaN automatically), so every average uses real data only.
    """
    df = _rename_match_columns(raw.copy())

    # ---- text columns: remove stray spaces (e.g. 'Central Midfielder ') ----
    for col in ["player", "club", "position", "season", "injury_raw"]:
        df[col] = df[col].astype("string").str.strip()
    df["position"] = df["position"].str.replace(r"\s+", " ", regex=True).str.title()

    # ---- injury type: unify capitalisation ('hamstring injury' = 'Hamstring injury') ----
    df["injury"] = df["injury_raw"].str.lower().str.strip().str.capitalize()
    df["injury_category"] = df["injury_raw"].apply(categorise_injury)

    # ---- dates ----
    df["injury_date"] = parse_dates(df["injury_date_raw"])
    df["return_date"] = parse_dates(df["return_date_raw"])
    # 'Present' means the player had not returned when the data was collected
    df["still_injured"] = df["return_date_raw"].astype("string").str.strip().str.lower().eq("present")

    # A return date before the injury date is impossible (typos such as 'Mar 7, 0202').
    df["date_invalid"] = df["return_date"].notna() & df["injury_date"].notna() & (df["return_date"] < df["injury_date"])
    df.loc[df["date_invalid"], "return_date"] = pd.NaT

    df["injury_duration_days"] = (df["return_date"] - df["injury_date"]).dt.days

    # ---- calendar helpers ----
    df["injury_year"] = df["injury_date"].dt.year
    df["injury_month_num"] = df["injury_date"].dt.month
    df["injury_month"] = df["injury_date"].dt.strftime("%b")

    # ---- numeric columns ----
    for i in (1, 2, 3):
        for ph in PHASES:
            df[f"{ph}{i}_gd"] = pd.to_numeric(df[f"{ph}{i}_gd"], errors="coerce")
            df[f"{ph}{i}_result"] = df[f"{ph}{i}_result"].astype("string").str.lower().str.strip()
        for ph in ("before", "after"):
            col = f"{ph}{i}_rating"
            text = df[col].astype("string")
            # '6.1(S)' = came on as a substitute; keep the number and remember the flag
            df[f"{ph}{i}_sub"] = text.str.contains(r"\(S\)", na=False)
            df[col] = pd.to_numeric(text.str.extract(r"^(\d+(?:\.\d+)?)")[0], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # ---- rating sanity check: valid football ratings are between 0 and 10 ----
    for ph in ("before", "after"):
        for i in (1, 2, 3):
            bad = (df[f"{ph}{i}_rating"] < 0) | (df[f"{ph}{i}_rating"] > 10)
            df.loc[bad, f"{ph}{i}_rating"] = np.nan

    df = add_features(df)

    df = _to_plain_dtypes(df)
    df.insert(0, "case_id", np.arange(1, len(df) + 1))
    df["case_label"] = (df["player"] + " - " + df["injury"].fillna("Unknown")
                        + " (" + df["injury_date"].dt.strftime("%b %Y").fillna("date n/a") + ")").astype(object)
    return df


def _to_plain_dtypes(df):
    """
    Convert pandas 'extension' dtypes (string, Float64, pyarrow ...) to plain
    NumPy dtypes so Plotly and Streamlit behave identically on every pandas version.
    """
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dt):
            continue
        if pd.api.types.is_bool_dtype(dt):
            df[col] = df[col].fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(dt):
            df[col] = df[col].astype(float) if pd.api.types.is_float_dtype(dt) else df[col]
            if str(dt) not in ("float64", "int64", "int32"):
                df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(object)
    return df


def add_features(df):
    """Create the calculated columns requested in the assignment."""
    for ph in PHASES:
        res_cols = [f"{ph}{i}_result" for i in (1, 2, 3)]
        gd_cols = [f"{ph}{i}_gd" for i in (1, 2, 3)]

        points = df[res_cols].apply(lambda c: c.map(POINTS)).astype(float)
        df[f"{ph}_matches"] = df[res_cols].notna().sum(axis=1)
        df[f"{ph}_ppm"] = points.mean(axis=1)                    # points per match (0 - 3)
        df[f"{ph}_gd_avg"] = df[gd_cols].mean(axis=1)            # average goal difference
        df[f"{ph}_wins"] = (df[res_cols] == "win").sum(axis=1)
        df[f"{ph}_draws"] = (df[res_cols] == "draw").sum(axis=1)
        df[f"{ph}_losses"] = (df[res_cols] == "lose").sum(axis=1)

    # Player rating averages (only matches where a rating exists)
    df["rating_before"] = df[[f"before{i}_rating" for i in (1, 2, 3)]].mean(axis=1)
    df["rating_after"] = df[[f"after{i}_rating" for i in (1, 2, 3)]].mean(axis=1)
    df["rating_change"] = df["rating_after"] - df["rating_before"]

    # TEAM PERFORMANCE DROP INDEX = avg goal difference BEFORE minus avg goal difference DURING absence.
    # Positive value = the team did worse while the player was out.
    df["perf_drop_index"] = df["before_gd_avg"] - df["during_gd_avg"]
    # Same idea with points per match (easier for managers to read)
    df["ppm_drop"] = df["before_ppm"] - df["during_ppm"]
    # Did the team bounce back after the player returned?
    df["team_recovery_gd"] = df["after_gd_avg"] - df["during_gd_avg"]

    df["matches_missed"] = df["during_matches"]
    return df


def load_clean_data(path=DATA_PATH):
    """Convenience wrapper: returns (raw, clean)."""
    raw = load_raw_data(path)
    return raw, clean_data(raw)
