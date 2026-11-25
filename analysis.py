"""
analysis.py
===========
All statistical calculations: groupby / agg / pivot_table summaries,
before-during-after comparisons, rankings and the automatic text findings.

Every function takes the (already filtered) DataFrame and returns a
DataFrame, a number or text. Nothing here draws charts.
"""

import numpy as np
import pandas as pd

MONTH_ORDER = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]  # football season order
SMALL_SAMPLE = 5   # clubs / players with fewer cases than this are flagged as small samples


def _fmt(x, d=2):
    """Format a number safely (handles NaN)."""
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
def compute_kpis(df):
    """Headline numbers for the Overview cards."""
    rating_cols = [f"{ph}{i}_rating" for ph in ("before", "after") for i in (1, 2, 3)]
    all_ratings = df[rating_cols].to_numpy(dtype=float).ravel()
    all_ratings = all_ratings[~np.isnan(all_ratings)]
    dur = df["injury_duration_days"].dropna()
    return {
        "players": df["player"].nunique(),
        "cases": len(df),
        "clubs": df["club"].nunique(),
        "avg_rating": all_ratings.mean() if len(all_ratings) else np.nan,
        "rated_matches": len(all_ratings),
        "avg_duration": dur.mean() if len(dur) else np.nan,
        "median_duration": dur.median() if len(dur) else np.nan,
        "duration_n": len(dur),
    }


# --------------------------------------------------------------------------
# INJURY ANALYSIS (Page 2)
# --------------------------------------------------------------------------
def injuries_per_player(df, top_n=10):
    """Players with the most injury cases (groupby + agg)."""
    out = (df.groupby(["player", "club"])
             .agg(injuries=("case_id", "count"),
                  avg_days_out=("injury_duration_days", "mean"),
                  most_common_injury=("injury_category", lambda s: s.mode().iat[0]))
             .reset_index()
             .sort_values(["injuries", "avg_days_out"], ascending=[False, False]))
    return out.head(top_n)


def injuries_per_club(df):
    """Injury cases, players, and average duration per club."""
    return (df.groupby("club")
              .agg(injuries=("case_id", "count"),
                   players=("player", "nunique"),
                   avg_days_out=("injury_duration_days", "mean"),
                   median_days_out=("injury_duration_days", "median"))
              .assign(injuries_per_player=lambda t: t["injuries"] / t["players"])
              .reset_index()
              .sort_values("injuries", ascending=False))


def injuries_by_category(df):
    return (df.groupby("injury_category").agg(cases=("case_id", "count"))
              .reset_index().sort_values("cases", ascending=False))


def injuries_by_month(df):
    """Injury count per calendar month (football-season order)."""
    t = df.dropna(subset=["injury_month"]).groupby("injury_month").size().reindex(MONTH_ORDER, fill_value=0)
    return t.rename("injuries").reset_index().rename(columns={"index": "injury_month"})


def month_club_pivot(df):
    """pivot_table: rows = club, columns = month, values = number of injuries."""
    d = df.dropna(subset=["injury_month"])
    if d.empty:
        return pd.DataFrame()
    pv = pd.pivot_table(d, index="club", columns="injury_month", values="case_id",
                        aggfunc="count", fill_value=0)
    return pv.reindex(columns=MONTH_ORDER, fill_value=0)


def injuries_by_season(df):
    return (df.groupby(["season", "club"]).size().rename("injuries").reset_index()
              .sort_values("season"))


def duration_by_category(df, min_cases=3):
    """Mean / median / std of recovery days per injury category."""
    t = (df.dropna(subset=["injury_duration_days"])
           .groupby("injury_category")["injury_duration_days"]
           .agg(["count", "mean", "median", "std"]).reset_index())
    t = t[t["count"] >= min_cases].sort_values("median", ascending=False)
    return t.rename(columns={"count": "cases", "mean": "mean_days", "median": "median_days", "std": "std_days"})


def peak_month_info(df):
    m = injuries_by_month(df)
    if m["injuries"].sum() == 0:
        return None
    top = m.loc[m["injuries"].idxmax()]
    share = top["injuries"] / m["injuries"].sum() * 100
    return top["injury_month"], int(top["injuries"]), share


# --------------------------------------------------------------------------
# TEAM PERFORMANCE (Page 3)
# --------------------------------------------------------------------------
def team_phase_summary(df):
    """Per club: average points-per-match and goal difference BEFORE / DURING / AFTER."""
    t = (df.groupby("club")
           .agg(cases=("case_id", "count"),
                ppm_before=("before_ppm", "mean"), ppm_during=("during_ppm", "mean"), ppm_after=("after_ppm", "mean"),
                gd_before=("before_gd_avg", "mean"), gd_during=("during_gd_avg", "mean"), gd_after=("after_gd_avg", "mean"),
                drop_index=("perf_drop_index", "mean"), drop_median=("perf_drop_index", "median"),
                drop_std=("perf_drop_index", "std"), ppm_drop=("ppm_drop", "mean"))
           .reset_index())
    return t.sort_values("drop_index", ascending=False)


def win_loss_record(df):
    """
    Business question 2: team results in the matches the player missed.
    Returns one row per club plus an 'All clubs' row.
    """
    g = df.groupby("club").agg(wins=("during_wins", "sum"), draws=("during_draws", "sum"),
                               losses=("during_losses", "sum"), cases=("case_id", "count")).reset_index()
    g["matches"] = g["wins"] + g["draws"] + g["losses"]
    g["win_pct"] = g["wins"] / g["matches"] * 100
    g["draw_pct"] = g["draws"] / g["matches"] * 100
    g["loss_pct"] = g["losses"] / g["matches"] * 100
    g["points_per_match"] = (g["wins"] * 3 + g["draws"]) / g["matches"]
    return g.sort_values("win_pct", ascending=False)


def overall_record(df):
    """Combined W/D/L before, during and after (all clubs in the filtered data)."""
    rows = []
    for ph, label in [("before", "Before injury"), ("during", "During absence"), ("after", "After return")]:
        w, d, l = df[f"{ph}_wins"].sum(), df[f"{ph}_draws"].sum(), df[f"{ph}_losses"].sum()
        n = w + d + l
        rows.append({"phase": label, "wins": w, "draws": d, "losses": l, "matches": n,
                     "win_pct": w / n * 100 if n else np.nan,
                     "draw_pct": d / n * 100 if n else np.nan,
                     "loss_pct": l / n * 100 if n else np.nan,
                     "points_per_match": (3 * w + d) / n if n else np.nan,
                     "avg_gd": df[[f"{ph}{i}_gd" for i in (1, 2, 3)]].stack().mean()})
    return pd.DataFrame(rows)


def top_drop_cases(df, n=10):
    """Business question 1: individual injury cases with the biggest team drop."""
    cols = ["case_label", "player", "club", "injury", "season", "age", "before_gd_avg", "during_gd_avg",
            "perf_drop_index", "before_ppm", "during_ppm", "ppm_drop", "matches_missed"]
    return (df.dropna(subset=["perf_drop_index"])
              .sort_values("perf_drop_index", ascending=False)[cols].head(n).reset_index(drop=True))


def drop_by_category(df, min_cases=5):
    """Which injury types are associated with the biggest average drop?"""
    t = (df.dropna(subset=["perf_drop_index"]).groupby("injury_category")["perf_drop_index"]
           .agg(["count", "mean", "median", "std"]).reset_index())
    t = t[t["count"] >= min_cases].sort_values("mean", ascending=False)
    return t.rename(columns={"count": "cases", "mean": "mean_drop", "median": "median_drop", "std": "std_drop"})


def age_drop_correlation(df):
    """Pearson r and best-fit line (numpy) for age vs drop index."""
    d = df.dropna(subset=["age", "perf_drop_index"])
    if len(d) < 3 or d["age"].nunique() < 2:
        return None
    r = np.corrcoef(d["age"], d["perf_drop_index"])[0, 1]
    slope, intercept = np.polyfit(d["age"], d["perf_drop_index"], 1)
    return {"r": r, "slope": slope, "intercept": intercept, "n": len(d)}


def drop_summary_stats(df):
    """Mean, median, std and an approximate 95% confidence interval of the drop index."""
    x = df["perf_drop_index"].dropna()
    if len(x) < 2:
        return None
    se = x.std(ddof=1) / np.sqrt(len(x))
    return {"n": len(x), "mean": x.mean(), "median": x.median(), "std": x.std(ddof=1),
            "ci_low": x.mean() - 1.96 * se, "ci_high": x.mean() + 1.96 * se,
            "share_worse": (x > 0).mean() * 100, "share_better": (x < 0).mean() * 100}


# --------------------------------------------------------------------------
# PLAYER RECOVERY (Page 4)
# --------------------------------------------------------------------------
def player_summary(df):
    """
    Group by player name and summarise all three phases
    (individual rating before/after + team results before/during/after).
    """
    t = (df.groupby("player")
           .agg(club=("club", lambda s: s.mode().iat[0]),
                age=("age", "max"),
                injuries=("case_id", "count"),
                rating_before=("rating_before", "mean"),
                rating_after=("rating_after", "mean"),
                rating_change=("rating_change", "mean"),
                cases_with_both=("rating_change", "count"),
                team_gd_before=("before_gd_avg", "mean"),
                team_gd_during=("during_gd_avg", "mean"),
                team_gd_after=("after_gd_avg", "mean"),
                drop_index=("perf_drop_index", "mean"),
                avg_days_out=("injury_duration_days", "mean"))
           .reset_index())
    return t


def comeback_leaderboard(df, min_cases=1):
    """
    Players ranked by rating improvement (after - before).
    Only injury cases where BOTH a before and an after rating exist are used.
    """
    d = df.dropna(subset=["rating_change"])
    if d.empty:
        return pd.DataFrame()
    t = (d.groupby("player")
           .agg(club=("club", lambda s: s.mode().iat[0]),
                cases=("case_id", "count"),
                rating_before=("rating_before", "mean"),
                rating_after=("rating_after", "mean"),
                rating_change=("rating_change", "mean"))
           .reset_index())
    t = t[t["cases"] >= min_cases].sort_values("rating_change", ascending=False).reset_index(drop=True)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    return t


def rating_change_stats(df):
    """Paired comparison of player rating before vs after (mean difference, spread, share improved)."""
    x = df["rating_change"].dropna()
    if len(x) < 2:
        return None
    se = x.std(ddof=1) / np.sqrt(len(x))
    return {"n": len(x), "mean_before": df.loc[x.index, "rating_before"].mean(),
            "mean_after": df.loc[x.index, "rating_after"].mean(),
            "mean_change": x.mean(), "median_change": x.median(), "std_change": x.std(ddof=1),
            "ci_low": x.mean() - 1.96 * se, "ci_high": x.mean() + 1.96 * se,
            "t_stat": x.mean() / se if se > 0 else np.nan,
            "share_improved": (x > 0).mean() * 100, "share_declined": (x < 0).mean() * 100}


def position_recovery(df, min_cases=5):
    """Average rating change by position (recovery comparison between player types)."""
    t = (df.dropna(subset=["rating_change"]).groupby("position")["rating_change"]
           .agg(["count", "mean", "median", "std"]).reset_index())
    t = t[t["count"] >= min_cases].sort_values("mean", ascending=False)
    return t.rename(columns={"count": "cases", "mean": "mean_change", "median": "median_change", "std": "std_change"})


def recovery_by_duration(df):
    """Does a longer injury relate to a bigger rating change? (bands of days out)"""
    d = df.dropna(subset=["rating_change", "injury_duration_days"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["duration_band"] = pd.cut(d["injury_duration_days"], [0, 14, 30, 60, 10000],
                                labels=["1-14 days", "15-30 days", "31-60 days", "60+ days"])
    return (d.groupby("duration_band", observed=True)["rating_change"]
              .agg(["count", "mean", "median"]).reset_index()
              .rename(columns={"count": "cases", "mean": "mean_change", "median": "median_change"}))


def player_timeline(df, case_ids):
    """
    Long-format rating timeline (3 matches before, 3 after) for chosen cases.
    ASSUMPTION: 'Match1' is the match closest to the injury / return date.
    """
    rows = []
    order = {("before", 3): -3, ("before", 2): -2, ("before", 1): -1,
             ("after", 1): 1, ("after", 2): 2, ("after", 3): 3}
    for _, r in df[df["case_id"].isin(case_ids)].iterrows():
        for (ph, i), pos in order.items():
            rows.append({"case_id": r["case_id"], "case_label": r["case_label"], "player": r["player"],
                         "position": pos, "phase": ph, "match": f"{ph.title()} {i}",
                         "rating": r[f"{ph}{i}_rating"], "substitute": bool(r[f"{ph}{i}_sub"]),
                         "opponent": r[f"{ph}{i}_opponent"], "result": r[f"{ph}{i}_result"]})
    return pd.DataFrame(rows)


def average_timeline(df):
    """Average rating at each of the six timeline positions across all filtered cases."""
    out = []
    for ph, sign in (("before", -1), ("after", 1)):
        for i in (1, 2, 3):
            col = f"{ph}{i}_rating"
            out.append({"position": sign * i, "avg_rating": df[col].mean(), "n": int(df[col].notna().sum())})
    return pd.DataFrame(out).sort_values("position")


# --------------------------------------------------------------------------
# CLUB IMPACT (business question 5)
# --------------------------------------------------------------------------
def club_impact_table(df):
    """One row per club combining the key indicators of how affected it is."""
    inj = injuries_per_club(df)[["club", "injuries", "players", "avg_days_out"]]
    team = team_phase_summary(df)[["club", "drop_index", "ppm_drop", "gd_before", "gd_during"]]
    t = inj.merge(team, on="club", how="left")
    t["games_missed_total"] = t["club"].map(df.groupby("club")["matches_missed"].sum())
    t["small_sample"] = t["injuries"] < SMALL_SAMPLE
    return t.sort_values("drop_index", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# AUTOMATIC FINDINGS (Overview + Insights pages)
# --------------------------------------------------------------------------
def overview_summary(df):
    """Short bullet-point summary generated from the filtered data."""
    if df.empty:
        return []
    k = compute_kpis(df)
    out = [f"The selection contains **{k['cases']:,} injury cases** involving **{k['players']:,} players** from **{k['clubs']} clubs**."]
    clubs = injuries_per_club(df)
    top = clubs.iloc[0]
    out.append(f"**{top['club']}** has the most recorded injuries ({int(top['injuries'])}, "
               f"{top['injuries'] / k['cases'] * 100:.0f}% of the selection).")
    cat = injuries_by_category(df).iloc[0]
    out.append(f"The most common injury type is **{cat['injury_category']}** ({int(cat['cases'])} cases).")
    if not np.isnan(k["avg_duration"]):
        out.append(f"Players were out for **{_fmt(k['avg_duration'], 1)} days on average** "
                   f"(median {_fmt(k['median_duration'], 0)} days), so a few very long injuries pull the average above the median.")
    pk = peak_month_info(df)
    if pk:
        out.append(f"Injuries peak in **{pk[0]}** ({pk[1]} cases, {pk[2]:.0f}% of dated cases).")
    ds = drop_summary_stats(df)
    if ds:
        out.append(f"On average the team's goal difference was **{_fmt(ds['mean'], 2)} goals per match lower** during a player's absence "
                   f"than before it (based on {ds['n']} cases).")
    return out


def top_insights(df):
    """
    Five major insights, each calculated from the filtered data.
    Returns a list of dicts: {title, text}.
    """
    ins = []
    if df.empty:
        return ins

    # 1. Most injury-affected club by volume + by impact
    clubs = injuries_per_club(df)
    c = clubs.iloc[0]
    ins.append({"title": "Injury volume differs between clubs",
                "text": f"{c['club']} recorded the most injuries ({int(c['injuries'])} cases, "
                        f"{c['injuries'] / len(df) * 100:.0f}% of all cases). "
                        f"The lowest was {clubs.iloc[-1]['club']} with {int(clubs.iloc[-1]['injuries'])}. "
                        f"Note: the dataset only contains a sample of clubs, so this is volume in the dataset, not the league."})

    # 2. Team performance drop
    ds = drop_summary_stats(df)
    if ds:
        direction = "lower" if ds["mean"] > 0 else "higher"
        sig = ("The 95% confidence interval includes zero, so the overall average change is not clearly different from no change."
               if ds["ci_low"] <= 0 <= ds["ci_high"] else
               "The 95% confidence interval excludes zero, so the average change is unlikely to be pure chance.")
        clear = not (ds["ci_low"] <= 0 <= ds["ci_high"])
        title = ("Teams tend to do " + direction + " while a player is absent") if clear else "No clear average team drop while a player is absent"
        ins.append({"title": title,
                    "text": f"Average goal difference during absence was {_fmt(abs(ds['mean']))} goals per match {direction} than before "
                            f"(median {_fmt(ds['median'])}). {ds['share_worse']:.0f}% of cases got worse, {ds['share_better']:.0f}% improved. {sig}"})

    # 3. Individual rating after recovery
    rs = rating_change_stats(df)
    if rs:
        if rs["ci_low"] <= 0 <= rs["ci_high"]:
            rtitle = "Players return at about the same level on average"
        else:
            rtitle = "Players return " + ("better" if rs["mean_change"] > 0 else "worse") + " on average"
        ins.append({"title": rtitle,
                    "text": f"Across {rs['n']} comparable cases the average player rating moved from {_fmt(rs['mean_before'])} to "
                            f"{_fmt(rs['mean_after'])} (change {rs['mean_change']:+.2f}). "
                            f"{rs['share_improved']:.0f}% improved and {rs['share_declined']:.0f}% declined. "
                            f"Individual swings are large (std {_fmt(rs['std_change'])}) because only 3 matches are averaged on each side."})

    # 4. Seasonality
    pk = peak_month_info(df)
    if pk:
        m = injuries_by_month(df).sort_values("injuries", ascending=False)
        ins.append({"title": "Injuries cluster in particular months",
                    "text": f"{pk[0]} has the most injuries ({pk[1]} cases, {pk[2]:.0f}% of dated cases), followed by "
                            f"{m.iloc[1]['injury_month']} ({int(m.iloc[1]['injuries'])}) and {m.iloc[2]['injury_month']} ({int(m.iloc[2]['injuries'])}). "
                            f"The quietest month is {m.iloc[-1]['injury_month']} ({int(m.iloc[-1]['injuries'])})."})

    # 5. Duration / injury type
    dc = duration_by_category(df)
    if not dc.empty:
        longest = dc.iloc[0]
        ins.append({"title": "Injury type matters for time out",
                    "text": f"{longest['injury_category']} injuries have the longest median absence "
                            f"({_fmt(longest['median_days'], 0)} days, {int(longest['cases'])} cases). "
                            f"The shortest median is {dc.iloc[-1]['injury_category']} ({_fmt(dc.iloc[-1]['median_days'], 0)} days)."})
    return ins


def key_facts(df):
    """Single-number facts for the Insights page cards."""
    facts = {}
    if df.empty:
        return facts
    clubs = injuries_per_club(df)
    facts["most_injured_club"] = (clubs.iloc[0]["club"], int(clubs.iloc[0]["injuries"]))

    pl = injuries_per_player(df, 1)
    facts["most_injured_player"] = (pl.iloc[0]["player"], int(pl.iloc[0]["injuries"]), pl.iloc[0]["club"])

    top = top_drop_cases(df, 1)
    if not top.empty:
        r = top.iloc[0]
        facts["highest_drop_case"] = (r["case_label"], r["club"], r["perf_drop_index"])

    ts = team_phase_summary(df)
    ts = ts[ts["cases"] >= SMALL_SAMPLE]
    if not ts.empty:
        r = ts.iloc[0]
        facts["highest_drop_club"] = (r["club"], r["drop_index"], int(r["cases"]))

    cb = comeback_leaderboard(df)
    if not cb.empty:
        facts["top_comeback"] = cb.head(3)
    return facts


def business_answers(df):
    """
    The five business questions: calculation, result and interpretation text.
    Returns a list of dicts with keys: q, calculation, result, interpretation, chart.
    """
    if df.empty:
        return []
    out = []

    # Q1 ---------------------------------------------------------------
    top = top_drop_cases(df, 5)
    cat = drop_by_category(df)
    if not top.empty:
        lines = [f"{i + 1}. {r['case_label']} ({r['club']}): drop index {r['perf_drop_index']:.2f}"
                 for i, r in top.iterrows()]
        res = "Top 5 cases:\n" + "\n".join(lines)
        if not cat.empty:
            res += (f"\n\nBy injury type (min 5 cases): highest average drop = {cat.iloc[0]['injury_category']} "
                    f"({cat.iloc[0]['mean_drop']:.2f}, n={int(cat.iloc[0]['cases'])}).")
        out.append({"q": "1. Which injuries are associated with the biggest drop in team performance?",
                    "calculation": "Drop index = average goal difference in the 3 matches BEFORE the injury minus average goal difference in the matches MISSED. Cases are ranked high to low.",
                    "result": res,
                    "interpretation": "These are the cases where results were much worse during the absence. Each is based on at most 3 matches per phase, so a single heavy defeat can create a large value; treat them as examples, not proof that the injury caused the drop.",
                    "chart": "Bar chart: Top 10 injuries by team performance drop"})

    # Q2 ---------------------------------------------------------------
    rec = overall_record(df)
    b, d, a = rec.iloc[0], rec.iloc[1], rec.iloc[2]
    out.append({"q": "2. What was the team's win/loss record during a player's absence?",
                "calculation": "Count wins, draws and losses in the 'missed match' columns (up to 3 matches per case) and compare with the 3 matches before and after.",
                "result": (f"During absence: {int(d['wins'])}W {int(d['draws'])}D {int(d['losses'])}L from {int(d['matches'])} matches "
                           f"(win rate {_fmt(d['win_pct'], 1)}%, {_fmt(d['points_per_match'])} points per match).\n"
                           f"Before injury: win rate {_fmt(b['win_pct'], 1)}%, {_fmt(b['points_per_match'])} points per match.\n"
                           f"After return: win rate {_fmt(a['win_pct'], 1)}%, {_fmt(a['points_per_match'])} points per match."),
                "interpretation": "Compare the win rates: a lower win rate during absence suggests teams struggled without the player, but opposition strength and other absences are not controlled for.",
                "chart": "Stacked bar chart: win/draw/loss share before, during and after"})

    # Q3 ---------------------------------------------------------------
    rs = rating_change_stats(df)
    cb = comeback_leaderboard(df)
    if rs:
        best = cb.iloc[0] if not cb.empty else None
        res = (f"Average rating before: {_fmt(rs['mean_before'])}; after: {_fmt(rs['mean_after'])}; change {rs['mean_change']:+.2f} "
               f"(95% CI {rs['ci_low']:+.2f} to {rs['ci_high']:+.2f}, n={rs['n']}). "
               f"{rs['share_improved']:.0f}% improved, {rs['share_declined']:.0f}% declined.")
        if best is not None:
            res += f"\nStrongest comeback: {best['player']} ({best['club']}), {best['rating_change']:+.2f}."
        out.append({"q": "3. How did individual players perform after recovery compared with before injury?",
                    "calculation": "Rating change = average rating in the first 3 matches after return minus average rating in the 3 matches before injury (only matches with a recorded rating).",
                    "result": res,
                    "interpretation": "If the confidence interval includes zero, there is no clear average decline or improvement. Individual changes vary a lot, so the leaderboard shows who changed most, not who is permanently better.",
                    "chart": "Line chart (rating timeline) and leaderboard table"})

    # Q4 ---------------------------------------------------------------
    pv = month_club_pivot(df)
    if not pv.empty:
        total_by_month = pv.sum()
        peak = total_by_month.idxmax()
        flat = pv.stack().sort_values(ascending=False)
        (club, month), val = flat.index[0], flat.iloc[0]
        out.append({"q": "4. Are there particular months or clubs with frequent injury clusters?",
                    "calculation": "pivot_table of injury count by club (rows) and injury month (columns).",
                    "result": (f"Busiest month overall: {peak} ({int(total_by_month.max())} injuries). "
                               f"Largest single club-month cluster: {club} in {month} ({int(val)} injuries). "
                               f"Quietest month: {total_by_month.idxmin()} ({int(total_by_month.min())})."),
                    "interpretation": "Look for months that are consistently dark across several clubs. If they match busy parts of the fixture calendar (for example winter), that is a possible explanation, but the data cannot prove fixture congestion is the cause.",
                    "chart": "Heatmap: injuries by month and club"})

    # Q5 ---------------------------------------------------------------
    imp = club_impact_table(df)
    if not imp.empty:
        big = imp[~imp["small_sample"]]
        if big.empty:
            big = imp
        worst = big.iloc[0]
        most = imp.sort_values("injuries", ascending=False).iloc[0]
        out.append({"q": "5. Which clubs appear to be most affected by player injuries?",
                    "calculation": "Per club: number of injuries, average days out, and average team drop index. Clubs with fewer than 5 cases are flagged as small samples.",
                    "result": (f"Most injuries: {most['club']} ({int(most['injuries'])}). "
                               f"Largest average performance drop: {worst['club']} ({worst['drop_index']:.2f} goals per match, {int(worst['injuries'])} cases)."),
                    "interpretation": "A club can have many injuries without a big performance drop (deep squad) or few injuries with a big drop (key players). Looking at both gives a fairer picture.",
                    "chart": "Bar charts of injuries per club and drop index per club"})
    return out


def summary_report_text(df, filters_text=""):
    """Plain-text analysis summary for the download button."""
    lines = ["FOOTLENS ANALYTICS - ANALYSIS SUMMARY", "=" * 40, ""]
    if filters_text:
        lines += ["Filters applied: " + filters_text, ""]
    if df.empty:
        return "\n".join(lines + ["No data for the selected filters."])
    lines.append("OVERVIEW")
    lines += ["- " + s.replace("**", "") for s in overview_summary(df)]
    lines += ["", "MAJOR INSIGHTS"]
    for i, ins in enumerate(top_insights(df), 1):
        lines.append(f"{i}. {ins['title']}: {ins['text']}")
    lines += ["", "BUSINESS QUESTIONS"]
    for b in business_answers(df):
        lines += [b["q"], "  Calculation: " + b["calculation"], "  Result: " + b["result"].replace("\n", "\n    "),
                  "  Interpretation: " + b["interpretation"], ""]
    lines += ["LIMITATIONS",
              "- Only 3 matches are recorded before, during and after each injury.",
              "- Results show association, not proof that an injury caused a change.",
              "- Opposition strength, other absences and home/away are not controlled for."]
    return "\n".join(lines)
