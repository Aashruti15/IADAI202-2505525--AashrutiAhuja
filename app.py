"""
app.py  -  FOOTLENS ANALYTICS
Main Streamlit application: page navigation, sidebar filters, KPI cards and layout.

Run locally with:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import analysis as an
import visualizations as viz
from data_processing import DATA_PATH, data_quality_report, load_clean_data

st.set_page_config(page_title="FootLens Analytics", page_icon="⚽", layout="wide",
                   initial_sidebar_state="expanded")

PAGES = ["Overview", "Injury Analysis", "Team Performance", "Player Recovery",
         "Insights & Findings", "About Project", "Data Information"]

# ---------------------------------------------------------------------------
# STYLING (cards, header, spacing)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px;}
.fl-header {display:flex; align-items:center; gap:16px; padding:18px 24px; margin-bottom:18px;
    border-radius:16px; border:1px solid #24324B;
    background: linear-gradient(120deg, #0F2A24 0%, #121B2E 55%, #0B1220 100%); position:relative; overflow:hidden;}
.fl-header:after {content:""; position:absolute; right:-40px; top:-40px; width:200px; height:200px;
    border:3px solid rgba(0,195,122,.25); border-radius:50%;}
.fl-header:before {content:""; position:absolute; right:60px; top:0; width:3px; height:100%; background:rgba(0,195,122,.18);}
.fl-logo {font-size:2.4rem;}
.fl-title {font-size:1.9rem; font-weight:800; letter-spacing:3px; color:#E8EEF7; line-height:1.1;}
.fl-title span {color:#00C37A;}
.fl-sub {color:#9AA9C2; font-size:1rem; margin-top:2px;}
.kpi {background:#121B2E; border:1px solid #24324B; border-left:4px solid #00C37A; border-radius:12px;
      padding:14px 16px; height:100%;}
.kpi .label {color:#9AA9C2; font-size:.78rem; text-transform:uppercase; letter-spacing:1px;}
.kpi .value {color:#E8EEF7; font-size:1.9rem; font-weight:700; line-height:1.25;}
.kpi .note {color:#7CF0B8; font-size:.78rem;}
.card {background:#121B2E; border:1px solid #24324B; border-radius:12px; padding:16px 18px; margin-bottom:12px;}
.card h4 {margin:0 0 6px 0; color:#00C37A; font-size:1.02rem;}
.card p {margin:0; color:#E8EEF7; font-size:.95rem;}
.small-muted {color:#9AA9C2; font-size:.85rem;}
.caveat {background:rgba(255,184,77,.08); border:1px solid rgba(255,184,77,.35); border-radius:10px;
         padding:12px 16px; color:#E8EEF7; font-size:.92rem;}
section[data-testid="stSidebar"] {border-right:1px solid #24324B;}
h1,h2,h3 {letter-spacing:.3px;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def show_chart(fig):
    """Display a Plotly chart full width (works on old and new Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(data, **kwargs):
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=True, **kwargs)


def kpi_card(label, value, note=""):
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                f'<div class="note">{note}&nbsp;</div></div>', unsafe_allow_html=True)


def info_card(title, text):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)


def section(title, description=""):
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def num(x, d=2, suffix=""):
    return "n/a" if pd.isna(x) else f"{x:,.{d}f}{suffix}"


# ---------------------------------------------------------------------------
# DATA LOADING (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    raw, clean = load_clean_data()
    return raw, clean, data_quality_report(raw)


try:
    with st.spinner("Loading and cleaning the dataset..."):
        raw_df, df_all, quality = get_data()
except FileNotFoundError:
    st.error(f"Dataset not found. Place the CSV at `{DATA_PATH}` (data/dataset.csv) and restart the app.")
    st.stop()
except Exception as exc:  # any unexpected problem is shown in plain English
    st.error(f"The dataset could not be processed: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR: NAVIGATION + FILTERS
# ---------------------------------------------------------------------------
all_clubs = sorted(df_all["club"].unique())
all_seasons = sorted(df_all["season"].unique())
all_players = sorted(df_all["player"].unique())
all_types = sorted(df_all["injury_category"].unique())
age_min, age_max = int(df_all["age"].min()), int(df_all["age"].max())
date_min, date_max = df_all["injury_date"].min().date(), df_all["injury_date"].max().date()

# Reset works by changing the widget keys: a new key = a brand-new, empty widget.
st.session_state.setdefault("reset_n", 0)


def reset_filters():
    st.session_state["reset_n"] += 1


n = st.session_state["reset_n"]

with st.sidebar:
    st.markdown("## ⚽ FOOTLENS")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("#### Filters")
    st.caption("Pick one or more values to narrow the data. Leave a filter empty to include everything.")
    f_clubs = st.multiselect("Club", all_clubs, key=f"f_clubs_{n}")
    f_seasons = st.multiselect("Season", all_seasons, key=f"f_seasons_{n}")
    f_types = st.multiselect("Injury type", all_types, key=f"f_types_{n}")
    f_players = st.multiselect("Player", all_players, key=f"f_players_{n}")
    f_age = st.slider("Player age", age_min, age_max, (age_min, age_max), key=f"f_age_{n}")
    f_dates = st.slider("Injury date range", min_value=date_min, max_value=date_max,
                        value=(date_min, date_max), format="MMM YYYY", key=f"f_dates_{n}")
    st.button("↺ Reset filters", on_click=reset_filters)


def apply_filters(data):
    """Return only the rows that match every sidebar filter (an empty filter means 'all')."""
    out = data
    if f_clubs:
        out = out[out["club"].isin(f_clubs)]
    if f_seasons:
        out = out[out["season"].isin(f_seasons)]
    if f_types:
        out = out[out["injury_category"].isin(f_types)]
    if f_players:
        out = out[out["player"].isin(f_players)]
    out = out[out["age"].between(f_age[0], f_age[1])]
    if tuple(f_dates) != (date_min, date_max):          # only filter when the slider was moved
        out = out[(out["injury_date"] >= pd.Timestamp(f_dates[0])) & (out["injury_date"] <= pd.Timestamp(f_dates[1]))]
    return out


df = apply_filters(df_all)


def filters_description():
    parts = []
    if f_clubs:
        parts.append("clubs=" + ", ".join(f_clubs))
    if f_seasons:
        parts.append("seasons=" + ", ".join(f_seasons))
    if f_types:
        parts.append("injury types=" + ", ".join(f_types))
    if f_players:
        parts.append("players=" + ", ".join(f_players))
    if tuple(f_age) != (age_min, age_max):
        parts.append("age=%d-%d" % tuple(f_age))
    if tuple(f_dates) != (date_min, date_max):
        parts.append(f"dates={f_dates[0]} to {f_dates[1]}")
    return "; ".join(parts) if parts else "none (all data)"


with st.sidebar:
    st.markdown("---")
    st.markdown("#### Downloads")
    st.caption(f"{len(df):,} of {len(df_all):,} cases selected")
    st.download_button("⬇ Filtered dataset (CSV)", df.drop(columns=["case_label"]).to_csv(index=False).encode("utf-8"),
                       file_name="footlens_filtered_data.csv", mime="text/csv", disabled=df.empty)
    st.download_button("⬇ Analysis summary (TXT)", an.summary_report_text(df, filters_description()).encode("utf-8"),
                       file_name="footlens_analysis_summary.txt", mime="text/plain", disabled=df.empty)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="fl-header"><div class="fl-logo">⚽</div>
<div><div class="fl-title">FOOT<span>LENS</span> ANALYTICS</div>
<div class="fl-sub">Player Injuries &amp; Team Performance Intelligence</div></div></div>
""", unsafe_allow_html=True)

# Pages that explain the project do not depend on the filters
needs_data = page not in ("About Project", "Data Information")
if needs_data and df.empty:
    st.warning("No injury cases match the current filters. Widen the filters or press **Reset filters** in the sidebar.")
    st.stop()
if needs_data:
    st.caption(f"Showing **{len(df):,}** of {len(df_all):,} injury cases  |  Filters: {filters_description()}")


# ===========================================================================
# PAGE 1: OVERVIEW
# ===========================================================================
def page_overview():
    k = an.compute_kpis(df)
    c = st.columns(5)
    with c[0]: kpi_card("Total players", f"{k['players']:,}", "unique names")
    with c[1]: kpi_card("Injury cases", f"{k['cases']:,}", "rows in selection")
    with c[2]: kpi_card("Clubs", k["clubs"])
    with c[3]: kpi_card("Avg player rating", num(k["avg_rating"]), f"{k['rated_matches']:,} rated matches")
    with c[4]: kpi_card("Avg injury duration", num(k["avg_duration"], 1, " days"),
                        f"median {num(k['median_duration'], 0)} days (n={k['duration_n']})")
    st.write("")

    left, right = st.columns([3, 2])
    with left:
        show_chart(viz.fig_injuries_per_club(df))
    with right:
        section("Key findings (auto-generated)")
        for line in an.overview_summary(df):
            st.markdown(f"- {line}")

    a, b = st.columns(2)
    with a:
        show_chart(viz.fig_injury_category(df))
    with b:
        show_chart(viz.fig_rating_by_phase(df))
        st.caption("Player rating = average of the recorded match ratings (substitute appearances included).")


# ===========================================================================
# PAGE 2: INJURY ANALYSIS
# ===========================================================================
def page_injuries():
    section("Injury Analysis", "Who gets injured, where and when. Use the sidebar to filter by club, player and injury type.")
    a, b = st.columns(2)
    with a:
        show_chart(viz.fig_top_players(df))
    with b:
        show_chart(viz.fig_injuries_per_club(df))

    # MANDATORY 3: heatmap
    show_chart(viz.fig_injury_heatmap(df))
    st.caption("Darker green = more injuries. Each cell counts injuries that started in that month for that club.")

    a, b = st.columns(2)
    with a:
        show_chart(viz.fig_monthly_bar(df))
    with b:
        show_chart(viz.fig_season_trend(df))

    show_chart(viz.fig_duration_box(df))
    st.caption("Duration = return date minus injury date. Cases with a missing or impossible date are left out of duration "
               "calculations only. Log scale is used because a few injuries lasted more than a year.")

    with st.expander("Detailed tables"):
        st.markdown("**Injury duration by type**")
        show_table(an.duration_by_category(df).round(1))
        st.markdown("**Club summary**")
        show_table(an.injuries_per_club(df).round(1))


# ===========================================================================
# PAGE 3: TEAM PERFORMANCE
# ===========================================================================
def page_team():
    section("Team Performance Analysis",
            "Team results in the 3 matches before the injury, the matches missed (up to 3) and the 3 matches after return.")
    st.markdown('<div class="caveat"><b>Read with care:</b> these charts show an <i>association</i>, not proof that an injury '
                'caused a change. Only up to 3 matches are recorded per phase, opposition strength and other absences are not '
                'controlled for, and extreme before-injury results tend to move back towards average anyway.</div>',
                unsafe_allow_html=True)
    st.write("")

    ds = an.drop_summary_stats(df)
    if ds:
        c = st.columns(4)
        with c[0]: kpi_card("Avg performance drop", num(ds["mean"]), "goal difference per match")
        with c[1]: kpi_card("Median drop", num(ds["median"]), f"std {num(ds['std'])}")
        with c[2]: kpi_card("Cases worse / better", f"{ds['share_worse']:.0f}% / {ds['share_better']:.0f}%", f"n = {ds['n']}")
        with c[3]: kpi_card("95% CI of mean", f"{ds['ci_low']:+.2f} to {ds['ci_high']:+.2f}",
                            "includes 0 = no clear drop" if ds["ci_low"] <= 0 <= ds["ci_high"] else "excludes 0")
        st.write("")

    # MANDATORY 1: bar chart
    show_chart(viz.fig_top_drop_bar(df))
    st.caption("Drop index = average goal difference in the 3 matches before the injury minus average goal difference in the matches missed.")

    a, b = st.columns(2)
    with a:
        show_chart(viz.fig_record_stacked(df))
    with b:
        show_chart(viz.fig_team_phase_bars(df))

    a, b = st.columns(2)
    with a:
        show_chart(viz.fig_drop_by_club(df))
    with b:
        show_chart(viz.fig_win_loss_by_club(df))

    # MANDATORY 4: scatter
    show_chart(viz.fig_age_vs_drop_scatter(df))
    corr = an.age_drop_correlation(df)
    if corr:
        strength = "very weak" if abs(corr["r"]) < 0.1 else "weak" if abs(corr["r"]) < 0.3 else "moderate"
        st.caption(f"Correlation between age and drop index: r = {corr['r']:.2f} ({strength}, n = {corr['n']}). "
                   "Bubble size = matches missed.")

    show_chart(viz.fig_drop_box_by_club(df))
    with st.expander("Performance table by club"):
        show_table(an.team_phase_summary(df).round(2))


# ===========================================================================
# PAGE 4: PLAYER RECOVERY
# ===========================================================================
def page_recovery():
    section("Player Recovery Analysis", "Average match rating in the 3 games before the injury vs the first 3 games after return.")
    rs = an.rating_change_stats(df)
    if rs:
        c = st.columns(4)
        with c[0]: kpi_card("Rating before", num(rs["mean_before"]), f"{rs['n']} comparable cases")
        with c[1]: kpi_card("Rating after", num(rs["mean_after"]))
        with c[2]: kpi_card("Average change", f"{rs['mean_change']:+.2f}", f"95% CI {rs['ci_low']:+.2f} to {rs['ci_high']:+.2f}")
        with c[3]: kpi_card("Improved / declined", f"{rs['share_improved']:.0f}% / {rs['share_declined']:.0f}%")
        st.write("")
    else:
        st.info("Not enough cases with both a before and an after rating in this selection.")

    # MANDATORY 2: line chart with player selector
    section("Player rating timeline", "Pick a player and one or more of their injuries. The dashed line is the average of all cases in the current filter.")
    with_ratings = df.dropna(subset=["rating_before", "rating_after"], how="all")
    players = sorted(with_ratings["player"].unique())
    if players:
        lb = an.comeback_leaderboard(df)
        default_idx = players.index(lb.iloc[0]["player"]) if not lb.empty and lb.iloc[0]["player"] in players else 0
        p1, p2 = st.columns([1, 2])
        with p1:
            player = st.selectbox("Player", players, index=default_idx)
        cases = with_ratings[with_ratings["player"] == player]
        label_map = dict(zip(cases["case_id"], cases["case_label"]))
        with p2:
            chosen = st.multiselect("Injury cases", list(label_map), default=list(label_map)[:3],
                                    format_func=lambda cid: label_map[cid])
        if chosen:
            show_chart(viz.fig_player_timeline(df, chosen))
            st.caption("Circles = starter, triangles = came on as a substitute. 'Match 1' is assumed to be the game closest to the injury / return. "
                       "Gaps in a line mean no rating was recorded (the dataset marks these N.A.).")
        else:
            st.info("Select at least one injury case.")
    else:
        st.info("No rating data available for the selected filters.")

    a, b = st.columns(2)
    with a:
        show_chart(viz.fig_rating_before_after(df))
    with b:
        show_chart(viz.fig_rating_change_hist(df))

    # MANDATORY 5: leaderboard
    section("Comeback leaderboard", "Players ranked by rating improvement (after recovery minus before injury).")
    c1, c2 = st.columns(2)
    with c1:
        top_n = st.slider("Players to show", 5, 30, 15)
    with c2:
        min_cases = st.slider("Minimum injury cases per player", 1, 5, 1)
    show_chart(viz.fig_leaderboard_table(df, top_n, min_cases))
    lb_full = an.comeback_leaderboard(df, min_cases)
    if not lb_full.empty:
        st.markdown("**Biggest declines** (bottom of the same ranking)")
        show_table(lb_full.tail(5).iloc[::-1].round(2))

    a, b = st.columns(2)
    with a:
        show_chart(viz.fig_position_recovery(df))
    with b:
        st.markdown("**Rating change by length of injury**")
        t = an.recovery_by_duration(df)
        if t.empty:
            st.info("Not enough data.")
        else:
            show_table(t.round(2))
            st.caption("Shows whether longer injuries are linked to a bigger change in rating.")
    with st.expander("Full player summary (before / during / after)"):
        show_table(an.player_summary(df).round(2))


# ===========================================================================
# PAGE 5: INSIGHTS AND FINDINGS
# ===========================================================================
def page_insights():
    section("Insights & Findings", "Everything on this page is calculated from the currently filtered data.")
    facts = an.key_facts(df)
    c = st.columns(4)
    if "most_injured_club" in facts:
        with c[0]: kpi_card("Most injury-affected club", facts["most_injured_club"][0], f"{facts['most_injured_club'][1]} injury cases")
    if "most_injured_player" in facts:
        p = facts["most_injured_player"]
        with c[1]: kpi_card("Most frequently injured player", p[0], f"{p[1]} injuries ({p[2]})")
    if "highest_drop_club" in facts:
        h = facts["highest_drop_club"]
        with c[2]: kpi_card("Highest team performance drop", h[0], f"avg drop {h[1]:+.2f} (n={h[2]})")
    if "highest_drop_case" in facts:
        h = facts["highest_drop_case"]
        with c[3]: kpi_card("Single biggest drop (case)", f"{h[2]:.2f}", h[0])
    st.write("")

    section("Five major insights")
    for i, ins in enumerate(an.top_insights(df), 1):
        info_card(f"{i}. {ins['title']}", ins["text"])

    if "top_comeback" in facts:
        section("Strongest post-recovery improvement")
        show_table(facts["top_comeback"].round(2))

    section("The five business questions")
    figs = [viz.fig_top_drop_bar, viz.fig_record_stacked, viz.fig_rating_change_hist,
            viz.fig_injury_heatmap, viz.fig_drop_by_club]
    for ans, fig_fn in zip(an.business_answers(df), figs):
        with st.expander(ans["q"], expanded=False):
            st.markdown(f"**Calculation:** {ans['calculation']}")
            st.markdown("**Result:**")
            st.text(ans["result"])
            st.markdown(f"**Interpretation:** {ans['interpretation']}")
            show_chart(fig_fn(df))

    section("Limitations")
    st.markdown("""
- Only **3 matches** are recorded before, during (max) and after each injury, so single results can swing averages.
- Results show **association, not causation**; opposition strength, home/away and other absences are not in the data.
- The dataset covers **8 clubs** only, so club totals describe this dataset, not the whole league.
- Missing values (`N.A.`) were **kept as missing**, not guessed. Averages use only matches that have data.
""")


# ===========================================================================
# ABOUT + DATA INFORMATION
# ===========================================================================
def page_about():
    section("About FootLens Analytics")
    st.markdown("""
**FootLens Analytics** is a (fictional) football analytics company that helps managers and technical directors understand how
player injuries affect individual performance and overall team results.

This dashboard was built for the *Mathematics for AI-II* individual summative assessment (IBCP). It cleans the supplied
player-injury dataset, calculates before / during / after comparisons and presents them in interactive charts.

**Business questions**
1. Which injuries are associated with the biggest drop in team performance?
2. What was the team's win/loss record during a player's absence?
3. How did individual players perform after recovery compared with before injury?
4. Are there particular months or clubs with frequent injury clusters?
5. Which clubs appear to be most affected by player injuries?

**How to use it:** choose a page in the sidebar, narrow the data with the filters, hover over charts for details, and use the
download buttons for the filtered data or a written summary.

**Technology:** Python, Pandas, NumPy, Plotly, Streamlit, GitHub, Streamlit Community Cloud.
""")


COLUMN_GUIDE = pd.DataFrame([
    ("Name", "player", "Player name"),
    ("Team Name", "club", "Club the player belonged to (8 clubs)"),
    ("Position", "position", "Playing position (spacing/case cleaned)"),
    ("Age", "age", "Player age at the time of the injury"),
    ("Season", "season", "Season of the injury (2019/20 - 2023/24)"),
    ("FIFA rating", "fifa_rating", "Video-game style overall player rating"),
    ("Injury", "injury / injury_category", "Injury description (capitalisation unified) and a broader category created for filtering"),
    ("Date of Injury", "injury_date", "Start of injury, converted to a date"),
    ("Date of return", "return_date", "Return date, converted to a date ('Present' = still injured)"),
    ("MatchN_before_injury_*", "beforeN_result / opponent / gd / rating", "3 matches before the injury: result, opponent, goal difference, player rating"),
    ("MatchN_missed_match_*", "duringN_result / opponent / gd", "Up to 3 team matches missed by the player: result, opponent, goal difference"),
    ("MatchN_after_injury_*", "afterN_result / opponent / gd / rating", "First 3 matches after return: result, opponent, goal difference, player rating"),
], columns=["Original column", "Cleaned name", "Meaning"])

CALC_GUIDE = pd.DataFrame([
    ("injury_duration_days", "return_date - injury_date (valid dates only)"),
    ("before_ppm / during_ppm / after_ppm", "Average points per match (win 3, draw 1, loss 0) in each phase"),
    ("before_gd_avg / during_gd_avg / after_gd_avg", "Average goal difference in each phase"),
    ("perf_drop_index", "before_gd_avg - during_gd_avg (positive = team did worse while the player was out)"),
    ("ppm_drop", "before_ppm - during_ppm"),
    ("rating_before / rating_after", "Mean player rating in the 3 matches before / after (missing ratings skipped)"),
    ("rating_change", "rating_after - rating_before"),
    ("matches_missed", "Number of recorded missed matches (0-3)"),
], columns=["Calculated column", "Definition"])


def page_data_info():
    section("Data Information", "The original file is preserved; all cleaning happens on a copy.")
    c = st.columns(4)
    with c[0]: kpi_card("Rows", quality["rows"])
    with c[1]: kpi_card("Columns (original)", quality["columns"])
    with c[2]: kpi_card("Duplicate rows", quality["duplicate_rows"])
    with c[3]: kpi_card("Missing cells", f"{quality['total_missing_cells']:,}", "'N.A.' counted as missing")
    st.write("")
    st.markdown("**Column guide**")
    show_table(COLUMN_GUIDE)
    st.markdown("**Calculated columns**")
    show_table(CALC_GUIDE)

    section("Cleaning steps and decisions")
    st.markdown("""
- **'N.A.' text** was converted to real missing values. It means the match did not exist or no rating was recorded
  (for example an injury shorter than 3 matches). Rows were **not deleted** and values were **not invented**.
- **Dates** came in mixed styles (`Nov 9, 2019`, `Dec 6,2019`, `January 4, 2023`, `Present`). They were standardised and converted to datetime.
- **Impossible dates** (return before injury, e.g. `Mar 7, 0202`) were flagged and the return date left blank, so no duration is calculated for them.
- **Injury names** differed only by capitalisation (`hamstring injury` / `Hamstring injury`); they were unified and grouped into broader categories.
- **Positions** had stray spaces (`Central Midfielder `); these were trimmed.
- **Ratings** such as `6.1(S)` mean the player came on as a substitute; the number was kept and the substitute flag stored separately.
- **Goal difference of 9** appears twice; both are real results (large wins), so outliers were kept.
- **Assumption:** `Match1` is the match closest to the injury/return date. Averages do not depend on this; only the timeline order does.
""")
    st.markdown("**Missing values in the original data (N.A. counted as missing)**")
    show_table(quality["missing_by_column"].reset_index().rename(columns={"index": "Column"}))
    with st.expander("Preview: original data"):
        st.dataframe(raw_df.head(50))
    with st.expander("Preview: cleaned data with calculated columns"):
        st.dataframe(df_all.drop(columns=["case_label"]).head(50))


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------
{"Overview": page_overview, "Injury Analysis": page_injuries, "Team Performance": page_team,
 "Player Recovery": page_recovery, "Insights & Findings": page_insights,
 "About Project": page_about, "Data Information": page_data_info}[page]()