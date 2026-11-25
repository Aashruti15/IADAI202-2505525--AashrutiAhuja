"""
visualizations.py
=================
Plotly chart functions. The five MANDATORY visualisations are:

    1. fig_top_drop_bar()        - bar chart: Top 10 injuries with the highest team performance drop
    2. fig_player_timeline()     - line chart: player rating timeline before and after injury
    3. fig_injury_heatmap()      - heatmap: injury frequency across months and clubs
    4. fig_age_vs_drop_scatter() - scatter: player age vs performance drop index
    5. fig_leaderboard_table()   - leaderboard table: comeback players ranked by rating improvement

The rest are supporting charts used on the dashboard pages.
Every function returns a Plotly Figure (or a message-figure when there is no data).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import analysis as an

# ---- Football-inspired dark theme ------------------------------------------
NAVY = "#0B1220"
CARD = "#121B2E"
GRID = "#24324B"
GREEN = "#00C37A"
LIGHT_GREEN = "#7CF0B8"
TEXT = "#E8EEF7"
MUTED = "#9AA9C2"
RED = "#FF5A6A"
AMBER = "#FFB84D"
BLUE = "#4DA3FF"
CLUB_COLORS = ["#00C37A", "#4DA3FF", "#FFB84D", "#FF5A6A", "#B388FF", "#2DD4BF", "#F472B6", "#A3E635"]


def style(fig, title, x_title=None, y_title=None, height=420):
    """Apply the same look to every chart."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=TEXT), x=0.01),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
        height=height, margin=dict(l=20, r=20, t=60, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor=CARD, font_color=TEXT, bordercolor=GREEN),
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(title=y_title, gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


def empty_fig(message="No data available for the selected filters."):
    """Friendly placeholder used when a filter removes all rows."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, "", height=300)


# ===========================================================================
# MANDATORY 1: BAR CHART - Top 10 injuries with the highest team performance drop
# ===========================================================================
def fig_top_drop_bar(df, n=10):
    top = an.top_drop_cases(df, n)
    if top.empty:
        return empty_fig("No cases with both 'before' and 'missed' match data.")
    top = top.sort_values("perf_drop_index")          # largest bar at the top of a horizontal chart
    fig = go.Figure(go.Bar(
        x=top["perf_drop_index"], y=top["case_label"], orientation="h",
        marker=dict(color=top["perf_drop_index"], colorscale=[[0, "#1B7F5B"], [1, GREEN]], line=dict(width=0)),
        text=top["perf_drop_index"].round(2), textposition="outside", textfont=dict(color=TEXT),
        customdata=np.stack([top["club"], top["injury"], top["before_gd_avg"].round(2),
                             top["during_gd_avg"].round(2), top["matches_missed"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Club: %{customdata[0]}<br>Injury: %{customdata[1]}"
                       "<br>Avg goal diff BEFORE: %{customdata[2]}<br>Avg goal diff DURING: %{customdata[3]}"
                       "<br>Matches missed (max 3): %{customdata[4]}<br>Drop index: %{x:.2f}<extra></extra>")))
    fig.update_xaxes(range=[0, top["perf_drop_index"].max() * 1.18])
    return style(fig, f"Top {len(top)} Injuries with the Highest Team Performance Drop",
                 "Team performance drop index (goal difference before - during absence)", "Injury case",
                 height=max(420, 44 * len(top) + 140))


# ===========================================================================
# MANDATORY 2: LINE CHART - Player rating timeline before and after injury
# ===========================================================================
def fig_player_timeline(df, case_ids, show_average=True):
    """
    case_ids: list of injury cases to draw (one line each).
    A dashed grey line shows the average of ALL cases in the current filter for comparison.
    """
    tl = an.player_timeline(df, case_ids)
    if tl.empty or tl["rating"].notna().sum() == 0:
        return empty_fig("No ratings recorded for the selected player / injury.")
    fig = go.Figure()
    labels = {-3: "Before 3", -2: "Before 2", -1: "Before 1", 1: "After 1", 2: "After 2", 3: "After 3"}

    if show_average:
        avg = an.average_timeline(df)
        fig.add_trace(go.Scatter(x=avg["position"], y=avg["avg_rating"], mode="lines+markers",
                                 name="Average of filtered cases", line=dict(color=MUTED, dash="dash"),
                                 marker=dict(size=6), customdata=avg["n"],
                                 hovertemplate="Average rating: %{y:.2f}<br>Based on %{customdata} ratings<extra></extra>"))
    for k, (cid, g) in enumerate(tl.groupby("case_id")):
        g = g.sort_values("position")
        symbols = ["triangle-up" if s else "circle" for s in g["substitute"]]
        fig.add_trace(go.Scatter(
            x=g["position"], y=g["rating"], mode="lines+markers", name=g["case_label"].iloc[0],
            line=dict(color=CLUB_COLORS[k % len(CLUB_COLORS)], width=3), connectgaps=False,
            marker=dict(size=10, symbol=symbols),
            customdata=np.stack([g["match"], g["opponent"].fillna("n/a"), g["result"].fillna("n/a"),
                                 g["substitute"].map({True: "Yes", False: "No"})], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b> vs %{customdata[1]} (%{customdata[2]})"
                           "<br>Rating: %{y}<br>Came on as sub: %{customdata[3]}<extra></extra>")))
    # shaded injury gap between the "before" and "after" blocks
    fig.add_vrect(x0=-0.5, x1=0.5, fillcolor=RED, opacity=0.10, line_width=0,
                  annotation_text="Injury period", annotation_position="top", annotation_font_color=MUTED)
    fig.update_xaxes(tickmode="array", tickvals=list(labels), ticktext=list(labels.values()))
    fig.update_yaxes(range=[0, 10.2])
    return style(fig, "Player Rating Timeline: 3 Matches Before vs 3 Matches After Injury",
                 "Match (closest to injury in the middle)", "Match rating (0-10)", height=460)


# ===========================================================================
# MANDATORY 3: HEATMAP - Injury frequency across months and clubs
# ===========================================================================
def fig_injury_heatmap(df):
    pv = an.month_club_pivot(df)
    if pv.empty:
        return empty_fig("No dated injuries for the selected filters.")
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=pv.columns, y=pv.index,
        colorscale=[[0, "#0F1A2E"], [0.35, "#0E5A45"], [1, GREEN]],
        text=pv.values, texttemplate="%{text}", textfont=dict(color=TEXT),
        colorbar=dict(title="Injuries", tickfont=dict(color=MUTED)),
        hovertemplate="<b>%{y}</b> in %{x}<br>Injuries: %{z}<extra></extra>", xgap=3, ygap=3))
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Injury Frequency by Month and Club", "Month of injury (football-season order)", "Club", height=430)


# ===========================================================================
# MANDATORY 4: SCATTER - Player age vs performance drop index
# ===========================================================================
def fig_age_vs_drop_scatter(df):
    d = df.dropna(subset=["age", "perf_drop_index"])
    if d.empty:
        return empty_fig("No cases with age and team performance data.")
    fig = px.scatter(d, x="age", y="perf_drop_index", color="club", color_discrete_sequence=CLUB_COLORS,
                     size=d["matches_missed"].clip(lower=1), size_max=14, opacity=0.75,
                     hover_data={"player": True, "injury": True, "season": True, "age": True,
                                 "perf_drop_index": ":.2f", "matches_missed": True}, )
    stats = an.age_drop_correlation(d)
    if stats:
        xs = np.array([d["age"].min(), d["age"].max()])
        fig.add_trace(go.Scatter(x=xs, y=stats["slope"] * xs + stats["intercept"], mode="lines",
                                 name=f"Trend line (r = {stats['r']:.2f})", line=dict(color="white", dash="dash", width=2)))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig = style(fig, "Player Age vs Team Performance Drop Index", "Player age (years)",
                "Performance drop index (positive = team did worse)", height=470)
    fig.update_layout(legend_title_text="Club")
    return fig


# ===========================================================================
# MANDATORY 5: LEADERBOARD TABLE - Comeback players ranked by rating improvement
# ===========================================================================
def fig_leaderboard_table(df, top_n=15, min_cases=1):
    lb = an.comeback_leaderboard(df, min_cases).head(top_n)
    if lb.empty:
        return empty_fig("No players with both a before and an after rating.")
    change_colors = [GREEN if v > 0 else RED for v in lb["rating_change"]]
    fig = go.Figure(go.Table(
        columnwidth=[50, 190, 110, 70, 110, 110, 120],
        header=dict(values=["Rank", "Player", "Club", "Cases", "Rating before", "Rating after", "Improvement"],
                    fill_color=GREEN, font=dict(color=NAVY, size=14), align="left", height=34),
        cells=dict(values=[lb["rank"], lb["player"], lb["club"], lb["cases"],
                           lb["rating_before"].round(2), lb["rating_after"].round(2),
                           lb["rating_change"].map(lambda v: f"{v:+.2f}")],
                   fill_color=[[CARD if i % 2 == 0 else "#0F1727" for i in range(len(lb))]] * 6 + [[CARD] * len(lb)],
                   font=dict(color=[[TEXT] * len(lb)] * 6 + [change_colors], size=13), align="left", height=30)))
    return style(fig, "Comeback Leaderboard: Rating After Recovery minus Rating Before Injury",
                 height=min(80 + 34 * (len(lb) + 1), 700))


# ===========================================================================
# SUPPORTING CHARTS
# ===========================================================================
def fig_injury_category(df):
    t = an.injuries_by_category(df)
    if t.empty:
        return empty_fig()
    t = t.sort_values("cases")
    fig = go.Figure(go.Bar(x=t["cases"], y=t["injury_category"], orientation="h", marker_color=GREEN,
                           hovertemplate="%{y}: %{x} cases<extra></extra>"))
    return style(fig, "Injury Distribution by Type", "Number of injury cases", "Injury type", height=470)


def fig_injuries_per_club(df):
    t = an.injuries_per_club(df)
    if t.empty:
        return empty_fig()
    fig = go.Figure(go.Bar(x=t["club"], y=t["injuries"], marker_color=GREEN, text=t["injuries"], textposition="outside",
                           customdata=np.stack([t["players"], t["avg_days_out"].round(1)], axis=-1),
                           hovertemplate="<b>%{x}</b><br>Injuries: %{y}<br>Players injured: %{customdata[0]}"
                                         "<br>Avg days out: %{customdata[1]}<extra></extra>"))
    fig.update_yaxes(range=[0, t["injuries"].max() * 1.15])
    return style(fig, "Injury Cases by Club", "Club", "Number of injury cases")


def fig_top_players(df, n=10):
    t = an.injuries_per_player(df, n).sort_values("injuries")
    if t.empty:
        return empty_fig()
    fig = go.Figure(go.Bar(x=t["injuries"], y=t["player"] + " (" + t["club"] + ")", orientation="h", marker_color=BLUE,
                           customdata=np.stack([t["most_common_injury"], t["avg_days_out"].round(1)], axis=-1),
                           hovertemplate="<b>%{y}</b><br>Injuries: %{x}<br>Most common: %{customdata[0]}"
                                         "<br>Avg days out: %{customdata[1]}<extra></extra>"))
    return style(fig, f"Most Frequently Injured Players (Top {len(t)})", "Number of injuries", "Player", height=420)


def fig_monthly_bar(df):
    t = an.injuries_by_month(df)
    colors = [GREEN if v == t["injuries"].max() else "#1B7F5B" for v in t["injuries"]]
    fig = go.Figure(go.Bar(x=t["injury_month"], y=t["injuries"], marker_color=colors,
                           hovertemplate="%{x}: %{y} injuries<extra></extra>"))
    return style(fig, "Injury Frequency by Month", "Month (football-season order)", "Number of injuries")


def fig_season_trend(df):
    t = an.injuries_by_season(df)
    if t.empty:
        return empty_fig()
    fig = px.line(t, x="season", y="injuries", color="club", markers=True, color_discrete_sequence=CLUB_COLORS)
    fig.update_traces(line=dict(width=3), marker=dict(size=9),
                      hovertemplate="%{x}<br>%{fullData.name}: %{y} injuries<extra></extra>")
    fig = style(fig, "Injury Trend Across Seasons", "Season", "Number of injuries")
    fig.update_layout(legend_title_text="Club")
    return fig


def fig_duration_box(df):
    d = df.dropna(subset=["injury_duration_days"])
    if d.empty:
        return empty_fig("No injury duration data (needs valid injury and return dates).")
    order = an.duration_by_category(d)["injury_category"].tolist()
    d = d[d["injury_category"].isin(order)]
    fig = px.box(d, x="injury_category", y="injury_duration_days", category_orders={"injury_category": order},
                 color_discrete_sequence=[GREEN], points="outliers",
                 hover_data=["player", "injury", "season"])
    fig.update_yaxes(type="log")
    return style(fig, "Injury Duration by Type (log scale, days)", "Injury type (ordered by median days)",
                 "Days between injury and return", height=450)


def fig_team_phase_bars(df):
    t = an.team_phase_summary(df)
    if t.empty:
        return empty_fig()
    fig = go.Figure()
    for col, name, color in [("ppm_before", "Before injury", BLUE), ("ppm_during", "During absence", RED), ("ppm_after", "After return", GREEN)]:
        fig.add_trace(go.Bar(x=t["club"], y=t[col], name=name, marker_color=color,
                             hovertemplate="%{x}<br>" + name + ": %{y:.2f} points per match<extra></extra>"))
    fig.update_layout(barmode="group")
    return style(fig, "Team Points per Match: Before, During and After Player Absence", "Club", "Average points per match (max 3)")


def fig_drop_by_club(df):
    t = an.team_phase_summary(df).sort_values("drop_index")
    if t.empty:
        return empty_fig()
    colors = [RED if v > 0 else GREEN for v in t["drop_index"]]
    fig = go.Figure(go.Bar(x=t["drop_index"], y=t["club"], orientation="h", marker_color=colors,
                           customdata=np.stack([t["cases"], t["ppm_drop"].round(2)], axis=-1),
                           text=t["drop_index"].round(2), textposition="outside",
                           hovertemplate="<b>%{y}</b><br>Avg drop index: %{x:.2f}<br>Cases: %{customdata[0]}"
                                         "<br>Avg points-per-match drop: %{customdata[1]}<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED)
    return style(fig, "Average Team Performance Drop by Club (red = worse without player)",
                 "Average drop index (goal difference before - during)", "Club", height=400)


def fig_drop_box_by_club(df):
    d = df.dropna(subset=["perf_drop_index"])
    if d.empty:
        return empty_fig()
    fig = px.box(d, x="club", y="perf_drop_index", color="club", color_discrete_sequence=CLUB_COLORS,
                 hover_data=["player", "injury", "season"])
    fig.add_hline(y=0, line_color=MUTED)
    fig = style(fig, "Spread of Performance Drop Index Across Clubs", "Club", "Performance drop index", height=420)
    fig.update_layout(showlegend=False)
    return fig


def fig_record_stacked(df):
    r = an.overall_record(df)
    fig = go.Figure()
    for col, name, color in [("win_pct", "Win", GREEN), ("draw_pct", "Draw", AMBER), ("loss_pct", "Loss", RED)]:
        fig.add_trace(go.Bar(x=r["phase"], y=r[col], name=name, marker_color=color,
                             text=r[col].round(1).astype(str) + "%", textposition="inside",
                             customdata=r["matches"], hovertemplate="%{x}<br>" + name + ": %{y:.1f}% (of %{customdata} matches)<extra></extra>"))
    fig.update_layout(barmode="stack")
    return style(fig, "Team Results Before, During and After a Player's Absence", "Phase", "Share of matches (%)", height=400)


def fig_win_loss_by_club(df):
    t = an.win_loss_record(df)
    if t.empty:
        return empty_fig()
    t = t.sort_values("win_pct")
    fig = go.Figure()
    for col, name, color in [("win_pct", "Win", GREEN), ("draw_pct", "Draw", AMBER), ("loss_pct", "Loss", RED)]:
        fig.add_trace(go.Bar(y=t["club"], x=t[col], name=name, orientation="h", marker_color=color,
                             hovertemplate="%{y}<br>" + name + ": %{x:.1f}%<extra></extra>"))
    fig.update_layout(barmode="stack")
    return style(fig, "Win / Draw / Loss Share While the Injured Player Was Absent", "Share of matches missed (%)", "Club", height=420)


def fig_rating_before_after(df):
    d = df.dropna(subset=["rating_before", "rating_after"])
    if d.empty:
        return empty_fig()
    fig = px.scatter(d, x="rating_before", y="rating_after", color="club", color_discrete_sequence=CLUB_COLORS,
                     opacity=0.75, hover_data=["player", "injury", "season"])
    fig.add_trace(go.Scatter(x=[2, 9], y=[2, 9], mode="lines", name="No change", line=dict(color="white", dash="dash")))
    fig.update_xaxes(range=[3, 9])
    fig.update_yaxes(range=[3, 9])
    fig = style(fig, "Player Rating Before vs After Injury (above line = improved)",
                "Average rating before injury", "Average rating after return", height=450)
    fig.update_layout(legend_title_text="Club")
    return fig


def fig_rating_change_hist(df):
    d = df.dropna(subset=["rating_change"])
    if d.empty:
        return empty_fig()
    fig = px.histogram(d, x="rating_change", nbins=30, color_discrete_sequence=[GREEN])
    fig.add_vline(x=0, line_color="white", line_dash="dash")
    fig.add_vline(x=d["rating_change"].mean(), line_color=AMBER, annotation_text=f"mean {d['rating_change'].mean():+.2f}")
    fig.update_traces(hovertemplate="Change: %{x}<br>Cases: %{y}<extra></extra>")
    return style(fig, "Distribution of Rating Change After Recovery", "Rating change (after - before)", "Number of cases", height=400)


def fig_position_recovery(df):
    t = an.position_recovery(df)
    if t.empty:
        return empty_fig("Not enough cases per position.")
    t = t.sort_values("mean_change")
    fig = go.Figure(go.Bar(x=t["mean_change"], y=t["position"], orientation="h",
                           marker_color=[GREEN if v >= 0 else RED for v in t["mean_change"]],
                           customdata=t["cases"], hovertemplate="<b>%{y}</b><br>Mean change: %{x:+.2f}<br>Cases: %{customdata}<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED)
    return style(fig, "Average Rating Change After Recovery by Position", "Mean rating change", "Position", height=420)


def fig_rating_by_phase(df):
    """Histogram-style comparison of rating distributions before vs after."""
    b = df["rating_before"].dropna()
    a = df["rating_after"].dropna()
    if b.empty and a.empty:
        return empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=b, name="Before injury", marker_color=BLUE, opacity=0.65, nbinsx=25))
    fig.add_trace(go.Histogram(x=a, name="After return", marker_color=GREEN, opacity=0.65, nbinsx=25))
    fig.update_layout(barmode="overlay")
    return style(fig, "Player Performance Overview: Rating Distribution", "Average match rating", "Number of injury cases", height=400)
