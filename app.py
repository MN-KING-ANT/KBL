import os
import pandas as pd
import plotly.express as px
import streamlit as st

from core.metrics import add_advanced_metrics, build_team_ratings
from core.simulator import simulate_matchup

st.set_page_config(page_title="K-Basket Analytics", page_icon="🏀", layout="wide")


@st.cache_data(ttl=1800)
def load_games(league: str) -> pd.DataFrame:
    path = f"data/{league.lower()}_games.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return add_advanced_metrics(df) if not df.empty else df


@st.cache_data(ttl=1800)
def load_kbl_schedule() -> pd.DataFrame:
    path = "data/kbl_schedule.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# 사이드바 설정
st.sidebar.title("🏀 K-Basket 설정")
league_label = st.sidebar.radio("리그 선택", ["KBL (남자프로농구)", "WKBL (여자프로농구)"])
league = "KBL" if league_label.startswith("KBL") else "WKBL"

ewma_span = st.sidebar.slider("최근 폼(EWMA) 기준 경기 수", 3, 10, 5)
recent_weight = st.sidebar.slider("최근 폼 가중치", 0.0, 0.8, 0.4, 0.1)
use_q_split = (
    st.sidebar.checkbox("KBL 1·4Q / 2·3Q 쿼터 분리 모델", value=True)
    if league == "KBL"
    else False
)

df = load_games(league)
if df.empty:
    st.warning(f"{league} 데이터 파일(`data/{league.lower()}_games.csv`)이 아직 없습니다. GitHub Actions에서 `Update KBL & WKBL Data` 워크플로우를 1회 실행해 주세요.")
    st.stop()

ratings = build_team_ratings(df, ewma_span=ewma_span, recent_weight=recent_weight)

st.title(f"🏀 {league} 2026-27 시즌 분석 & 시뮬레이터")
st.caption(f"누적 반영 경기: {df['game_id'].nunique()}경기 | 시즌 평균 {int((1-recent_weight)*100)}% + 최근 {ewma_span}경기 EWMA {int(recent_weight*100)}%")

tabs = ["📊 팀 효율 & 랭킹", "🎯 매치업 시뮬레이터"]
if league == "KBL":
    tabs.append("📅 예정 경기 자동 예측")
tab_objs = st.tabs(tabs)

# ---------- TAB 1: 팀 효율 ----------
with tab_objs[0]:
    cols_show = ["team", "games", "wins", "losses", "ortg", "drtg", "net_rtg", "pace", "efg", "tov_pct", "ftr"]
    if league == "KBL":
        cols_show += ["q14_diff", "q23_diff"]

    rename_map = {
        "team": "팀", "games": "경기", "wins": "승", "losses": "패",
        "ortg": "ORtg", "drtg": "DRtg", "net_rtg": "NetRtg", "pace": "Pace",
        "efg": "eFG%", "tov_pct": "TOV%", "ftr": "FTr",
        "q14_diff": "1·4Q 마진", "q23_diff": "2·3Q 마진",
    }
    st.dataframe(
        ratings[cols_show].rename(columns=rename_map).round(2),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            ratings, x="ortg", y="drtg", text="team", size="games",
            title=f"{league} 공격효율(ORtg) vs 수비효율(DRtg)",
            labels={"ortg": "공격 효율 (높을수록 우수)", "drtg": "수비 효율 (낮을수록 우수)"},
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_traces(textposition="top center")
        fig.add_vline(x=ratings["ortg"].mean(), line_dash="dash", line_color="gray")
        fig.add_hline(y=ratings["drtg"].mean(), line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if league == "KBL":
            q_plot = ratings[["team", "q14_diff", "q23_diff"]].melt(
                id_vars="team", var_name="구간", value_name="평균 득실마진"
            )
            q_plot["구간"] = q_plot["구간"].map({"q14_diff": "1·4Q (외인 1명)", "q23_diff": "2·3Q (외인 2명)"})
            fig_q = px.bar(
                q_plot, x="team", y="평균 득실마진", color="구간", barmode="group",
                title="KBL 쿼터 구간별 팀 마진 (외국인 선수 규정 효과)",
            )
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            fig_n = px.bar(
                ratings, x="team", y="net_rtg", color="net_rtg",
                color_continuous_scale="RdBu", title="WKBL 팀별 Net Rating",
            )
            st.plotly_chart(fig_n, use_container_width=True)

# ---------- TAB 2: 매치업 시뮬레이터 ----------
with tab_objs[1]:
    teams = sorted(ratings["team"].unique())
    c1, c2 = st.columns(2)
    home = c1.selectbox("홈 팀", teams, index=0)
    away = c2.selectbox("원정 팀", teams, index=1 if len(teams) > 1 else 0)

    if home == away:
        st.info("서로 다른 두 팀을 선택해 주세요.")
    else:
        res = simulate_matchup(ratings, home, away, league=league, use_quarter_split=use_q_split)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{home} 승률", f"{res['home_win_prob']*100:.1f}%")
        m2.metric("예상 스코어", f"{res['home_exp_pts']:.1f} : {res['away_exp_pts']:.1f}")
        m3.metric("예상 마진 (홈 기준)", f"{res['exp_margin']:+.1f}")
        m4.metric("예상 합산 총점", f"{res['exp_total']:.1f}")

        if res["split_info"]:
            sp = res["split_info"]
            st.info(
                f"**쿼터 구간 예상** — "
                f"1·4Q (외인 1명): {home} {sp['home_q14']:.1f} vs {away} {sp['away_q14']:.1f} ({sp['home_q14']-sp['away_q14']:+.1f})  |  "
                f"2·3Q (외인 2명): {home} {sp['home_q23']:.1f} vs {away} {sp['away_q23']:.1f} ({sp['home_q23']-sp['away_q23']:+.1f})"
            )

        # 핸디캡 / 언오버 기준점 확률 계산기
        st.subheader("기준점 돌파 확률 계산 (10,000회 시뮬레이션)")
        b1, b2 = st.columns(2)
        with b1:
            h_line = st.number_input("홈 팀 핸디캡 (예: -3.5 또는 +4.5)", value=0.0, step=0.5)
            cover_prob = float(((res["margins"] + h_line) > 0).mean()) * 100
            st.write(f"**{home} ({h_line:+.1f}) 승리 확률:** `{cover_prob:.1f}%`  |  **{away} ({-h_line:+.1f}):** `{100-cover_prob:.1f}%`")
        with b2:
            t_line = st.number_input("언더/오버 기준점", value=round(res["exp_total"] * 2) / 2, step=0.5)
            over_prob = float((res["totals"] > t_line).mean()) * 100
            st.write(f"**오버 ({t_line:.1f}↑) 확률:** `{over_prob:.1f}%`  |  **언더 ({t_line:.1f}↓):** `{100-over_prob:.1f}%`")

        # 점수차 분포 히스토그램
        fig_hist = px.histogram(
            x=res["margins"], nbins=50,
            title=f"{home} 기준 예상 점수차 분포",
            labels={"x": f"{home} - {away} 점수차", "y": "시뮬레이션 횟수"},
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)

        # 시즌 맞대결 기록
        h2h = df[(df["team"] == home) & (df["opp"] == away)][
            ["game_date", "is_home", "pts", "opp_pts", "q14_diff", "q23_diff", "ortg", "drtg"]
        ]
        st.subheader(f"올 시즌 맞대결 기록 ({home} 기준)")
        if h2h.empty:
            st.caption("아직 올 시즌 맞대결 기록이 없습니다.")
        else:
            st.dataframe(h2h.round(1), use_container_width=True, hide_index=True)

# ---------- TAB 3: KBL 예정 경기 자동 예측 ----------
if league == "KBL":
    with tab_objs[2]:
        sched = load_kbl_schedule()
        if sched.empty:
            st.info("예정 경기 데이터가 없습니다.")
        else:
            preds = []
            valid_teams = set(ratings["team"])
            for _, row in sched.head(15).iterrows():
                if row["home"] in valid_teams and row["away"] in valid_teams:
                    sim = simulate_matchup(ratings, row["home"], row["away"], league="KBL", use_quarter_split=use_q_split, n_sims=5000)
                    preds.append({
                        "날짜": row["game_date"],
                        "시간": row["game_time"],
                        "홈": row["home"],
                        "원정": row["away"],
                        "홈 승률(%)": round(sim["home_win_prob"] * 100, 1),
                        "예상 스코어": f"{sim['home_exp_pts']:.1f} : {sim['away_exp_pts']:.1f}",
                        "예상 마진": round(sim["exp_margin"], 1),
                        "예상 총점": round(sim["exp_total"], 1),
                    })
            st.dataframe(pd.DataFrame(preds), use_container_width=True, hide_index=True)
