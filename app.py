import json
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import plotly.express as px
import streamlit as st

from core.metrics import add_advanced_metrics, build_team_ratings
from core.simulator import simulate_matchup
from core.tracker import (
    LOG_PATH,
    auto_grade_predictions,
    build_eval_frame,
    compute_accuracy_summary,
    load_predictions,
    normalize_pick,
    save_predictions,
)

st.set_page_config(page_title="K-Basket Analytics", page_icon="🏀", layout="wide")

KST = timezone(timedelta(hours=9))


@st.cache_data(ttl=1800)
def load_games(league: str) -> pd.DataFrame:
    path = f"data/{league.lower()}_games.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return add_advanced_metrics(df) if not df.empty else df


@st.cache_data(ttl=1800)
def load_schedule(league: str) -> pd.DataFrame:
    path = f"data/{league.lower()}_schedule.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if not df.empty and "game_date" in df.columns:
        df["game_date"] = df["game_date"].astype(str).str.strip()
    return df


def _get_github_secrets() -> tuple[str | None, str | None, str]:
    try:
        token = st.secrets.get("GITHUB_TOKEN", None)
        repo = st.secrets.get("GITHUB_REPO", None)
        branch = st.secrets.get("GITHUB_BRANCH", "main")
        return token, repo, branch
    except Exception:
        return None, None, "main"


def _get_upcoming_schedule(sched_df: pd.DataFrame, valid_teams: set[str], today_str: str) -> pd.DataFrame:
    if sched_df.empty:
        return pd.DataFrame()
    sub = sched_df[
        (sched_df["game_date"].astype(str) >= today_str)
        & (sched_df["home"].isin(valid_teams))
        & (sched_df["away"].isin(valid_teams))
    ].copy()
    if sub.empty:
        return sub
    sort_cols = [c for c in ["game_date", "game_time", "home"] if c in sub.columns]
    return sub.sort_values(sort_cols).reset_index(drop=True)


def _format_h2h_summary(games_df: pd.DataFrame, home: str, away: str) -> str:
    h2h = games_df[(games_df["team"] == home) & (games_df["opp"] == away)].sort_values("game_date")
    if h2h.empty:
        return "올 시즌 맞대결 없음"
    wins = int((h2h["pts"] > h2h["opp_pts"]).sum())
    losses = len(h2h) - wins
    scores = ", ".join(
        f"{r['game_date']}({ '홈' if int(r['is_home'])==1 else '원정' } {int(r['pts'])}:{int(r['opp_pts'])})"
        for _, r in h2h.tail(3).iterrows()
    )
    return f"{home} 기준 {wins}승 {losses}패 [{scores}]"


def _build_gemini_prompt(target_date: str, entries: list[dict]) -> str:
    game_blocks = []
    for i, e in enumerate(entries, start=1):
        lg = e["league"]
        hm, aw = e["home"], e["away"]
        sim = e["sim"]
        h_r, a_r = e["home_rating"], e["away_rating"]

        quarter_line = ""
        if lg == "KBL" and sim.get("split_info"):
            sp = sim["split_info"]
            quarter_line = (
                f"\n  · KBL 쿼터 구간 예상: 1·4Q(외인1명) {hm} {sp['home_q14']:.1f} vs {aw} {sp['away_q14']:.1f} ({sp['home_q14']-sp['away_q14']:+.1f}) / "
                f"2·3Q(외인2명) {hm} {sp['home_q23']:.1f} vs {aw} {sp['away_q23']:.1f} ({sp['home_q23']-sp['away_q23']:+.1f})"
                f"\n  · 팀별 시즌 쿼터 마진: {hm}(1·4Q {h_r['q14_diff']:+.1f}, 2·3Q {h_r['q23_diff']:+.1f}) vs {aw}(1·4Q {a_r['q14_diff']:+.1f}, 2·3Q {a_r['q23_diff']:+.1f})"
            )

        note_line = f"\n  · 추가 메모/배당: {e['extra_note']}" if e["extra_note"] else ""

        block = f"""- 경기 {i}: [{lg}] {hm}(홈) vs {aw}(원정) | 날짜: {e['game_date']}
  · 팀 효율(ORtg/DRtg/Net/Pace/eFG%):
    - {hm}(홈): {h_r['wins']}승 {h_r['losses']}패 | ORtg {h_r['ortg']:.1f} / DRtg {h_r['drtg']:.1f} / Net {h_r['net_rtg']:+.1f} / Pace {h_r['pace']:.1f} / eFG {h_r['efg']:.1f}%
    - {aw}(원정): {a_r['wins']}승 {a_r['losses']}패 | ORtg {a_r['ortg']:.1f} / DRtg {a_r['drtg']:.1f} / Net {a_r['net_rtg']:+.1f} / Pace {a_r['pace']:.1f} / eFG {a_r['efg']:.1f}%
  · 올 시즌 맞대결: {e['h2h_text']}{quarter_line}
  · 시장 기준점: 홈 핸디캡 {e['handi_line']:+.1f} / 언오버 {e['ou_line']:.1f}
  · 몬테카를로 시뮬레이션(10,000회):
    - 예상 스코어: {hm} {sim['home_exp_pts']:.1f} : {aw} {sim['away_exp_pts']:.1f} (예상 마진 {sim['exp_margin']:+.1f}, 예상 총점 {sim['exp_total']:.1f})
    - 승패 픽: {e['m_ml_pick']} ({e['m_ml_conf']:.1f}% | 홈 승률 {sim['home_win_prob']*100:.1f}%)
    - 핸디({e['handi_line']:+.1f}) 픽: {e['m_hd_pick']} ({e['m_hd_conf']:.1f}%)
    - 언오버({e['ou_line']:.1f}) 픽: {e['m_ou_pick']} ({e['m_ou_conf']:.1f}%)
  · 라스(시장) 예상: 승패 [{e['las_ml'] or '미입력'}] / 핸디 [{e['las_handi'] or '미입력'}] / 언오버 [{e['las_ou'] or '미입력'}]{note_line}"""
        game_blocks.append(block)

    games_text = "\n\n".join(game_blocks)

    return f"""너는 2026-27 시즌 한국 프로농구(KBL, WKBL) 전문 퀀트·매치업 애널리스트다.
아래 제공한 {target_date} 경기별 팀 효율, 몬테카를로 시뮬레이션 수치, 라스(시장) 예상을 바탕으로 최신 부상·로스터·일정 변수를 웹 검색으로 확인한 뒤, 최종 분석과 앱 일괄 등록용 JSON을 출력해 줘.

[분석 원칙]
1. 최신 변수 확인: 오늘({target_date}) 기준 양 팀 핵심 국내 선수·외국인 선수·아시아쿼터 부상/결장/복귀, 외인 교체 여부, 백투백(연전) 및 장거리 이동 피로도를 반드시 검색해 반영한다.
2. 리그별 핵심 기준:
   - KBL: 2026-27 시즌 2·3쿼터 외인 2명 동시 출전 규정(1·4Q 국내 뎁스 vs 2·3Q 외인 2옵션 화력)과 홈 이점(평균 +2.5점)을 반영한다.
   - WKBL: 6개 구단 체제 특성상 상대 전적·매치업 상성과 리바운드/3점슛 기복을 중점 반영하며, 홈 이점은 거의 없는 것(0.0점)으로 간주한다.
3. 모델 충돌 판별: 몬테카를로(정량)와 라스(시장) 방향이 다를 경우, 부상·로스터 변화 때문인지 시장 과잉 반응인지 근거를 밝히고 제미나이 최종 픽을 선택한다.

[제미나이 5단계 등급 기준 (남발 금지)]
- S: 핵심 전력·상성 우위 확실 + 정량/정성 근거 강하게 일치 (주력)
- A: 뚜렷한 우위 근거가 있는 추천 픽
- B: 우세하나 슈팅 기복·파울 트러블 등 변수 존재 (일반)
- C: 박빙 또는 모델 충돌 구간, 미세 우위 (소액/참고)
- D: 불확실성 매우 높음 (패스 권장, 픽은 "" 처리 가능)

[출력 형식]
1. 경기별 핵심 분석(부상/로스터 변수, 쿼터/매치업 포인트, 몬테 vs 라스 충돌 해설, 승패·핸디·언오버 최종 픽 및 등급)을 간결하게 정리한다.
2. 맨 마지막에 반드시 아래 스키마를 정확히 지킨 순수 JSON 배열 코드 블록 1개만 출력한다. (주석 금지, 입력된 팀명·기준점·라스 예상 값 그대로 유지)
   - 키 목록: league, game_date, home, away, handi_line, ou_line, las_ml, las_handi, las_ou, gemini_ml, gemini_ml_grade, gemini_handi, gemini_handi_grade, gemini_ou, gemini_ou_grade, memo
   - 픽 값: 승패/핸디는 반드시 "홈", "원정", "" 중 하나 / 언오버는 "오버", "언더", "" 중 하나
   - 등급 값: "S", "A", "B", "C", "D", "" 중 하나
   - memo: 25자 이내 핵심 근거 요약

========================================
[분석 대상 경기 데이터 ({target_date}, 총 {len(entries)}경기)]

{games_text}
"""


# ---------- 사이드바 (백테스트 최적 기본값 자동 세팅) ----------
st.sidebar.title("🏀 K-Basket 설정")
league_label = st.sidebar.radio("리그 선택", ["KBL (남자프로농구)", "WKBL (여자프로농구)"])
league = "KBL" if league_label.startswith("KBL") else "WKBL"

default_span = 3 if league == "KBL" else 5
default_weight = 0.15 if league == "KBL" else 0.0

ewma_span = st.sidebar.slider("최근 폼(EWMA) 기준 경기 수", 3, 10, default_span, key=f"span_{league}")
recent_weight = st.sidebar.slider("최근 폼 가중치", 0.0, 0.8, default_weight, 0.05, key=f"w_{league}")
use_q_split = (
    st.sidebar.checkbox("KBL 1·4Q / 2·3Q 쿼터 분리 모델", value=True)
    if league == "KBL"
    else False
)

today_kst = datetime.now(KST).strftime("%Y-%m-%d")
df = load_games(league)
if df.empty:
    st.warning(f"{league} 데이터(`data/{league.lower()}_games.csv`)가 아직 없습니다. GitHub Actions를 1회 실행해 주세요.")
    st.stop()

ratings = build_team_ratings(df, ewma_span=ewma_span, recent_weight=recent_weight)
sched = load_schedule(league)
valid_teams = set(ratings["team"])
upcoming_sched = _get_upcoming_schedule(sched, valid_teams, today_kst)

st.title(f"🏀 {league} 2026-27 시즌 분석 & 시뮬레이터")
st.caption(
    f"오늘(KST): {today_kst} | 누적 반영 경기: {df['game_id'].nunique()}경기 | "
    f"시즌 평균 {int(round((1-recent_weight)*100))}% + 최근 {ewma_span}경기 EWMA {int(round(recent_weight*100))}% | "
    f"HCA: {'2.5점' if league == 'KBL' else '0.0점'}"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 팀 효율 & 랭킹",
    "🎯 매치업 시뮬레이터",
    "📋 라스 입력 & 프롬프트 추출",
    "📅 예정 경기 전체 예측",
    "📝 픽 기록 & 적중률 검증",
])

# ==============================================================================
# TAB 1: 팀 효율 & 랭킹
# ==============================================================================
with tab1:
    cols_show = ["team", "games", "wins", "losses", "ortg", "drtg", "net_rtg", "pace", "efg", "tov_pct", "ftr"]
    if league == "KBL":
        cols_show += ["q14_diff", "q23_diff"]

    rename_map = {
        "team": "팀", "games": "경기", "wins": "승", "losses": "패",
        "ortg": "ORtg", "drtg": "DRtg", "net_rtg": "NetRtg", "pace": "Pace",
        "efg": "eFG%", "tov_pct": "TOV%", "ftr": "FTr",
        "q14_diff": "1·4Q 마진", "q23_diff": "2·3Q 마진",
    }
    st.dataframe(ratings[cols_show].rename(columns=rename_map).round(2), use_container_width=True, hide_index=True)

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

# ==============================================================================
# TAB 2: 매치업 시뮬레이터 (오늘/가장 가까운 예정 경기 자동 불러오기)
# ==============================================================================
with tab2:
    teams = sorted(valid_teams)
    manual_mode = False
    selected_game_date = today_kst

    if not upcoming_sched.empty:
        avail_dates = sorted(upcoming_sched["game_date"].unique())
        default_date = today_kst if today_kst in avail_dates else avail_dates[0]
        is_today = (default_date == today_kst)

        d_col, m_col = st.columns([1, 2.2])
        with d_col:
            selected_game_date = st.selectbox(
                "경기 날짜 (자동 선택)",
                options=avail_dates,
                index=avail_dates.index(default_date),
                format_func=lambda d: f"{d} (오늘)" if d == today_kst else d,
                key=f"sim_date_{league}",
            )
        day_games = upcoming_sched[upcoming_sched["game_date"] == selected_game_date].reset_index(drop=True)

        match_labels = [
            f"{r['home']} (홈) vs {r['away']} (원정)" + (f" · {r['game_time']}" if str(r.get("game_time", "")).strip() not in ("", "nan") else "")
            for _, r in day_games.iterrows()
        ]
        match_labels.append("🔧 직접 팀 선택 (수동)")

        with m_col:
            choice_idx = st.selectbox(
                "매치업 선택",
                options=range(len(match_labels)),
                format_func=lambda i: match_labels[i],
                key=f"sim_match_{league}_{selected_game_date}",
            )

        if not is_today and selected_game_date == default_date:
            st.caption(f"오늘({today_kst}) {league} 경기가 없어 가장 가까운 예정 경기({default_date})를 자동으로 불러왔습니다.")

        if choice_idx < len(day_games):
            home = day_games.iloc[choice_idx]["home"]
            away = day_games.iloc[choice_idx]["away"]
        else:
            manual_mode = True
    else:
        st.caption("예정 경기 일정이 없어 수동 선택 모드로 실행합니다.")
        manual_mode = True

    if manual_mode:
        c1, c2 = st.columns(2)
        home = c1.selectbox("홈 팀", teams, index=0, key=f"sim_h_{league}")
        away = c2.selectbox("원정 팀", teams, index=1 if len(teams) > 1 else 0, key=f"sim_a_{league}")

    if home == away:
        st.info("서로 다른 두 팀을 선택해 주세요.")
    else:
        res = simulate_matchup(ratings, home, away, league=league, use_quarter_split=use_q_split)
        st.subheader(f"🏟️ {home} (홈) vs {away} (원정)")

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

        st.subheader("기준점 돌파 확률 계산 (10,000회 시뮬레이션)")
        b1, b2 = st.columns(2)
        default_hl = float(round(-res["exp_margin"] * 2) / 2)
        default_tl = float(round(res["exp_total"] * 2) / 2)
        with b1:
            h_line = st.number_input(
                "홈 팀 핸디캡 (예: -3.5 또는 +4.5)",
                value=default_hl,
                step=0.5,
                key=f"hl_{league}_{home}_{away}",
            )
            cover_prob = float(((res["margins"] + h_line) > 0).mean()) * 100
            st.write(f"**{home} ({h_line:+.1f}) 확률:** `{cover_prob:.1f}%`  |  **{away} ({-h_line:+.1f}):** `{100-cover_prob:.1f}%`")
        with b2:
            t_line = st.number_input(
                "언더/오버 기준점",
                value=default_tl,
                step=0.5,
                key=f"tl_{league}_{home}_{away}",
            )
            over_prob = float((res["totals"] > t_line).mean()) * 100
            st.write(f"**오버 ({t_line:.1f}↑) 확률:** `{over_prob:.1f}%`  |  **언더 ({t_line:.1f}↓):** `{100-over_prob:.1f}%`")

        fig_hist = px.histogram(
            x=res["margins"], nbins=50,
            title=f"{home} 기준 예상 점수차 분포",
            labels={"x": f"{home} - {away} 점수차", "y": "시뮬레이션 횟수"},
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)

        h2h = df[(df["team"] == home) & (df["opp"] == away)][
            ["game_date", "is_home", "pts", "opp_pts", "q14_diff", "q23_diff", "ortg", "drtg"]
        ]
        st.subheader(f"올 시즌 맞대결 기록 ({home} 기준)")
        if h2h.empty:
            st.caption("아직 올 시즌 맞대결 기록이 없습니다.")
        else:
            st.dataframe(h2h.round(1), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: 라스 입력 & 제미나이 프롬프트 자동 추출
# ==============================================================================
with tab3:
    st.subheader("📋 라스(시장) 입력 & 제미나이 프롬프트 추출기")
    st.caption("오늘(또는 선택한 날짜) 경기의 시뮬레이션 수치에 라스 기준점·예상만 입력하면, 제미나이 분석 & JSON 출력용 프롬프트가 자동으로 생성됩니다.")

    kbl_games_all = load_games("KBL")
    wkbl_games_all = load_games("WKBL")
    kbl_sched_all = load_schedule("KBL")
    wkbl_sched_all = load_schedule("WKBL")

    # 리그별 레이팅 준비 (현재 선택 리그는 사이드바 설정 반영, 타 리그는 최적 기본값)
    ratings_by_lg = {}
    games_by_lg = {"KBL": kbl_games_all, "WKBL": wkbl_games_all}
    if not kbl_games_all.empty:
        ratings_by_lg["KBL"] = ratings if league == "KBL" else build_team_ratings(kbl_games_all, ewma_span=3, recent_weight=0.15)
    if not wkbl_games_all.empty:
        ratings_by_lg["WKBL"] = ratings if league == "WKBL" else build_team_ratings(wkbl_games_all, ewma_span=5, recent_weight=0.0)

    p_col1, p_col2 = st.columns([1.2, 1])
    scope = p_col1.radio(
        "추출 대상 범위",
        ["BOTH", "CURRENT"],
        horizontal=True,
        format_func=lambda x: "🔥 KBL + WKBL 통합 추출" if x == "BOTH" else f"현재 선택 리그만 ({league})",
    )

    target_leagues = ["KBL", "WKBL"] if scope == "BOTH" else [league]
    sched_frames = []
    for lg in target_leagues:
        if lg not in ratings_by_lg:
            continue
        raw_sc = kbl_sched_all if lg == "KBL" else wkbl_sched_all
        up_sc = _get_upcoming_schedule(raw_sc, set(ratings_by_lg[lg]["team"]), today_kst)
        if not up_sc.empty:
            up_sc["league"] = lg
            sched_frames.append(up_sc)

    combined_sched = pd.concat(sched_frames, ignore_index=True) if sched_frames else pd.DataFrame()

    prompt_entries = []
    prompt_date = today_kst

    if not combined_sched.empty:
        p_dates = sorted(combined_sched["game_date"].unique())
        p_default = today_kst if today_kst in p_dates else p_dates[0]
        prompt_date = p_col2.selectbox(
            "분석할 경기 날짜",
            options=p_dates,
            index=p_dates.index(p_default),
            format_func=lambda d: f"{d} (오늘)" if d == today_kst else d,
            key=f"prompt_date_{scope}_{league}",
        )
        day_df = combined_sched[combined_sched["game_date"] == prompt_date].reset_index(drop=True)

        st.markdown(f"**{prompt_date} 예정 경기 ({len(day_df)}경기)** — 각 경기의 라스 기준점과 예상을 입력하세요.")

        for idx, row in day_df.iterrows():
            lg = row["league"]
            hm, aw = row["home"], row["away"]
            r_tbl = ratings_by_lg[lg]
            g_tbl = games_by_lg[lg]
            q_flag = use_q_split if lg == "KBL" else False
            sim_p = simulate_matchup(r_tbl, hm, aw, league=lg, use_quarter_split=q_flag)

            k_prefix = f"pr_{prompt_date}_{lg}_{hm}_{aw}_{idx}"
            with st.container(border=True):
                h_c1, h_c2 = st.columns([3, 1])
                h_c1.markdown(
                    f"#### [{lg}] {hm} (홈) vs {aw} (원정)  "
                    f"<span style='font-size:0.85em;color:gray;'>예상 {sim_p['home_exp_pts']:.1f} : {sim_p['away_exp_pts']:.1f} (홈 승률 {sim_p['home_win_prob']*100:.1f}%)</span>",
                    unsafe_allow_html=True,
                )
                include_game = h_c2.checkbox("프롬프트 포함", value=True, key=f"{k_prefix}_inc")

                if include_game:
                    c_in1, c_in2, c_in3, c_in4, c_in5 = st.columns(5)
                    def_hl = float(round(-sim_p["exp_margin"] * 2) / 2)
                    def_tl = float(round(sim_p["exp_total"] * 2) / 2)

                    in_hl = c_in1.number_input("홈 핸디캡", value=def_hl, step=0.5, key=f"{k_prefix}_hl")
                    in_ol = c_in2.number_input("언오버 기준점", value=def_tl, step=0.5, key=f"{k_prefix}_ol")
                    las_ml = c_in3.selectbox("라스 승패", ["", "홈", "원정"], key=f"{k_prefix}_lml")
                    las_hd = c_in4.selectbox("라스 핸디", ["", "홈", "원정"], key=f"{k_prefix}_lhd")
                    las_ou = c_in5.selectbox("라스 언오버", ["", "오버", "언더"], key=f"{k_prefix}_lou")
                    extra_note = st.text_input("추가 메모/배당/결장 정보 (선택)", value="", key=f"{k_prefix}_note")

                    # 입력된 기준점으로 몬테카를로 확률 즉시 재계산
                    p_ml = sim_p["home_win_prob"] * 100
                    m_ml_pick = "홈" if p_ml >= 50 else "원정"
                    m_ml_conf = max(p_ml, 100 - p_ml)

                    p_hd = float(((sim_p["margins"] + in_hl) > 0).mean()) * 100
                    m_hd_pick = "홈" if p_hd >= 50 else "원정"
                    m_hd_conf = max(p_hd, 100 - p_hd)

                    p_ou = float((sim_p["totals"] > in_ol).mean()) * 100
                    m_ou_pick = "오버" if p_ou >= 50 else "언더"
                    m_ou_conf = max(p_ou, 100 - p_ou)

                    st.caption(
                        f"🎯 **몬테 자동 계산** — "
                        f"승패: `{m_ml_pick} {m_ml_conf:.1f}%` | "
                        f"핸디({in_hl:+.1f}): `{m_hd_pick} {m_hd_conf:.1f}%` | "
                        f"언오버({in_ol:.1f}): `{m_ou_pick} {m_ou_conf:.1f}%`"
                    )

                    prompt_entries.append({
                        "league": lg,
                        "game_date": prompt_date,
                        "home": hm,
                        "away": aw,
                        "sim": sim_p,
                        "home_rating": r_tbl.loc[r_tbl["team"] == hm].iloc[0],
                        "away_rating": r_tbl.loc[r_tbl["team"] == aw].iloc[0],
                        "h2h_text": _format_h2h_summary(g_tbl, hm, aw),
                        "handi_line": in_hl,
                        "ou_line": in_ol,
                        "m_ml_pick": m_ml_pick, "m_ml_conf": m_ml_conf,
                        "m_hd_pick": m_hd_pick, "m_hd_conf": m_hd_conf,
                        "m_ou_pick": m_ou_pick, "m_ou_conf": m_ou_conf,
                        "las_ml": las_ml, "las_handi": las_hd, "las_ou": las_ou,
                        "extra_note": extra_note.strip(),
                    })
    else:
        st.info("불러올 예정 경기가 없습니다. 아래 '수동 경기 추가'로 프롬프트를 생성할 수 있습니다.")

    # 예정 경기에 없는 경기 수동 추가 옵션
    with st.expander("➕ 일정 외 경기 수동 추가 (선택)"):
        mc1, mc2, mc3 = st.columns(3)
        m_lg = mc1.selectbox("리그", [lg for lg in ["KBL", "WKBL"] if lg in ratings_by_lg], key="man_pr_lg")
        m_teams = sorted(ratings_by_lg[m_lg]["team"].unique())
        m_home = mc2.selectbox("홈 팀", m_teams, index=0, key="man_pr_h")
        m_away = mc3.selectbox("원정 팀", m_teams, index=1 if len(m_teams) > 1 else 0, key="man_pr_a")
        use_manual_pr = st.checkbox("이 수동 경기도 프롬프트에 추가", value=False, key="man_pr_use")

        if use_manual_pr and m_home != m_away:
            r_tbl = ratings_by_lg[m_lg]
            g_tbl = games_by_lg[m_lg]
            sim_m = simulate_matchup(r_tbl, m_home, m_away, league=m_lg, use_quarter_split=(m_lg == "KBL"))
            mi1, mi2, mi3, mi4, mi5 = st.columns(5)
            m_hl = mi1.number_input("홈 핸디캡", value=float(round(-sim_m["exp_margin"] * 2) / 2), step=0.5, key="man_pr_hl")
            m_ol = mi2.number_input("언오버 기준점", value=float(round(sim_m["exp_total"] * 2) / 2), step=0.5, key="man_pr_ol")
            m_lml = mi3.selectbox("라스 승패", ["", "홈", "원정"], key="man_pr_lml")
            m_lhd = mi4.selectbox("라스 핸디", ["", "홈", "원정"], key="man_pr_lhd")
            m_lou = mi5.selectbox("라스 언오버", ["", "오버", "언더"], key="man_pr_lou")
            m_note = st.text_input("추가 메모", value="", key="man_pr_note")

            p_ml = sim_m["home_win_prob"] * 100
            p_hd = float(((sim_m["margins"] + m_hl) > 0).mean()) * 100
            p_ou = float((sim_m["totals"] > m_ol).mean()) * 100

            prompt_entries.append({
                "league": m_lg, "game_date": prompt_date, "home": m_home, "away": m_away,
                "sim": sim_m,
                "home_rating": r_tbl.loc[r_tbl["team"] == m_home].iloc[0],
                "away_rating": r_tbl.loc[r_tbl["team"] == m_away].iloc[0],
                "h2h_text": _format_h2h_summary(g_tbl, m_home, m_away),
                "handi_line": m_hl, "ou_line": m_ol,
                "m_ml_pick": "홈" if p_ml >= 50 else "원정", "m_ml_conf": max(p_ml, 100 - p_ml),
                "m_hd_pick": "홈" if p_hd >= 50 else "원정", "m_hd_conf": max(p_hd, 100 - p_hd),
                "m_ou_pick": "오버" if p_ou >= 50 else "언더", "m_ou_conf": max(p_ou, 100 - p_ou),
                "las_ml": m_lml, "las_handi": m_lhd, "las_ou": m_lou,
                "extra_note": m_note.strip(),
            })

    st.markdown("---")
    st.subheader("🚀 생성된 제미나이 분석용 프롬프트")
    if not prompt_entries:
        st.info("프롬프트에 포함된 경기가 없습니다.")
    else:
        final_prompt = _build_gemini_prompt(prompt_date, prompt_entries)
        st.caption("오른쪽 위 복사 아이콘을 눌러 제미나이에 그대로 붙여넣고, 나온 JSON을 5번 탭에 붙여넣으세요.")
        st.code(final_prompt, language="text")

# ==============================================================================
# TAB 4: 예정 경기 전체 예측
# ==============================================================================
with tab4:
    if upcoming_sched.empty:
        st.info(f"{league} 예정 경기 파일(`data/{league.lower()}_schedule.csv`)이 비어 있거나 예정 경기가 없습니다.")
    else:
        preds = []
        for _, row in upcoming_sched.head(20).iterrows():
            sim = simulate_matchup(
                ratings, row["home"], row["away"],
                league=league, use_quarter_split=use_q_split, n_sims=5000,
            )
            preds.append({
                "날짜": row["game_date"],
                "시간": row.get("game_time", ""),
                "홈": row["home"],
                "원정": row["away"],
                "홈 승률(%)": round(sim["home_win_prob"] * 100, 1),
                "예상 스코어": f"{sim['home_exp_pts']:.1f} : {sim['away_exp_pts']:.1f}",
                "예상 마진": round(sim["exp_margin"], 1),
                "예상 총점": round(sim["exp_total"], 1),
                "구장": row.get("stadium", ""),
            })
        st.dataframe(pd.DataFrame(preds), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 5: 픽 기록 & 몬테/제미/라스 적중률 검증
# ==============================================================================
with tab5:
    kbl_all = load_games("KBL")
    wkbl_all = load_games("WKBL")
    gh_token, gh_repo, gh_branch = _get_github_secrets()

    log_df = load_predictions(LOG_PATH)
    log_df, graded_now = auto_grade_predictions(log_df, kbl_all, wkbl_all)
    if graded_now > 0:
        _, msg = save_predictions(log_df, LOG_PATH, gh_token, gh_repo, gh_branch)
        st.success(f"종료된 {graded_now}경기 결과를 자동 채점했습니다! ({msg})")

    sub_json, sub_input, sub_stats = st.tabs([
        "📋 제미나이 JSON 일괄 등록",
        "✍️ 경기별 개별 등록",
        "📈 적중률 · 합의 · 충돌 대시보드",
    ])

    # --- 5-1. 제미나이 JSON 일괄 등록 (가장 자주 쓰는 탭을 첫 번째로) ---
    with sub_json:
        st.caption("3번 탭에서 추출한 프롬프트로 제미나이에게 분석받은 뒤, 맨 아래 JSON 배열만 복사해서 여기에 붙여넣으세요.")
        sample_json = """[
  {
    "league": "KBL",
    "game_date": "2026-12-03",
    "home": "창원 LG",
    "away": "안양 정관장",
    "handi_line": -3.5,
    "ou_line": 157.5,
    "las_ml": "홈",
    "las_handi": "홈",
    "las_ou": "언더",
    "gemini_ml": "홈",
    "gemini_ml_grade": "A",
    "gemini_handi": "원정",
    "gemini_handi_grade": "B",
    "gemini_ou": "언더",
    "gemini_ou_grade": "A",
    "memo": "LG 수비 우위, 핸디는 빡빡해 언더 주력"
  }
]"""
        raw_json = st.text_area("제미나이 JSON 붙여넣기", height=230, placeholder=sample_json)
        if st.button("📥 JSON 일괄 등록", type="primary"):
            try:
                cleaned = raw_json.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`").replace("json", "", 1).strip()
                items = json.loads(cleaned)
                if isinstance(items, dict):
                    items = [items]

                ratings_cache = {
                    "KBL": build_team_ratings(kbl_all, ewma_span=3, recent_weight=0.15) if not kbl_all.empty else pd.DataFrame(),
                    "WKBL": build_team_ratings(wkbl_all, ewma_span=5, recent_weight=0.0) if not wkbl_all.empty else pd.DataFrame(),
                }
                added = 0
                for it in items:
                    lg = str(it.get("league", league)).upper()
                    gdate = str(it.get("game_date", "")).strip()
                    hm = str(it.get("home", "")).strip()
                    aw = str(it.get("away", "")).strip()
                    if not hm or not aw:
                        continue

                    hline = float(it.get("handi_line", 0.0))
                    oline = float(it.get("ou_line", 0.0))
                    r_tbl = ratings_cache.get(lg, pd.DataFrame())

                    if not r_tbl.empty and hm in set(r_tbl["team"]) and aw in set(r_tbl["team"]):
                        sim_j = simulate_matchup(r_tbl, hm, aw, league=lg, use_quarter_split=(lg == "KBL"))
                        m_hp, m_ap = f"{sim_j['home_exp_pts']:.1f}", f"{sim_j['away_exp_pts']:.1f}"
                        p_ml = sim_j["home_win_prob"] * 100
                        m_ml, m_ml_c = ("홈" if p_ml >= 50 else "원정"), f"{max(p_ml, 100-p_ml):.1f}"
                        p_hd = float(((sim_j["margins"] + hline) > 0).mean()) * 100
                        m_hd, m_hd_c = ("홈" if p_hd >= 50 else "원정"), f"{max(p_hd, 100-p_hd):.1f}"
                        p_ou = float((sim_j["totals"] > oline).mean()) * 100 if oline > 0 else 50.0
                        m_ou, m_ou_c = (("오버" if p_ou >= 50 else "언더") if oline > 0 else ""), (f"{max(p_ou, 100-p_ou):.1f}" if oline > 0 else "")
                    else:
                        m_hp = m_ap = m_ml = m_ml_c = m_hd = m_hd_c = m_ou = m_ou_c = ""

                    rid = f"{lg}_{gdate}_{hm}_{aw}"
                    default_g = str(it.get("gemini_grade", "")).upper()
                    new_row = {
                        "record_id": rid, "league": lg, "game_date": gdate, "home": hm, "away": aw,
                        "handi_line": str(hline), "ou_line": str(oline) if oline > 0 else "",
                        "monte_home_pts": m_hp, "monte_away_pts": m_ap,
                        "monte_ml_pick": m_ml, "monte_ml_prob": m_ml_c,
                        "monte_handi_pick": m_hd, "monte_handi_prob": m_hd_c,
                        "monte_ou_pick": m_ou, "monte_ou_prob": m_ou_c,
                        "las_ml_pick": normalize_pick(it.get("las_ml", ""), "ml", hm, aw),
                        "las_handi_pick": normalize_pick(it.get("las_handi", ""), "handi", hm, aw),
                        "las_ou_pick": normalize_pick(it.get("las_ou", ""), "ou"),
                        "gemini_ml_pick": normalize_pick(it.get("gemini_ml", ""), "ml", hm, aw),
                        "gemini_handi_pick": normalize_pick(it.get("gemini_handi", ""), "handi", hm, aw),
                        "gemini_ou_pick": normalize_pick(it.get("gemini_ou", ""), "ou"),
                        "gemini_ml_grade": str(it.get("gemini_ml_grade", default_g)).upper(),
                        "gemini_handi_grade": str(it.get("gemini_handi_grade", default_g)).upper(),
                        "gemini_ou_grade": str(it.get("gemini_ou_grade", default_g)).upper(),
                        "memo": str(it.get("memo", "")),
                        "actual_home_pts": "", "actual_away_pts": "", "status": "pending",
                    }
                    log_df = log_df[log_df["record_id"] != rid]
                    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
                    added += 1

                log_df, _ = auto_grade_predictions(log_df, kbl_all, wkbl_all)
                _, msg = save_predictions(log_df, LOG_PATH, gh_token, gh_repo, gh_branch)
                st.success(f"{added}경기 일괄 등록 완료! ({msg})")
                st.rerun()
            except Exception as e:
                st.error(f"JSON 파싱 오류: {e}")

    # --- 5-2. 경기별 개별 등록 ---
    with sub_input:
        teams_lg = sorted(valid_teams)
        sched_opts = ["직접 선택"]
        sched_rows = []
        if not upcoming_sched.empty:
            for _, sr in upcoming_sched.head(15).iterrows():
                sched_opts.append(f"{sr['game_date']} | {sr['home']} vs {sr['away']}")
                sched_rows.append(sr)

        pick_idx = st.selectbox("예정 경기 불러오기", range(len(sched_opts)), format_func=lambda i: sched_opts[i], key=f"trk_sel_{league}")
        if pick_idx > 0:
            sr = sched_rows[pick_idx - 1]
            init_date, init_home, init_away = str(sr["game_date"]), sr["home"], sr["away"]
        else:
            init_date, init_home, init_away = today_kst, teams_lg[0], teams_lg[1 if len(teams_lg) > 1 else 0]

        r1, r2, r3 = st.columns(3)
        in_date = r1.text_input("경기 날짜 (YYYY-MM-DD)", value=init_date, key=f"trk_dt_{league}_{pick_idx}")
        in_home = r2.selectbox("홈 팀", teams_lg, index=teams_lg.index(init_home), key=f"trk_h_{league}_{pick_idx}")
        in_away = r3.selectbox("원정 팀", teams_lg, index=teams_lg.index(init_away), key=f"trk_a_{league}_{pick_idx}")

        if in_home != in_away:
            sim_in = simulate_matchup(ratings, in_home, in_away, league=league, use_quarter_split=use_q_split)
            l1, l2 = st.columns(2)
            in_hline = l1.number_input("홈 기준 핸디캡", value=float(round(-sim_in["exp_margin"] * 2) / 2), step=0.5, key=f"trk_hl_{league}_{pick_idx}")
            in_oline = l2.number_input("언오버 기준점", value=float(round(sim_in["exp_total"] * 2) / 2), step=0.5, key=f"trk_ol_{league}_{pick_idx}")

            m_ml_p = sim_in["home_win_prob"] * 100
            m_ml_pick, m_ml_conf = ("홈" if m_ml_p >= 50 else "원정"), max(m_ml_p, 100 - m_ml_p)
            m_hd_p = float(((sim_in["margins"] + in_hline) > 0).mean()) * 100
            m_hd_pick, m_hd_conf = ("홈" if m_hd_p >= 50 else "원정"), max(m_hd_p, 100 - m_hd_p)
            m_ou_p = float((sim_in["totals"] > in_oline).mean()) * 100
            m_ou_pick, m_ou_conf = ("오버" if m_ou_p >= 50 else "언더"), max(m_ou_p, 100 - m_ou_p)

            st.info(
                f"**몬테카를로 자동 픽** — "
                f"승패: `{m_ml_pick} ({m_ml_conf:.1f}%)` | "
                f"핸디({in_hline:+.1f}): `{m_hd_pick} ({m_hd_conf:.1f}%)` | "
                f"언오버({in_oline:.1f}): `{m_ou_pick} ({m_ou_conf:.1f}%)`"
            )

            with st.form(f"add_pick_form_{league}"):
                st.markdown("**라스(시장) 예상 입력**")
                la1, la2, la3 = st.columns(3)
                las_ml = la1.selectbox("라스 승패", ["", "홈", "원정"])
                las_hd = la2.selectbox("라스 핸디캡", ["", "홈", "원정"])
                las_ou = la3.selectbox("라스 언오버", ["", "오버", "언더"])

                st.markdown("**제미나이 예상 & 등급(S~D) 입력**")
                g1, g2, g3 = st.columns(3)
                gem_ml = g1.selectbox("제미나이 승패", ["", "홈", "원정"])
                gem_ml_g = g1.selectbox("승패 등급", ["", "S", "A", "B", "C", "D"])
                gem_hd = g2.selectbox("제미나이 핸디캡", ["", "홈", "원정"])
                gem_hd_g = g2.selectbox("핸디 등급", ["", "S", "A", "B", "C", "D"])
                gem_ou = g3.selectbox("제미나이 언오버", ["", "오버", "언더"])
                gem_ou_g = g3.selectbox("언오버 등급", ["", "S", "A", "B", "C", "D"])

                memo = st.text_input("메모 (부상/결장/특이사항)")
                if st.form_submit_button("💾 픽 저장하기", type="primary"):
                    rid = f"{league}_{in_date}_{in_home}_{in_away}"
                    new_row = {
                        "record_id": rid, "league": league, "game_date": in_date,
                        "home": in_home, "away": in_away,
                        "handi_line": str(in_hline), "ou_line": str(in_oline),
                        "monte_home_pts": f"{sim_in['home_exp_pts']:.1f}",
                        "monte_away_pts": f"{sim_in['away_exp_pts']:.1f}",
                        "monte_ml_pick": m_ml_pick, "monte_ml_prob": f"{m_ml_conf:.1f}",
                        "monte_handi_pick": m_hd_pick, "monte_handi_prob": f"{m_hd_conf:.1f}",
                        "monte_ou_pick": m_ou_pick, "monte_ou_prob": f"{m_ou_conf:.1f}",
                        "las_ml_pick": las_ml, "las_handi_pick": las_hd, "las_ou_pick": las_ou,
                        "gemini_ml_pick": gem_ml, "gemini_handi_pick": gem_hd, "gemini_ou_pick": gem_ou,
                        "gemini_ml_grade": gem_ml_g, "gemini_handi_grade": gem_hd_g, "gemini_ou_grade": gem_ou_g,
                        "memo": memo,
                        "actual_home_pts": "", "actual_away_pts": "", "status": "pending",
                    }
                    log_df = log_df[log_df["record_id"] != rid]
                    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
                    log_df, _ = auto_grade_predictions(log_df, kbl_all, wkbl_all)
                    _, msg = save_predictions(log_df, LOG_PATH, gh_token, gh_repo, gh_branch)
                    st.success(f"저장 완료: {in_home} vs {in_away} ({msg})")
                    st.rerun()

    # --- 5-3. 적중률 · 방향 일치 · 충돌 대시보드 ---
    with sub_stats:
        f1, f2 = st.columns(2)
        lg_filter = f1.radio("리그 범위", ["ALL", "KBL", "WKBL"], horizontal=True, format_func=lambda x: "전체 (KBL+WKBL)" if x == "ALL" else x)
        mk_filter = f2.radio("마켓 필터", ["ALL", "승패", "핸디캡", "언오버"], horizontal=True, format_func=lambda x: "전체 마켓" if x == "ALL" else x)

        ev = build_eval_frame(log_df, league_filter=lg_filter, market_filter=mk_filter)
        if ev.empty:
            st.info("아직 채점 완료된 픽 기록이 없습니다. 픽을 등록해 두면 경기 종료 후 CSV 업데이트 시 자동으로 채점됩니다.")
        else:
            stats = compute_accuracy_summary(ev)

            st.subheader("1️⃣ 모델별 단독 적중률 (몬테 vs 제미나이 vs 라스)")
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.dataframe(stats["solo"], use_container_width=True, hide_index=True)
            with c2:
                fig_s = px.bar(
                    stats["solo"].dropna(subset=["적중률(%)"]),
                    x="구분", y="적중률(%)", text="적중률(%)", range_y=[0, 100],
                    title="소스별 단독 적중률(%)",
                )
                fig_s.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig_s, use_container_width=True)

            st.subheader("2️⃣ 방향 일치(합의) 적중률 & 3️⃣ 모델 간 충돌 시 승률 비교")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**방향 일치(합의) 적중률**")
                st.dataframe(stats["consensus"], use_container_width=True, hide_index=True)
            with cc2:
                st.markdown("**모델 간 의견 충돌 시 승률**")
                st.dataframe(stats["conflict"], use_container_width=True, hide_index=True)

            st.subheader("4️⃣ 확률 구간별(몬테) & 등급별(제미나이) 정밀 적중률")
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**몬테카를로 확률 구간별 (50 / 60 / 70%+)**")
                st.dataframe(stats["monte_bins"], use_container_width=True, hide_index=True)
            with g2:
                st.markdown("**제미나이 5단계 등급별 (S ~ D)**")
                st.dataframe(stats["gemini_grades"], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📋 전체 픽 기록 (직접 수정·수동 채점 가능)")
        if log_df.empty:
            st.caption("등록된 기록이 없습니다.")
        else:
            edited = st.data_editor(
                log_df.iloc[::-1].reset_index(drop=True),
                use_container_width=True,
                num_rows="dynamic",
                key="pred_editor",
            )
            if st.button("💾 수정사항 저장"):
                save_df = edited.iloc[::-1].reset_index(drop=True)
                has_score = (save_df["actual_home_pts"].astype(str).str.strip() != "") & (save_df["actual_away_pts"].astype(str).str.strip() != "")
                save_df.loc[has_score, "status"] = "graded"
                _, msg = save_predictions(save_df, LOG_PATH, gh_token, gh_repo, gh_branch)
                st.success(f"수정 완료 ({msg})")
                st.rerun()
