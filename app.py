import json
import os
from datetime import datetime, timedelta
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
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def _get_github_secrets() -> tuple[str | None, str | None, str]:
    try:
        token = st.secrets.get("GITHUB_TOKEN", None)
        repo = st.secrets.get("GITHUB_REPO", None)
        branch = st.secrets.get("GITHUB_BRANCH", "main")
        return token, repo, branch
    except Exception:
        return None, None, "main"


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

df = load_games(league)
if df.empty:
    st.warning(f"{league} 데이터(`data/{league.lower()}_games.csv`)가 아직 없습니다. GitHub Actions를 1회 실행해 주세요.")
    st.stop()

ratings = build_team_ratings(df, ewma_span=ewma_span, recent_weight=recent_weight)
sched = load_schedule(league)

st.title(f"🏀 {league} 2026-27 시즌 분석 & 시뮬레이터")
st.caption(
    f"누적 반영 경기: {df['game_id'].nunique()}경기 | "
    f"시즌 평균 {int(round((1-recent_weight)*100))}% + 최근 {ewma_span}경기 EWMA {int(round(recent_weight*100))}% | "
    f"HCA: {'2.5점' if league == 'KBL' else '0.0점'}"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 팀 효율 & 랭킹",
    "🎯 매치업 시뮬레이터",
    "📅 예정 경기 자동 예측",
    "📝 픽 기록 & 적중률 검증",
])

# ---------- TAB 1: 팀 효율 ----------
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

# ---------- TAB 2: 매치업 시뮬레이터 ----------
with tab2:
    teams = sorted(ratings["team"].unique())
    c1, c2 = st.columns(2)
    home = c1.selectbox("홈 팀", teams, index=0, key=f"sim_h_{league}")
    away = c2.selectbox("원정 팀", teams, index=1 if len(teams) > 1 else 0, key=f"sim_a_{league}")

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

        st.subheader("기준점 돌파 확률 계산 (10,000회 시뮬레이션)")
        b1, b2 = st.columns(2)
        with b1:
            h_line = st.number_input("홈 팀 핸디캡 (예: -3.5 또는 +4.5)", value=0.0, step=0.5, key=f"hl_{league}")
            cover_prob = float(((res["margins"] + h_line) > 0).mean()) * 100
            st.write(f"**{home} ({h_line:+.1f}) 확률:** `{cover_prob:.1f}%`  |  **{away} ({-h_line:+.1f}):** `{100-cover_prob:.1f}%`")
        with b2:
            t_line = st.number_input("언더/오버 기준점", value=round(res["exp_total"] * 2) / 2, step=0.5, key=f"tl_{league}")
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

# ---------- TAB 3: 예정 경기 자동 예측 (KBL · WKBL 공용) ----------
with tab3:
    if sched.empty:
        st.info(f"{league} 예정 경기 파일(`data/{league.lower()}_schedule.csv`)이 없습니다. GitHub Actions를 1회 실행하면 생성됩니다.")
    else:
        preds = []
        valid_teams = set(ratings["team"])
        for _, row in sched.head(20).iterrows():
            if row["home"] in valid_teams and row["away"] in valid_teams:
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
        if preds:
            st.dataframe(pd.DataFrame(preds), use_container_width=True, hide_index=True)
        else:
            st.info("매칭 가능한 예정 경기가 없습니다.")

# ---------- TAB 4: 픽 기록 & 몬테/제미/라스 적중률 검증 ----------
with tab4:
    kbl_all = load_games("KBL")
    wkbl_all = load_games("WKBL")
    gh_token, gh_repo, gh_branch = _get_github_secrets()

    # 로드 + 종료 경기 자동 채점
    log_df = load_predictions(LOG_PATH)
    log_df, graded_now = auto_grade_predictions(log_df, kbl_all, wkbl_all)
    if graded_now > 0:
        ok, msg = save_predictions(log_df, LOG_PATH, gh_token, gh_repo, gh_branch)
        st.success(f"종료된 {graded_now}경기 결과를 자동 채점했습니다! ({msg})")

    sub_input, sub_json, sub_stats = st.tabs(["✍️ 경기별 픽 등록", "📋 제미나이 JSON 일괄 등록", "📈 적중률 · 합의 · 충돌 대시보드"])

    # --- 4-1. 개별 경기 등록 ---
    with sub_input:
        teams_lg = sorted(ratings["team"].unique())
        today_kst = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")

        # 예정 경기 빠른 선택
        sched_opts = ["직접 선택"]
        sched_rows = []
        if not sched.empty:
            for _, sr in sched.head(15).iterrows():
                if sr["home"] in teams_lg and sr["away"] in teams_lg:
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
            in_hline = l1.number_input("홈 기준 핸디캡 (예: -3.5)", value=round(-sim_in["exp_margin"] * 2) / 2, step=0.5, key=f"trk_hl_{league}")
            in_oline = l2.number_input("언오버 기준점", value=round(sim_in["exp_total"] * 2) / 2, step=0.5, key=f"trk_ol_{league}")

            # 몬테카를로 자동 산출
            m_ml_p = sim_in["home_win_prob"] * 100
            m_ml_pick = "홈" if m_ml_p >= 50 else "원정"
            m_ml_conf = max(m_ml_p, 100 - m_ml_p)

            m_hd_p = float(((sim_in["margins"] + in_hline) > 0).mean()) * 100
            m_hd_pick = "홈" if m_hd_p >= 50 else "원정"
            m_hd_conf = max(m_hd_p, 100 - m_hd_p)

            m_ou_p = float((sim_in["totals"] > in_oline).mean()) * 100
            m_ou_pick = "오버" if m_ou_p >= 50 else "언더"
            m_ou_conf = max(m_ou_p, 100 - m_ou_p)

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
                submitted = st.form_submit_button("💾 픽 저장하기", type="primary")

                if submitted:
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

    # --- 4-2. 제미나이 JSON 일괄 등록 ---
    with sub_json:
        st.caption("제미나이에게 경기 분석을 받은 뒤 JSON 배열을 붙여넣으면 몬테카를로 예측까지 자동 결합해 일괄 등록합니다.")
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
    "memo": "LG 홈 수비 우위"
  }
]"""
        with st.expander("JSON 입력 형식 예시 보기"):
            st.code(sample_json, language="json")

        raw_json = st.text_area("JSON 붙여넣기", height=220, placeholder=sample_json)
        if st.button("📥 JSON 일괄 등록", type="primary"):
            try:
                items = json.loads(raw_json)
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

    # --- 4-3. 적중률 · 방향 일치 · 충돌 대시보드 ---
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
                # 점수 수동 입력 시 status 자동 graded 처리
                has_score = (save_df["actual_home_pts"].astype(str).str.strip() != "") & (save_df["actual_away_pts"].astype(str).str.strip() != "")
                save_df.loc[has_score, "status"] = "graded"
                _, msg = save_predictions(save_df, LOG_PATH, gh_token, gh_repo, gh_branch)
                st.success(f"수정 완료 ({msg})")
                st.rerun()
