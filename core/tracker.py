import base64
import json
import os
import requests
import numpy as np
import pandas as pd

LOG_PATH = "data/predictions_log.csv"

LOG_COLUMNS = [
    "record_id", "league", "game_date", "home", "away",
    "handi_line", "ou_line",
    "monte_home_pts", "monte_away_pts",
    "monte_ml_pick", "monte_ml_prob",
    "monte_handi_pick", "monte_handi_prob",
    "monte_ou_pick", "monte_ou_prob",
    "las_ml_pick", "las_handi_pick", "las_ou_pick",
    "gemini_ml_pick", "gemini_handi_pick", "gemini_ou_pick",
    "gemini_ml_grade", "gemini_handi_grade", "gemini_ou_grade",
    "memo",
    "actual_home_pts", "actual_away_pts", "status",
]


def load_predictions(path: str = LOG_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(path, dtype=str)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[LOG_COLUMNS].fillna("")


def save_predictions(
    df: pd.DataFrame,
    path: str = LOG_PATH,
    github_token: str | None = None,
    github_repo: str | None = None,
    github_branch: str = "main",
) -> tuple[bool, str]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df[LOG_COLUMNS].copy()
    out.to_csv(path, index=False, encoding="utf-8-sig")

    if not github_token or not github_repo:
        return True, "로컬 저장 완료 (GitHub 영구 백업은 Streamlit Secrets에 GITHUB_TOKEN, GITHUB_REPO 설정 시 활성화)"

    api_url = f"https://api.github.com/repos/{github_repo}/contents/{path}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    try:
        sha = None
        r_get = requests.get(api_url, headers=headers, params={"ref": github_branch}, timeout=15)
        if r_get.status_code == 200:
            sha = r_get.json().get("sha")

        content_b64 = base64.b64encode(out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")).decode("utf-8")
        payload = {
            "message": "chore: update basket predictions log",
            "content": content_b64,
            "branch": github_branch,
        }
        if sha:
            payload["sha"] = sha

        r_put = requests.put(api_url, headers=headers, json=payload, timeout=15)
        if r_put.status_code in (200, 201):
            return True, "GitHub 영구 백업 완료"
        return False, f"로컬 저장됨 · GitHub 백업 실패 ({r_put.status_code})"
    except Exception as e:
        return False, f"로컬 저장됨 · GitHub 연결 오류 ({e})"


def normalize_pick(val: str, market: str, home: str = "", away: str = "") -> str:
    s = str(val or "").strip()
    if not s or s in ("-", "패스", "없음", "None", "nan"):
        return ""
    u = s.upper()
    if market in ("ml", "handi"):
        if s in ("홈", "홈승", "홈핸디", "홈마핸", "홈플핸") or u in ("H", "HOME") or (home and home in s):
            return "홈"
        if s in ("원정", "원정승", "원정핸디", "원정마핸", "원정플핸") or u in ("A", "AWAY") or (away and away in s):
            return "원정"
    elif market == "ou":
        if "오버" in s or u in ("O", "OVER"):
            return "오버"
        if "언더" in s or u in ("U", "UNDER"):
            return "언더"
    return ""


def auto_grade_predictions(
    log_df: pd.DataFrame,
    kbl_games: pd.DataFrame,
    wkbl_games: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """종료된 경기 CSV와 매칭해 실제 점수와 status='graded'를 자동 반영"""
    if log_df.empty:
        return log_df, 0

    df = log_df.copy()
    games_map = {
        "KBL": kbl_games[kbl_games["is_home"] == 1].copy() if not kbl_games.empty else pd.DataFrame(),
        "WKBL": wkbl_games[wkbl_games["is_home"] == 1].copy() if not wkbl_games.empty else pd.DataFrame(),
    }

    updated_cnt = 0
    for idx, row in df.iterrows():
        if row["status"] == "graded" and str(row["actual_home_pts"]).strip() != "":
            continue

        lg = row["league"]
        gdf = games_map.get(lg, pd.DataFrame())
        if gdf.empty:
            continue

        # 1순위: 날짜 + 홈 + 원정 정확 일치
        m = gdf[
            (gdf["game_date"].astype(str) == str(row["game_date"]))
            & (gdf["team"].astype(str) == str(row["home"]))
            & (gdf["opp"].astype(str) == str(row["away"]))
        ]
        # 2순위: WKBL 날짜 누락 대비 (홈 + 원정 최신 종료 경기)
        if m.empty and lg == "WKBL":
            m2 = gdf[(gdf["team"].astype(str) == str(row["home"])) & (gdf["opp"].astype(str) == str(row["away"]))]
            if not m2.empty:
                m = m2.sort_values("game_no").tail(1)

        if not m.empty:
            g = m.iloc[0]
            df.at[idx, "actual_home_pts"] = str(int(g["pts"]))
            df.at[idx, "actual_away_pts"] = str(int(g["opp_pts"]))
            df.at[idx, "status"] = "graded"
            updated_cnt += 1

    return df, updated_cnt


def _to_float(val, default=np.nan) -> float:
    try:
        s = str(val).strip()
        return float(s) if s != "" else default
    except Exception:
        return default


def build_eval_frame(log_df: pd.DataFrame, league_filter: str = "ALL", market_filter: str = "ALL") -> pd.DataFrame:
    """채점 완료된 기록을 마켓 단위 평가 테이블로 변환"""
    if log_df.empty:
        return pd.DataFrame()

    df = log_df.copy()
    if league_filter != "ALL":
        df = df[df["league"] == league_filter]

    rows = []
    for _, r in df.iterrows():
        ah = _to_float(r["actual_home_pts"])
        aa = _to_float(r["actual_away_pts"])
        if np.isnan(ah) or np.isnan(aa):
            continue

        margin = ah - aa
        total = ah + aa

        # 1) 승패 (ML)
        ml_actual = "홈" if margin > 0 else ("원정" if margin < 0 else "")
        if ml_actual:
            rows.append({
                "league": r["league"], "game_date": r["game_date"],
                "matchup": f"{r['home']} vs {r['away']}", "market": "승패",
                "actual": ml_actual,
                "monte_pick": normalize_pick(r["monte_ml_pick"], "ml", r["home"], r["away"]),
                "monte_prob": _to_float(r["monte_ml_prob"]),
                "las_pick": normalize_pick(r["las_ml_pick"], "ml", r["home"], r["away"]),
                "gemini_pick": normalize_pick(r["gemini_ml_pick"], "ml", r["home"], r["away"]),
                "gemini_grade": str(r["gemini_ml_grade"]).strip().upper(),
            })

        # 2) 핸디캡 (홈 기준)
        hline = _to_float(r["handi_line"])
        if not np.isnan(hline):
            adj = margin + hline
            if adj != 0:  # 적특 제외
                rows.append({
                    "league": r["league"], "game_date": r["game_date"],
                    "matchup": f"{r['home']} vs {r['away']}", "market": "핸디캡",
                    "actual": "홈" if adj > 0 else "원정",
                    "monte_pick": normalize_pick(r["monte_handi_pick"], "handi", r["home"], r["away"]),
                    "monte_prob": _to_float(r["monte_handi_prob"]),
                    "las_pick": normalize_pick(r["las_handi_pick"], "handi", r["home"], r["away"]),
                    "gemini_pick": normalize_pick(r["gemini_handi_pick"], "handi", r["home"], r["away"]),
                    "gemini_grade": str(r["gemini_handi_grade"]).strip().upper(),
                })

        # 3) 언오버
        oline = _to_float(r["ou_line"])
        if not np.isnan(oline) and oline > 0 and total != oline:
            rows.append({
                "league": r["league"], "game_date": r["game_date"],
                "matchup": f"{r['home']} vs {r['away']}", "market": "언오버",
                "actual": "오버" if total > oline else "언더",
                "monte_pick": normalize_pick(r["monte_ou_pick"], "ou"),
                "monte_prob": _to_float(r["monte_ou_prob"]),
                "las_pick": normalize_pick(r["las_ou_pick"], "ou"),
                "gemini_pick": normalize_pick(r["gemini_ou_pick"], "ou"),
                "gemini_grade": str(r["gemini_ou_grade"]).strip().upper(),
            })

    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    if market_filter != "ALL":
        ev = ev[ev["market"] == market_filter]
    return ev.reset_index(drop=True)


def _acc_row(label: str, sub: pd.DataFrame, pick_col: str) -> dict:
    valid = sub[sub[pick_col] != ""]
    n = len(valid)
    hits = int((valid[pick_col] == valid["actual"]).sum()) if n else 0
    return {
        "구분": label,
        "적중": hits,
        "전체": n,
        "적중률(%)": round(hits / n * 100, 1) if n else np.nan,
    }


def compute_accuracy_summary(ev: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """단독 · 방향 일치 · 모델 충돌 · 몬테 확률구간 · 제미 등급별 집계"""
    if ev.empty:
        empty = pd.DataFrame()
        return {"solo": empty, "consensus": empty, "conflict": empty, "monte_bins": empty, "gemini_grades": empty}

    # 1. 단독 적중률
    solo = pd.DataFrame([
        _acc_row("몬테카를로", ev, "monte_pick"),
        _acc_row("제미나이", ev, "gemini_pick"),
        _acc_row("라스(시장)", ev, "las_pick"),
    ])

    # 2. 방향 일치(합의) 적중률
    both_mg = ev[(ev["monte_pick"] != "") & (ev["gemini_pick"] != "") & (ev["monte_pick"] == ev["gemini_pick"])]
    both_ml = ev[(ev["monte_pick"] != "") & (ev["las_pick"] != "") & (ev["monte_pick"] == ev["las_pick"])]
    both_gl = ev[(ev["gemini_pick"] != "") & (ev["las_pick"] != "") & (ev["gemini_pick"] == ev["las_pick"])]
    all_3 = ev[
        (ev["monte_pick"] != "")
        & (ev["monte_pick"] == ev["gemini_pick"])
        & (ev["monte_pick"] == ev["las_pick"])
    ]
    consensus = pd.DataFrame([
        _acc_row("🔥 3자 전원 일치 (몬테=제미=라스)", all_3, "monte_pick"),
        _acc_row("🤝 몬테 + 제미 일치", both_mg, "monte_pick"),
        _acc_row("🤝 몬테 + 라스 일치", both_ml, "monte_pick"),
        _acc_row("🤝 제미 + 라스 일치", both_gl, "gemini_pick"),
    ])

    # 3. 모델 간 충돌 시 승률 비교
    conflict_rows = []
    pairs = [
        ("몬테 vs 제미 충돌", "monte_pick", "gemini_pick", "몬테 승률(%)", "제미 승률(%)"),
        ("몬테 vs 라스 충돌", "monte_pick", "las_pick", "몬테 승률(%)", "라스 승률(%)"),
        ("제미 vs 라스 충돌", "gemini_pick", "las_pick", "제미 승률(%)", "라스 승률(%)"),
    ]
    for title, c1, c2, l1, l2 in pairs:
        sub = ev[(ev[c1] != "") & (ev[c2] != "") & (ev[c1] != ev[c2])]
        n = len(sub)
        w1 = int((sub[c1] == sub["actual"]).sum()) if n else 0
        w2 = int((sub[c2] == sub["actual"]).sum()) if n else 0
        conflict_rows.append({
            "충돌 조합": title,
            "충돌 건수": n,
            "A 승(좌측)": f"{w1}승 ({round(w1/n*100, 1)}%)" if n else "-",
            "B 승(우측)": f"{w2}승 ({round(w2/n*100, 1)}%)" if n else "-",
        })
    conflict = pd.DataFrame(conflict_rows)

    # 4. 몬테카를로 확률 구간별 (50~60 / 60~70 / 70%+)
    mv = ev[(ev["monte_pick"] != "") & (~ev["monte_prob"].isna())].copy()
    bin_rows = []
    for label, lo, hi in [("50% ~ 60%", 50.0, 60.0), ("60% ~ 70%", 60.0, 70.0), ("70% 이상", 70.0, 101.0)]:
        sub = mv[(mv["monte_prob"] >= lo) & (mv["monte_prob"] < hi)]
        bin_rows.append(_acc_row(label, sub, "monte_pick"))
    monte_bins = pd.DataFrame(bin_rows)

    # 5. 제미나이 5단계 등급별 (S / A / B / C / D)
    grade_rows = []
    for g in ["S", "A", "B", "C", "D"]:
        sub = ev[(ev["gemini_pick"] != "") & (ev["gemini_grade"] == g)]
        grade_rows.append(_acc_row(f"{g} 등급", sub, "gemini_pick"))
    gemini_grades = pd.DataFrame(grade_rows)

    return {
        "solo": solo,
        "consensus": consensus,
        "conflict": conflict,
        "monte_bins": monte_bins,
        "gemini_grades": gemini_grades,
    }
