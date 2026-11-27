import io
import os
import re
import time
import requests
import pandas as pd

WKBL_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.wkbl.or.kr/game/result.asp",
}
WKBL_AJAX_HEADERS = {**WKBL_HEADERS, "X-Requested-With": "XMLHttpRequest"}

SEASON_GU = "047"   # 2026-27 시즌
GAME_TYPE = "01"    # 정규리그
MAX_GAMES = 95      # 6개 구단 6라운드 = 90경기


def _clean_team(raw: str) -> str:
    return re.sub(r"\s*\[.*\]", "", str(raw)).strip()


def _parse_ma(val) -> tuple[int, int]:
    parts = str(val).strip().split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return 0, 0


def _extract_team_total(df: pd.DataFrame) -> dict | None:
    df = df.copy()
    df.columns = [c[1] if isinstance(c, tuple) else str(c) for c in df.columns]
    if "선수" not in df.columns:
        return None
    mask = df["선수"].astype(str).str.contains("팀합계", na=False)
    row = df.loc[mask].iloc[-1] if mask.any() else df.iloc[-1]

    p2m, p2a = _parse_ma(row.get("2PM-A", "0-0"))
    p3m, p3a = _parse_ma(row.get("3PM-A", "0-0"))
    ftm, fta = _parse_ma(row.get("FTM-A", "0-0"))

    def to_int(col):
        try:
            return int(float(row.get(col, 0)))
        except Exception:
            return 0

    return {
        "pts": to_int("PTS"),
        "fgm": p2m + p3m,
        "fga": p2a + p3a,
        "fg3m": p3m,
        "fg3a": p3a,
        "ftm": ftm,
        "fta": fta,
        "orb": to_int("OFF"),
        "drb": to_int("DEF"),
        "ast": to_int("AST"),
        "tov": to_int("TO"),
        "stl": to_int("ST"),
        "blk": to_int("BS"),
        "pf": to_int("PF"),
    }


def fetch_wkbl_game(game_num: int) -> list[dict] | None:
    gno = f"{game_num:03d}"
    params = {"season_gu": SEASON_GU, "game_type": GAME_TYPE, "game_no": gno}

    # 1) 쿼터별 점수 & 팀명
    r1 = requests.get("https://www.wkbl.or.kr/game/result.asp", headers=WKBL_HEADERS, params=params, timeout=15)
    r1.encoding = r1.apparent_encoding
    try:
        t1 = pd.read_html(io.StringIO(r1.text))
    except ValueError:
        return None
    if not t1 or len(t1[0]) < 2 or "TOTAL" not in t1[0].columns:
        return None

    qdf = t1[0]
    t0_tot = int(pd.to_numeric(qdf.iloc[0]["TOTAL"], errors="coerce") or 0)
    t1_tot = int(pd.to_numeric(qdf.iloc[1]["TOTAL"], errors="coerce") or 0)
    if t0_tot == 0 and t1_tot == 0:
        return None

    team0 = _clean_team(qdf.iloc[0]["팀명"])
    team1 = _clean_team(qdf.iloc[1]["팀명"])

    # 날짜 추출 시도 (없으면 game_no 기준 순서 유지)
    m_date = re.search(r"(202\d)[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})", r1.text)
    gdate = f"{m_date.group(1)}-{int(m_date.group(2)):02d}-{int(m_date.group(3)):02d}" if m_date else f"G{gno}"

    # 2) 상세 팀합계 박스스코어 (AJAX tab 2)
    r2 = requests.post(
        "https://www.wkbl.or.kr/game/ajax/ajax_game_result_2.asp",
        headers=WKBL_AJAX_HEADERS,
        data=params,
        timeout=15,
    )
    r2.encoding = r2.apparent_encoding
    try:
        t2 = pd.read_html(io.StringIO(r2.text))
    except ValueError:
        return None
    if len(t2) < 3:
        return None

    stat0 = _extract_team_total(t2[1])
    stat1 = _extract_team_total(t2[2])
    if not stat0 or not stat1 or (stat0["pts"] == 0 and stat1["pts"] == 0):
        return None

    # 테이블 순서가 뒤바뀐 경우 점수로 자동 보정
    if stat0["pts"] == t1_tot and stat1["pts"] == t0_tot and t0_tot != t1_tot:
        stat0, stat1 = stat1, stat0

    def q_vals(idx):
        row = qdf.iloc[idx]
        return [int(pd.to_numeric(row.get(f"{i}Q", 0), errors="coerce") or 0) for i in range(1, 5)]

    q0, q1 = q_vals(0), q_vals(1)
    game_id = f"WKBL_{SEASON_GU}_{gno}"

    def build_row(team, opp, is_home, st, opp_st, q, opp_q):
        return {
            "league": "WKBL",
            "game_id": game_id,
            "game_no": game_num,
            "game_date": gdate,
            "team": team,
            "opp": opp,
            "is_home": is_home,
            "pts": st["pts"],
            "opp_pts": opp_st["pts"],
            "q1": q[0], "q2": q[1], "q3": q[2], "q4": q[3],
            "opp_q1": opp_q[0], "opp_q2": opp_q[1], "opp_q3": opp_q[2], "opp_q4": opp_q[3],
            "fgm": st["fgm"], "fga": st["fga"],
            "fg3m": st["fg3m"], "fg3a": st["fg3a"],
            "ftm": st["ftm"], "fta": st["fta"],
            "orb": st["orb"], "drb": st["drb"],
            "ast": st["ast"], "tov": st["tov"],
            "stl": st["stl"], "blk": st["blk"], "pf": st["pf"],
        }

    time.sleep(0.35)
    return [
        build_row(team0, team1, 1, stat0, stat1, q0, q1),
        build_row(team1, team0, 0, stat1, stat0, q1, q0),
    ]


def update_wkbl_data(games_path: str = "data/wkbl_games.csv"):
    os.makedirs(os.path.dirname(games_path), exist_ok=True)
    existing_df = pd.read_csv(games_path) if os.path.exists(games_path) else pd.DataFrame()
    existing_ids = set(existing_df["game_id"].astype(str)) if not existing_df.empty else set()

    new_rows = []
    empty_streak = 0

    for n in range(1, MAX_GAMES + 1):
        gid = f"WKBL_{SEASON_GU}_{n:03d}"
        if gid in existing_ids:
            empty_streak = 0
            continue

        try:
            rows = fetch_wkbl_game(n)
        except Exception as e:
            print(f"[WKBL] {gid} 수집 실패: {e}")
            rows = None

        if rows:
            new_rows.extend(rows)
            empty_streak = 0
        else:
            empty_streak += 1
            if empty_streak >= 3:  # 연속 3경기 미진행 시 종료
                break

    if new_rows:
        updated_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        updated_df = updated_df.drop_duplicates(subset=["game_id", "team"]).sort_values(["game_no", "is_home"])
        updated_df.to_csv(games_path, index=False, encoding="utf-8-sig")
        print(f"[WKBL] 신규 {len(new_rows)//2}경기 추가 (총 {len(updated_df)//2}경기)")
    elif existing_df.empty:
        print("[WKBL] 수집된 종료 경기가 없습니다.")
    else:
        print(f"[WKBL] 최신 상태 유지 (총 {len(existing_df)//2}경기)")
