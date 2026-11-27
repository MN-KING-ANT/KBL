import io
import os
import re
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
from bs4 import BeautifulSoup

WKBL_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.wkbl.or.kr/game/result.asp",
}
WKBL_AJAX_HEADERS = {**WKBL_HEADERS, "X-Requested-With": "XMLHttpRequest"}

SEASON_GU = "047"   # 2026-27 시즌
GAME_TYPE = "01"    # 정규리그
MAX_GAMES = 95

WKBL_TEAM_ALIASES = [
    ("삼성생명", "삼성생명"),
    ("신한은행", "신한은행"),
    ("우리은행", "우리은행"),
    ("하나은행", "하나은행"),
    ("KB스타즈", "KB스타즈"),
    ("BNK 썸", "BNK 썸"),
    ("BNK썸", "BNK 썸"),
]


def _clean_team(raw: str) -> str:
    txt = re.sub(r"\s*\[.*\]", "", str(raw)).strip()
    if txt == "BNK썸":
        return "BNK 썸"
    return txt


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


def fetch_wkbl_game(game_num: int, season_gu: str = SEASON_GU) -> list[dict] | None:
    gno = f"{game_num:03d}"
    params = {"season_gu": season_gu, "game_type": GAME_TYPE, "game_no": gno}

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

    m_date = re.search(r"(202\d)[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})", r1.text)
    gdate = f"{m_date.group(1)}-{int(m_date.group(2)):02d}-{int(m_date.group(3)):02d}" if m_date else f"G{gno}"

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

    if stat0["pts"] == t1_tot and stat1["pts"] == t0_tot and t0_tot != t1_tot:
        stat0, stat1 = stat1, stat0

    def q_vals(idx):
        row = qdf.iloc[idx]
        return [int(pd.to_numeric(row.get(f"{i}Q", 0), errors="coerce") or 0) for i in range(1, 5)]

    q0, q1 = q_vals(0), q_vals(1)
    game_id = f"WKBL_{season_gu}_{gno}"

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


def _extract_two_teams(text: str) -> tuple[str, str] | None:
    hits = []
    for raw, canonical in WKBL_TEAM_ALIASES:
        for m in re.finditer(re.escape(raw), text):
            hits.append((m.start(), canonical))
    hits.sort(key=lambda x: x[0])
    ordered = []
    for _, name in hits:
        if not ordered or ordered[-1] != name:
            ordered.append(name)
    if len(ordered) >= 2:
        return ordered[0], ordered[1]
    return None


def fetch_wkbl_schedule_and_dates(
    season_gu: str = SEASON_GU,
    months: tuple[str, ...] = ("202610", "202611", "202612", "202701", "202702", "202703"),
) -> tuple[list[dict], dict[tuple[str, str, int, int], str]]:
    """WKBL 리스트형 일정표(viewType=1)에서 예정 경기와 종료 경기 날짜를 함께 추출"""
    today_str = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")
    upcoming = []
    finished_date_map: dict[tuple[str, str, int, int], str] = {}
    seen_upcoming = set()

    for ym in months:
        year = int(ym[:4])
        url = "https://www.wkbl.or.kr/game/sch/schedule.asp"
        params = {
            "gun": "1",
            "season_gu": season_gu,
            "game_type": GAME_TYPE,
            "ym": ym,
            "viewType": "1",
        }
        try:
            res = requests.get(url, headers=WKBL_HEADERS, params=params, timeout=15)
            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.text, "lxml")
        except Exception as e:
            print(f"[WKBL 일정 {ym}] 조회 실패: {e}")
            continue

        last_date = None
        for tr in soup.select("table tr"):
            tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(tds) < 2:
                continue

            row_txt = " ".join(tds)
            teams = _extract_two_teams(row_txt)
            if not teams:
                continue
            home, away = teams

            # 날짜 추출 (예: 11/27(금) 또는 2026.11.27)
            m_md = re.search(r"(\d{1,2})[/.\-](\d{1,2})", tds[0])
            if m_md:
                gdate = f"{year}-{int(m_md.group(1)):02d}-{int(m_md.group(2)):02d}"
                last_date = gdate
            else:
                gdate = last_date
            if not gdate:
                continue

            # 시간 & 경기장 추출
            m_time = re.search(r"\b(\d{2}:\d{2})\b", row_txt)
            gtime = m_time.group(1) if m_time else ""
            stadium = next((c for c in tds if any(k in c for k in ["체육관", "아레나", "실내", "스포원", "도원", "이순신"])), "")

            # 대진 셀에서 점수 유무 판별 (종료 경기 vs 예정 경기)
            match_cell = next((c for c in tds if _extract_two_teams(c)), row_txt)
            scores = [int(n) for n in re.findall(r"\b(\d{2,3})\b", match_cell) if 25 <= int(n) <= 150]

            if len(scores) >= 2:
                finished_date_map[(home, away, scores[0], scores[1])] = gdate
            elif gdate >= today_str:
                key = (gdate, home, away)
                if key not in seen_upcoming:
                    seen_upcoming.add(key)
                    upcoming.append({
                        "game_id": f"WKBL_{season_gu}_{gdate.replace('-', '')}_{home}_{away}",
                        "game_date": gdate,
                        "game_time": gtime,
                        "home": home,
                        "away": away,
                        "stadium": stadium,
                    })
        time.sleep(0.25)

    return upcoming, finished_date_map


def update_wkbl_data(
    games_path: str = "data/wkbl_games.csv",
    sched_path: str = "data/wkbl_schedule.csv",
    season_gu: str = SEASON_GU,
):
    os.makedirs(os.path.dirname(games_path), exist_ok=True)
    existing_df = pd.read_csv(games_path) if os.path.exists(games_path) else pd.DataFrame()
    existing_ids = set(existing_df["game_id"].astype(str)) if not existing_df.empty else set()

    new_rows = []
    empty_streak = 0

    for n in range(1, MAX_GAMES + 1):
        gid = f"WKBL_{season_gu}_{n:03d}"
        if gid in existing_ids:
            empty_streak = 0
            continue

        try:
            rows = fetch_wkbl_game(n, season_gu=season_gu)
        except Exception as e:
            print(f"[WKBL] {gid} 수집 실패: {e}")
            rows = None

        if rows:
            new_rows.extend(rows)
            empty_streak = 0
        else:
            empty_streak += 1
            if empty_streak >= 3:
                break

    updated_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else existing_df.copy()

    # 예정 경기 일정 수집 & 종료 경기 실제 날짜 보정
    upcoming, finished_date_map = fetch_wkbl_schedule_and_dates(season_gu=season_gu)

    if not updated_df.empty:
        # G001 같은 임시 날짜를 일정표의 실제 날짜(YYYY-MM-DD)로 채움
        if finished_date_map:
            for idx, r in updated_df.iterrows():
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(r.get("game_date", ""))):
                    if int(r["is_home"]) == 1:
                        k = (_clean_team(r["team"]), _clean_team(r["opp"]), int(r["pts"]), int(r["opp_pts"]))
                    else:
                        k = (_clean_team(r["opp"]), _clean_team(r["team"]), int(r["opp_pts"]), int(r["pts"]))
                    if k in finished_date_map:
                        updated_df.at[idx, "game_date"] = finished_date_map[k]

        updated_df = updated_df.drop_duplicates(subset=["game_id", "team"]).sort_values(["game_no", "is_home"])
        updated_df.to_csv(games_path, index=False, encoding="utf-8-sig")
        print(f"[WKBL] 신규 {len(new_rows)//2}경기 추가 (총 {len(updated_df)//2}경기)")
    else:
        print("[WKBL] 수집된 종료 경기가 없습니다.")

    sched_df = pd.DataFrame(upcoming, columns=["game_id", "game_date", "game_time", "home", "away", "stadium"])
    if not sched_df.empty:
        sched_df = sched_df.sort_values(["game_date", "game_time"])
    sched_df.to_csv(sched_path, index=False, encoding="utf-8-sig")
    print(f"[WKBL] 예정 경기 {len(sched_df)}건 저장 완료 -> {sched_path}")
