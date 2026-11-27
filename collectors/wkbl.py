import io
import os
import re
import time
from datetime import datetime, timedelta, timezone
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

WKBL_STADIUMS = {
    "삼성생명": "용인실내체육관",
    "신한은행": "인천도원체육관",
    "우리은행": "아산이순신체육관",
    "하나은행": "부천체육관",
    "KB스타즈": "청주체육관",
    "BNK 썸": "부산사직실내체육관",
}

TEAM_PAT = r"(삼성생명|신한은행|우리은행|하나은행|KB스타즈|BNK\s*썸)"
UPCOMING_RE = re.compile(rf"{TEAM_PAT}\s*vs\s*{TEAM_PAT}", re.IGNORECASE)
SCORE_RE = re.compile(
    rf"{TEAM_PAT}\s*(\d{{2,3}})\s*(?:[:\-]|vs|\s)\s*(\d{{2,3}})\s*{TEAM_PAT}",
    re.IGNORECASE,
)


def _clean_team(raw: str) -> str:
    txt = re.sub(r"\s*\[.*\]", "", str(raw)).strip()
    txt = re.sub(r"BNK\s*썸", "BNK 썸", txt)
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

    # 1) 쿼터별 점수 및 팀명
    r1 = requests.get(
        "https://www.wkbl.or.kr/game/result.asp",
        headers=WKBL_HEADERS,
        params=params,
        timeout=15,
    )
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
    gdate = (
        f"{m_date.group(1)}-{int(m_date.group(2)):02d}-{int(m_date.group(3)):02d}"
        if m_date
        else f"G{gno}"
    )

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
        return [
            int(pd.to_numeric(row.get(f"{i}Q", 0), errors="coerce") or 0)
            for i in range(1, 5)
        ]

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


def fetch_wkbl_schedule_and_dates(
    season_gu: str = SEASON_GU,
    months: tuple[str, ...] = ("202610", "202611", "202612", "202701", "202702", "202703", "202704"),
) -> tuple[list[dict], dict[tuple[str, str, int, int], str]]:
    """WKBL 달력형 일정표(td 셀 단위)에서 예정 경기와 종료 경기 날짜를 추출"""
    kst = timezone(timedelta(hours=9))
    today_str = datetime.now(kst).strftime("%Y-%m-%d")
    upcoming = []
    finished_date_map: dict[tuple[str, str, int, int], str] = {}
    seen_upcoming = set()

    for ym in months:
        year, month = int(ym[:4]), int(ym[4:])
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

        for td in soup.select("table td"):
            cell_txt = td.get_text(" ", strip=True)
            m_day = re.match(r"^(\d{1,2})\b", cell_txt)
            if not m_day:
                continue

            day = int(m_day.group(1))
            if not (1 <= day <= 31):
                continue
            gdate = f"{year}-{month:02d}-{day:02d}"
            body = cell_txt[m_day.end():].strip()
            if not body:
                continue

            # 1) 점수가 표시된 종료 경기 -> 날짜 보정 맵에 저장
            for m in SCORE_RE.finditer(body):
                home = _clean_team(m.group(1))
                sh, sa = int(m.group(2)), int(m.group(3))
                away = _clean_team(m.group(4))
                finished_date_map[(home, away, sh, sa)] = gdate

            # 2) '홈 vs 원정' 형태의 예정 경기 -> 오늘 이후 일정 수집 (하루 2경기 대응)
            if gdate >= today_str:
                for m in UPCOMING_RE.finditer(body):
                    home = _clean_team(m.group(1))
                    away = _clean_team(m.group(2))
                    key = (gdate, home, away)
                    if key not in seen_upcoming:
                        seen_upcoming.add(key)
                        upcoming.append({
                            "game_id": f"WKBL_{season_gu}_{gdate.replace('-', '')}_{home}_{away}",
                            "game_date": gdate,
                            "game_time": "",
                            "home": home,
                            "away": away,
                            "stadium": WKBL_STADIUMS.get(home, ""),
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

    updated_df = (
        pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        if new_rows
        else existing_df.copy()
    )

    # 예정 경기 일정 수집 & 종료 경기 실제 날짜 보정
    upcoming, finished_date_map = fetch_wkbl_schedule_and_dates(season_gu=season_gu)

    if not updated_df.empty:
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
        sched_df = sched_df.sort_values(["game_date", "home"])
    sched_df.to_csv(sched_path, index=False, encoding="utf-8-sig")
    print(f"[WKBL] 예정 경기 {len(sched_df)}건 저장 완료 -> {sched_path}")
