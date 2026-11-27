import os
import time
import calendar
from datetime import datetime, timedelta
import requests
import pandas as pd

KBL_HEADERS = {
    "Channel": "WEB",
    "TeamCode": "XX",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.kbl.or.kr/",
}

SEASON_CODE = 49       # 2026-27 시즌
GLKEY_PREFIX = "S49G01"


def _month_ranges(start_ym: str = "202610", end_ym: str = "202704"):
    y, m = int(start_ym[:4]), int(start_ym[4:])
    ey, em = int(end_ym[:4]), int(end_ym[4:])
    ranges = []
    while (y, m) <= (ey, em):
        last_day = calendar.monthrange(y, m)[1]
        ranges.append((f"{y}{m:02d}01", f"{y}{m:02d}{last_day:02d}"))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return ranges


def fetch_kbl_season_list() -> list[dict]:
    games = []
    seen = set()
    for fdate, tdate in _month_ranges():
        r = requests.get(
            "https://api.kbl.or.kr/match/list",
            headers=KBL_HEADERS,
            params={"fromDate": fdate, "toDate": tdate, "seasonGrade": 1},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        rows = data if isinstance(data, list) else data.get("data", [])
        for g in rows:
            gmkey = g.get("gmkey")
            if not gmkey or gmkey in seen:
                continue
            if g.get("seasonCode") != SEASON_CODE and not str(g.get("glkey", "")).startswith(GLKEY_PREFIX):
                continue
            seen.add(gmkey)
            games.append(g)
        time.sleep(0.2)
    return games


def fetch_kbl_game_detail(g: dict) -> list[dict]:
    gmkey = g["gmkey"]
    # 1) 쿼터별 점수 조회
    r_info = requests.get(f"https://api.kbl.or.kr/match/{gmkey}", headers=KBL_HEADERS, timeout=15)
    r_info.raise_for_status()
    info = r_info.json()
    q_home = info.get("teamrecords", {}).get("home", {})
    q_away = info.get("teamrecords", {}).get("away", {})

    # 2) 팀 박스스코어 조회
    r_box = requests.get(f"https://api.kbl.or.kr/match/{gmkey}/team-record", headers=KBL_HEADERS, timeout=15)
    r_box.raise_for_status()
    box_list = r_box.json()
    box_map = {str(item["tcode"]): item.get("records", {}) for item in box_list}

    tcode_h, tcode_a = str(g["tcodeH"]), str(g["tcodeA"])
    rec_h, rec_a = box_map.get(tcode_h), box_map.get(tcode_a)
    if not rec_h or not rec_a:
        return []

    gdate = str(g.get("gameDate", ""))
    if len(gdate) == 8:
        gdate = f"{gdate[:4]}-{gdate[4:6]}-{gdate[6:]}"

    def build_row(team, opp, is_home, rec, opp_rec, q, opp_q):
        return {
            "league": "KBL",
            "game_id": gmkey,
            "game_no": int(g.get("gameNo", 0)),
            "game_date": gdate,
            "team": team,
            "opp": opp,
            "is_home": is_home,
            "pts": int(rec.get("score", 0)),
            "opp_pts": int(opp_rec.get("score", 0)),
            "q1": int(q.get("scoreq1", 0)),
            "q2": int(q.get("scoreq2", 0)),
            "q3": int(q.get("scoreq3", 0)),
            "q4": int(q.get("scoreq4", 0)),
            "opp_q1": int(opp_q.get("scoreq1", 0)),
            "opp_q2": int(opp_q.get("scoreq2", 0)),
            "opp_q3": int(opp_q.get("scoreq3", 0)),
            "opp_q4": int(opp_q.get("scoreq4", 0)),
            "fgm": int(rec.get("fgt", 0)),       # 전체 야투 성공 (2P+3P)
            "fga": int(rec.get("fgtA", 0)),      # 전체 야투 시도
            "fg3m": int(rec.get("threep", 0)),
            "fg3a": int(rec.get("threepA", 0)),
            "ftm": int(rec.get("ft", 0)),
            "fta": int(rec.get("ftA", 0)),
            "orb": int(rec.get("offr", 0)),
            "drb": int(rec.get("defr", 0)),
            "ast": int(rec.get("ast", 0)),
            "tov": int(rec.get("to", 0)),
            "stl": int(rec.get("stl", 0)),
            "blk": int(rec.get("bs", 0)),
            "pf": int(rec.get("foul", 0)),
        }

    time.sleep(0.25)
    return [
        build_row(g["tnameH"], g["tnameA"], 1, rec_h, rec_a, q_home, q_away),
        build_row(g["tnameA"], g["tnameH"], 0, rec_a, rec_h, q_away, q_home),
    ]


def update_kbl_data(games_path: str = "data/kbl_games.csv", sched_path: str = "data/kbl_schedule.csv"):
    os.makedirs(os.path.dirname(games_path), exist_ok=True)
    existing_df = pd.read_csv(games_path) if os.path.exists(games_path) else pd.DataFrame()
    existing_ids = set(existing_df["game_id"].astype(str)) if not existing_df.empty else set()

    all_matches = fetch_kbl_season_list()
    new_rows = []
    upcoming_rows = []

    today_str = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y%m%d")

    for g in all_matches:
        gmkey = str(g["gmkey"])
        if g.get("isEnded") == 1:
            if gmkey not in existing_ids:
                try:
                    new_rows.extend(fetch_kbl_game_detail(g))
                except Exception as e:
                    print(f"[KBL] {gmkey} 수집 실패: {e}")
        else:
            gdate = str(g.get("gameDate", ""))
            if gdate >= today_str:
                upcoming_rows.append({
                    "game_id": gmkey,
                    "game_date": f"{gdate[:4]}-{gdate[4:6]}-{gdate[6:]}" if len(gdate) == 8 else gdate,
                    "game_time": str(g.get("gameStart", "")),
                    "home": g.get("tnameH", ""),
                    "away": g.get("tnameA", ""),
                    "stadium": g.get("stadiumname", ""),
                })

    if new_rows:
        updated_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        updated_df = updated_df.drop_duplicates(subset=["game_id", "team"]).sort_values(["game_date", "game_no", "is_home"])
        updated_df.to_csv(games_path, index=False, encoding="utf-8-sig")
        print(f"[KBL] 신규 {len(new_rows)//2}경기 추가 (총 {len(updated_df)//2}경기)")
    elif existing_df.empty:
        print("[KBL] 수집된 종료 경기가 없습니다.")
    else:
        print(f"[KBL] 최신 상태 유지 (총 {len(existing_df)//2}경기)")

    pd.DataFrame(upcoming_rows).sort_values(["game_date", "game_time"]).to_csv(
        sched_path, index=False, encoding="utf-8-sig"
    )
