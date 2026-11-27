import numpy as np
import pandas as pd


def add_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["poss"] = (df["fga"] - df["orb"] + df["tov"] + 0.44 * df["fta"]).clip(lower=1)
    df["ortg"] = (df["pts"] / df["poss"]) * 100
    df["drtg"] = (df["opp_pts"] / df["poss"]) * 100
    df["net_rtg"] = df["ortg"] - df["drtg"]
    df["efg"] = ((df["fgm"] + 0.5 * df["fg3m"]) / df["fga"].clip(lower=1)) * 100
    df["tov_pct"] = (df["tov"] / df["poss"]) * 100
    df["ftr"] = (df["fta"] / df["fga"].clip(lower=1)) * 100
    df["win"] = (df["pts"] > df["opp_pts"]).astype(int)

    # KBL 1·4Q(외인 1명) vs 2·3Q(외인 2명) 구간 지표
    df["q14_pts"] = df["q1"] + df["q4"]
    df["q14_opp"] = df["opp_q1"] + df["opp_q4"]
    df["q14_diff"] = df["q14_pts"] - df["q14_opp"]
    df["q23_pts"] = df["q2"] + df["q3"]
    df["q23_opp"] = df["opp_q2"] + df["opp_q3"]
    df["q23_diff"] = df["q23_pts"] - df["q23_opp"]
    return df


def build_team_ratings(df: pd.DataFrame, ewma_span: int = 5, recent_weight: float = 0.4) -> pd.DataFrame:
    df = df.sort_values(["game_no", "game_date"])
    sw = 1.0 - recent_weight
    rows = []

    for team, g in df.groupby("team"):
        def blend(col):
            s_mean = g[col].mean()
            r_mean = g[col].ewm(span=ewma_span, adjust=False).mean().iloc[-1]
            return sw * s_mean + recent_weight * r_mean

        def safe_std(col, default_val):
            val = g[col].std(ddof=1)
            return float(default_val if pd.isna(val) or val < 4.0 else val)

        ortg = blend("ortg")
        drtg = blend("drtg")
        rows.append({
            "team": team,
            "games": len(g),
            "wins": int(g["win"].sum()),
            "losses": int(len(g) - g["win"].sum()),
            "win_pct": g["win"].mean() * 100,
            "pace": blend("poss"),
            "ortg": ortg,
            "drtg": drtg,
            "net_rtg": ortg - drtg,
            "efg": g["efg"].mean(),
            "tov_pct": g["tov_pct"].mean(),
            "ftr": g["ftr"].mean(),
            "pts_avg": g["pts"].mean(),
            "opp_pts_avg": g["opp_pts"].mean(),
            "pts_std": safe_std("pts", 9.5),
            "q14_pts": blend("q14_pts"),
            "q14_opp": blend("q14_opp"),
            "q14_diff": blend("q14_diff"),
            "q14_std": safe_std("q14_pts", 6.5),
            "q23_pts": blend("q23_pts"),
            "q23_opp": blend("q23_opp"),
            "q23_diff": blend("q23_diff"),
            "q23_std": safe_std("q23_pts", 6.5),
        })

    return pd.DataFrame(rows).sort_values("net_rtg", ascending=False).reset_index(drop=True)
