import numpy as np
import pandas as pd


def simulate_matchup(
    ratings: pd.DataFrame,
    home_team: str,
    away_team: str,
    league: str = "KBL",
    use_quarter_split: bool = True,
    n_sims: int = 10000,
) -> dict:
    h = ratings.loc[ratings["team"] == home_team].iloc[0]
    a = ratings.loc[ratings["team"] == away_team].iloc[0]
    hca = 2.5 if league == "KBL" else 0.0

    rng = np.random.default_rng(42)

    if league == "KBL" and use_quarter_split:
        lg_q14 = ratings["q14_pts"].mean()
        lg_q23 = ratings["q23_pts"].mean()

        h_q14_mu = (h["q14_pts"] + a["q14_opp"] - lg_q14) + (hca / 4)
        a_q14_mu = (a["q14_pts"] + h["q14_opp"] - lg_q14) - (hca / 4)

        h_q23_mu = (h["q23_pts"] + a["q23_opp"] - lg_q23) + (hca / 4)
        a_q23_mu = (a["q23_pts"] + h["q23_opp"] - lg_q23) - (hca / 4)

        h_sim = rng.normal(h_q14_mu, h["q14_std"], n_sims) + rng.normal(h_q23_mu, h["q23_std"], n_sims)
        a_sim = rng.normal(a_q14_mu, a["q14_std"], n_sims) + rng.normal(a_q23_mu, a["q23_std"], n_sims)

        home_mu = h_q14_mu + h_q23_mu
        away_mu = a_q14_mu + a_q23_mu
        split_info = {
            "home_q14": float(h_q14_mu), "away_q14": float(a_q14_mu),
            "home_q23": float(h_q23_mu), "away_q23": float(a_q23_mu),
        }
    else:
        lg_ortg = ratings["ortg"].mean()
        lg_pace = ratings["pace"].mean()

        exp_pace = h["pace"] + a["pace"] - lg_pace
        home_mu = ((h["ortg"] + a["drtg"] - lg_ortg) * exp_pace / 100.0) + (hca / 2)
        away_mu = ((a["ortg"] + h["drtg"] - lg_ortg) * exp_pace / 100.0) - (hca / 2)

        h_sim = rng.normal(home_mu, h["pts_std"], n_sims)
        a_sim = rng.normal(away_mu, a["pts_std"], n_sims)
        split_info = None

    margins = h_sim - a_sim
    totals = h_sim + a_sim

    return {
        "home_win_prob": float((margins > 0).mean()),
        "home_exp_pts": float(home_mu),
        "away_exp_pts": float(away_mu),
        "exp_total": float(home_mu + away_mu),
        "exp_margin": float(home_mu - away_mu),
        "margins": margins,
        "totals": totals,
        "split_info": split_info,
    }
