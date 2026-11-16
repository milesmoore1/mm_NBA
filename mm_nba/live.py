"""Live recommendations: the same signals, model and threshold as the backtest, computed for games that have not started.

`signals_at` reads every input as of one moment `at`: the price path up to `at` and the sportsbook lines posted by `at`.
Called with at = tipoff - 60 minutes it reproduces the backtest's cutoff exactly (what scripts/collect.py stores), and
fetching a little after that moment changes nothing, because both sources are filtered to `at`.
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import lean, sources

CUTOFF_MINUTES = 60
PRICE_RANGE = (0.02, 0.98)          # the backtest table holds only games priced inside this range at the cutoff


@dataclass
class Model:
    dev: pd.DataFrame                 # the 2024-25 games the model is fitted on
    floor: float                      # bet when |estimate| >= floor cents

    @classmethod
    def load(cls):
        history = lean.load_games()
        _, floor = lean.backtest(history)
        return cls(history[history.split == "dev"], floor)

    def estimate(self, rows):
        return lean.estimate(self.dev, lean.add_signals(pd.DataFrame(rows)))


def build_row(game, points, spreads, at_ts):
    """One live row from a price path [(unix s, home price)] and the spreads read as of at_ts; (row, None) or (None, reason).
    The path checks are those of scripts/collect.py: 10+ points, the newest within an hour of `at`, price in range."""
    points = np.asarray(points, float).reshape(-1, 2)
    before = points[points[:, 0] <= at_ts]
    if len(before) < 10 or before[-1, 0] < at_ts - 3600:
        return None, "no price within an hour of the cutoff"
    if not np.isfinite(spreads["home_team_spread"]):
        return None, "no sportsbook spread before the cutoff"
    price = before[-1, 1]
    if not PRICE_RANGE[0] <= price <= PRICE_RANGE[1]:
        return None, f"price {price:.3f} outside {PRICE_RANGE[0]} to {PRICE_RANGE[1]}"
    return {"game_id": game["game_id"], "home_abbr": game["home_abbr"], "away_abbr": game["away_abbr"], "price_home": price,
            **lean.path_signals((before[:, 0] - at_ts) / 60.0, before[:, 1] * 100), **spreads}, None


def signals_at(game, market, at_ts):
    """Fetch the inputs for one game as of at_ts (unix seconds). market: (home token, away token, slug)."""
    points = sources.price_history(market[0], int(at_ts) - 24 * 3600, int(at_ts))
    minutes_before = (game["scheduled_start_utc"].timestamp() - at_ts) / 60
    spreads, _ = sources.game_spreads(game, minutes_before=minutes_before)
    time.sleep(sources.SPREAD_PAUSE_SECONDS)                  # TeamRankings asks for one page per 10 seconds
    return build_row(game, points, spreads, at_ts)


def decide(model, row):
    """The lean for one row: {estimate, side, bet}. side is "home" or "away"."""
    estimate = float(model.estimate([row])[0])
    return {"estimate": estimate, "side": "home" if estimate > 0 else "away", "bet": abs(estimate) >= model.floor}


def upcoming_games(now):
    """Today's (US Eastern) regular-season games that have not started, each with its Polymarket market or None."""
    today = str(now.tz_convert("America/New_York").date())
    games = sources.espn_games(today, today, finished=False)
    listing = sources.polymarket_moneylines(now - pd.Timedelta(days=1), now + pd.Timedelta(days=10), closed=False) if games else []
    return [(game, sources.home_market(game, listing)) for game in games]
