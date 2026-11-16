"""Which side to bid on for games that have not started, using prices and lines as of now.

    python scripts/tonight.py

The strategy was tested with signals read 60 minutes before tipoff. For a game less than an hour away the signals are
read as of that cutoff, exactly as in the backtest and as scripts/trade.py does; for a later game they are read as of
now, so run this about an hour before each game. The model is fitted on the 2024-25 season. It places no orders.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mm_nba import lean, live

now = pd.Timestamp.now(tz="UTC")
rows = []
for game, market in live.upcoming_games(now):
    minutes_left = (game["scheduled_start_utc"] - now).total_seconds() / 60
    if market is None or minutes_left <= 0:
        continue
    at = min(now, game["scheduled_start_utc"] - pd.Timedelta(minutes=live.CUTOFF_MINUTES))
    row, _ = live.signals_at(game, market, at.timestamp())
    if row is not None:
        rows.append({**row, "game": f"{game['away_abbr']} at {game['home_abbr']}", "minutes to tipoff": round(minutes_left)})
if not rows:
    sys.exit("No upcoming game today has a Polymarket market, a price path and a sportsbook line.")

model = live.Model.load()
out = pd.DataFrame(rows)
out["estimate"] = model.estimate(rows)
home = out.estimate > 0
out["lean"] = np.where(home, out.home_abbr, out.away_abbr)
out["price of that side"] = np.where(home, out.price_home, 1 - out.price_home).round(3)
out["action"] = np.where(out.estimate.abs() >= model.floor, "BID", "skip")
out["expected move, cents"] = out.estimate.abs().round(2)
print(f"Bid when the expected move is at least {model.floor:.2f} cents. trade.py bids at the best bid of the lean side.\n")
print(out[["game", "minutes to tipoff", "lean", "price of that side", "expected move, cents", "action"]].to_string(index=False))
