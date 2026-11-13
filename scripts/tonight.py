"""Which side to bid on for games that have not started, using prices and lines as of now.

    python scripts/tonight.py

The strategy was tested with signals read 60 minutes before tipoff, so run this about an hour before each game; the
"minutes to tipoff" column shows how far off that a game is. The model is fitted on the 2024-25 season, exactly as in
the backtest. It prints one line per game; it places no orders.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mm_nba import lean, sources

now = pd.Timestamp.now(tz="UTC")
today = str(now.tz_convert("America/New_York").date())
games = sources.espn_games(today, today, finished=False)
listing = sources.polymarket_moneylines(now - pd.Timedelta(days=1), now + pd.Timedelta(days=10), closed=False) if games else []
rows = []
for game in games:
    minutes_left = (game["scheduled_start_utc"] - now).total_seconds() / 60
    market = sources.home_market(game, listing)
    if market is None or minutes_left <= 0:
        continue
    points = np.array(sources.price_history(market[0], int(now.timestamp()) - 24 * 3600, int(now.timestamp())), float)
    if len(points) < 10:
        continue
    spreads, line_time = sources.game_spreads(game, minutes_before=minutes_left)
    time.sleep(sources.SPREAD_PAUSE_SECONDS)
    if not np.isfinite(spreads["home_team_spread"]):
        continue
    rows.append({"game": f"{game['away_abbr']} at {game['home_abbr']}", "home_abbr": game["home_abbr"], "away_abbr": game["away_abbr"],
                 "minutes to tipoff": round(minutes_left), "price_home": points[-1, 1],
                 **lean.path_signals((points[:, 0] - points[-1, 0]) / 60.0, points[:, 1] * 100), **spreads})
if not rows:
    sys.exit("No upcoming game today has a Polymarket market, a price path and a sportsbook line.")

history = lean.load_games()
dev = history[history.split == "dev"]
_, floor = lean.backtest(history)
live = lean.add_signals(pd.DataFrame(rows))
live["estimate"] = lean.estimate(dev, live)
home = live.estimate > 0
live["lean"] = np.where(home, live.home_abbr, live.away_abbr)
live["price of that side"] = np.where(home, live.price_home, 1 - live.price_home).round(3)
live["bid"] = (live["price of that side"] - lean.PRICE_OFFSET).round(3)
live["action"] = np.where(live.estimate.abs() >= floor, "BID", "skip")
live["expected move, cents"] = live.estimate.abs().round(2)
print(f"Bid when the expected move is at least {floor:.2f} cents.\n")
print(live[["game", "minutes to tipoff", "lean", "price of that side", "bid", "expected move, cents", "action"]].to_string(index=False))
