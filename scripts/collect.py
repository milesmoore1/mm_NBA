"""Add finished games to data/games.parquet so the forward test can be extended.

    python scripts/collect.py 2026-11-12 2026-11-20

For every finished regular-season game between the two US Eastern dates that is not already in the table: the result and
tipoff time from ESPN, the Polymarket home price path up to the cutoff (tipoff minus 60 minutes) and the price at tipoff,
and the sportsbook spread history from TeamRankings (one page per 10 seconds). Games without a Polymarket market, a price
within an hour of the cutoff, or a spread are skipped. New rows get split "new". Existing rows are never changed.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mm_nba import lean, sources

first, last = sys.argv[1], sys.argv[2]
table = pd.read_parquet(lean.GAMES)
games = [g for g in sources.espn_games(first, last, finished=True) if g["game_id"] not in set(table.game_id)]
print(f"{len(games)} finished games to add", flush=True)
listing = sources.polymarket_moneylines(pd.Timestamp(first, tz="UTC") - pd.Timedelta(days=1), pd.Timestamp(last, tz="UTC") + pd.Timedelta(days=10),
                                        closed=True) if games else []
rows, skipped = [], []
for game in games:
    market = sources.home_market(game, listing)
    if market is None:
        skipped.append((game["game_id"], "no Polymarket market"))
        continue
    tip = int(game["scheduled_start_utc"].timestamp())
    points = np.array(sources.price_history(market[0], tip - 25 * 3600, tip), float)
    before = points[points[:, 0] <= tip - 3600] if len(points) else points
    if len(before) < 10 or before[-1, 0] < tip - 7200:
        skipped.append((game["game_id"], "no price within an hour of the cutoff"))
        continue
    spreads, _ = sources.game_spreads(game)
    time.sleep(sources.SPREAD_PAUSE_SECONDS)
    if not np.isfinite(spreads["home_team_spread"]):
        skipped.append((game["game_id"], "no sportsbook spread before the cutoff"))
        continue
    if not 0.02 <= before[-1, 1] <= 0.98:
        skipped.append((game["game_id"], "price outside 0.02 to 0.98"))
        continue
    rows.append({"game_id": game["game_id"], "split": "new", "game_date_et": game["game_date_et"], "scheduled_start_utc": game["scheduled_start_utc"],
                 "home_win": game["home_win"], "price_home": before[-1, 1], "price_tip": points[-1, 1],
                 **lean.path_signals((before[:, 0] - (tip - 3600)) / 60.0, before[:, 1] * 100), **spreads})
    print(f"  {game['game_date_et']} {game['away_abbr']} at {game['home_abbr']}: added", flush=True)
for game_id, reason in skipped:
    print(f"  skipped {game_id}: {reason}")
if rows:
    pd.concat([table, pd.DataFrame(rows)], ignore_index=True).to_parquet(lean.GAMES, index=False)
print(f"added {len(rows)}, skipped {len(skipped)}; the table now has {len(table) + len(rows):,} games")
