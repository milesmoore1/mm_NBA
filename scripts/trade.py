"""Trade the lean on Polymarket with real money, from recommendation to order, unattended.

    python scripts/trade.py

Runs until stopped. Every 30 minutes it reloads today's games; for each game, at tipoff - 60 minutes, it computes the
lean and, when the estimate clears the floor, posts a post-only bid for config.json's flat stake at the best bid of the
side expected to rise. At tipoff it records the fill and cancels anything unfilled. Everything goes to data/orders.csv.

Stop placing orders (and cancel every open one): create a file named STOP in the repository root. Delete it to resume.
Limits per order, per day and open at once are in config.json under "trading". Credentials: see mm_nba/execute.py.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mm_nba import execute, live
from mm_nba.trader import Trader

REFRESH_SECONDS, STEP_SECONDS = 30 * 60, 15

exchange = execute.Polymarket()
print(f"Connected. USDC balance ${exchange.usdc_balance():,.2f}. Limits: {execute.TRADING}", flush=True)
model = live.Model.load()
print(f"Model fitted on 2024-25; bet when the expected move is at least {model.floor:.3f} cents.", flush=True)
trader = Trader(exchange, model, execute.OrderLog(), say=lambda text: print(text, flush=True))
last_refresh = 0.0
while True:
    now = pd.Timestamp.now(tz="UTC")
    if time.time() - last_refresh >= REFRESH_SECONDS:
        try:
            pairs = live.upcoming_games(now)
            trader.add_games(pairs)
            last_refresh = time.time()
            schedule = ", ".join(f"{g['away_abbr']}@{g['home_abbr']} {(g['scheduled_start_utc'] - pd.Timedelta(minutes=60)).tz_convert('America/New_York'):%H:%M}"
                                 for g, m in pairs if m is not None)
            print(f"{now:%Y-%m-%d %H:%M} {len(pairs)} upcoming games; cutoffs (ET): {schedule or 'none'}", flush=True)
        except Exception as error:
            print(f"{now:%H:%M:%S} could not load games ({error}); retrying in a minute", flush=True)
            last_refresh = time.time() - REFRESH_SECONDS + 60
    trader.step(now)
    time.sleep(STEP_SECONDS)
