"""Check the Polymarket connection without placing anything: sign in, read the USDC balance, list today's markets.

    python scripts/check_polymarket.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mm_nba import execute, live

exchange = execute.Polymarket()
print(f"Signed in. USDC balance: ${exchange.usdc_balance():,.2f}")
for game, market in live.upcoming_games(pd.Timestamp.now(tz="UTC")):
    name = f"{game['away_abbr']} at {game['home_abbr']}"
    if market is None:
        print(f"  {name}: no Polymarket market")
        continue
    home, _, _ = exchange.best_bid(market[0])
    away, _, minimum = exchange.best_bid(market[1])
    print(f"  {name} ({market[2]}): best bids home {home}, away {away}; minimum order {minimum:g} shares")
