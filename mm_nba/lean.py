"""The lean: predict the last-hour move of the Polymarket home price, bid one cent under on the side expected to rise.

Everything the strategy does is in this file: the signals, the model, the bet threshold, the trade and its scoring.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / "data/games.parquet"
CONFIG = json.loads((ROOT / "config.json").read_text())
SPREAD_COEF, SPREAD_INTERCEPT = CONFIG["spread_to_probability"]["coef"], CONFIG["spread_to_probability"]["intercept"]
PRICE_OFFSET, FEE_RATE, MAKER_REBATE = CONFIG["price_offset"], CONFIG["fee_rate"], CONFIG["maker_rebate"]
FLAT_STAKE, START_BANKROLL, BET_SHARE = CONFIG["flat_stake"], CONFIG["start_bankroll"], CONFIG["bet_share"]

EARLIER_SPREADS = (180, 480, 1500)      # minutes before tipoff at which the sportsbook line is also read
PATH_HORIZONS = {"pm_23h": 1380, "pm_7h": 420, "pm_3h": 180, "pm_1h": 60, "pm_15m": 15}      # minutes before the cutoff
BOOK_MOVES = ["book_2h", "book_7h", "book_24h"]      # tipoff minus 180, 480 and 1500 minutes, measured from the cutoff
SIGNALS = [*PATH_HORIZONS, "book_gap", *BOOK_MOVES]


def path_signals(minutes, cents):
    """How far the home price has moved over each horizon. minutes: time relative to the cutoff (all <= 0, ascending);
    cents: price x 100."""
    at = lambda ago: cents[max(np.searchsorted(minutes, -ago, side="right") - 1, 0)]
    return {name: cents[-1] - at(ago) for name, ago in PATH_HORIZONS.items()}


def spread_probability(spread):
    """Home spread (positive = home favoured) -> home-win probability in cents; NaN where there is no spread.
    A logistic curve fitted once on the closing spreads and results of the 2012-13 to 2023-24 seasons."""
    return 100 / (1 + np.exp(-(SPREAD_COEF * np.asarray(spread, float) + SPREAD_INTERCEPT)))


def add_signals(frame):
    """Add the sportsbook signals (cents) to a frame with price_home and the home spread at each horizon, and the
    training target `move` where the tipoff price is known."""
    now = spread_probability(frame.home_team_spread)
    out = frame.assign(book_gap=now - 100 * frame.price_home)
    for minutes, name in zip(EARLIER_SPREADS, BOOK_MOVES):
        out[name] = np.nan_to_num(now - spread_probability(frame[f"home_team_spread_{minutes}"]))
    if "price_tip" in out:
        out["move"] = 100 * (out.price_tip - out.price_home)
    return out


def load_games():
    """Every game with its signals, ordered by tipoff. split is "dev" (2024-25), "test" (2025-26) or "new" (2026-27)."""
    return add_signals(pd.read_parquet(GAMES)).sort_values(["scheduled_start_utc", "game_id"]).reset_index(drop=True)


def estimate(fit, apply):
    """Fit on one frame; return the expected last-hour move of the home price, in cents, for each game of another."""
    model = HistGradientBoostingRegressor(max_depth=2, max_iter=120, learning_rate=0.03, min_samples_leaf=40, random_state=0)
    return model.fit(fit[SIGNALS].to_numpy(float), fit.move).predict(apply[SIGNALS].to_numpy(float))


def cross_validated(frame, blocks=10):
    """Out-of-fold estimates for a frame ordered by tipoff, holding out one block of consecutive days at a time."""
    days = pd.unique(frame.game_date_et)
    block = frame.game_date_et.map({d: i * blocks // len(days) for i, d in enumerate(days)}).to_numpy()
    out = np.full(len(frame), np.nan)
    for b in range(blocks):
        out[block == b] = estimate(frame[block != b], frame[block == b])
    return out


def maker_cost(fill):
    """Cost per share of a maker fill, net of the sports maker rebate."""
    return fill - MAKER_REBATE * FEE_RATE * fill * (1 - fill)


def trade(frame, estimates, floor):
    """One flat bet per game whose estimate is at least `floor` cents either way: bid one cent under the price on the
    side expected to rise, assume the fill, hold to the final result."""
    keep = np.abs(estimates) >= floor
    picked, home = frame[keep], estimates[keep] > 0
    price = np.where(home, picked.price_home, 1 - picked.price_home)
    won = np.where(home, picked.home_win == 1, picked.home_win == 0)
    cost = maker_cost(np.clip(price - PRICE_OFFSET, 0.01, 0.99))
    return pd.DataFrame({"game_id": picked.game_id.to_numpy(), "split": picked.split.to_numpy(), "game_date_et": picked.game_date_et.to_numpy(),
                         "side": np.where(home, "home", "away"), "estimate": estimates[keep], "price": price, "won": won,
                         "pnl": np.where(won, FLAT_STAKE / cost - FLAT_STAKE, -FLAT_STAKE),
                         "pnl_at_price": np.where(won, FLAT_STAKE / price - FLAT_STAKE, -FLAT_STAKE),
                         "move": (picked.move * np.where(home, 1, -1)).to_numpy()})


def backtest(games, bet_share=BET_SHARE):
    """(bets, floor). DEV bets are cross-validated; every later season comes from one fit on all of DEV. The floor is the
    quantile of the cross-validated DEV estimates that leaves `bet_share` of DEV games bet."""
    dev, later = games[games.split == "dev"], games[games.split != "dev"]
    dev_estimates = cross_validated(dev)
    floor = float(np.quantile(np.abs(dev_estimates), 1 - bet_share)) if bet_share < 1 else 0.0
    return pd.concat([trade(dev, dev_estimates, floor), trade(later, estimate(dev, later), floor)], ignore_index=True), floor


def mean_interval(values, days, level=95.0, draws=4000, seed=0):
    """Mean and its interval, resampling whole days."""
    values, days = np.asarray(values, float), np.asarray(days)
    by_day = [values[days == d] for d in pd.unique(days)]
    rng = np.random.default_rng(seed)
    boot = [np.concatenate([by_day[i] for i in rng.integers(len(by_day), size=len(by_day))]).mean() for _ in range(draws)]
    return (values.mean(), *np.percentile(boot, [(100 - level) / 2, 100 - (100 - level) / 2]))
