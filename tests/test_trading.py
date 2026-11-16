"""The live path must make the backtest's decisions, and the trader must place, limit, de-duplicate and close orders
correctly. A fake exchange stands in for Polymarket; nothing here touches the network."""
import numpy as np
import pandas as pd
import pytest

from mm_nba import execute, lean, live
from mm_nba.trader import Trader

TIP = pd.Timestamp("2026-11-16 00:30", tz="UTC")
GAME = {"game_id": "espn:1", "game_date_et": "2026-11-15", "scheduled_start_utc": TIP, "home_abbr": "BOS", "away_abbr": "ORL",
        "home_team": "Boston Celtics", "away_team": "Orlando Magic"}
MARKET = ("home-token", "away-token", "nba-orl-bos-2026-11-15")
SPREADS = {"home_team_spread": 8.0, "home_team_spread_180": 8.0, "home_team_spread_480": 7.5, "home_team_spread_1500": 7.0}


class FakeExchange:
    def __init__(self, bid=0.23, balance=100.0, matched=0.0, refuse=None):
        self.bid, self.balance, self.matched, self.refuse = bid, balance, matched, refuse
        self.placed, self.cancelled = [], []

    def best_bid(self, token):
        return self.bid, 0.01, 5.0

    def usdc_balance(self):
        return self.balance

    def place_bid(self, token, price, size):
        self.placed.append((token, price, size))
        return (None, self.refuse) if self.refuse else (f"order-{len(self.placed)}", None)

    def size_matched(self, order_id):
        return self.matched

    def cancel(self, order_id):
        self.cancelled.append(order_id)

    def cancel_token(self, token):
        self.cancelled.append(token)


class FakeModel:
    def __init__(self, estimate, floor=0.128):
        self.value, self.floor = estimate, floor

    def estimate(self, rows):
        return np.full(len(rows), self.value)


def row_for(game, market, at_ts):
    return {"game_id": game["game_id"], "price_home": 0.765, **dict.fromkeys(lean.PATH_HORIZONS, 0.0), **SPREADS}, None


@pytest.fixture
def trader(tmp_path):
    def make(estimate=-0.5, **exchange):
        t = Trader(FakeExchange(**exchange), FakeModel(estimate), execute.OrderLog(tmp_path / "orders.csv"),
                   signals=row_for, say=lambda text: None, stop_file=tmp_path / "STOP")
        t.add_games([(GAME, MARKET)])
        return t
    return make


def test_live_model_makes_the_backtest_decisions():
    games = lean.load_games()
    later = games[games.split == "test"].head(200)
    model = live.Model.load()
    assert np.array_equal(model.estimate(later.to_dict("records")), lean.estimate(model.dev, later))
    _, floor = lean.backtest(games)
    assert model.floor == floor


def test_build_row_reads_the_path_as_of_the_cutoff_like_collect():
    cutoff = TIP.timestamp() - 3600
    t = np.arange(cutoff - 24 * 3600, cutoff + 1800, 60.0)              # the path runs on past the cutoff
    points = np.column_stack([t, 0.5 + 0.1 * np.sin(t / 5000)])
    row, reason = live.build_row(GAME, points, SPREADS, cutoff)
    before = points[points[:, 0] <= cutoff]                            # what scripts/collect.py keeps
    assert reason is None and row["price_home"] == before[-1, 1]
    assert {k: row[k] for k in lean.PATH_HORIZONS} == lean.path_signals((before[:, 0] - cutoff) / 60.0, before[:, 1] * 100)
    assert live.build_row(GAME, np.column_stack([t, np.full_like(t, 0.99)]), SPREADS, cutoff) == (None, "price 0.990 outside 0.02 to 0.98")


def test_nothing_happens_before_the_cutoff(trader):
    t = trader()
    t.step(TIP - pd.Timedelta(minutes=61))
    assert t.exchange.placed == [] and t.log.rows == []


def test_bids_the_best_bid_of_the_lean_side_for_a_flat_stake(trader):
    t = trader(estimate=-0.5)                                          # away (Orlando) expected to rise
    t.step(TIP - pd.Timedelta(minutes=60))
    assert t.exchange.placed == [("away-token", 0.23, 21.73)]          # floor($5 / 0.23, 0.01 share)
    row = t.log.get("espn:1")
    assert row["status"] == "open" and row["side"] == "away" and row["stake_usd"] == pytest.approx(5.0, abs=0.01)


def test_one_order_per_game_even_after_a_restart(trader, tmp_path):
    t = trader()
    t.step(TIP - pd.Timedelta(minutes=60))
    t.step(TIP - pd.Timedelta(minutes=59))
    again = Trader(t.exchange, t.model, execute.OrderLog(tmp_path / "orders.csv"), signals=row_for, say=lambda text: None, stop_file=tmp_path / "STOP")
    again.add_games([(GAME, MARKET)])
    again.step(TIP - pd.Timedelta(minutes=58))
    assert len(t.exchange.placed) == 1


def test_below_the_floor_is_logged_and_not_ordered(trader):
    t = trader(estimate=0.05)
    t.step(TIP - pd.Timedelta(minutes=60))
    assert t.exchange.placed == [] and t.log.get("espn:1")["reason"] == "estimate below floor"


@pytest.mark.parametrize("kwargs, reason", [({"bid": None}, "no bids on the book"), ({"balance": 1.0}, "not enough USDC")])
def test_skips_without_ordering(trader, kwargs, reason):
    t = trader(**kwargs)
    t.step(TIP - pd.Timedelta(minutes=60))
    assert t.exchange.placed == [] and t.log.get("espn:1")["reason"] == reason


def test_daily_limit(trader, monkeypatch):
    monkeypatch.setitem(execute.TRADING, "max_day_usd", 4.0)
    t = trader()
    t.step(TIP - pd.Timedelta(minutes=60))
    assert t.exchange.placed == [] and t.log.get("espn:1")["reason"] == "max_day_usd reached"


def test_rejection_is_logged(trader):
    t = trader(refuse="order crosses book")
    t.step(TIP - pd.Timedelta(minutes=60))
    assert t.log.get("espn:1")["status"] == "rejected" and t.log.get("espn:1")["reason"] == "order crosses book"


def test_late_start_does_not_order(trader):
    t = trader()
    t.step(TIP - pd.Timedelta(minutes=3))
    assert t.exchange.placed == [] and t.log.get("espn:1")["status"] == "skipped"


@pytest.mark.parametrize("matched, status, cancelled", [(0.0, "unfilled", True), (10.0, "partial", True), (21.73, "filled", False)])
def test_closes_at_tipoff(trader, matched, status, cancelled):
    t = trader(matched=matched)
    t.step(TIP - pd.Timedelta(minutes=60))
    t.step(TIP - pd.Timedelta(minutes=1))
    assert t.log.get("espn:1")["status"] == "open"
    t.step(TIP)
    assert t.log.get("espn:1")["status"] == status and (t.exchange.cancelled == ["order-1"]) == cancelled


def test_stop_file_cancels_and_blocks(trader, tmp_path):
    t = trader()
    t.step(TIP - pd.Timedelta(minutes=60))
    (tmp_path / "STOP").touch()
    t.step(TIP - pd.Timedelta(minutes=50))
    assert t.exchange.cancelled == ["order-1"] and t.log.get("espn:1")["status"] == "unfilled"


def test_order_size():
    assert execute.order_size(0.23, stake=5) == 21.73
    assert execute.order_size(0.99, stake=5) == 5.05
    assert execute.order_size(0.99, stake=4) is None                   # under the 5-share minimum


def test_failed_placement_counts_against_limits_and_is_cancelled_at_tipoff(trader, tmp_path):
    t = trader()
    def boom(*args):
        raise TimeoutError("timed out")
    t.exchange.place_bid = boom
    t.step(TIP - pd.Timedelta(minutes=60))
    reloaded = execute.OrderLog(tmp_path / "orders.csv")
    assert reloaded.open_usd() == pytest.approx(5.0, abs=0.01) and reloaded.staked_on(TIP.tz_convert("America/New_York").date()) > 0
    t.step(TIP)
    assert t.exchange.cancelled == ["away-token"] and t.log.get("espn:1")["status"] == "check Polymarket"
