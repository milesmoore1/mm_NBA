"""The repository must reproduce the results it was built from (the forward-test table of the source notebook)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mm_nba import lean

EXPECTED = json.loads((Path(__file__).parent / "expected.json").read_text())
THROUGH = EXPECTED["through"]            # games collected later extend the forward test and are left out of these checks


@pytest.fixture(scope="module")
def games():
    table = lean.load_games()
    return table[table.game_date_et <= THROUGH]


@pytest.fixture(scope="module")
def result(games):
    return lean.backtest(games)


def test_game_counts(games):
    assert games.game_id.is_unique
    assert games.split.value_counts().to_dict() == EXPECTED["games"]
    assert games[["price_home", "price_tip", "home_win", "home_team_spread", *lean.PATH_HORIZONS]].notna().all().all()


def test_signals_match_the_source(games):
    source = pd.read_parquet(Path(__file__).parent / "expected_signals.parquet").set_index("game_id").loc[games.game_id]
    for column in ("book_gap", *lean.BOOK_MOVES, "move"):
        assert np.allclose(games[column].to_numpy(), source[column].to_numpy(), atol=1e-9), column


def test_threshold(result):
    assert result[1] == pytest.approx(EXPECTED["floor"], abs=1e-9)


@pytest.mark.parametrize("split", ["dev", "test", "new"])
def test_season_results(result, split):
    bets = result[0][result[0].split == split]
    assert len(bets) == EXPECTED["seasons"][split]["bets"]
    assert bets.pnl.sum() == pytest.approx(EXPECTED["seasons"][split]["profit"], abs=0.01)


def test_one_bet_per_game_and_no_look_ahead(games, result):
    bets = result[0]
    assert bets.game_id.is_unique
    assert set(lean.SIGNALS).isdisjoint({"price_tip", "home_win", "move"})
    later = games[games.split != "dev"]
    changed = later.assign(home_win=1 - later.home_win, price_tip=0.5)          # outcomes of later seasons cannot move their estimates
    assert np.array_equal(lean.estimate(games[games.split == "dev"], lean.add_signals(changed)), lean.estimate(games[games.split == "dev"], later))


def test_trade_arithmetic():
    frame = pd.DataFrame({"game_id": ["a", "b", "c"], "split": "x", "game_date_et": "2026-01-01", "price_home": [0.60, 0.60, 0.60],
                          "home_win": [1, 1, 0], "move": [2.0, 2.0, 2.0]})
    bets = lean.trade(frame, np.array([1.0, -1.0, 0.1]), floor=0.5)
    assert list(bets.side) == ["home", "away"] and list(bets.won) == [True, False]
    cost = 0.59 - lean.MAKER_REBATE * lean.FEE_RATE * 0.59 * 0.41
    assert bets.pnl.tolist() == pytest.approx([lean.FLAT_STAKE / cost - lean.FLAT_STAKE, -lean.FLAT_STAKE])
    assert bets.move.tolist() == [2.0, -2.0]
