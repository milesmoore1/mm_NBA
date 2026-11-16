"""Order placement on Polymarket, the order log, and the risk limits.

Credentials come from the environment (.env, copied into each workspace by .paradev/setup.sh):
    POLYMARKET_PRIVATE_KEY     the key that signs orders
    POLYMARKET_SIGNATURE_TYPE  0 = a wallet you hold directly, 1 = email/Magic login, 2 = browser-wallet login
    POLYMARKET_FUNDER          the address that holds the USDC (your Polymarket profile address; omit for type 0)
"""
import math
import os
from pathlib import Path

import pandas as pd

from .lean import CONFIG, ROOT

HOST, CHAIN_ID = "https://clob.polymarket.com", 137
TRADING = CONFIG["trading"]
LOG = ROOT / "data/orders.csv"
STOP_FILE = ROOT / "STOP"
LOG_COLUMNS = ["game_id", "game", "tipoff_utc", "status", "reason", "estimate", "floor", "side", "token_id", "bid", "size",
               "stake_usd", "order_id", "size_matched", "placed_utc", "closed_utc"]
OPEN, UNKNOWN = "open", "unknown"               # UNKNOWN: placing failed mid-request, so an order may exist


def load_env(path=ROOT / ".env"):
    """Read KEY=VALUE lines from .env into the environment without overriding variables already set."""
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class Polymarket:
    """The few CLOB calls the trader needs."""

    def __init__(self):
        from py_clob_client.client import ClobClient
        load_env()
        missing = [k for k in ("POLYMARKET_PRIVATE_KEY", "POLYMARKET_SIGNATURE_TYPE") if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"Set {', '.join(missing)} in .env (see mm_nba/execute.py).")
        self.client = ClobClient(HOST, chain_id=CHAIN_ID, key=os.environ["POLYMARKET_PRIVATE_KEY"],
                                 signature_type=int(os.environ["POLYMARKET_SIGNATURE_TYPE"]), funder=os.environ.get("POLYMARKET_FUNDER") or None)
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

    def usdc_balance(self):
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
        reply = self.client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        return int(reply["balance"]) / 1e6

    def best_bid(self, token_id):
        """(best bid price, tick size, minimum order size in shares); price is None when nobody is bidding."""
        book = self.client.get_order_book(token_id)
        bids = [float(level.price) for level in book.bids or [] if float(level.size) > 0]
        return (max(bids) if bids else None), float(book.tick_size), float(book.min_order_size)

    def place_bid(self, token_id, price, size):
        """Post-only GTC buy: rests on the book or is rejected; never trades against an ask. Returns (order id, error)."""
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY
        order = self.client.create_order(OrderArgs(token_id=token_id, price=price, size=size, side=BUY))
        reply = self.client.post_order(order, OrderType.GTC, post_only=True)
        if reply.get("success") and reply.get("orderID"):
            return reply["orderID"], None
        return None, reply.get("errorMsg") or str(reply)

    def size_matched(self, order_id):
        return float(self.client.get_order(order_id).get("size_matched") or 0)

    def cancel(self, order_id):
        self.client.cancel(order_id)

    def cancel_token(self, token_id):
        """Cancel every order we have resting on one token (used when it is unknown whether an order was placed)."""
        self.client.cancel_market_orders(asset_id=token_id)


def order_size(price, stake=None, minimum=5.0):
    """Shares for a flat stake at `price`, rounded down to 0.01 share; None if below the market's minimum order."""
    stake = TRADING["stake_usd"] if stake is None else stake
    size = math.floor(stake / price * 100) / 100
    return size if size >= minimum else None


class OrderLog:
    """data/orders.csv: one row per game, the trader's memory across restarts. A game with a row is never ordered again."""

    def __init__(self, path=LOG):
        self.path = Path(path)
        self.rows = pd.read_csv(self.path, dtype={"game_id": str, "order_id": str, "token_id": str}).to_dict("records") if self.path.exists() else []

    def get(self, game_id):
        return next((r for r in self.rows if r["game_id"] == game_id), None)

    def write(self, row):
        old = self.get(row["game_id"])
        if old is not None:
            old.update(row)
        else:
            self.rows.append({c: row.get(c) for c in LOG_COLUMNS})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows, columns=LOG_COLUMNS).to_csv(self.path, index=False)

    def open_orders(self):
        return [r for r in self.rows if r["status"] == OPEN]

    def staked_on(self, date_et):
        """Dollars committed by orders placed (or possibly placed) for games on one US Eastern date."""
        return sum(stake(r) for r in self.rows if pd.Timestamp(r["tipoff_utc"]).tz_convert("America/New_York").date() == date_et)

    def open_usd(self):
        return sum(stake(r) for r in self.open_orders())


def stake(row):
    value = row.get("stake_usd")
    return 0.0 if value is None or pd.isna(value) else float(value)


def limit_breach(log, stake, date_et):
    """The first risk limit an order of `stake` dollars would break, or None."""
    if stake > TRADING["max_order_usd"]:
        return f"stake ${stake:.2f} over max_order_usd"
    if log.staked_on(date_et) + stake > TRADING["max_day_usd"]:
        return "max_day_usd reached"
    if log.open_usd() + stake > TRADING["max_open_usd"]:
        return "max_open_usd reached"
    return None
