"""The trading loop's decisions, kept apart from the clock and the network so they can be tested.

For each game: at tipoff - 60 minutes compute the lean; if it clears the floor, post a post-only bid for a flat stake at
the best bid of the side expected to rise; at tipoff read the fill and cancel what is left. Every outcome, including
every skipped game, is one row in data/orders.csv.
"""
import pandas as pd

from . import execute, live

LATEST_ORDER_MINUTES = 5            # do not place an order this close to tipoff (e.g. after a restart)
SIGNAL_ATTEMPTS = 3                 # data fetch failures tolerated per game before it is logged as an error


class Trader:
    def __init__(self, exchange, model, log, signals=live.signals_at, say=print, stop_file=execute.STOP_FILE):
        self.exchange, self.model, self.log, self.signals, self.say, self.stop_file = exchange, model, log, signals, say, stop_file
        self.games = {}             # game_id -> (game, market)
        self.failures = {}

    def add_games(self, pairs):
        for game, market in pairs:
            self.games[game["game_id"]] = (game, market)

    def step(self, now):
        """Do whatever is due at `now` (a UTC Timestamp)."""
        if self.stop_file.exists():
            for row in self.log.open_orders():
                self.close(row, now, "STOP file present")
            return
        for row in self.log.open_orders():
            if now >= pd.Timestamp(row["tipoff_utc"]):
                self.close(row, now)
        for game_id, (game, market) in sorted(self.games.items(), key=lambda kv: kv[1][0]["scheduled_start_utc"]):
            tipoff = game["scheduled_start_utc"]
            cutoff = tipoff - pd.Timedelta(minutes=live.CUTOFF_MINUTES)
            if self.log.get(game_id) is None and cutoff <= now:
                if now >= tipoff - pd.Timedelta(minutes=LATEST_ORDER_MINUTES):
                    self.record(game, now, status="skipped", reason="missed the cutoff (trader was not running)")
                else:
                    self.enter(game, market, cutoff, now)

    def record(self, game, now, **fields):
        row = {"game_id": game["game_id"], "game": f"{game['away_abbr']} at {game['home_abbr']}",
               "tipoff_utc": game["scheduled_start_utc"].isoformat(), "floor": self.model.floor, **fields}
        if fields.get("status") != execute.OPEN:
            row["closed_utc"] = now.isoformat()
        self.log.write(row)
        detail = ", ".join(f"{k}={v}" for k, v in fields.items() if k in ("reason", "side", "bid", "size", "estimate") and v is not None)
        self.say(f"{now:%H:%M:%S} {row['game']}: {fields['status']} ({detail})")

    def enter(self, game, market, cutoff, now):
        if market is None:
            return self.record(game, now, status="skipped", reason="no Polymarket market")
        try:
            row, reason = self.signals(game, market, cutoff.timestamp())
        except Exception as error:                                  # network trouble: try again on the next step
            self.failures[game["game_id"]] = self.failures.get(game["game_id"], 0) + 1
            if self.failures[game["game_id"]] >= SIGNAL_ATTEMPTS:
                self.record(game, now, status="error", reason=f"could not read signals: {error}")
            return
        if row is None:
            return self.record(game, now, status="skipped", reason=reason)
        lean = live.decide(self.model, row)
        side, token = lean["side"], market[0] if lean["side"] == "home" else market[1]
        base = {"estimate": round(lean["estimate"], 4), "side": side, "token_id": token}
        if not lean["bet"]:
            return self.record(game, now, status="skipped", reason="estimate below floor", **base)
        try:
            bid, _, minimum = self.exchange.best_bid(token)
            if bid is None:
                return self.record(game, now, status="skipped", reason="no bids on the book", **base)
            size = execute.order_size(bid, minimum=minimum)
            if size is None:
                return self.record(game, now, status="skipped", reason=f"stake buys fewer than {minimum:g} shares at {bid}", bid=bid, **base)
            stake = round(bid * size, 2)
            breach = execute.limit_breach(self.log, stake, game["scheduled_start_utc"].tz_convert("America/New_York").date())
            if breach:
                return self.record(game, now, status="skipped", reason=breach, bid=bid, size=size, **base)
            if self.exchange.usdc_balance() < stake:
                return self.record(game, now, status="skipped", reason="not enough USDC", bid=bid, size=size, **base)
        except Exception as error:
            return self.record(game, now, status="error", reason=f"before placing: {error}", **base)
        try:
            order_id, refusal = self.exchange.place_bid(token, bid, size)
        except Exception as error:                  # the order may or may not exist: never retry, cancel the token at tipoff
            return self.record(game, now, status=execute.OPEN, reason=f"placing failed, an order may exist: {error}",
                               bid=bid, size=size, stake_usd=stake, order_id=execute.UNKNOWN, placed_utc=now.isoformat(), **base)
        if order_id is None:
            return self.record(game, now, status="rejected", reason=refusal, bid=bid, size=size, **base)
        self.record(game, now, status=execute.OPEN, bid=bid, size=size, stake_usd=stake, order_id=order_id, placed_utc=now.isoformat(), **base)

    def close(self, row, now, reason=None):
        """At tipoff (or on STOP): read how much filled and cancel the rest."""
        if row["order_id"] == execute.UNKNOWN:
            try:
                self.exchange.cancel_token(row["token_id"])
            except Exception as error:
                self.say(f"{now:%H:%M:%S} {row['game']}: could not cancel orders on {row['token_id']} ({error}); retrying")
                return
            self.log.write({**row, "status": "check Polymarket", "closed_utc": now.isoformat()})
            self.say(f"{now:%H:%M:%S} {row['game']}: cancelled anything resting on the token; check Polymarket for a fill")
            return
        try:
            matched = self.exchange.size_matched(row["order_id"])
            if matched < float(row["size"]):
                self.exchange.cancel(row["order_id"])
        except Exception as error:
            self.say(f"{now:%H:%M:%S} {row['game']}: could not close order {row['order_id']} ({error}); retrying")
            return
        status = "filled" if matched >= float(row["size"]) else "partial" if matched > 0 else "unfilled"
        self.log.write({**row, "status": status, "size_matched": matched, "closed_utc": now.isoformat(),
                        "reason": reason if reason else row.get("reason")})
        self.say(f"{now:%H:%M:%S} {row['game']}: {status}, {matched:g} of {float(row['size']):g} shares at {row['bid']}")
