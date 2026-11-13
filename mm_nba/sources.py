"""Public data sources for the lean: ESPN scoreboard, Polymarket markets and price history, TeamRankings spread history.

Nothing here needs an account or a key. TeamRankings asks for one page per 10 seconds; callers must sleep between pages.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .lean import EARLIER_SPREADS

AGENT = "nba-lean-research"
EASTERN = ZoneInfo("America/New_York")
SPREAD_PAUSE_SECONDS = 10
TEAM_SLUG = dict(ATL="hawks", BOS="celtics", BKN="nets", CHA="hornets", CHI="bulls", CLE="cavaliers", DAL="mavericks",
                 DEN="nuggets", DET="pistons", GSW="warriors", HOU="rockets", IND="pacers", LAC="clippers", LAL="lakers",
                 MEM="grizzlies", MIA="heat", MIL="bucks", MIN="timberwolves", NOP="pelicans", NYK="knicks", OKC="thunder",
                 ORL="magic", PHI="76ers", PHX="suns", POR="trail-blazers", SAC="kings", SAS="spurs", TOR="raptors",
                 UTA="jazz", WAS="wizards")
ESPN_FIX = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}


def get(url, agent="python-urllib", timeout=45):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": agent}), timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def espn_games(first_day, last_day, finished):
    """Regular-season games between two US Eastern dates (inclusive). finished=True: completed games with the result;
    finished=False: games that have not started."""
    games = {}
    for day in pd.date_range(first_day, pd.Timestamp(last_day) + pd.Timedelta(days=1)):       # scoreboard days are US dates
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={day:%Y%m%d}&limit=100"
        for event in json.loads(get(url)).get("events", []):
            competition = event["competitions"][0]
            status = competition["status"]["type"]
            if event.get("season", {}).get("type") != 2:
                continue
            if finished != bool(status.get("completed")) or (not finished and status.get("state") != "pre"):
                continue
            side = {c["homeAway"]: c for c in competition["competitors"]}
            abbr = {k: ESPN_FIX.get(c["team"]["abbreviation"], c["team"]["abbreviation"]) for k, c in side.items()}
            start = pd.Timestamp(event["date"])
            date_et = str(start.tz_convert("America/New_York").date())
            if abbr["home"] not in TEAM_SLUG or abbr["away"] not in TEAM_SLUG or not str(first_day) <= date_et <= str(last_day):
                continue
            game = {"game_id": "espn:" + event["id"], "game_date_et": date_et, "scheduled_start_utc": start, "home_abbr": abbr["home"],
                    "away_abbr": abbr["away"], "home_team": side["home"]["team"]["displayName"], "away_team": side["away"]["team"]["displayName"]}
            if finished:
                game["home_win"] = int(int(side["home"]["score"]) > int(side["away"]["score"]))
            games[game["game_id"]] = game
    return sorted(games.values(), key=lambda g: (g["scheduled_start_utc"], g["game_id"]))


def polymarket_moneylines(first_utc, last_utc, closed):
    """NBA moneyline markets whose end date falls between two UTC times."""
    listing, window = [], pd.Timestamp(first_utc)
    while window <= pd.Timestamp(last_utc):
        offset = 0
        while True:
            query = {"tag_id": 745, "sports_market_types": "moneyline", "closed": "true" if closed else "false", "limit": 100, "offset": offset,
                     "end_date_min": f"{window:%Y-%m-%dT%H:%M:%SZ}", "end_date_max": f"{window + pd.Timedelta(days=7):%Y-%m-%dT%H:%M:%SZ}"}
            page = json.loads(get("https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode(query), AGENT))
            listing += page
            offset += 100
            if len(page) < 100:
                break
        window += pd.Timedelta(days=7)
    return listing


def home_market(game, listing):
    """(home token id, away token id, market slug) of the moneyline market for this game, or None.
    The market must name both teams and start within three hours of the game; the plain dated slug wins a tie."""
    found = None
    for market in listing:
        try:
            outcomes, tokens = json.loads(market["outcomes"]), json.loads(market["clobTokenIds"])
            begins = pd.Timestamp(market["gameStartTime"])
            begins = begins.tz_localize("UTC") if begins.tzinfo is None else begins
        except (KeyError, TypeError, ValueError):
            continue
        if len(outcomes) != 2 or re.search(r"\d\.5|O/U|\bover\b|\bunder\b|spread|\([+-]\d", market.get("question") or "", re.I):
            continue
        if abs((begins - game["scheduled_start_utc"]).total_seconds()) > 180 * 60:
            continue
        sides = [[s for s in ("home", "away") if (game[s + "_team"].lower() + " ").endswith(" " + name.lower() + " ") or game[s + "_team"].lower() == name.lower()]
                 for name in outcomes]
        if sorted(map(str, sides)) != ["['away']", "['home']"]:
            continue
        plain = bool(re.fullmatch(r"nba-[a-z]+-[a-z]+-\d{4}-\d{2}-\d{2}", market["slug"]))
        if found is None or (plain and not found[3]):
            order = [s[0] for s in sides]
            found = (tokens[order.index("home")], tokens[order.index("away")], market["slug"], plain)
    return found[:3] if found else None


def price_history(token, start_ts, end_ts):
    """Polymarket's public one-minute CLOB price history for one token, as [(unix seconds, price)]."""
    url = "https://clob.polymarket.com/prices-history?" + urllib.parse.urlencode({"market": token, "startTs": start_ts, "endTs": end_ts, "fidelity": 1})
    for attempt in range(5):
        try:
            return [(int(pt["t"]), float(pt["p"])) for pt in json.loads(get(url, AGENT, timeout=30))["history"] if start_ts <= pt["t"] <= end_ts]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def spread_url(game):
    return f"https://www.teamrankings.com/nba/matchup/{TEAM_SLUG[game['away_abbr']]}-{TEAM_SLUG[game['home_abbr']]}-{game['game_date_et']}/spread-movement"


def parse_spread_page(html):
    """A TeamRankings spread-movement page -> (favourite's abbreviation, [(time text, one cell per sportsbook), ...])."""
    tables = re.findall(r"<table.*?</table>", html, flags=re.S)
    labels = re.findall(r"\b([A-Z]{2,4}) [+-]\d+\.\d\b", " ".join(re.sub(r"<[^>]+>", " ", t) for t in tables[:2]))
    history = []
    for table in tables:
        if "Book 1" not in table:
            continue
        for row in re.findall(r"<tr.*?</tr>", table, flags=re.S):
            cells = [re.sub(r"\s+", " ", c).strip()
                     for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", re.sub(r"<(?!/?t[dh])[^>]+>", "", row), flags=re.S)]
            if cells and re.match(r"\d{1,2}/\d{1,2} \d{1,2}:\d{2} [AP]M", cells[0]):
                history.append(cells)
    return (labels[0] if labels else None), history


def spread_before(listed_team, history, home_abbr, away_abbr, tipoff_utc, minutes_before=60):
    """Mean over sportsbooks of each book's last posted spread at or before (tipoff - minutes_before), as the home team's
    spread with positive = home favoured. Returns (spread, time of the newest line used, sportsbooks used).
    Page times are US Eastern with no year."""
    tipoff = pd.Timestamp(tipoff_utc).to_pydatetime()
    cutoff = tipoff - timedelta(minutes=minutes_before)
    if listed_team not in (home_abbr, away_abbr):
        return np.nan, pd.NaT, 0
    latest = {}                                                    # sportsbook column -> (time, spread of the listed team)
    for cells in history:
        stamp = datetime.strptime(cells[0], "%m/%d %I:%M %p")
        year = tipoff.astimezone(EASTERN).year
        when = stamp.replace(year=year, tzinfo=EASTERN)
        if when > tipoff + timedelta(hours=6):                     # a December line for a January game
            when = stamp.replace(year=year - 1, tzinfo=EASTERN)
        if when > cutoff:
            continue
        for book, text in enumerate(cells[1:]):
            try:
                value = float(text.replace("+", "").replace("PK", "0"))
            except ValueError:
                continue
            if book not in latest or when > latest[book][0]:
                latest[book] = (when, value)
    if not latest:
        return np.nan, pd.NaT, 0
    listed = float(np.mean([value for _, value in latest.values()]))
    newest = max(when for when, _ in latest.values())
    return (-listed if listed_team == home_abbr else listed), pd.Timestamp(newest).tz_convert("UTC"), len(latest)


def game_spreads(game, minutes_before=60):
    """Home spread at the cutoff and at each earlier horizon for one game, from one TeamRankings page.
    Returns ({"home_team_spread": ..., "home_team_spread_180": ..., ...}, time of the newest line used at the cutoff)."""
    listed_team, history = parse_spread_page(get(spread_url(game)))
    read = lambda minutes: spread_before(listed_team, history, game["home_abbr"], game["away_abbr"], game["scheduled_start_utc"], minutes)
    now, line_time, _ = read(minutes_before)
    return {"home_team_spread": now, **{f"home_team_spread_{m}": read(max(m, minutes_before))[0] for m in EARLIER_SPREADS}}, line_time
