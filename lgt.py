"""
MLB Data Warehouse — Live Game Tracker (v2)

Drop this file into your Streamlit repo and run:
    streamlit run live_tracker_v2.py

What this version fixes vs. last year:
- NO infinite loops / NO time.sleep polling in the app thread
- One shared server-side cache across users (so 5+ users don't 5x your API calls)
- TTL-based refresh (every 30s by default) via streamlit-autorefresh (optional dependency)
- Requests Session w/ retries + timeouts (prevents hanging and crashing)
- Parallel fetching per game (boxscore + pbp)
- Refactors "fetch" away from parsing functions, so caching actually works

Dependencies:
- streamlit
- pandas, numpy
- requests, pytz
Optional (recommended):
- streamlit-autorefresh  -> pip install streamlit-autorefresh
"""

from __future__ import annotations

import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="⚾ MLB Live Game Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Global CSS — clean light theme
# -----------------------------
st.markdown(
    """
    <style>
    /* ---------- base ---------- */
    html, body, [data-testid="stAppViewContainer"] {
        background: #f4f6fa !important;
    }
    [data-testid="stSidebar"] {
        background: #1a2744 !important;
    }
    [data-testid="stSidebar"] * {
        color: #e8edf6 !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #c5d0e8 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2e3f6a !important;
    }

    /* ---------- page header ---------- */
    .tracker-header {
        background: linear-gradient(135deg, #0d2060 0%, #1a4fa8 60%, #2563eb 100%);
        border-radius: 14px;
        padding: 20px 28px 16px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 18px rgba(13,32,96,0.18);
    }
    .tracker-header h1 {
        margin: 0;
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.3px;
    }
    .tracker-header .sub {
        color: #93b4e8;
        font-size: 0.82rem;
        margin-top: 3px;
    }
    .update-badge {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 6px 14px;
        color: #d0e4ff;
        font-size: 0.78rem;
        white-space: nowrap;
    }

    /* ---------- section headers ---------- */
    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1a2744;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border-left: 4px solid #2563eb;
        padding-left: 10px;
        margin: 18px 0 10px;
    }

    /* ---------- scoreboard cards ---------- */
    .scoreboard-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 16px;
    }
    .score-card {
        background: #ffffff;
        border: 1px solid #dce3f0;
        border-radius: 12px;
        padding: 12px 16px;
        min-width: 230px;
        flex: 1 1 230px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        position: relative;
        overflow: hidden;
    }
    .score-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2563eb, #38bdf8);
    }
    .score-card.final::before {
        background: linear-gradient(90deg, #64748b, #94a3b8);
    }
    .score-card.live::before {
        background: linear-gradient(90deg, #16a34a, #4ade80);
        animation: pulse-bar 2s ease-in-out infinite;
    }
    @keyframes pulse-bar {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.55; }
    }
    .score-card .matchup {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .score-card .teams-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .score-card .team-block {
        text-align: center;
    }
    .score-card .team-abbr {
        font-size: 1.3rem;
        font-weight: 800;
        color: #1a2744;
    }
    .score-card .team-runs {
        font-size: 2rem;
        font-weight: 900;
        color: #2563eb;
        line-height: 1;
    }
    .score-card .team-runs.winning {
        color: #16a34a;
    }
    .score-card .separator {
        font-size: 1.1rem;
        color: #cbd5e1;
        font-weight: 300;
    }
    .score-card .status-row {
        margin-top: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .badge-live {
        background: #dcfce7;
        color: #15803d;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-final {
        background: #f1f5f9;
        color: #64748b;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        text-transform: uppercase;
    }
    .badge-inning {
        background: #eff6ff;
        color: #2563eb;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 20px;
    }

    /* ---------- stat tiles ---------- */
    .stat-tiles {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 16px;
    }
    .stat-tile {
        background: #ffffff;
        border: 1px solid #dce3f0;
        border-radius: 10px;
        padding: 12px 18px;
        text-align: center;
        flex: 1 1 100px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    }
    .stat-tile .tile-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #2563eb;
        line-height: 1.1;
    }
    .stat-tile .tile-lbl {
        font-size: 0.68rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    /* ---------- dataframe overrides ---------- */
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid #dce3f0 !important;
    }

    /* ---------- sidebar nav ---------- */
    [data-testid="stSidebar"] .stRadio [role="radio"] {
        border-radius: 8px;
        padding: 4px 8px;
    }
    .sidebar-logo {
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .sidebar-sub {
        font-size: 0.72rem;
        color: #7b8fc2;
        margin-bottom: 16px;
    }

    /* ---------- alerts ---------- */
    .alert-info {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 10px 14px;
        color: #1e40af;
        font-size: 0.83rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Utilities
# -----------------------------
def dropUnnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


# -----------------------------
# Data loading (local CSVs)
# -----------------------------
@st.cache_resource
def load_local_files() -> Dict[str, Any]:
    """
    Loads your local lookup CSVs one time per server process.
    Make sure your repo has:
      Files/mlbteamnamechange.csv
      Files/LeagueLevels.csv
      Files/Team_Affiliates.csv
      Files/IDLookupTable.csv
      Files/pitchmovement25.csv
      Files/lsaclass.csv
    """
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "Files")

    teamnamechangedf = pd.read_csv(os.path.join(file_path, "mlbteamnamechange.csv"))
    teamnamedict = dict(zip(teamnamechangedf.Full, teamnamechangedf.Abbrev))

    league_lev_df = pd.read_csv(os.path.join(file_path, "LeagueLevels.csv"))
    levdict = dict(zip(league_lev_df.league_name, league_lev_df.level))

    affdf = pd.read_csv(os.path.join(file_path, "Team_Affiliates.csv"))
    affdict = dict(zip(affdf.team_id, affdf.parent_id))
    affdict_abbrevs = dict(zip(affdf.team_id, affdf.parent_abbrev))
    team_abbrev_look = dict(zip(affdf.team_name, affdf.team_abbrev))

    idlookup_df = pd.read_csv(os.path.join(file_path, "IDLookupTable.csv"))
    p_lookup_dict = dict(zip(idlookup_df.MLBID, idlookup_df.PLAYERNAME))

    pmove25 = pd.read_csv(os.path.join(file_path, "pitchmovement25.csv"))
    pmove25 = pmove25.rename(
        {
            "pfx_x": "Avg Horiz",
            "pfx_z": "Avg Vert",
            "release_speed": "Avg Velo",
            "player_name": "Pitcher",
            "pitch_type": "Pitch",
        },
        axis=1,
    )

    lsaclass = pd.read_csv(os.path.join(file_path, "lsaclass.csv"))
    lsaclass = dropUnnamed(lsaclass)
    lsaclass["launch_speed"] = round(lsaclass["launch_speed"], 0)
    lsaclass["launch_angle"] = round(lsaclass["launch_angle"], 0)
    lsaclass.columns = ["launch_speed_round", "launch_angle_round", "launch_speed_angle"]

    return {
        "teamnamedict": teamnamedict,
        "levdict": levdict,
        "affdict": affdict,
        "affdict_abbrevs": affdict_abbrevs,
        "team_abbrev_look": team_abbrev_look,
        "p_lookup_dict": p_lookup_dict,
        "pmove25": pmove25,
        "lsaclass": lsaclass,
    }


DATA = load_local_files()
teamnamedict = DATA["teamnamedict"]
levdict = DATA["levdict"]
affdict = DATA["affdict"]
affdict_abbrevs = DATA["affdict_abbrevs"]
pmove25 = DATA["pmove25"]
lsaclass = DATA["lsaclass"]


# -----------------------------
# HTTP client (retries + pooling)
# -----------------------------
@st.cache_resource
def get_http() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "MLB-DW-Live-Tracker/2.0"})
    return s


def fetch_json(url: str, timeout: Tuple[float, float] = (3.05, 12.0)) -> Dict[str, Any]:
    s = get_http()
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# -----------------------------
# Navigation
# -----------------------------
def sidebar_menu() -> str:
    st.sidebar.markdown('<div class="sidebar-logo">⚾ MLB Live Tracker</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-sub">MLB Data Warehouse</div>', unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigate",
        ["📊 Scores & Leaders", "🎯 Pitcher Detail", "🔀 Pitch Mix", "💥 Exit Velos", "📈 Game Pace"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Hide finished games — persisted in session_state
    if "hide_finished" not in st.session_state:
        st.session_state["hide_finished"] = False

    st.session_state["hide_finished"] = st.sidebar.checkbox(
        "Hide Finished Games",
        value=st.session_state["hide_finished"],
        key="hide_finished_cb",
    )

    return page


# -----------------------------
# Schedule / Live games
# -----------------------------
@st.cache_data(ttl=60, show_spinner=False)
def get_live_games(date_string: str) -> List[Dict[str, Any]]:
    games: List[Dict[str, Any]] = []

    sportIds = [1]  # MLB only (add MiLB if you want)
    sport_id_mappings = {1: "MLB", 11: "AAA", 12: "AA", 13: "A+", 14: "A", 16: "ROK", 17: "WIN"}

    for sportId in sportIds:
        url = f"https://statsapi.mlb.com/api/v1/schedule/?sportId={sportId}&date={date_string}"
        schedule = fetch_json(url)

        for date in schedule.get("dates", []):
            for game_data in date.get("games", []):
                games.append(
                    {
                        "date": date_string,
                        "game_id": game_data.get("gamePk"),
                        "game_type": game_data.get("gameType"),
                        "venue_id": game_data.get("venue", {}).get("id"),
                        "venue_name": game_data.get("venue", {}).get("name"),
                        "away_team": game_data.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                        "home_team": game_data.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                        "league_id": sportId,
                        "league_level": sport_id_mappings.get(sportId),
                        "game_status": game_data.get("status", {}).get("codedGameState"),
                        "game_status_full": game_data.get("status", {}).get("abstractGameState"),
                        "game_start_time": game_data.get("gameDate"),
                    }
                )
    return games


@st.cache_data(ttl=30, show_spinner=False)
def fetch_boxscore(game_id: int) -> Dict[str, Any]:
    return fetch_json(f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_pbp(game_id: int) -> Dict[str, Any]:
    return fetch_json(f"https://statsapi.mlb.com/api/v1/game/{game_id}/playByPlay")


# -----------------------------
# Parsing: Boxscore -> hitter & pitcher logs
# -----------------------------
def get_game_logs_from_boxjson(game: Dict[str, Any], game_info: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    batting_logs: List[Dict[str, Any]] = []
    pitching_logs: List[Dict[str, Any]] = []

    away = game_info.get("teams", {}).get("away", {}).get("team", {}) or {}
    lgname = (away.get("league", {}) or {}).get("name")

    away_team = away.get("name")
    away_team_id = away.get("id")

    home = game_info.get("teams", {}).get("home", {}).get("team", {}) or {}
    home_team = home.get("name")
    home_team_id = home.get("id")

    teams_obj = game_info.get("teams", {})
    if not teams_obj:
        return batting_logs, pitching_logs

    for team_key, team in teams_obj.items():
        team_id = (team.get("team") or {}).get("id")
        team_name = (team.get("team") or {}).get("name")

        for player in (team.get("players") or {}).values():
            stats = player.get("stats") or {}
            person = player.get("person") or {}
            pid = safe_int(person.get("id"), 0)
            pname = person.get("fullName")

            # batting
            batting = stats.get("batting") or {}
            if batting:
                batting_log = {
                    "game_date": game["date"],
                    "game_id": safe_int(game["game_id"]),
                    "league_name": lgname,
                    "level": game["league_level"],
                    "Team": team_name,
                    "team_id": safe_int(team_id),
                    "home_team": home_team,
                    "game_type": game["game_type"],
                    "venue_id": safe_int(game["venue_id"]),
                    "league_id": safe_int(game["league_id"]),
                    "Player": pname,
                    "player_id": pid,
                    "batting_order": player.get("battingOrder", ""),
                    "AB": safe_int(batting.get("atBats"), 0),
                    "R": safe_int(batting.get("runs"), 0),
                    "H": safe_int(batting.get("hits"), 0),
                    "2B": safe_int(batting.get("doubles"), 0),
                    "3B": safe_int(batting.get("triples"), 0),
                    "HR": safe_int(batting.get("homeRuns"), 0),
                    "RBI": safe_int(batting.get("rbi"), 0),
                    "SB": safe_int(batting.get("stolenBases"), 0),
                    "CS": safe_int(batting.get("caughtStealing"), 0),
                    "BB": safe_int(batting.get("baseOnBalls"), 0),
                    "SO": safe_int(batting.get("strikeOuts"), 0),
                    "IBB": safe_int(batting.get("intentionalWalks"), 0),
                    "HBP": safe_int(batting.get("hitByPitch"), 0),
                    "SH": safe_int(batting.get("sacBunts"), 0),
                    "SF": safe_int(batting.get("sacFlies"), 0),
                    "GIDP": safe_int(batting.get("groundIntoDoublePlay"), 0),
                }
                batting_logs.append(batting_log)

            # pitching
            pitching = stats.get("pitching") or {}
            if pitching:
                ip = safe_float(pitching.get("inningsPitched"), 0.0)
                gs = safe_int(pitching.get("gamesStarted"), 0)
                er = safe_int(pitching.get("earnedRuns"), 0)

                pitching_log = {
                    "game_date": game["date"],
                    "game_id": safe_int(game["game_id"]),
                    "league_name": lgname,
                    "level": game["league_level"],
                    "Team": team_name,
                    "team_id": safe_int(team_id),
                    "home_team": home_team,
                    "game_type": game["game_type"],
                    "venue_id": safe_int(game["venue_id"]),
                    "league_id": safe_int(game["league_id"]),
                    "Player": pname,
                    "player_id": pid,
                    "W": safe_int(pitching.get("wins"), 0),
                    "L": safe_int(pitching.get("losses"), 0),
                    "G": safe_int(pitching.get("gamesPlayed"), 0),
                    "GS": gs,
                    "CG": safe_int(pitching.get("completeGames"), 0),
                    "SHO": safe_int(pitching.get("shutouts"), 0),
                    "SV": safe_int(pitching.get("saves"), 0),
                    "HLD": safe_int(pitching.get("holds"), 0),
                    "BFP": safe_int(pitching.get("battersFaced"), 0),
                    "IP": ip,
                    "H": safe_int(pitching.get("hits"), 0),
                    "ER": er,
                    "R": safe_int(pitching.get("runs"), 0),
                    "HR": safe_int(pitching.get("homeRuns"), 0),
                    "SO": safe_int(pitching.get("strikeOuts"), 0),
                    "BB": safe_int(pitching.get("baseOnBalls"), 0),
                    "IBB": safe_int(pitching.get("intentionalWalks"), 0),
                    "HBP": safe_int(pitching.get("hitByPitch"), 0),
                    "WP": safe_int(pitching.get("wildPitches"), 0),
                    "BK": safe_int(pitching.get("balks"), 0),
                }
                pitching_log["QS"] = 1 if (gs > 0 and ip >= 6 and er <= 3) else 0
                pitching_logs.append(pitching_log)

    return batting_logs, pitching_logs


# -----------------------------
# Parsing: PBP JSON -> pitch-by-pitch dataframe
# -----------------------------
def get_pbp_from_json(game_info_dict: Dict[str, Any], pbp_json: Dict[str, Any], box_json: Dict[str, Any] | None = None) -> pd.DataFrame:
    game_pk = game_info_dict.get("game_id")
    game_date = game_info_dict.get("date")
    venue_id = game_info_dict.get("venue_id")
    venue_name = game_info_dict.get("venue_name")
    league_id = game_info_dict.get("league_id")
    game_type = game_info_dict.get("game_type")

    # If box_json not passed, we still can attempt to derive teams/league from pbp,
    # but box_json is better. We’ll accept either.
    lgname = None
    away_team = None
    away_team_id = None
    home_team = None
    home_team_id = None

    if box_json:
        try:
            lgname = box_json.get("teams", {}).get("away", {}).get("team", {}).get("league", {}).get("name")
            away_team = box_json.get("teams", {}).get("away", {}).get("team", {}).get("name")
            away_team_id = box_json.get("teams", {}).get("away", {}).get("team", {}).get("id")
            home_team = box_json.get("teams", {}).get("home", {}).get("team", {}).get("name")
            home_team_id = box_json.get("teams", {}).get("home", {}).get("team", {}).get("id")
        except Exception:
            pass

    jsonstr = str(pbp_json)
    statcastflag = "Y" if ("startSpeed" in jsonstr) else "N"

    allplays = pbp_json.get("allPlays", []) or []

    frames: List[pd.DataFrame] = []
    for currplay in allplays:
        inningtopbot = (currplay.get("about") or {}).get("halfInning")
        inning = (currplay.get("about") or {}).get("inning")
        at_bat_number = safe_int(((currplay.get("about") or {}).get("atBatIndex")), 0) + 1

        result = currplay.get("result") or {}
        currplay_type = result.get("type")
        currplay_res = result.get("eventType")
        currplay_descrip = result.get("description")
        currplay_rbi = result.get("rbi")
        currplay_awayscore = result.get("awayscore")
        currplay_homescore = result.get("homescore")
        currplay_isout = result.get("isOut")

        playdata = currplay.get("playEvents", []) or []
        matchup = currplay.get("matchup") or {}
        batter = matchup.get("batter") or {}
        pitcher = matchup.get("pitcher") or {}
        bat_side = matchup.get("batSide") or {}
        pitch_hand = matchup.get("pitchHand") or {}

        bid = batter.get("id")
        bname = batter.get("fullName")
        bstand = bat_side.get("code")
        pid = pitcher.get("id")
        pname = pitcher.get("fullName")
        pthrows = pitch_hand.get("code")

        for pitch_event in playdata:
            pdetails = pitch_event.get("details") or {}
            checkadvise = pdetails.get("event")
            pitch_number = pitch_event.get("pitchNumber")

            # Skip advisories/non-pitches
            if checkadvise is not None:
                continue

            try:
                description = (pdetails.get("call") or {}).get("description")
            except Exception:
                description = None

            inplay = pdetails.get("isInPlay")
            isstrike = pdetails.get("isStrike")
            isball = pdetails.get("isBall")

            try:
                pitchname = (pdetails.get("type") or {}).get("description")
                pitchtype = (pdetails.get("type") or {}).get("code")
            except Exception:
                pitchname = None
                pitchtype = None

            count = pitch_event.get("count") or {}
            ballcount = count.get("balls")
            strikecount = count.get("strikes")

            pitchData = pitch_event.get("pitchData") or {}
            coords = pitchData.get("coordinates") or {}
            breaks = pitchData.get("breaks") or {}

            plate_x = coords.get("x")
            plate_y = coords.get("y")

            startspeed = pitchData.get("startSpeed")
            endspeed = pitchData.get("endspeed")

            kzonetop = pitchData.get("strikeZoneTop")
            kzonebot = pitchData.get("strikeZoneBottom")
            kzonewidth = pitchData.get("strikeZoneWidth")
            kzonedepth = pitchData.get("strikeZoneDepth")

            ay = coords.get("aY")
            ax = coords.get("aX")
            pfxx = coords.get("pfxX")
            pfxz = coords.get("pfxZ")
            px = coords.get("pX")
            pz = coords.get("pZ")

            breakangle = breaks.get("breakAngle")
            breaklength = breaks.get("breakLength")
            break_y = breaks.get("breakY")

            zone = pitchData.get("zone")

            hitdata = pitch_event.get("hitData") or {}
            launchspeed = hitdata.get("launchSpeed")
            launchangle = hitdata.get("launchAngle")
            bb_type = hitdata.get("trajectory")
            hardness = hitdata.get("hardness")
            location = hitdata.get("location")

            hcoords = (hitdata.get("coordinates") or {})
            coord_x = hcoords.get("coordX")
            coord_y = hcoords.get("coordY")

            row = {
                "StatcastGame": statcastflag,
                "game_pk": game_pk,
                "game_date": game_date,
                "game_type": game_type,
                "venue": venue_name,
                "venue_id": venue_id,
                "league_id": league_id,
                "league": lgname,
                "level": levdict.get(lgname),
                "away_team": away_team,
                "away_team_id": away_team_id,
                "home_team": home_team,
                "home_team_id": home_team_id,
                "player_name": pname,
                "pitcher": pid,
                "BatterName": bname,
                "batter": bid,
                "stand": bstand,
                "p_throws": pthrows,
                "inning_top_bot": inningtopbot,
                "plate_x": plate_x,
                "plate_y": plate_y,
                "inning": inning,
                "at_bat_number": at_bat_number,
                "pitch_number": pitch_number,
                "description": description,
                "play_type": currplay_type,
                "play_res": currplay_res,
                "play_desc": currplay_descrip,
                "rbi": currplay_rbi,
                "away_team_score": currplay_awayscore,
                "home_team_score": currplay_homescore,
                "isOut": currplay_isout,
                "isInPlay": inplay,
                "IsStrike": isstrike,
                "IsBall": isball,
                "pitch_name": pitchname,
                "pitch_type": pitchtype,
                "balls": ballcount,
                "strikes": strikecount,
                "release_speed": startspeed,
                "end_pitch_speed": endspeed,
                "zone_top": kzonetop,
                "zone_bot": kzonebot,
                "zone_width": kzonewidth,
                "zone_depth": kzonedepth,
                "ay": ay,
                "ax": ax,
                "pfx_x": pfxx,
                "pfx_z": pfxz,
                "px": px,
                "pz": pz,
                "break_angle": breakangle,
                "break_length": breaklength,
                "break_y": break_y,
                "zone": zone,
                "launch_speed": launchspeed,
                "launch_angle": launchangle,
                "bb_type": bb_type,
                "hit_location": location,
                "hit_coord_x": coord_x,
                "hit_coord_y": coord_y,
            }

            frames.append(pd.DataFrame(row, index=[0]))

    if not frames:
        return pd.DataFrame()

    gamepbp = pd.concat(frames, ignore_index=True)
    return gamepbp


# -----------------------------
# Your original "addons" logic (kept, but lightly hardened)
# -----------------------------
def savAddOns(savdata: pd.DataFrame) -> pd.DataFrame:
    if savdata.empty:
        return savdata

    pdf = savdata.copy()

    pdf["away_team_aff_id"] = pdf["away_team_id"].map(affdict)
    pdf["away_team_aff"] = pdf["away_team_aff_id"].map(affdict_abbrevs)
    pdf["home_team_aff_id"] = pdf["home_team_id"].map(affdict)
    pdf["home_team_aff"] = pdf["home_team_aff_id"].map(affdict_abbrevs)

    pdf["IsWalk"] = np.where(pdf["balls"] == 4, 1, 0)
    pdf["IsStrikeout"] = np.where(pdf["strikes"] == 3, 1, 0)
    pdf["BallInPlay"] = np.where(pdf["isInPlay"] == 1, 1, 0)
    pdf["IsHBP"] = np.where(pdf["description"] == "Hit By Pitch", 1, 0)
    pdf["PA_flag"] = np.where(
        (pdf["balls"] == 4) | (pdf["strikes"] == 3) | (pdf["BallInPlay"] == 1) | (pdf["IsHBP"] == 1),
        1,
        0,
    )

    pdf["IsHomer"] = np.where((pdf["play_res"] == "home_run") & (pdf["PA_flag"] == 1), 1, 0)

    pitchthrownlist = [
        "In play, out(s)",
        "Swinging Strike",
        "Ball",
        "Foul",
        "In play, no out",
        "Called Strike",
        "Foul Tip",
        "In play, run(s)",
        "Hit By Pitch",
        "Ball In Dirt",
        "Pitchout",
        "Swinging Strike (Blocked)",
        "Foul Bunt",
        "Missed Bunt",
        "Foul Pitchout",
        "Intent Ball",
        "Swinging Pitchout",
    ]
    pdf["PitchesThrown"] = np.where(pdf["description"].isin(pitchthrownlist), 1, 0)

    map_pitchnames = {"Two-Seam Fastball": "Sinker", "Slow Curve": "Curveball", "Knuckle Curve": "Curveball"}
    pdf["pitch_name"] = pdf["pitch_name"].replace(map_pitchnames)

    swstrlist = ["Swinging Strike", "Foul Tip", "Swinging Strike (Blocked)", "Missed Bunt"]
    cslist = ["Called Strike"]
    contlist = ["Foul", "In play, no out", "In play, out(s)", "Foul Pitchout", "In play, run(s)"]
    swinglist = [
        "Swinging Strike",
        "Foul",
        "In play, no out",
        "In play, out(s)",
        "In play, run(s)",
        "Swinging Strike (Blocked)",
        "Foul Pitchout",
    ]
    hitlist = ["single", "double", "triple", "home_run"]

    isstrikelist = [
        "Swinging Strike",
        "Foul",
        "Called Strike",
        "Foul Tip",
        "Swinging Strike (Blocked)",
        "Automatic Strike - Batter Pitch Timer Violation",
        "Foul Bunt",
        "Automatic Strike - Batter Timeout Violation",
        "Missed Bunt",
        "Automatic Strike",
        "Foul Pitchout",
        "Swinging Pitchout",
    ]
    isballlist = [
        "Ball",
        "Hit By Pitch",
        "Automatic Ball - Pitcher Pitch Timer Violation",
        "Ball In Dirt",
        "Pitchout",
        "Automatic Ball - Intentional",
        "Automatic Ball",
        "Automatic Ball - Defensive Shift Violation",
        "Automatic Ball - Catcher Pitch Timer Violation",
        "Intent Ball",
    ]

    pdf["IsStrike"] = np.where(pdf["description"].isin(isstrikelist), 1, 0)
    pdf["IsBall"] = np.where(pdf["description"].isin(isballlist), 1, 0)

    pdf["BatterTeam"] = np.where(pdf["inning_top_bot"] == "bottom", pdf["home_team"], pdf["away_team"])
    pdf["PitcherTeam"] = np.where(pdf["inning_top_bot"] == "bottom", pdf["away_team"], pdf["home_team"])

    pdf["BatterTeam_aff"] = np.where(pdf["inning_top_bot"] == "bottom", pdf["home_team_aff"], pdf["away_team_aff"])
    pdf["PitcherTeam_aff"] = np.where(pdf["inning_top_bot"] == "bottom", pdf["away_team_aff"], pdf["home_team_aff"])

    pdf["IsBIP"] = pdf["BallInPlay"]
    pdf["PA"] = pdf["PA_flag"]
    pdf["IsHit"] = np.where((pdf["PA"] == 1) & (pdf["play_res"].isin(hitlist)), 1, 0)

    pdf["IsSwStr"] = np.where(pdf["description"].isin(swstrlist), 1, 0)
    pdf["IsCalledStr"] = np.where(pdf["description"].isin(cslist), 1, 0)
    pdf["ContactMade"] = np.where(pdf["description"].isin(contlist), 1, 0)
    pdf["SwungOn"] = np.where(pdf["description"].isin(swinglist), 1, 0)

    pdf["IsGB"] = np.where(pdf["bb_type"] == "ground_ball", 1, 0)
    pdf["IsFB"] = np.where(pdf["bb_type"] == "fly_ball", 1, 0)
    pdf["IsLD"] = np.where(pdf["bb_type"] == "line_drive", 1, 0)
    pdf["IsPU"] = np.where(pdf["bb_type"] == "popup", 1, 0)

    pdf["InZone"] = np.where(pdf["zone"] < 10, 1, 0)
    pdf["OutZone"] = np.where(pdf["zone"] > 9, 1, 0)
    pdf["IsChase"] = np.where(((pdf["SwungOn"] == 1) & (pdf["InZone"] == 0)), 1, 0)
    pdf["IsZoneSwing"] = np.where(((pdf["SwungOn"] == 1) & (pdf["InZone"] == 1)), 1, 0)
    pdf["IsZoneContact"] = np.where(((pdf["ContactMade"] == 1) & (pdf["InZone"] == 1)), 1, 0)

    pdf["IsSingle"] = np.where((pdf["play_res"] == "single") & (pdf["PA_flag"] == 1), 1, 0)
    pdf["IsDouble"] = np.where((pdf["play_res"] == "double") & (pdf["PA_flag"] == 1), 1, 0)
    pdf["IsTriple"] = np.where((pdf["play_res"] == "triple") & (pdf["PA_flag"] == 1), 1, 0)

    ablist = [
        "field_out",
        "double",
        "strikeout",
        "single",
        "grounded_into_double_play",
        "home_run",
        "fielders_choice",
        "force_out",
        "double_play",
        "triple",
        "field_error",
        "fielders_choice_out",
        "strikeout_double_play",
        "other_out",
        "sac_fly_double_play",
        "triple_play",
    ]
    pdf["AB"] = np.where((pdf["play_res"].isin(ablist)) & (pdf["PA_flag"] == 1), 1, 0)

    pdf["launch_angle"] = pdf["launch_angle"].replace([None], np.nan)
    pdf["launch_speed"] = pdf["launch_speed"].replace([None], np.nan)

    pdf["launch_angle_round"] = round(pdf["launch_angle"], 0)
    pdf["launch_speed_round"] = round(pdf["launch_speed"], 0)

    pdf = pd.merge(pdf, lsaclass, how="left", on=["launch_speed_round", "launch_angle_round"])
    pdf["launch_speed_angle"] = np.where(pdf["launch_speed_round"] < 60, 1, pdf["launch_speed_angle"])
    pdf["launch_speed_angle"] = np.where((pdf["launch_speed_angle"].isna()) & (pdf["launch_speed"] > 1), 1, pdf["launch_speed_angle"])

    pdf["IsBrl"] = np.where(pdf["launch_speed_angle"] == 6, 1, 0)
    pdf["IsSolid"] = np.where(pdf["launch_speed_angle"] == 5, 1, 0)
    pdf["IsFlare"] = np.where(pdf["launch_speed_angle"] == 4, 1, 0)
    pdf["IsUnder"] = np.where(pdf["launch_speed_angle"] == 3, 1, 0)
    pdf["IsTopped"] = np.where(pdf["launch_speed_angle"] == 2, 1, 0)
    pdf["IsWeak"] = np.where(pdf["launch_speed_angle"] == 1, 1, 0)

    # Zone calc (your original)
    pdf["IsCalledStr"] = np.where(pdf["description"] == "Called Strike", 1, 0)
    pdf["zone_bot2"] = pdf["zone_bot"] * 100
    pdf["zone_top2"] = pdf["zone_top"] * 100
    pdf["inzone_y"] = np.where((pdf["plate_y"] >= pdf["zone_bot2"]) & (pdf["plate_y"] <= pdf["zone_top2"]), 1, 0)
    pdf["inzone_x"] = np.where((pdf["plate_x"] >= 70) & (pdf["plate_x"] <= 140), 1, 0)
    pdf["InZone2"] = np.where((pdf["inzone_y"] == 1) & (pdf["inzone_x"] == 1), 1, 0)
    pdf["OutZone2"] = np.where(pdf["InZone2"] == 1, 0, 1)
    pdf["IsZoneSwing2"] = np.where((pdf["InZone2"] == 1) & (pdf["SwungOn"] == 1), 1, 0)
    pdf["IsChase2"] = np.where((pdf["OutZone2"] == 1) & (pdf["SwungOn"] == 1), 1, 0)
    pdf["IsZoneContact2"] = np.where((pdf["IsZoneSwing2"] == 1) & (pdf["ContactMade"] == 1), 1, 0)

    # Dedupes / name collisions
    dupes_hitter_df = pdf.groupby(["BatterName", "batter"], as_index=False)["AB"].sum()
    hitter_dupes = dupes_hitter_df.groupby("BatterName", as_index=False)["batter"].count().sort_values(by="batter", ascending=False)
    hitter_dupes = hitter_dupes[hitter_dupes["batter"] > 1]
    hitter_dupes_list = list(hitter_dupes["BatterName"]) if "BatterName" in hitter_dupes.columns else []
    if hitter_dupes_list:
        pdf["BatterName"] = np.where(
            pdf["BatterName"].isin(hitter_dupes_list),
            pdf["BatterName"] + " - " + pdf["batter"].astype("Int64").astype(str),
            pdf["BatterName"],
        )

    dupes_pitcher_df = pdf.groupby(["player_name", "pitcher"], as_index=False)["PitchesThrown"].sum()
    pitcher_dupes = dupes_pitcher_df.groupby("player_name", as_index=False)["pitcher"].count().sort_values(by="pitcher", ascending=False)
    pitcher_dupes = pitcher_dupes[pitcher_dupes["pitcher"] > 1]
    pitcher_dupes_list = list(pitcher_dupes["player_name"]) if "player_name" in pitcher_dupes.columns else []
    if pitcher_dupes_list:
        pdf["player_name"] = np.where(
            pdf["player_name"].isin(pitcher_dupes_list),
            pdf["player_name"] + " - " + pdf["pitcher"].astype("Int64").astype(str),
            pdf["player_name"],
        )

    pdf = dropUnnamed(pdf)

    try:
        pdf["game_date"] = pd.to_datetime(pdf["game_date"])
    except Exception:
        pass

    # Drop duplicate pitch rows
    pdf = pdf.drop_duplicates(subset=["game_pk", "pitcher", "batter", "inning", "at_bat_number", "pitch_number"])

    return pdf


# -----------------------------
# Boxscore display helpers (kept very close to your original)
# -----------------------------
def getBoxDetails(game_box: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]):
    hitbox_json = game_box[0]
    pitbox_json = game_box[1]

    hitbox = pd.DataFrame(hitbox_json) if hitbox_json else pd.DataFrame()
    pitbox = pd.DataFrame(pitbox_json) if pitbox_json else pd.DataFrame()

    if hitbox.empty or pitbox.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    hitbox["1B"] = hitbox["H"] - hitbox["2B"] - hitbox["3B"] - hitbox["HR"]
    hitbox = hitbox[["Player", "player_id", "batting_order", "Team", "home_team", "AB", "R", "H", "1B", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "HBP"]]
    hitbox["Team"] = hitbox["Team"].replace(teamnamedict)
    hitbox["home_team"] = hitbox["home_team"].replace(teamnamedict)

    pitbox = pitbox[["Player", "player_id", "Team", "home_team", "G", "GS", "IP", "H", "ER", "R", "HR", "SO", "BB", "IBB", "HBP", "QS", "W"]]
    pitbox["Team"] = pitbox["Team"].replace(teamnamedict)
    pitbox["home_team"] = pitbox["home_team"].replace(teamnamedict)

    hitbox["DKPts"] = (hitbox["1B"] * 3) + (hitbox["2B"] * 5) + (hitbox["3B"] * 8) + (hitbox["HR"] * 10) + (hitbox["SB"] * 5) + (hitbox["BB"] * 2) + (hitbox["HBP"] * 2) + (hitbox["R"] * 2) + (hitbox["RBI"] * 2)
    pitbox["DKPts"] = (pitbox["IP"] * 2.25) + (pitbox["SO"] * 2) + (pitbox["W"] * 4) + (pitbox["ER"] * -2) + (pitbox["H"] * -0.6) + (pitbox["BB"] * -0.6)

    pitbox["Line"] = pitbox["IP"].astype(str) + "IP " + pitbox["H"].astype(str) + "H " + pitbox["ER"].astype(str) + "ER " + pitbox["SO"].astype(str) + "K " + pitbox["BB"].astype(str) + "BB"
    linebox = pitbox[["Player", "Team", "GS", "Line", "DKPts"]]
    linebox.columns = ["Pitcher", "Team", "GS", "Line", "DKPts"]

    show_hitbox = hitbox[["Player", "Team", "H", "R", "HR", "RBI", "SB", "2B", "3B", "SO", "BB", "DKPts"]]

    # Scoreboard (simple)
    teams = hitbox["Team"].unique()
    if len(teams) < 2:
        return show_hitbox, linebox, pd.DataFrame()

    this_mu = {teams[0]: teams[1], teams[1]: teams[0]}
    this_hometeams = dict(zip(hitbox.Team, hitbox.home_team))

    home_team = list(this_hometeams.values())[0]
    road_team = [t for t in teams if t != home_team][0]

    team_ip = pitbox.groupby("Team", as_index=False)["IP"].sum()
    curr_inning = int(np.min(team_ip["IP"]) + 1)
    inningprint = "F" if curr_inning >= 9 else str(curr_inning)

    team_runs = hitbox.groupby("Team", as_index=False)["R"].sum().sort_values(by="R", ascending=False)
    team_runs_dict = dict(zip(team_runs.Team, team_runs.R))

    game_dis = f"{road_team} @ {home_team}"
    score = f"{road_team} ({team_runs_dict.get(road_team,0)}) @ {home_team} ({team_runs_dict.get(home_team,0)})"
    this_score = pd.DataFrame({"Game": game_dis, "Score": score, "Inn": inningprint}, index=[0])

    return show_hitbox, linebox, this_score


# -----------------------------
# Derived tables
# -----------------------------
def getPData(livedb: pd.DataFrame, all_pitboxes: pd.DataFrame, cplist: List[str]) -> pd.DataFrame:
    pdata = livedb.groupby(["player_name", "pitcher", "PitcherTeam_aff"], as_index=False)[
        ["PitchesThrown", "IsStrike", "IsBall", "IsBIP", "IsHit", "IsHomer", "IsSwStr", "IsGB", "IsLD", "IsFB", "IsBrl", "PA_flag", "DP", "IsStrikeout", "IsWalk"]
    ].sum()

    pdata["Outs"] = pdata["PA_flag"] - pdata["IsHit"] - pdata["IsWalk"] + pdata["DP"]
    pdata["IP"] = round((pdata["Outs"] / 3), 2)

    pdata["SwStr%"] = round(pdata["IsSwStr"] / pdata["PitchesThrown"].replace(0, np.nan), 3)
    pdata["Strike%"] = round(pdata["IsStrike"] / pdata["PitchesThrown"].replace(0, np.nan), 3)
    pdata["Ball%"] = round(pdata["IsBall"] / pdata["PitchesThrown"].replace(0, np.nan), 3)

    pdata["GB%"] = round(pdata["IsGB"] / pdata["IsBIP"].replace(0, np.nan), 3)
    pdata["FB%"] = round(pdata["IsFB"] / pdata["IsBIP"].replace(0, np.nan), 3)
    pdata["LD%"] = round(pdata["IsLD"] / pdata["IsBIP"].replace(0, np.nan), 3)
    pdata["Brl%"] = round(pdata["IsBrl"] / pdata["IsBIP"].replace(0, np.nan), 3)

    pdata = pdata.sort_values(by="IsSwStr", ascending=False)
    pdata = pdata[["player_name", "pitcher", "PitcherTeam_aff", "PA_flag", "IP", "IsStrikeout", "IsWalk", "IsHit", "IsHomer", "PitchesThrown", "IsSwStr", "IsStrike", "SwStr%", "Strike%", "Ball%", "GB%", "LD%", "FB%", "Brl%"]]
    pdata.columns = ["Pitcher", "ID", "Team", "TBF", "IP", "SO", "BB", "H", "HR", "PC", "Whiffs", "Strikes", "SwStr%", "Strike%", "Ball%", "GB%", "LD%", "FB%", "Brl%"]

    pdata["Current Pitcher?"] = np.where(pdata["Pitcher"].isin(cplist), "Y", "N")

    showdf = pdata.copy()
    if not all_pitboxes.empty and "Pitcher" in all_pitboxes.columns:
        showdf = pd.merge(showdf, all_pitboxes[["Pitcher", "Line"]], how="left", on="Pitcher")

    pdatadf = showdf[["Pitcher", "Team", "Line", "PC", "SO", "BB", "Whiffs", "SwStr%", "Strike%", "Ball%", "Current Pitcher?"]].sort_values(by=["Whiffs"], ascending=False)
    return pdatadf


def getPMixData(livedb: pd.DataFrame, cplist: List[str]) -> pd.DataFrame:
    velodata = livedb.groupby(["player_name", "PitcherTeam_aff", "pitch_type"], as_index=False)[["release_speed", "pfx_x", "pfx_z"]].mean()
    velodata = velodata.round(1)
    velodata.columns = ["Pitcher", "Team", "Pitch", "Velo", "Horiz", "Vert"]

    mixdata = livedb.groupby(["player_name", "pitcher", "PitcherTeam_aff", "pitch_type"], as_index=False)[
        ["PitchesThrown", "IsStrike", "IsBall", "IsBIP", "IsHit", "IsHomer", "IsSwStr", "IsGB", "IsLD", "IsFB", "IsBrl", "PA_flag", "DP", "IsStrikeout", "IsWalk"]
    ].sum()

    mixdata["SwStr%"] = round(mixdata["IsSwStr"] / mixdata["PitchesThrown"].replace(0, np.nan), 3)
    mixdata["Strike%"] = round(mixdata["IsStrike"] / mixdata["PitchesThrown"].replace(0, np.nan), 3)
    mixdata["Ball%"] = round(mixdata["IsBall"] / mixdata["PitchesThrown"].replace(0, np.nan), 3)
    mixdata["Brl%"] = round(mixdata["IsBrl"] / mixdata["IsBIP"].replace(0, np.nan), 3)

    mixdata = mixdata[["player_name", "PitcherTeam_aff", "pitch_type", "PitchesThrown", "IsSwStr", "SwStr%", "Strike%", "Ball%", "Brl%"]]
    mixdata.columns = ["Pitcher", "Team", "Pitch", "PC", "Whiffs", "SwStr%", "Strike%", "Ball%", "Brl%"]

    mixdata = pd.merge(mixdata, velodata, on=["Pitcher", "Team", "Pitch"], how="left")
    mixdata = mixdata[["Pitcher", "Team", "Pitch", "PC", "Velo", "Whiffs", "SwStr%", "Strike%", "Ball%", "Brl%", "Horiz", "Vert"]]

    pm = pmove25[["Pitcher", "Pitch", "Avg Velo", "Avg Horiz", "Avg Vert"]].copy()
    mixdata = pd.merge(mixdata, pm, on=["Pitcher", "Pitch"], how="left")

    mixdata["Avg Velo"] = round(mixdata["Avg Velo"], 1)
    mixdata["Avg Horiz"] = round(mixdata["Avg Horiz"], 1)
    mixdata["Avg Vert"] = round(mixdata["Avg Vert"], 1)

    mixdata["Velo Diff"] = mixdata["Velo"] - mixdata["Avg Velo"]
    mixdata["Horiz Diff"] = mixdata["Horiz"] - mixdata["Avg Horiz"]
    mixdata["Vert Diff"] = mixdata["Vert"] - mixdata["Avg Vert"]

    mixdata = mixdata.drop(["Avg Velo", "Avg Horiz", "Avg Vert"], axis=1)
    return mixdata


# -----------------------------
# UI rendering (simple + fast)
# -----------------------------

def with_percent_format(df: pd.DataFrame, percent_cols: List[str]) -> Tuple[pd.DataFrame, dict]:
    """
    Streamlit percent formatting helper:
    - assumes percent columns are stored as fractions (e.g., 0.123)
    - displays them as 12.3% (0.0% style)
    """
    if df is None or df.empty:
        return df, {}

    show = df.copy()
    cfg: dict = {}

    for c in percent_cols:
        if c in show.columns:
            # convert fraction -> percent for display only
            show[c] = pd.to_numeric(show[c], errors="coerce").fillna(0) * 100.0
            cfg[c] = st.column_config.NumberColumn(c, format="%.1f%%")

    return show, cfg

def _render_game_detail_panel(
    game_key: str,
    hitboxes: pd.DataFrame,
    pitboxes: pd.DataFrame,
):
    """Renders an inline detail panel for the selected game card."""
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #dce3f0;border-radius:14px;
                    padding:20px 24px;margin:12px 0 20px;
                    box-shadow:0 4px 20px rgba(37,99,235,0.10);
                    border-top:4px solid #2563eb;">
            <div style="font-size:1.1rem;font-weight:800;color:#1a2744;margin-bottom:2px;">
                📋 {game_key} — Game Detail
            </div>
            <div style="font-size:0.75rem;color:#94a3b8;">Click the same card again to close</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filter hitboxes and pitboxes to just teams in this game
    try:
        away_t, home_t = game_key.split(" @ ")
        away_t = away_t.strip()
        home_t = home_t.strip()
        game_teams = {away_t, home_t}
    except Exception:
        st.warning("Could not parse game teams.")
        return

    game_hits = hitboxes[hitboxes["Team"].isin(game_teams)].copy() if not hitboxes.empty else pd.DataFrame()
    game_pits = pitboxes[pitboxes["Team"].isin(game_teams)].copy() if not pitboxes.empty else pd.DataFrame()

    tab_hit, tab_pit = st.tabs([f"🏏 Hitting — {game_key}", f"⚾ Pitching — {game_key}"])

    with tab_hit:
        if game_hits.empty:
            st.info("No hitting data for this game yet.")
        else:
            # Split by team
            c1, c2 = st.columns(2)
            for col, team in zip([c1, c2], [away_t, home_t]):
                team_hits = game_hits[game_hits["Team"] == team].copy()
                if team_hits.empty:
                    continue
                team_hits = team_hits.sort_values("DKPts", ascending=False)
                show_cols = [c for c in ["Player", "DKPts", "H", "R", "HR", "RBI", "SB", "2B", "3B", "BB", "SO"] if c in team_hits.columns]
                col.markdown(
                    f'<div style="font-size:0.78rem;font-weight:700;color:#1a2744;text-transform:uppercase;'
                    f'letter-spacing:0.5px;margin-bottom:6px;border-left:3px solid #2563eb;padding-left:8px;">{team}</div>',
                    unsafe_allow_html=True,
                )
                col.dataframe(team_hits[show_cols], hide_index=True, use_container_width=True)

    with tab_pit:
        if game_pits.empty:
            st.info("No pitching data for this game yet.")
        else:
            c1, c2 = st.columns(2)
            for col, team in zip([c1, c2], [away_t, home_t]):
                team_pits = game_pits[game_pits["Team"] == team].copy()
                if team_pits.empty:
                    continue
                team_pits = team_pits.sort_values("DKPts", ascending=False)
                show_cols = [c for c in ["Pitcher", "Line", "DKPts", "GS"] if c in team_pits.columns]
                col.markdown(
                    f'<div style="font-size:0.78rem;font-weight:700;color:#1a2744;text-transform:uppercase;'
                    f'letter-spacing:0.5px;margin-bottom:6px;border-left:3px solid #2563eb;padding-left:8px;">{team}</div>',
                    unsafe_allow_html=True,
                )
                col.dataframe(team_pits[show_cols], hide_index=True, use_container_width=True)


def render_scores_and_leaders(
    scoreboard_df: pd.DataFrame,
    hitboxes: pd.DataFrame,
    pitboxes: pd.DataFrame,
    current_time: str,
    today_games: List[Dict[str, Any]],
):
    hide_finished = st.session_state.get("hide_finished", False)

    # --- Header ---
    live_count = sum(1 for g in today_games if g.get("game_status") == "I")
    final_count = sum(1 for g in today_games if g.get("game_status") == "F")
    sched_count = sum(1 for g in today_games if g.get("game_status") not in ("I", "F"))

    st.markdown(
        f"""
        <div class="tracker-header">
            <div>
                <h1>⚾ MLB Live Game Tracker</h1>
                <div class="sub">MLB Data Warehouse · Live Scores &amp; Leaders</div>
            </div>
            <div class="update-badge">🕐 Last update: {current_time} ET</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Quick-stat tiles ---
    total_hrs = int(hitboxes["HR"].sum()) if not hitboxes.empty and "HR" in hitboxes.columns else 0
    total_ks = int(pitboxes["SO"].sum()) if not pitboxes.empty and "SO" in pitboxes.columns else 0
    total_sb = int(hitboxes["SB"].sum()) if not hitboxes.empty and "SB" in hitboxes.columns else 0
    total_h = int(hitboxes["H"].sum()) if not hitboxes.empty and "H" in hitboxes.columns else 0

    tiles = [
        (live_count, "Live Games"), (final_count, "Final"), (sched_count, "Scheduled"),
        (total_hrs, "Home Runs"), (total_ks, "Strikeouts"), (total_sb, "Stolen Bases"), (total_h, "Hits"),
    ]
    tile_cols = st.columns(len(tiles))
    for col, (val, lbl) in zip(tile_cols, tiles):
        col.markdown(
            f"""<div style="background:#ffffff;border:1px solid #dce3f0;border-radius:10px;
                            padding:12px 8px;text-align:center;box-shadow:0 1px 5px rgba(0,0,0,0.05);">
                    <div style="font-size:1.6rem;font-weight:800;color:#2563eb;line-height:1.1;">{val}</div>
                    <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">{lbl}</div>
                </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # --- Scorecard grid ---
    st.markdown('<div class="section-title">Live Scores</div>', unsafe_allow_html=True)

    # Init selected game in session state
    if "selected_game" not in st.session_state:
        st.session_state["selected_game"] = None

    if not scoreboard_df.empty:
        # Build status map: matchup string -> game_status code
        status_map: Dict[str, str] = {}
        for g in today_games:
            away_abbr = teamnamedict.get(g.get("away_team", ""), g.get("away_team", ""))
            home_abbr = teamnamedict.get(g.get("home_team", ""), g.get("home_team", ""))
            matchup_key = f"{away_abbr} @ {home_abbr}"
            status_map[matchup_key] = g.get("game_status", "")

        # Build list of card dicts
        card_data = []
        for _, row in scoreboard_df.iterrows():
            game_key = str(row.get("Game", ""))
            status_code = status_map.get(game_key, "")
            is_final = status_code == "F"
            is_live = status_code == "I"

            if hide_finished and is_final:
                continue

            score_str = str(row.get("Score", ""))
            inning_str = str(row.get("Inn", ""))

            away_abbr_s, home_abbr_s, away_r, home_r = game_key, "—", 0, 0
            try:
                parts = score_str.split(" @ ")
                away_part = parts[0]
                home_part = parts[1]
                away_abbr_s = away_part.split(" (")[0]
                away_r = int(away_part.split("(")[1].rstrip(")"))
                home_abbr_s = home_part.split(" (")[0]
                home_r = int(home_part.split("(")[1].rstrip(")"))
            except Exception:
                pass

            inning_display = "Final" if inning_str == "F" else (f"Inn {inning_str}" if inning_str.isdigit() else inning_str)
            card_data.append({
                "game_key": game_key,
                "away": away_abbr_s, "away_r": away_r,
                "home": home_abbr_s, "home_r": home_r,
                "is_final": is_final, "is_live": is_live,
                "inning_display": inning_display,
            })

        if not card_data:
            st.markdown('<div class="alert-info">All games finished. Uncheck "Hide Finished Games" to see results.</div>', unsafe_allow_html=True)
        else:
            # CSS for the card select buttons
            st.markdown("""
                <style>
                /* Game card select buttons — compact, ghost style */
                div[data-testid="stButton"] > button[kind="secondary"] {
                    width: 100% !important;
                    border-radius: 0 0 10px 10px !important;
                    border: 1px solid #e2e8f0 !important;
                    border-top: none !important;
                    background: #f8fafc !important;
                    color: #64748b !important;
                    font-size: 0.72rem !important;
                    padding: 4px 8px !important;
                    margin-top: -6px !important;
                    box-shadow: none !important;
                }
                div[data-testid="stButton"] > button[kind="secondary"]:hover {
                    background: #eff6ff !important;
                    color: #2563eb !important;
                    border-color: #bfdbfe !important;
                }
                </style>
            """, unsafe_allow_html=True)

            CARDS_PER_ROW = 5
            for row_start in range(0, len(card_data), CARDS_PER_ROW):
                row_cards = card_data[row_start : row_start + CARDS_PER_ROW]
                cols = st.columns(CARDS_PER_ROW)
                for col, c in zip(cols, row_cards):
                    is_selected = st.session_state["selected_game"] == c["game_key"]
                    top_color = "#16a34a" if c["is_live"] else ("#94a3b8" if c["is_final"] else "#2563eb")
                    away_color = "#16a34a" if c["away_r"] > c["home_r"] else "#1a2744"
                    home_color = "#16a34a" if c["home_r"] > c["away_r"] else "#1a2744"
                    border_style = f"2px solid {top_color}" if is_selected else "1px solid #dce3f0"
                    shadow_style = f"0 0 0 3px {top_color}33, 0 4px 16px rgba(0,0,0,0.10)" if is_selected else "0 2px 8px rgba(0,0,0,0.06)"

                    if c["is_live"]:
                        status_html = '<span style="background:#dcfce7;color:#15803d;font-size:0.68rem;font-weight:700;padding:2px 8px;border-radius:20px;">● LIVE</span>'
                    elif c["is_final"]:
                        status_html = '<span style="background:#f1f5f9;color:#64748b;font-size:0.68rem;font-weight:700;padding:2px 8px;border-radius:20px;">FINAL</span>'
                    else:
                        status_html = f'<span style="background:#eff6ff;color:#2563eb;font-size:0.68rem;font-weight:600;padding:2px 8px;border-radius:20px;">{c["inning_display"]}</span>'

                    click_hint = ' <span style="font-size:0.6rem;color:#2563eb;margin-left:6px;">▼ open</span>' if is_selected else ' <span style="font-size:0.6rem;color:#cbd5e1;margin-left:6px;">▼ details</span>'

                    card_html = f"""
                    <div style="background:#ffffff;border:{border_style};border-radius:12px 12px 0 0;padding:14px 16px 12px;
                                box-shadow:{shadow_style};position:relative;overflow:hidden;">
                        <div style="position:absolute;top:0;left:0;right:0;height:4px;background:{top_color};border-radius:12px 12px 0 0;"></div>
                        <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;margin-top:2px;">{c["game_key"]}</div>
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                            <div style="text-align:center;">
                                <div style="font-size:1.0rem;font-weight:800;color:#1a2744;">{c["away"]}</div>
                                <div style="font-size:2.2rem;font-weight:900;color:{away_color};line-height:1;">{c["away_r"]}</div>
                            </div>
                            <div style="color:#cbd5e1;font-size:0.9rem;">vs</div>
                            <div style="text-align:center;">
                                <div style="font-size:1.0rem;font-weight:800;color:#1a2744;">{c["home"]}</div>
                                <div style="font-size:2.2rem;font-weight:900;color:{home_color};line-height:1;">{c["home_r"]}</div>
                            </div>
                        </div>
                        <div style="display:flex;align-items:center;">{status_html}{click_hint}</div>
                    </div>
                    """
                    col.markdown(card_html, unsafe_allow_html=True)
                    # Button sits flush below the card (border-radius on top = 0)
                    btn_label = "▲ Close" if is_selected else "▼ View Stats"
                    if col.button(btn_label, key=f"card_btn_{c['game_key']}", use_container_width=True):
                        if st.session_state["selected_game"] == c["game_key"]:
                            st.session_state["selected_game"] = None  # toggle off
                        else:
                            st.session_state["selected_game"] = c["game_key"]
                        st.rerun()

            # --- Game detail panel ---
            selected = st.session_state.get("selected_game")
            if selected:
                _render_game_detail_panel(selected, hitboxes, pitboxes)

    else:
        st.markdown('<div class="alert-info">No scoreboard data yet — check back once games begin.</div>', unsafe_allow_html=True)

    # --- Leaders section ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-title">Pitching Leaders (Starters)</div>', unsafe_allow_html=True)
        if not pitboxes.empty:
            pit_show = pitboxes.copy().sort_values(by="DKPts", ascending=False)
            pit_show = pit_show[pit_show["GS"] == 1]
            if hide_finished:
                live_teams = set()
                for g in today_games:
                    if g.get("game_status") == "I":
                        live_teams.add(teamnamedict.get(g.get("away_team", ""), g.get("away_team", "")))
                        live_teams.add(teamnamedict.get(g.get("home_team", ""), g.get("home_team", "")))
                if live_teams:
                    pit_show = pit_show[pit_show["Team"].isin(live_teams)]
            pit_show = pit_show[["Pitcher", "Team", "Line", "DKPts"]].head(30)
            st.dataframe(pit_show, hide_index=True, width=450, height=620)
        else:
            st.info("No boxscore pitching data yet.")

    with col2:
        st.markdown('<div class="section-title">Hitting Leaders</div>', unsafe_allow_html=True)
        if not hitboxes.empty:
            hit_show = hitboxes.copy().sort_values(by="DKPts", ascending=False)
            if hide_finished:
                live_teams = set()
                for g in today_games:
                    if g.get("game_status") == "I":
                        live_teams.add(teamnamedict.get(g.get("away_team", ""), g.get("away_team", "")))
                        live_teams.add(teamnamedict.get(g.get("home_team", ""), g.get("home_team", "")))
                if live_teams:
                    hit_show = hit_show[hit_show["Team"].isin(live_teams)]
            hit_show = hit_show[["Player", "Team", "DKPts", "H", "R", "HR", "RBI", "SB", "2B", "3B", "SO", "BB"]].head(60)
            st.dataframe(hit_show, hide_index=True, width=950, height=900)
        else:
            st.info("No boxscore hitting data yet.")


def render_pitcher_detail(p_data: pd.DataFrame, current_time: str):
    st.markdown(
        f"""
        <div class="tracker-header">
            <div><h1>🎯 Pitcher Detail</h1><div class="sub">Live pitch-by-pitch stats</div></div>
            <div class="update-badge">🕐 {current_time} ET</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Pitcher Detailed Stats</div>', unsafe_allow_html=True)
    if p_data.empty:
        st.info("No pitch-by-pitch data yet.")
        return

    hide_finished = st.session_state.get("hide_finished", False)
    show_df = p_data.copy()
    if hide_finished:
        show_df = show_df[show_df.get("Current Pitcher?", pd.Series(["Y"] * len(show_df))) == "Y"]

    col_f, col_s = st.columns([1, 3])
    with col_f:
        filter_cp = st.checkbox("Current pitchers only", value=hide_finished)
    with col_s:
        pass

    if filter_cp and "Current Pitcher?" in show_df.columns:
        show_df = show_df[show_df["Current Pitcher?"] == "Y"]

    p_show, p_cfg = with_percent_format(show_df, ["SwStr%", "Strike%", "Ball%"])
    st.dataframe(p_show, column_config=p_cfg, hide_index=True, width=1200, height=520)


def render_pitch_mix(pmix_data: pd.DataFrame, current_time: str):
    st.markdown(
        f"""
        <div class="tracker-header">
            <div><h1>🔀 Pitch Mix</h1><div class="sub">Pitch type breakdown with movement vs. season avg</div></div>
            <div class="update-badge">🕐 {current_time} ET</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Pitch Mix Detailed Stats</div>', unsafe_allow_html=True)

    if pmix_data.empty:
        st.info("No pitch-by-pitch data yet.")
        return

    pmix_data = pmix_data.fillna(0).sort_values(by=["Pitcher", "PC"], ascending=[True, False])
    pitcher_options = ["All Pitchers"] + sorted(pmix_data["Pitcher"].unique().tolist())
    selected_pitcher = st.selectbox("Select Pitcher", pitcher_options)

    view = pmix_data if selected_pitcher == "All Pitchers" else pmix_data[pmix_data["Pitcher"] == selected_pitcher]
    view_show, view_cfg = with_percent_format(view, ["SwStr%", "Strike%", "Ball%", "Brl%"])
    st.dataframe(view_show, column_config=view_cfg, hide_index=True, width=1200, height=760)


def render_exit_velos(hrs: pd.DataFrame, evs: pd.DataFrame, current_time: str):
    st.markdown(
        f"""
        <div class="tracker-header">
            <div><h1>💥 Exit Velos</h1><div class="sub">Statcast batted ball data</div></div>
            <div class="update-badge">🕐 {current_time} ET</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">⚾ Home Runs</div>', unsafe_allow_html=True)
        if hrs.empty:
            st.info("No home runs recorded yet.")
        else:
            st.dataframe(hrs, hide_index=True, width=620)

    with col2:
        ev_threshold = st.slider("Min EV (mph)", min_value=80, max_value=115, value=100, step=1)
        st.markdown(f'<div class="section-title">🔥 Hard Hit Balls (EV ≥ {ev_threshold})</div>', unsafe_allow_html=True)
        if evs.empty:
            st.info("No batted ball data yet.")
        else:
            filtered_evs = evs[evs["EV"] >= ev_threshold]
            st.markdown(f"**{len(filtered_evs)} batted balls** at or above {ev_threshold} mph")
            st.dataframe(filtered_evs, hide_index=True, width=620, height=560)


# -----------------------------
# Snapshot builder (parallel + cached per request)
# -----------------------------
#@st.cache_data(ttl=30, show_spinner=False)
#string: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
@st.cache_data(ttl=30, show_spinner=False)
def build_snapshot(date_string: str, include_pbp: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    today_games = get_live_games(date_string)

    # Only games in progress or finished (you were doing I and F)
    target_games = [g for g in today_games if g.get("game_status") in ("I", "F")]
    if not target_games:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    scoreboard_frames: List[pd.DataFrame] = []
    hit_frames: List[pd.DataFrame] = []
    pit_frames: List[pd.DataFrame] = []
    pbp_frames: List[pd.DataFrame] = []

    max_workers = min(12, max(4, len(target_games) * 2))

    # Pre-fetch boxscores for all target games in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        box_futs = {ex.submit(fetch_boxscore, g["game_id"]): g for g in target_games}
        #pbp_futs = {ex.submit(fetch_pbp, g["game_id"]): g for g in target_games if g.get("game_status") == "I"}
        pbp_futs = {}
        if include_pbp:
            # For old dates you want PBP for finals too
            pbp_futs = {ex.submit(fetch_pbp, g["game_id"]): g for g in target_games if g.get("game_status") in ("I", "F")}


        box_by_gameid: Dict[int, Dict[str, Any]] = {}

        for fut in as_completed(box_futs):
            g = box_futs[fut]
            gid = g["game_id"]
            try:
                box_json = fut.result()
                box_by_gameid[gid] = box_json
            except Exception:
                continue

        # Process boxscore results
        for g in target_games:
            gid = g["game_id"]
            box_json = box_by_gameid.get(gid)
            if not box_json:
                continue

            bat_logs, pit_logs = get_game_logs_from_boxjson(g, box_json)
            hitbox, pitbox, game_score = getBoxDetails((bat_logs, pit_logs))

            if not hitbox.empty:
                hit_frames.append(hitbox)
            if not pitbox.empty:
                pit_frames.append(pitbox)
            if not game_score.empty:
                scoreboard_frames.append(game_score)

        # Process pbp results (only for live games)
        for fut in as_completed(pbp_futs):
            g = pbp_futs[fut]
            gid = g["game_id"]
            try:
                pbp_json = fut.result()
            except Exception:
                continue

            box_json = box_by_gameid.get(gid)
            gamedb = get_pbp_from_json(g, pbp_json, box_json=box_json)
            if not gamedb.empty and gamedb.get("StatcastGame", pd.Series(["N"])).iloc[0] == "Y":
                gamedb["game_status"] = g.get("game_status")
                pbp_frames.append(gamedb)

    scoreboard_df = pd.concat(scoreboard_frames, ignore_index=True) if scoreboard_frames else pd.DataFrame()
    all_hitboxes = pd.concat(hit_frames, ignore_index=True) if hit_frames else pd.DataFrame()
    all_pitboxes = pd.concat(pit_frames, ignore_index=True) if pit_frames else pd.DataFrame()
    livedb = pd.concat(pbp_frames, ignore_index=True) if pbp_frames else pd.DataFrame()

    return scoreboard_df, all_hitboxes, all_pitboxes, livedb


def render_game_pace(livedb: pd.DataFrame, scoreboard_df: pd.DataFrame, current_time: str):
    """New page: game-level pace stats — pitches/PA, K%, BB%, HR/PA, etc."""
    st.markdown(
        f"""
        <div class="tracker-header">
            <div><h1>📈 Game Pace</h1><div class="sub">Per-game stats &amp; pace metrics</div></div>
            <div class="update-badge">🕐 {current_time} ET</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if livedb.empty:
        st.info("No play-by-play data available yet.")
        return

    hide_finished = st.session_state.get("hide_finished", False)

    pace = livedb.groupby("game_pk", as_index=False).agg(
        Away=("away_team_aff", "first"),
        Home=("home_team_aff", "first"),
        Pitches=("PitchesThrown", "sum"),
        PA=("PA_flag", "sum"),
        K=("IsStrikeout", "sum"),
        BB=("IsWalk", "sum"),
        HR=("IsHomer", "sum"),
        Hits=("IsHit", "sum"),
        Whiffs=("IsSwStr", "sum"),
        BIP=("IsBIP", "sum"),
        Brls=("IsBrl", "sum"),
        game_status=("game_status", "first"),
    )

    if hide_finished:
        pace = pace[pace["game_status"] != "F"]

    if pace.empty:
        st.info("No live games to display. Uncheck 'Hide Finished Games' to see all.")
        return

    pace["Matchup"] = pace["Away"] + " @ " + pace["Home"]
    pace["P/PA"] = (pace["Pitches"] / pace["PA"].replace(0, np.nan)).round(2)
    pace["K%"] = (pace["K"] / pace["PA"].replace(0, np.nan) * 100).round(1)
    pace["BB%"] = (pace["BB"] / pace["PA"].replace(0, np.nan) * 100).round(1)
    pace["SwStr%"] = (pace["Whiffs"] / pace["Pitches"].replace(0, np.nan) * 100).round(1)
    pace["Brl%"] = (pace["Brls"] / pace["BIP"].replace(0, np.nan) * 100).round(1)
    pace["Status"] = pace["game_status"].map({"I": "🟢 Live", "F": "⚫ Final"}).fillna("⏳ Sched")

    display_cols = ["Matchup", "Status", "PA", "Pitches", "P/PA", "K", "BB", "HR", "Hits", "K%", "BB%", "SwStr%", "Brl%"]
    pace_show = pace[display_cols].sort_values(by="Pitches", ascending=False)

    st.markdown('<div class="section-title">Per-Game Pace Stats</div>', unsafe_allow_html=True)
    st.dataframe(pace_show, hide_index=True, width=1200, height=450)

    # Quick leaderboard: most Ks this game
    st.markdown('<div class="section-title">Strikeout Leaders (Pitchers)</div>', unsafe_allow_html=True)
    k_leaders = livedb.groupby(["player_name", "PitcherTeam_aff", "game_status"], as_index=False)["IsStrikeout"].sum()
    if hide_finished:
        k_leaders = k_leaders[k_leaders["game_status"] != "F"]
    k_leaders = k_leaders.sort_values("IsStrikeout", ascending=False).head(15)
    k_leaders.columns = ["Pitcher", "Team", "Status", "Ks"]
    k_leaders["Status"] = k_leaders["Status"].map({"I": "🟢 Live", "F": "⚫ Final"}).fillna("⏳")
    st.dataframe(k_leaders, hide_index=True, width=500)


# -----------------------------
# Main app
# -----------------------------
def main():
    selected_page = sidebar_menu()

    # Refresh controls
    st.sidebar.markdown("---")
    refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 10, 120, 30, step=5)
    st.sidebar.caption("Shared cache across all users · TTL-based")

    # Optional autorefresh
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore
        st_autorefresh(interval=refresh_seconds * 1000, key="mlbdw_refresh_v2")
    except Exception:
        st.sidebar.warning("💡 Install streamlit-autorefresh for auto updates.")

    eastern = pytz.timezone("US/Eastern")
    now_eastern = datetime.now(eastern)
    default_date = now_eastern.date()

    chosen_date = st.sidebar.date_input(
        "Game date",
        value=default_date,
        max_value=default_date,
    )

    date_str = chosen_date.strftime("%Y-%m-%d")
    current_time = now_eastern.strftime("%I:%M %p")

    # Pull snapshot (cached across all users)
    needs_pbp = selected_page in ("🎯 Pitcher Detail", "🔀 Pitch Mix", "💥 Exit Velos", "📈 Game Pace")
    scoreboard_df, all_hitboxes, all_pitboxes, livedb = build_snapshot(date_str, include_pbp=needs_pbp)

    # Today's games for status checking / tile counts
    today_games = get_live_games(date_str)

    # If no PBP yet, show Scores & Leaders gracefully
    if livedb.empty:
        if selected_page == "📊 Scores & Leaders":
            render_scores_and_leaders(scoreboard_df, all_hitboxes, all_pitboxes, current_time, today_games)
            st.caption("No Statcast play-by-play detected yet. Refreshing automatically.")
        elif selected_page == "📈 Game Pace":
            render_game_pace(pd.DataFrame(), scoreboard_df, current_time)
        else:
            st.info("No Statcast pitch-by-pitch data available yet. (Try Scores & Leaders for boxscore-only.)")
        return

    # Build pitch-by-pitch derived tables
    livedb = savAddOns(livedb)
    livedb["play_desc"] = livedb["play_desc"].fillna("")
    livedb["DP"] = np.where((livedb["play_desc"].str.contains("double play")) & (livedb["PA_flag"] == 1), 1, 0)

    finished = livedb[livedb["game_status"] == "F"]
    finished_pitchers = set(finished["player_name"].unique())

    lastpitcher = livedb.sort_values(by=["inning", "at_bat_number", "pitch_number"])[["PitcherTeam_aff", "player_name"]]
    currentpitchers = lastpitcher.drop_duplicates(subset=["PitcherTeam_aff"], keep="last")
    cplist = [p for p in currentpitchers["player_name"].tolist() if p not in finished_pitchers]

    p_data = getPData(livedb, all_pitboxes, cplist)
    pmix_data = getPMixData(livedb, cplist)

    # HR / EV tables
    hrs = livedb[livedb["IsHomer"] == 1][["BatterName", "BatterTeam_aff", "player_name", "launch_speed", "play_desc"]].sort_values(by="launch_speed", ascending=False)
    if not hrs.empty:
        hrs.columns = ["Hitter", "Team", "Pitcher", "EV", "Description"]
    else:
        hrs = pd.DataFrame(columns=["Hitter", "Team", "Pitcher", "EV", "Description"])

    evs = livedb[["BatterName", "BatterTeam_aff", "player_name", "launch_speed", "play_desc"]].sort_values(by="launch_speed", ascending=False)
    if not evs.empty:
        evs.columns = ["Hitter", "Team", "Pitcher", "EV", "Description"]
    else:
        evs = pd.DataFrame(columns=["Hitter", "Team", "Pitcher", "EV", "Description"])

    # Render selected page
    if selected_page == "📊 Scores & Leaders":
        render_scores_and_leaders(scoreboard_df, all_hitboxes, all_pitboxes, current_time, today_games)
    elif selected_page == "🎯 Pitcher Detail":
        render_pitcher_detail(p_data, current_time)
    elif selected_page == "🔀 Pitch Mix":
        render_pitch_mix(pmix_data, current_time)
    elif selected_page == "💥 Exit Velos":
        render_exit_velos(hrs, evs, current_time)
    elif selected_page == "📈 Game Pace":
        render_game_pace(livedb, scoreboard_df, current_time)


if __name__ == "__main__":
    main()
