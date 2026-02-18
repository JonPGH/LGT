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
st.set_page_config(page_title="MLB Data Warehouse Live Game Tracker (v2)", layout="wide")


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
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Select a page:",
        ["Scores & Leaders", "Pitcher Detail", "Pitch Mix Data", "Exit Velos"],
    )


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
def render_scores_and_leaders(scoreboard_df: pd.DataFrame, hitboxes: pd.DataFrame, pitboxes: pd.DataFrame, current_time: str):
    st.markdown(f"## MLB DW Live Game Tracker (Last Update {current_time})")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Live Scores")
        st.dataframe(scoreboard_df, width=350, height=260, hide_index=True)

        st.markdown("### Pitching Leaders (Starters)")
        if not pitboxes.empty:
            pit_show = pitboxes.copy()
            pit_show = pit_show.sort_values(by="DKPts", ascending=False)
            pit_show = pit_show[pit_show["GS"] == 1]
            pit_show = pit_show[["Pitcher", "Team", "Line", "DKPts"]].head(30)
            st.dataframe(pit_show, hide_index=True, width=450, height=620)
        else:
            st.info("No boxscore pitching data yet.")

    with col2:
        st.markdown("### Hitting Leaders")
        if not hitboxes.empty:
            hit_show = hitboxes.copy().sort_values(by="DKPts", ascending=False)
            hit_show = hit_show[["Player", "Team", "DKPts", "H", "R", "HR", "RBI", "SB", "2B", "3B", "SO", "BB"]].head(60)
            st.dataframe(hit_show, hide_index=True, width=950, height=900)
        else:
            st.info("No boxscore hitting data yet.")


def render_pitcher_detail(p_data: pd.DataFrame, current_time: str):
    st.markdown(f"## MLB DW Live Game Tracker (Last Update {current_time})")
    st.markdown("### Pitcher Detailed Stats")
    if p_data.empty:
        st.info("No pitch-by-pitch data yet.")
        return
    st.dataframe(p_data, hide_index=True, width=1200, height=520)


def render_pitch_mix(pmix_data: pd.DataFrame, current_time: str):
    st.markdown(f"## MLB DW Live Game Tracker (Last Update {current_time})")
    st.markdown("### Pitch Mix Detailed Stats")

    if pmix_data.empty:
        st.info("No pitch-by-pitch data yet.")
        return

    pmix_data = pmix_data.fillna(0).sort_values(by=["Pitcher", "PC"], ascending=[True, False])
    pitcher_options = ["All Pitchers"] + sorted(pmix_data["Pitcher"].unique().tolist())
    selected_pitcher = st.selectbox("Select Pitcher", pitcher_options)

    view = pmix_data if selected_pitcher == "All Pitchers" else pmix_data[pmix_data["Pitcher"] == selected_pitcher]
    st.dataframe(view, hide_index=True, width=1200, height=760)


def render_exit_velos(hrs: pd.DataFrame, evs: pd.DataFrame, current_time: str):
    st.markdown(f"## MLB DW Live Game Tracker (Last Update {current_time})")

    st.markdown("### Homers")
    if hrs.empty:
        st.info("No HR EVs yet.")
    else:
        st.dataframe(hrs, hide_index=True, width=1000)

    st.markdown("### Hardest Hit Balls (EV > 90)")
    if evs.empty:
        st.info("No batted ball EVs yet.")
    else:
        st.dataframe(evs[evs["EV"] > 90], hide_index=True, width=1000, height=650)


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


# -----------------------------
# Main app
# -----------------------------
def main():
    selected_page = sidebar_menu()

    # Refresh controls
    st.sidebar.markdown("---")
    refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 10, 120, 30, step=5)
    st.sidebar.caption("This is TTL-based; all users share cached snapshots.")

    # Optional autorefresh (recommended)
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore

        st_autorefresh(interval=refresh_seconds * 1000, key="mlbdw_refresh_v2")
    except Exception:
        st.sidebar.warning("Tip: install streamlit-autorefresh for automatic updates. (pip install streamlit-autorefresh)")

    eastern = pytz.timezone("US/Eastern")
    #now_eastern = datetime.now(eastern)
    #today_str = now_eastern.strftime("%Y-%m-%d")

    now_eastern = datetime.now(eastern)
    default_date = now_eastern.date()

    chosen_date = st.sidebar.date_input(
        "Game date",
        value=default_date,
        max_value=default_date,
    )

    date_str = chosen_date.strftime("%Y-%m-%d")

    current_time = now_eastern.strftime("%I:%M")

    # Pull snapshot (cached across all users)
    needs_pbp = selected_page in ("Pitcher Detail", "Pitch Mix Data", "Exit Velos")

    #scoreboard_df, all_hitboxes, all_pitboxes, livedb = build_snapshot(date_str)
    scoreboard_df, all_hitboxes, all_pitboxes, livedb = build_snapshot(date_str, include_pbp=needs_pbp)

    # Basic status messaging
    today_games = get_live_games(date_str)
    statuses = [g.get("game_status") for g in today_games]
    if ("I" not in statuses and "F" in statuses) and 1==2:
        st.info("All games today are complete.")
        if selected_page == "Scores & Leaders":
            # still show final scoreboards from snapshot if present
            render_scores_and_leaders(scoreboard_df, all_hitboxes, all_pitboxes, current_time)
        return
    if ("I" not in statuses) and 1==2:
        st.info("No games currently live.")
        if selected_page == "Scores & Leaders" and not scoreboard_df.empty:
            render_scores_and_leaders(scoreboard_df, all_hitboxes, all_pitboxes, current_time)
        return

    # If we don't have PBP yet, still allow Scores & Leaders via boxscore
    if livedb.empty:
        if selected_page == "Scores & Leaders":
            render_scores_and_leaders(scoreboard_df, all_hitboxes, all_pitboxes, current_time)
            st.caption("No Statcast play-by-play detected yet. Refreshing automatically.")
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
    if selected_page == "Scores & Leaders":
        render_scores_and_leaders(scoreboard_df, all_hitboxes, all_pitboxes, current_time)
    elif selected_page == "Pitcher Detail":
        render_pitcher_detail(p_data, current_time)
    elif selected_page == "Pitch Mix Data":
        render_pitch_mix(pmix_data, current_time)
    elif selected_page == "Exit Velos":
        render_exit_velos(hrs, evs, current_time)


if __name__ == "__main__":
    main()
