import streamlit as st
from datetime import datetime, timezone
import time 
import pytz, requests, pandas as pd, os, numpy as np

# Set page configuration
st.set_page_config(
    page_title="MLB Data Warehouse Live Game Tracker",
    layout="wide"
)

# Data Load
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'Files')
teamnamechangedf = pd.read_csv('{}/mlbteamnamechange.csv'.format(file_path))
teamnamechangedict = dict(zip(teamnamechangedf.Full, teamnamechangedf.Abbrev))
teamnamedict = dict(zip(teamnamechangedf.Full, teamnamechangedf.Abbrev))
league_lev_df = pd.read_csv('{}/LeagueLevels.csv'.format(file_path))
levdict = dict(zip(league_lev_df.league_name,league_lev_df.level))
affdf = pd.read_csv('{}/Team_Affiliates.csv'.format(file_path))
affdict = dict(zip(affdf.team_id, affdf.parent_id))
affdict_abbrevs = dict(zip(affdf.team_id, affdf.parent_abbrev))
team_abbrev_look = dict(zip(affdf.team_name,affdf.team_abbrev))
idlookup_df = pd.read_csv('{}/IDLookupTable.csv'.format(file_path))
p_lookup_dict = dict(zip(idlookup_df.MLBID, idlookup_df.PLAYERNAME))

lsaclass = pd.read_csv('{}/lsaclass.csv'.format(file_path))

def dropUnnamed(df):
  df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
  return(df)

lsaclass = dropUnnamed(lsaclass)
lsaclass['launch_speed'] = round(lsaclass['launch_speed'],0)
lsaclass['launch_angle'] = round(lsaclass['launch_angle'],0)
lsaclass.columns=['launch_speed_round','launch_angle_round','launch_speed_angle']

# Sidebar menu
def sidebar_menu():
    st.sidebar.title("Navigation")
    
    # Buttons for different pages
    page = st.sidebar.radio(
        "Select a page:",
        ["Scores & Leaders", "Pitcher Detail", "Pitch Mix Data", "Exit Velos"]  # Add or remove pages as needed
    )
    return page


# Get Data
def getLiveGames(date_string):
    games = []
    #sportIds = [1,11, 12, 13, 14, 15, 16, 17]
    sportIds = [1]
    sport_id_mappings = {1: 'MLB', 11: 'AAA', 12: 'AA', 13: 'A+', 14: 'A', 16: 'ROK', 17: 'WIN'}

    for sportId in sportIds:
        url = "https://statsapi.mlb.com/api/v1/schedule/?sportId={}&date={}".format(sportId,date_string)
        schedule = requests.get(url).json()

        for date in schedule["dates"]:
            for game_data in date["games"]:
                # Skip games that are not finished ("F")
                # If a game was delayed, it will show up again on a later calendar date
                #if game_data["status"]["codedGameState"] == "F":
                game = {}
                game["date"] = date_string
                game["game_id"] = game_data["gamePk"]
                game["game_type"] = game_data["gameType"]
                game["venue_id"] = game_data["venue"]["id"]
                game["venue_name"] = game_data["venue"]["name"]
                game["away_team"] = game_data["teams"]["away"]["team"]["name"]
                game["home_team"] = game_data["teams"]["home"]["team"]["name"]
                game["league_id"] = sportId
                game["league_level"] = sport_id_mappings.get(sportId)
                game["game_status"] = game_data["status"]["codedGameState"]
                game["game_status_full"] = game_data["status"]["abstractGameState"]
                game["game_start_time"] = game_data["gameDate"]
                games.append(game)
    return games

def get_MILB_PBP_Live(game_info_dict):
  game_pk = game_info_dict.get('game_id')
  game_date = game_info_dict.get('date')
  venue_id = game_info_dict.get('venue_id')
  venue_name = game_info_dict.get('venue_name')
  league_id = game_info_dict.get('league_id')
  game_type = game_info_dict.get('game_type')

  # get game play by play
  url = 'https://statsapi.mlb.com/api/v1/game/{}/playByPlay'.format(game_pk)

  boxurl = 'https://statsapi.mlb.com/api/v1/game/{}/boxscore'.format(game_pk)
  box_game_info = requests.get(boxurl).json()
  lgname=box_game_info.get('teams').get('away').get('team').get('league').get('name')
  away_team = box_game_info.get('teams').get('away').get('team').get('name')
  away_team_id =  box_game_info.get('teams').get('away').get('team').get('id')
  home_team = box_game_info.get('teams').get('home').get('team').get('name')
  home_team_id = box_game_info.get('teams').get('home').get('team').get('id')

  game_info = requests.get(url).json()
  jsonstr = str(game_info)

  savtest='startSpeed' in jsonstr
  if savtest is True:
    statcastflag='Y'
  else:
    statcastflag='N'

  allplays = game_info.get('allPlays')

  gamepbp = pd.DataFrame()
  for play in allplays:
    currplay = play
    inningtopbot = currplay.get('about').get('halfInning')
    inning = currplay.get('about').get('inning')
    actionindex = currplay.get('actionIndex')
    at_bat_number = currplay.get('about').get('atBatIndex')+1
    currplay_type = currplay.get('result').get('type')
    currplay_res = currplay.get('result').get('eventType')
    currplay_descrip = currplay.get('result').get('description')
    currplay_rbi = currplay.get('result').get('rbi')
    currplay_awayscore = currplay.get('result').get('awayscore')
    currplay_homescore = currplay.get('result').get('homescore')
    currplay_isout = currplay.get('result').get('isOut')
    playdata = currplay.get('playEvents')
    playmatchup = currplay.get('matchup')
    bid = playmatchup.get('batter').get('id')
    bname = playmatchup.get('batter').get('fullName')
    bstand = playmatchup.get('batSide').get('code')
    pid = playmatchup.get('pitcher').get('id')
    pname = playmatchup.get('pitcher').get('fullName')
    pthrows = playmatchup.get('pitchHand').get('code')

    for pitch in playdata:
      pdetails = pitch.get('details')
      checkadvise = pdetails.get('event')
      pitch_number = pitch.get('pitchNumber')
      if checkadvise is None:
        try:
          description = pdetails.get('call').get('description')
        except:
          description=None
        inplay = pdetails.get('isInPlay')
        isstrike = pdetails.get('isStrike')
        isball = pdetails.get('isBall')
        try:
          pitchname = pdetails.get('type').get('description')
          pitchtype = pdetails.get('type').get('code')
        except:
          pitchname=None
          pitchtype=None

        ballcount = pitch.get('count').get('balls')
        strikecount = pitch.get('count').get('strikes')
        try:
          plate_x = pitch.get('pitchData').get('coordinates').get('x')
          plate_y = pitch.get('pitchData').get('coordinates').get('y')
        except:
          plate_x = None
          plate_y = None

        try:
          startspeed = pitch.get('pitchData').get('startSpeed')
          endspeed = pitch.get('pitchData').get('endspeed')
        except:
          startspeed=None
          endspeed=None

        try:
          kzonetop = pitch.get('pitchData').get('strikeZoneTop')
          kzonebot = pitch.get('pitchData').get('strikeZoneBottom')
          kzonewidth = pitch.get('pitchData').get('strikeZoneWidth')
          kzonedepth = pitch.get('pitchData').get('strikeZoneDepth')
        except:
          kzonetop=None
          kzonebot=None
          kzonewidth=None
          kzonedepth=None


        try:
          ay = pitch.get('pitchData').get('coordinates').get('aY')
          ax = pitch.get('pitchData').get('coordinates').get('aX')
          pfxx = pitch.get('pitchData').get('coordinates').get('pfxX')
          pfxz = pitch.get('pitchData').get('coordinates').get('pfxZ')
          px = pitch.get('pitchData').get('coordinates').get('pX')
          pz = pitch.get('pitchData').get('coordinates').get('pZ')
          breakangle = pitch.get('pitchData').get('breaks').get('breakAngle')
          breaklength = pitch.get('pitchData').get('breaks').get('breakLength')
          break_y= pitch.get('pitchData').get('breaks').get('breakY')
          zone = pitch.get('pitchData').get('zone')
        except:
          ay=None
          ax=None
          pfxx=None
          pfxz=None
          px=None
          pz=None
          breakangle=None
          break_y=None
          breaklength=None
          zone=None

        try:
          hitdata = pitch.get('hitData')
          launchspeed = pitch.get('hitData').get('launchSpeed')
          launchspeed_round = round(launchspeed,0)
          launchangle = pitch.get('hitData').get('launchAngle')
          launchangle_round = round(launchangle,0)
          bb_type = pitch.get('hitData').get('trajectory')
          hardness = pitch.get('hitData').get('hardness')
          location = pitch.get('hitData').get('location')
          total_distance = pitch.get('hitData').get('totalDistance')
          coord_x = pitch.get('hitData').get('coordinates').get('coordX')
          coord_y = pitch.get('hitData').get('coordinates').get('coordY')

        except:
          #print('No hit data')
          launchspeed = None
          launchangle = None
          get_lsa = None
          bb_type = None
          hardness = None
          location = None
          coord_x = None
          coord_y = None


        this_gamepbp = pd.DataFrame({'StatcastGame': statcastflag,
                                  'game_pk': game_pk, 'game_date': game_date, 'game_type': game_type, 'venue': venue_name,
                                  'venue_id': venue_id,'league_id': league_id, 'league': lgname, 'level': levdict.get(lgname),
                                  'away_team': away_team,'away_team_id': away_team_id, 'away_team_aff': affdict.get(away_team),
                                  'home_team': home_team, 'home_team_id': home_team_id, 'home_team_aff': affdict.get(home_team),
                                  'player_name': pname, 'pitcher': pid, 'BatterName': bname, 'batter': bid,
                                  'stand': bstand, 'p_throws': pthrows,'inning_top_bot': inningtopbot,
                                  'plate_x': plate_x, 'plate_y': plate_y,
                                  'inning': inning, 'at_bat_number':at_bat_number, 'pitch_number': pitch_number,
                                  'description': description, 'play_type': currplay_type, 'play_res':currplay_res,
                                  'play_desc': currplay_descrip,
                                  'rbi': currplay_rbi,'away_team_score':currplay_awayscore,
                                  'isOut': currplay_isout,'home_team_score':currplay_homescore,
                                  'isInPlay': inplay,'IsStrike':isstrike,'IsBall':isball,'pitch_name':pitchname,
                                  'pitch_type':pitchtype,'balls': ballcount,'strikes':strikecount,
                                  'release_speed':startspeed, 'end_pitch_speed': endspeed,'zone_top':kzonetop,
                                  'zone_bot':kzonebot,'zone_width':kzonewidth,'zone_depth':kzonedepth,'ay':ay,
                                  'ax':ax,'pfx_x':pfxx,'pfx_z':pfxz,'px':px,'pz':pz,'break_angle':breakangle,
                                  'break_length':breaklength,'break_y':break_y,'zone':zone,
                                  'launch_speed': launchspeed, 'launch_angle': launchangle,# 'hit_distance': total_distance,
                                  'bb_type':bb_type,'hit_location': location,'hit_coord_x': coord_x,
                                  'hit_coord_y': coord_y},index=[0])
        gamepbp = pd.concat([gamepbp,this_gamepbp])


      else:
        #print('Found game advisory: {}'.format(checkadvise))
        pass
  gamepbp = gamepbp.reset_index(drop=True)
  #gamepbp.to_csv('/content/drive/My Drive/FLB/2024/LiveGames/AllPlayByPlay/{}_pbp.csv'.format(game_pk))
  return(gamepbp)

def get_game_logs(game):
    batting_logs = []
    pitching_logs = []
    url = "https://statsapi.mlb.com/api/v1/game/{}/boxscore".format(game["game_id"])
    game_info = requests.get(url).json()
    lgname=game_info.get('teams').get('away').get('team').get('league').get('name')

    away_team = game_info.get('teams').get('away').get('team').get('name')
    away_team_id =  game_info.get('teams').get('away').get('team').get('id')
    home_team = game_info.get('teams').get('home').get('team').get('name')
    home_team_id = game_info.get('teams').get('home').get('team').get('id')

    if "teams" in game_info:
        for team in game_info["teams"].values():
            team_id = team.get('team').get('id')
            for player in team["players"].values():
                if player["stats"]["batting"]:
                    batting_log = {}
                    batting_log["game_date"] = game["date"]
                    batting_log["game_id"] = int(game["game_id"])
                    batting_log["league_name"] = lgname
                    batting_log["level"] = game["league_level"]
                    batting_log["Team"] = team.get('team').get('name')
                    batting_log["team_id"] = team_id
                    batting_log["home_team"] = home_team
                    batting_log["game_type"] = game["game_type"]
                    batting_log["venue_id"] = int(game["venue_id"])
                    batting_log["league_id"] = int(game["league_id"])
                    batting_log["Player"] = player["person"]["fullName"]
                    batting_log["player_id"] = int(player["person"]["id"])
                    batting_log["batting_order"] = player.get("battingOrder", "")
                    batting_log["AB"] = int(player["stats"]["batting"]["atBats"])
                    batting_log["R"] = int(player["stats"]["batting"]["runs"])
                    batting_log["H"] = int(player["stats"]["batting"]["hits"])
                    batting_log["2B"] = int(player["stats"]["batting"]["doubles"])
                    batting_log["3B"] = int(player["stats"]["batting"]["triples"])
                    batting_log["HR"] = int(player["stats"]["batting"]["homeRuns"])
                    batting_log["RBI"] = int(player["stats"]["batting"]["rbi"])
                    batting_log["SB"] = int(player["stats"]["batting"]["stolenBases"])
                    batting_log["CS"] = int(player["stats"]["batting"]["caughtStealing"])
                    batting_log["BB"] = int(player["stats"]["batting"]["baseOnBalls"])
                    batting_log["SO"] = int(player["stats"]["batting"]["strikeOuts"])
                    batting_log["IBB"] = int(player["stats"]["batting"]["intentionalWalks"])
                    batting_log["HBP"] = int(player["stats"]["batting"]["hitByPitch"])
                    batting_log["SH"] = int(player["stats"]["batting"]["sacBunts"])
                    batting_log["SF"] = int(player["stats"]["batting"]["sacFlies"])
                    batting_log["GIDP"] = int(player["stats"]["batting"]["groundIntoDoublePlay"])

                    batting_logs.append(batting_log)

                if player["stats"]["pitching"]:
                    pitching_log = {}
                    pitching_log["game_date"] = game["date"]
                    pitching_log["game_id"] = int(game["game_id"])
                    pitching_log["league_name"] = lgname
                    pitching_log["level"] = game["league_level"]
                    pitching_log["Team"] = team.get('team').get('name')
                    pitching_log["team_id"] = team_id
                    pitching_log["home_team"] = home_team
                    pitching_log["game_type"] = game["game_type"]
                    pitching_log["venue_id"] = int(game["venue_id"])
                    pitching_log["league_id"] = int(game["league_id"])
                    pitching_log["Player"] = player["person"]["fullName"]
                    pitching_log["player_id"] = int(player["person"]["id"])
                    pitching_log["W"] = int(player["stats"]["pitching"].get("wins", ""))
                    pitching_log["L"] = int(player["stats"]["pitching"].get("losses", ""))
                    pitching_log["G"] = int(player["stats"]["pitching"].get("gamesPlayed", ""))
                    pitching_log["GS"] = int(player["stats"]["pitching"].get("gamesStarted", ""))
                    pitching_log["CG"] = int(player["stats"]["pitching"].get("completeGames", ""))
                    pitching_log["SHO"] = int(player["stats"]["pitching"].get("shutouts", ""))
                    pitching_log["SV"] = int(player["stats"]["pitching"].get("saves", ""))
                    pitching_log["HLD"] = int(player["stats"]["pitching"].get("holds", ""))
                    pitching_log["BFP"] = int(player["stats"]["pitching"].get("battersFaced", ""))
                    pitching_log["IP"] = float(player["stats"]["pitching"].get("inningsPitched", ""))
                    pitching_log["H"] = int(player["stats"]["pitching"].get("hits", ""))
                    pitching_log["ER"] = int(player["stats"]["pitching"].get("earnedRuns", ""))
                    pitching_log["R"] = int(player["stats"]["pitching"].get("runs", ""))
                    pitching_log["HR"] = int(player["stats"]["pitching"].get("homeRuns", ""))
                    pitching_log["SO"] = int(player["stats"]["pitching"].get("strikeOuts", ""))
                    pitching_log["BB"] = int(player["stats"]["pitching"].get("baseOnBalls", ""))
                    pitching_log["IBB"] = int(player["stats"]["pitching"].get("intentionalWalks", ""))
                    pitching_log["HBP"] = int(player["stats"]["pitching"].get("hitByPitch", ""))
                    pitching_log["WP"] = int(player["stats"]["pitching"].get("wildPitches", ""))
                    pitching_log["BK"] = int(player["stats"]["pitching"].get("balks", ""))

                    if pitching_log["GS"] > 0 and pitching_log["IP"] >= 6 and pitching_log["ER"] <= 3:
                        pitching_log["QS"] = 1
                    else:
                        pitching_log["QS"] = 0

                    pitching_logs.append(pitching_log)

    return batting_logs, pitching_logs

def savAddOns(savdata):
  pdf = savdata.copy()
  #st.write(pdf)

  pdf['away_team_aff_id'] = pdf['away_team_id'].map(affdict)
  pdf['away_team_aff'] = pdf['away_team_aff_id'].map(affdict_abbrevs)
  pdf['home_team_aff_id'] = pdf['home_team_id'].map(affdict)
  pdf['home_team_aff'] = pdf['home_team_aff_id'].map(affdict_abbrevs)

  pdf['IsWalk'] = np.where(pdf['balls']==4,1,0)
  pdf['IsStrikeout'] = np.where(pdf['strikes']==3,1,0)
  pdf['BallInPlay'] = np.where(pdf['isInPlay']==1,1,0)
  pdf['IsHBP'] = np.where(pdf['description']=='Hit By Pitch',1,0)
  pdf['PA_flag'] = np.where((pdf['balls']==4)|(pdf['strikes']==3)|(pdf['BallInPlay']==1)|(pdf['IsHBP']==1),1,0)


  pdf['IsHomer'] = np.where((pdf['play_res']=='home_run')&(pdf['PA_flag']==1),1,0)

  pitchthrownlist = ['In play, out(s)', 'Swinging Strike', 'Ball', 'Foul',
        'In play, no out', 'Called Strike', 'Foul Tip', 'In play, run(s)','Hit By Pitch',
        'Ball In Dirt','Pitchout', 'Swinging Strike (Blocked)',
        'Foul Bunt', 'Missed Bunt', 'Foul Pitchout',
        'Intent Ball', 'Swinging Pitchout']

  pdf['PitchesThrown'] = np.where(pdf['description'].isin(pitchthrownlist),1,0)

  map_pitchnames = {'Two-Seam Fastball': 'Sinker', 'Slow Curve': 'Curveball', 'Knuckle Curve': 'Curveball'}
  pdf['pitch_name'] = pdf['pitch_name'].replace(map_pitchnames)

  swstrlist = ['Swinging Strike','Foul Tip','Swinging Strike (Blocked)', 'Missed Bunt']
  cslist = ['Called Strike']
  cswlist = ['Swinging Strike','Foul Tip','Swinging Strike (Blocked)', 'Missed Bunt','Called Strike']
  contlist = ['Foul','In play, no out', 'In play, out(s)', 'Foul Pitchout','In play, run(s)']
  swinglist = ['Swinging Strike','Foul','In play, no out', 'In play, out(s)', 'In play, run(s)', 'Swinging Strike (Blocked)', 'Foul Pitchout']
  klist = ['strikeout', 'strikeout_double_play']
  bblist = ['walk','intent_walk']
  hitlist = ['single','double','triple','home_run']
  #balllist = ['Ball','Automatic Ball','Intent Ball','Pitchout']
  palist = ['strikeout','walk']

  isstrikelist = [ 'Swinging Strike', 'Foul','Called Strike', 'Foul Tip','Swinging Strike (Blocked)',
                  'Automatic Strike - Batter Pitch Timer Violation', 'Foul Bunt',
                  'Automatic Strike - Batter Timeout Violation', 'Missed Bunt',
                  'Automatic Strike','Foul Pitchout','Swinging Pitchout']

  isballlist = ['Ball', 'Hit By Pitch','Automatic Ball - Pitcher Pitch Timer Violation',
                'Ball In Dirt','Pitchout', 'Automatic Ball - Intentional', 'Automatic Ball',
                'Automatic Ball - Defensive Shift Violation','Automatic Ball - Catcher Pitch Timer Violation',
                'Intent Ball']

  pdf['IsStrike'] = np.where(pdf['description'].isin(isstrikelist),1,0)
  pdf['IsBall'] = np.where(pdf['description'].isin(isballlist),1,0)


  pdf['BatterTeam'] = np.where(pdf['inning_top_bot']=='bottom', pdf['home_team'], pdf['away_team'])
  pdf['PitcherTeam'] = np.where(pdf['inning_top_bot']=='bottom', pdf['away_team'], pdf['home_team'])

  pdf['BatterTeam_aff'] = np.where(pdf['inning_top_bot']=='bottom', pdf['home_team_aff'], pdf['away_team_aff'])
  pdf['PitcherTeam_aff'] = np.where(pdf['inning_top_bot']=='bottom', pdf['away_team_aff'], pdf['home_team_aff'])

  pdf['IsBIP'] = pdf['BallInPlay']

  pdf['PA'] = pdf['PA_flag']
  #pdf['AB'] = np.where((pdf['IsBIP']+pdf['IsStrikeout'])>0,1,0)
  pdf['IsHit'] = np.where((pdf['PA']==1)&(pdf['play_res'].isin(hitlist)),1,0)

  pdf['IsSwStr'] = np.where(pdf['description'].isin(swstrlist),1,0)
  pdf['IsCalledStr'] = np.where(pdf['description'].isin(cslist),1,0)
  pdf['ContactMade'] = np.where(pdf['description'].isin(contlist),1,0)
  pdf['SwungOn'] = np.where(pdf['description'].isin(swinglist),1,0)
  pdf['IsGB'] = np.where(pdf['bb_type']=='ground_ball',1,0)
  pdf['IsFB'] = np.where(pdf['bb_type']=='fly_ball',1,0)
  pdf['IsLD'] = np.where(pdf['bb_type']=='line_drive',1,0)
  pdf['IsPU'] = np.where(pdf['bb_type']=='popup',1,0)

  pdf['InZone'] = np.where(pdf['zone']<10,1,0)
  pdf['OutZone'] = np.where(pdf['zone']>9,1,0)
  pdf['IsChase'] = np.where(((pdf['SwungOn']==1)&(pdf['InZone']==0)),1,0)
  pdf['IsZoneSwing'] = np.where(((pdf['SwungOn']==1)&(pdf['InZone']==1)),1,0)
  pdf['IsZoneContact'] = np.where(((pdf['ContactMade']==1)&(pdf['InZone']==1)),1,0)

  pdf['IsSingle'] = np.where((pdf['play_res']=='single')&(pdf['PA_flag']==1),1,0)
  pdf['IsDouble'] = np.where((pdf['play_res']=='double')&(pdf['PA_flag']==1),1,0)
  pdf['IsTriple'] = np.where((pdf['play_res']=='triple')&(pdf['PA_flag']==1),1,0)

  ablist = ['field_out', 'double', 'strikeout', 'single','grounded_into_double_play',
            'home_run','fielders_choice', 'force_out', 'double_play', 'triple','field_error',
            'fielders_choice_out','strikeout_double_play','other_out', 'sac_fly_double_play','triple_play']

  pdf['AB'] = np.where((pdf['play_res'].isin(ablist))&(pdf['PA_flag']==1),1,0)

  try:
    pdf = pdf.drop(['launch_speed_angle'],axis=1)
  except:
    pass

  pdf['launch_angle'] = pdf['launch_angle'].replace([None], np.nan)
  pdf['launch_speed'] = pdf['launch_speed'].replace([None], np.nan)

  pdf['launch_angle_round'] = round(pdf['launch_angle'],0)
  pdf['launch_speed_round'] = round(pdf['launch_speed'],0)

  pdf = pd.merge(pdf, lsaclass, how='left', on=['launch_speed_round','launch_angle_round'])

  pdf['launch_speed_angle'] = np.where(pdf['launch_speed_round']<60,1,pdf['launch_speed_angle'])
  pdf['launch_speed_angle'] = np.where((pdf['launch_speed_angle'].isna())&(pdf['launch_speed']>1),1,pdf['launch_speed_angle'])

  pdf['IsBrl'] = np.where(pdf['launch_speed_angle']==6,1,0)
  pdf['IsSolid'] = np.where(pdf['launch_speed_angle']==5,1,0)
  pdf['IsFlare'] = np.where(pdf['launch_speed_angle']==4,1,0)
  pdf['IsUnder'] = np.where(pdf['launch_speed_angle']==3,1,0)
  pdf['IsTopped'] = np.where(pdf['launch_speed_angle']==2,1,0)
  pdf['IsWeak'] = np.where(pdf['launch_speed_angle']==1,1,0)
  ###

  ## zone stuff
  pdf['IsCalledStr'] = np.where(pdf['description']=='Called Strike',1,0)
  pdf['zone_bot2'] = pdf['zone_bot']*100
  pdf['zone_top2'] = pdf['zone_top']*100
  pdf['inzone_y'] = np.where((pdf['plate_y']>=pdf['zone_bot2'])&(pdf['plate_y']<=pdf['zone_top2']),1,0)
  pdf['inzone_x'] = np.where((pdf['plate_x']>=70)&(pdf['plate_x']<=140),1,0)
  pdf['InZone2'] = np.where((pdf['inzone_y']==1)&(pdf['inzone_x']==1),1,0)
  pdf['OutZone2'] = np.where(pdf['InZone2']==1,0,1)
  pdf['IsZoneSwing2'] = np.where((pdf['InZone2']==1)&(pdf['SwungOn']==1),1,0)
  pdf['IsChase2'] = np.where((pdf['OutZone2']==1)&(pdf['SwungOn']==1),1,0)
  pdf['IsZoneContact2'] = np.where((pdf['IsZoneSwing2']==1)&(pdf['ContactMade']==1),1,0)

  # HANDLE DUPLICATES
  dupes_hitter_df = pdf.groupby(['BatterName','batter'],as_index=False)['AB'].sum()
  hitter_dupes = dupes_hitter_df.groupby('BatterName',as_index=False)['batter'].count().sort_values(by='batter',ascending=False)
  hitter_dupes = hitter_dupes[hitter_dupes['batter']>1]
  hitter_dupes.columns=['Player','Count']
  hitter_dupes['Pos'] = 'Hitter'
  hitter_dupes_list = list(hitter_dupes['Player'])
  pdf['BatterName'] = np.where(pdf['BatterName'].isin(hitter_dupes_list),pdf['BatterName'] + ' - ' + pdf['batter'].astype(int).astype(str), pdf['BatterName'])

  dupes_pitcher_df = pdf.groupby(['player_name','pitcher'],as_index=False)['PitchesThrown'].sum()
  pitcher_dupes = dupes_pitcher_df.groupby('player_name',as_index=False)['pitcher'].count().sort_values(by='pitcher',ascending=False)
  pitcher_dupes = pitcher_dupes[pitcher_dupes['pitcher']>1]
  pitcher_dupes.columns=['Player','Count']
  pitcher_dupes['Pos'] = 'Pitcher'
  pitcher_dupes_list = list(pitcher_dupes['Player'])
  pdf['player_name'] = np.where(pdf['player_name'].isin(pitcher_dupes_list),pdf['player_name'] + ' - ' + pdf['pitcher'].astype(int).astype(str), pdf['player_name'])

  pdf = dropUnnamed(pdf)
  pdf['game_date'] = pd.to_datetime(pdf['game_date'])
  pdf['player_name'] = pdf['player_name'].replace({'Luis L. Ortiz': 'Luis Ortiz - 682847'})

  # drop dupes
  pdf = pdf.drop_duplicates(subset=['game_pk','pitcher','batter','inning','at_bat_number','pitch_number'])

  return(pdf)

def checkStatus(gamestatuslist):
    if ('I' not in gamestatuslist) and ('S' not in gamestatuslist) and ('P' not in gamestatuslist):
        return_status = 'Stop'
    if 'I' not in gamestatuslist:
        if 'I' not in gamestatuslist:
            st.write('No live games, but more coming, waiting 10 minutes')
            return_status = 'Wait'
        pass
    else:
       return_status = 'live'
    
    return(return_status)   

def getBoxDetails(game_box):
    hitbox_json = game_box[0]
    hitbox = pd.DataFrame(hitbox_json)

    hitbox['1B'] = hitbox['H']-hitbox['2B']-hitbox['3B']-hitbox['HR']
    hitbox = hitbox[['Player','player_id','batting_order','Team','home_team','AB','R','H','1B','2B','3B','HR','RBI','SB','CS','BB','SO','HBP']]
    hitbox['Team'] = hitbox['Team'].replace(teamnamedict)
    hitbox['home_team'] = hitbox['home_team'].replace(teamnamedict)
    pitbox_json = game_box[1]
    pitbox = pd.DataFrame(pitbox_json)
    pitbox = pitbox[['Player','player_id','Team','home_team','G','GS','IP','H','ER','R','HR','SO','BB','IBB','HBP','QS','W']]
    pitbox['Team'] = pitbox['Team'].replace(teamnamedict)
    pitbox['home_team'] = pitbox['home_team'].replace(teamnamedict)

    hitbox['DKPts'] = (hitbox['1B']*3)+(hitbox['2B']*5)+(hitbox['3B']*8)+(hitbox['HR']*10)+(hitbox['SB']*5)+(hitbox['BB']*2)+(hitbox['HBP']*2)+(hitbox['R']*2)+(hitbox['RBI']*2)
    pitbox['DKPts'] = (pitbox['IP']*2.25)+(pitbox['SO']*2)+(pitbox['W']*4)+(pitbox['ER']*-2)+(pitbox['H']*-.6)+(pitbox['BB']*-.6)

    pitbox['Line'] = pitbox['IP'].astype(str) + 'IP ' + pitbox['H'].astype(str) + 'H ' + pitbox['ER'].astype(str) + 'ER ' + pitbox['SO'].astype(str) + 'K ' + pitbox['BB'].astype(str) + 'BB'
    linebox = pitbox[['Player','Team','GS','Line','DKPts']]
    linebox.columns=['Pitcher','Team','GS','Line','DKPts']

    show_hitbox = hitbox[['Player','Team','H','R','HR','RBI','SB','2B','3B','SO','BB','DKPts']]

    ## CREATE SCOREBOARD OUT OF HIT BOX 
    teams = hitbox['Team'].unique()
    this_mu = {teams[0]:teams[1], teams[1]:teams[0]}
    this_hometeams = dict(zip(hitbox.Team,hitbox.home_team))

    home_team = list(this_hometeams.values())[0]
    for xteam in teams:
        if xteam == home_team:
           pass
        else:
           road_team = xteam

    team_ip = pitbox.groupby('Team',as_index=False)['IP'].sum()
    curr_inning = np.min(team_ip['IP'])+1
    curr_inning = int(curr_inning)

    if curr_inning >= 9:
        inningprint = 'F'
    else:
        inningprint = str(curr_inning)

    team_runs = hitbox.groupby('Team',as_index=False)['R'].sum()
    team_runs['Opp'] = team_runs['Team'].map(this_mu)
    team_runs['Home'] = team_runs['Team'].map(this_hometeams)
    team_runs = team_runs.sort_values(by='R',ascending=False)
    team_runs_dict = dict(zip(team_runs.Team,team_runs.R))

    show_df = pd.DataFrame({'Inning': inningprint, 'Road': road_team, 'Home': home_team, 
                            'Road Score': team_runs_dict.get(road_team),
                            'Home Score': team_runs_dict.get(home_team)}, index=[0])

    game_dis = show_df['Road'].iloc[0] + ' @ ' + show_df['Home'].iloc[0]
    score = road_team + ' (' + str(team_runs_dict.get(road_team)) + ') @ ' + home_team + ' (' + str(team_runs_dict.get(home_team)) + ')'

    this_score = pd.DataFrame({'Game': game_dis, 'Score': score, 'Inn': inningprint}, index=[0])
    
    box_steals = hitbox[['Player','Team','SB']]
    box_homers = hitbox[['Player','Team','HR']]
    box_dkpts = hitbox[['Player','Team','DKPts']]

    return(show_hitbox,linebox,this_score)

def getPData(livedb,all_pitboxes,cplist):
    # Start
    pdata = livedb.groupby(['player_name','pitcher','PitcherTeam_aff'],as_index=False)[['PitchesThrown','IsStrike','IsBall','IsBIP','IsHit','IsHomer','IsSwStr','IsGB','IsLD','IsFB','IsBrl','PA_flag','DP','IsStrikeout','IsWalk']].sum()
    pdata['Outs'] = pdata['PA_flag']-pdata['IsHit']-pdata['IsWalk']+pdata['DP']
    pdata['IP'] = round((pdata['Outs']/3),2)
    pdata['SwStr%'] = round(pdata['IsSwStr']/pdata['PitchesThrown'],3)
    pdata['Strike%'] = round(pdata['IsStrike']/pdata['PitchesThrown'],3)
    pdata['Ball%'] = round(pdata['IsBall']/pdata['PitchesThrown'],3)
    pdata['GB%'] = round(pdata['IsGB']/pdata['IsBIP'],3)
    pdata['FB%'] = round(pdata['IsFB']/pdata['IsBIP'],3)
    pdata['LD%'] = round(pdata['IsLD']/pdata['IsBIP'],3)
    pdata['Brl%'] = round(pdata['IsBrl']/pdata['IsBIP'],3)

    pdata = pdata.sort_values(by='IsSwStr',ascending=False)
    pdata = pdata[['player_name','pitcher','PitcherTeam_aff','PA_flag','IP','IsStrikeout','IsWalk','IsHit','IsHomer','PitchesThrown','IsSwStr','IsStrike','SwStr%','Strike%','Ball%','GB%','LD%','FB%','Brl%']]
    pdata.columns=['Pitcher','ID','Team','TBF','IP','SO','BB','H','HR','PC','Whiffs','Strikes','SwStr%','Strike%','Ball%','GB%','LD%','FB%','Brl%']
    pdata['Current Pitcher?'] = np.where(pdata['Pitcher'].isin(cplist),'Y','N')
    
    showdf = pdata.copy()
    showdf = pd.merge(showdf,all_pitboxes[['Pitcher','Line']],how='left',on='Pitcher')
    pdatadf = showdf[['Pitcher','Team','Line','PC','SO','BB','Whiffs','SwStr%','Strike%','Ball%','Current Pitcher?']].sort_values(by=['Whiffs'],ascending=False)

    return(pdatadf)

def getPMixData(livedb,cplist):
    velodata = livedb.groupby(['player_name','PitcherTeam_aff','pitch_type'],as_index=False)['release_speed'].mean()
    velodata = velodata.round(1)
    velodata.columns=['Pitcher','Team','Pitch','Velo']

    mixdata = livedb.groupby(['player_name','pitcher','PitcherTeam_aff','pitch_type'],as_index=False)[['PitchesThrown','IsStrike','IsBall','IsBIP','IsHit','IsHomer','IsSwStr','IsGB','IsLD','IsFB','IsBrl','PA_flag','DP','IsStrikeout','IsWalk']].sum()
    mixdata['SwStr%'] = round(mixdata['IsSwStr']/mixdata['PitchesThrown'],3)
    mixdata['Strike%'] = round(mixdata['IsStrike']/mixdata['PitchesThrown'],3)
    mixdata['Ball%'] = round(mixdata['IsBall']/mixdata['PitchesThrown'],3)
    mixdata['Brl%'] = round(mixdata['IsBrl']/mixdata['IsBIP'],3)
    mixdata = mixdata[['player_name','PitcherTeam_aff','pitch_type','PitchesThrown','IsSwStr','SwStr%','Strike%','Ball%','Brl%']]
    mixdata.columns=['Pitcher','Team','Pitch','PC','Whiffs','SwStr%','Strike%','Ball%','Brl%']
    mixdata = pd.merge(mixdata,velodata,on=['Pitcher','Team','Pitch'])
    mixdata = mixdata[['Pitcher','Team','Pitch','PC','Velo','Whiffs','SwStr%','Strike%','Ball%','Brl%']]
    return(mixdata)

# Main content functions
def home_page(scoreboard_df, hitboxes, pitboxes, currentime, 
              scoreboard_container=None, pitching_container=None, hitting_container=None):
    # Preprocess data
    pitboxes = pitboxes.sort_values(by='DKPts', ascending=False)
    pitboxes = pitboxes[pitboxes['GS'] == 1]
    pitboxes = pitboxes[['Pitcher', 'Team', 'Line', 'DKPts']]
    hitboxes = hitboxes.sort_values(by='DKPts', ascending=False)

    # If no containers are provided (first call), set up the layout and placeholders
    if scoreboard_container is None:
        st.markdown(f"<h1>MLB DW Live Game Tracker (Last Update {currentime})</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("<h4>Live Scores:</h4>", unsafe_allow_html=True)
            scoreboard_container = st.empty()  # Placeholder for scoreboard

            st.markdown("<h4>Pitching Leaders:</h4>", unsafe_allow_html=True)
            pitching_container = st.empty()  # Placeholder for pitching leaders

        with col2:
            st.markdown("<h4>Hitting Leaders:</h4>", unsafe_allow_html=True)
            hitting_container = st.empty()  # Placeholder for hitting leaders

    # Style the pitching dataframe
    styled_pitboxes = pitboxes.style.format({'DKPts': '{:.1f}'})
    show_hitdf = hitboxes[['Player', 'Team', 'DKPts', 'H', 'R', 'HR', 'RBI', 'SB', '2B', '3B', 'SO', 'BB']].sort_values(by='DKPts', ascending=False)

    # Update the placeholders with the new dataframes
    with scoreboard_container.container():
        scoreboard_container.dataframe(scoreboard_df, width=300, height=270, hide_index=True)
    
    with pitching_container.container():
        pitching_container.dataframe(styled_pitboxes, hide_index=True, width=420, height=600)
    
    with hitting_container.container():
        hitting_container.dataframe(show_hitdf, hide_index=True, width=825, height=850)

    # Return the containers to reuse them in the next call
    return scoreboard_container, pitching_container, hitting_container

def pitcher_detail_page(p_data, current_time, pitcher_detail_dataframe_container=None):
    # Sort data by Whiffs
    p_data = p_data.sort_values(by='Whiffs', ascending=False)

    # If no container is provided (first call), create placeholders
    if pitcher_detail_dataframe_container is None:
        # Display static header and title (these won't update)
        st.markdown(f"<h1>MLB DW Live Game Tracker (Last Update {current_time})</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Pitcher Detailed Stats</h3>", unsafe_allow_html=True)
        # Create a placeholder for the dataframe
        pitcher_detail_dataframe_container = st.empty()
    
    # Style the dataframe
    styled_df = p_data.style.format({
        'PC': '{:.0f}', 'SO': '{:.0f}', 'BB': '{:.0f}', 'Whiffs': '{:.0f}',
        'SwStr%': '{:.1%}', 'Strike%': '{:.1%}', 'Ball%': '{:.1%}'
    })

    # Update the placeholder with the new dataframe
    pitcher_detail_dataframe_container.dataframe(styled_df, hide_index=True, width=1000, height=450)

    return pitcher_detail_dataframe_container


def pitch_mix_detail(pmix_data, current_time, dropdown_container=None, dataframe_container=None):
    # Sort and fill NaN values
    pmix_data = pmix_data.sort_values(by=['Pitcher', 'PC'], ascending=[True, False])
    pmix_data = pmix_data.fillna(0)

    # If no containers are provided (first call), set up the layout and placeholders
    if dropdown_container is None:
        st.markdown(f"<h1>MLB DW Live Game Tracker (Last Update {current_time})</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Pitch Mix Detailed Stats</h3>", unsafe_allow_html=True)
        dropdown_container = st.empty()  # Placeholder for the dropdown
        dataframe_container = st.empty()  # Placeholder for the dataframe

    # Get unique pitcher names and add "All Pitchers" option
    pitcher_options = ['All Pitchers'] + sorted(pmix_data['Pitcher'].unique().tolist())

    # Add dropdown for pitcher selection within its placeholder
    with dropdown_container.container():
        selected_pitcher = st.selectbox("Select Pitcher", pitcher_options, key=f"pitch_mix_select_{id(pmix_data)}")

    # Filter dataframe based on selected pitcher
    if selected_pitcher == 'All Pitchers':
        filtered_data = pmix_data
    else:
        filtered_data = pmix_data[pmix_data['Pitcher'] == selected_pitcher]

    # Apply styling to the filtered dataframe
    styled_df = filtered_data.style.format({
        'PC': '{:.0f}', 'Velo': '{:.1f}', 'Whiffs': '{:.0f}',
        'SwStr%': '{:.1%}', 'Strike%': '{:.1%}', 'Ball%': '{:.1%}',
        'Brl%': '{:.1%}'
    })

    # Update the dataframe placeholder with the new dataframe
    with dataframe_container.container():
        dataframe_container.dataframe(styled_df, hide_index=True, width=1000, height=700)

    # Return both containers to reuse them in the next call
    return dropdown_container, dataframe_container

def pitch_mix_detail3(pmix_data, current_time, dataframe_container=None):
    # Sort and fill NaN values
    pmix_data = pmix_data.sort_values(by=['Pitcher', 'PC'], ascending=[True, False])
    pmix_data = pmix_data.fillna(0)

    # If no container is provided (first call), set up the layout and placeholder
    if dataframe_container is None:
        st.markdown(f"<h1>MLB DW Live Game Tracker (Last Update {current_time})</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Pitch Mix Detailed Stats</h3>", unsafe_allow_html=True)
        dataframe_container = st.empty()  # Placeholder for the dataframe

    # Get unique pitcher names and add "All Pitchers" option
    pitcher_options = ['All Pitchers'] + sorted(pmix_data['Pitcher'].unique().tolist())

    # Add dropdown for pitcher selection (outside the container to persist)
    selected_pitcher = st.selectbox("Select Pitcher", pitcher_options, key=f"pitch_mix_select_{id(pmix_data)}")

    # Filter dataframe based on selected pitcher
    if selected_pitcher == 'All Pitchers':
        filtered_data = pmix_data
    else:
        filtered_data = pmix_data[pmix_data['Pitcher'] == selected_pitcher]

    # Apply styling to the filtered dataframe
    styled_df = filtered_data.style.format({
        'PC': '{:.0f}', 'Velo': '{:.1f}', 'Whiffs': '{:.0f}',
        'SwStr%': '{:.1%}', 'Strike%': '{:.1%}', 'Ball%': '{:.1%}',
        'Brl%': '{:.1%}'
    })

    # Update the placeholder with the new dataframe
    dataframe_container.dataframe(styled_df, hide_index=True, width=1000, height=700)

    # Return the container to reuse it in the next call
    return dataframe_container

def pitch_mix_detail2(pmix_data, current_time):
    # Sort and fill NaN values
    pmix_data = pmix_data.sort_values(by=['Pitcher', 'PC'], ascending=[True, False])
    pmix_data = pmix_data.fillna(0)

    # Header and title
    st.markdown(f"<h1>MLB DW Live Game Tracker (Last Update {current_time})</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Pitch Mix Detailed Stats</h3>", unsafe_allow_html=True)

    # Get unique pitcher names and add "All Pitchers" option
    pitcher_options = ['All Pitchers'] + sorted(pmix_data['Pitcher'].unique().tolist())

    # Add dropdown for pitcher selection
    selected_pitcher = st.selectbox("Select Pitcher", pitcher_options)

    # Filter dataframe based on selected pitcher
    if selected_pitcher == 'All Pitchers':
        filtered_data = pmix_data
    else:
        filtered_data = pmix_data[pmix_data['Pitcher'] == selected_pitcher]

    # Apply styling to the filtered dataframe
    styled_df = filtered_data.style.format({
        'PC': '{:.0f}', 'Velo': '{:.1f}', 'Whiffs': '{:.0f}',
        'SwStr%': '{:.1%}', 'Strike%': '{:.1%}', 'Ball%': '{:.1%}',
        'Brl%': '{:.1%}'
    })

    # Display the dataframe
    st.dataframe(styled_df, hide_index=True, width=1000, height=700)

def exit_velo_page(hrs, evs, current_time, hr_container=None, ev_container=None):
    # Filter EVs > 90
    evs = evs[evs['EV'] > 90]

    # If no containers are provided (first call), set up the layout and placeholders
    if hr_container is None:
        st.markdown(f"<h1>MLB DW Live Game Tracker (Last Update {current_time})</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Homers</h3>", unsafe_allow_html=True)
        hr_container = st.empty()  # Placeholder for homers dataframe
        st.markdown("<h3>Hardest Hit Balls</h3>", unsafe_allow_html=True)
        ev_container = st.empty()  # Placeholder for hardest hit balls dataframe

    # Update the placeholders with the new dataframes
    with hr_container.container():
        hr_container.dataframe(hrs, hide_index=True, width=900)

    with ev_container.container():
        ev_container.dataframe(evs, hide_index=True, width=900, height=600)

    # Return the containers to reuse them in the next call
    return hr_container, ev_container

def exit_velo_page2(hrs,evs,current_time):
    evs = evs[evs['EV']>90]
    st.markdown(f"<h1>MLB DW Live Game Tracker (Last Update {current_time})</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Homers</h3>", unsafe_allow_html=True)
    st.dataframe(hrs,hide_index=True,width=900)
    st.markdown("<h3>Hardest Hit Balls</h3>", unsafe_allow_html=True)
    st.dataframe(evs,hide_index=True,width=900,height=600)

# Main app logic
def main():
    from datetime import datetime

    selected_page = sidebar_menu()
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)
    now_date = now_eastern.date()
    today_str = now_date.strftime("%Y-%m-%d")
    pitcher_detail_dataframe_container = None
    scoreboard_container = None
    hitting_container = None
    pitching_container = None
    pmix_container = None
    pitch_mix_dropdown_container = None
    ev_container = None
    hr_container = None

    while True:
        today_games = getLiveGames(today_str)
        gamestatuslist = []
        for game in today_games:
            game_status = game.get('game_status')
            gamestatuslist.append(game_status)
    
        check_status_result = 'Wait'
        while check_status_result == 'Wait':
            check_status_result = checkStatus(gamestatuslist)
            if check_status_result == 'live':
                
                ## LOOP THROUGH GAMES ##
                scoreboard_df = pd.DataFrame()
                all_hitboxes = pd.DataFrame()
                all_pitboxes = pd.DataFrame()
                livedb = pd.DataFrame()
                for game in today_games:
                    game_status = game.get('game_status')
                    if game_status != 'I' and game_status != 'F':
                        continue
                    game_box = get_game_logs(game)
                    return_boxes = getBoxDetails(game_box)
                    hitbox = return_boxes[0]
                    all_hitboxes = pd.concat([all_hitboxes,hitbox])
                    pitbox = return_boxes[1]
                    all_pitboxes = pd.concat([all_pitboxes,pitbox])
                    game_score = return_boxes[2]
                    scoreboard_df = pd.concat([scoreboard_df,game_score])
                    
                    ## PLAY BY PLAY STUFF
                    gamedb = get_MILB_PBP_Live(game)
                    if len(gamedb) == 0:
                        pass
                    else:
                        if gamedb['StatcastGame'].iloc[0] == 'Y':
                            gamedb['game_status'] = game_status
                            livedb = pd.concat([livedb,gamedb])
                        else:
                            pass

                livedb = savAddOns(livedb)
                finishedgames = livedb[livedb['game_status']=='F']
                finished_game_pitchers = list(finishedgames['player_name'].unique())
                lastpitcher = livedb.sort_values(by=['inning','at_bat_number','pitch_number'])[['PitcherTeam_aff','player_name','inning','at_bat_number','pitch_number']]
                cutlist=lastpitcher[['PitcherTeam_aff','player_name']].drop_duplicates()
                currentpitchers = cutlist.drop_duplicates(subset=['PitcherTeam_aff'],keep='last')
                cplist_1 = list(currentpitchers['player_name'])

                cplist = []
                for cp in cplist_1:
                    if cp in finished_game_pitchers:
                        pass
                    else:
                        cplist.append(cp)
                
                livedb['DP'] = np.where((livedb['play_desc'].str.contains('double play'))&(livedb['PA_flag']==1),1,0)

                p_data = getPData(livedb,all_pitboxes,cplist)
                pmix_data = getPMixData(livedb,cplist)

                ## Hitter stuff
                try:
                    hrs = livedb[livedb['IsHomer']==1][['BatterName','BatterTeam_aff','player_name','launch_speed','play_desc']].sort_values(by='launch_speed',ascending=False)
                    hrs.columns=['Hitter','Team','Pitcher','EV','Description']
                except:
                    hrs = pd.DataFrame(columns=['Hitter','Team','Pitcher','EV','Description'])
                
                try:
                    evs = livedb[['BatterName','BatterTeam_aff','player_name','launch_speed','play_desc']].sort_values(by='launch_speed',ascending=False)
                    evs.columns=['Hitter','Team','Pitcher','EV','Description']
                except:
                    evs = pd.DataFrame(columns=['Hitter','Team','Pitcher','EV','Description'])

                if selected_page == "Scores & Leaders":
                    import datetime
                    now_eastern = datetime.datetime.now(eastern)
                    current_time = now_eastern.strftime("%I:%M")
                    scoreboard_container, pitching_container, hitting_container = home_page(
                       scoreboard_df, all_hitboxes, all_pitboxes, current_time, 
                       scoreboard_container, pitching_container, hitting_container
                )
                    #home_page(scoreboard_df,all_hitboxes,all_pitboxes,current_time)
                elif selected_page == "Pitcher Detail":
                    import datetime
                    now_eastern = datetime.datetime.now(eastern)
                    current_time = now_eastern.strftime("%I:%M")
                    pitcher_detail_dataframe_container=pitcher_detail_page(p_data,current_time, pitcher_detail_dataframe_container)

                elif selected_page == 'Pitch Mix Data':
                    import datetime
                    now_eastern = datetime.datetime.now(eastern)
                    current_time = now_eastern.strftime("%I:%M")
                    pitch_mix_dropdown_container, pmix_container = pitch_mix_detail(
                        pmix_data, current_time, pitch_mix_dropdown_container, pmix_container
                    )
                elif selected_page == 'Exit Velos':
                    import datetime
                    now_eastern = datetime.datetime.now(eastern)
                    current_time = now_eastern.strftime("%I:%M")
                    #exit_velo_page(hrs,evs,current_time)
                    hr_container, ev_container = exit_velo_page(hrs, evs, current_time, hr_container, ev_container)
                

            elif check_status_result == 'Stop':
                st.write('All games today are complete.')
                st.stop()
            else:
                st.write('No games currently live, waiting ten minutes to check again...')
                check_status_result == 'Wait'
                time.sleep(60*10)
        
        secs_to_wait = 30
        #st.write(f'Waiting {secs_to_wait} seconds')
        time.sleep(secs_to_wait)


if __name__ == "__main__":
    main()