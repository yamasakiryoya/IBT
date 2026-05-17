import pandas as pd
import numpy as np

data = pd.read_csv('gl2024.txt', header=None)

teams = set()

for _, row in data.iterrows():
    visitor = row[3]
    home = row[6]
    teams.add(visitor)
    teams.add(home)

teams = sorted(list(teams))

wins = pd.DataFrame(0.0, index=teams, columns=teams)
total_games = pd.DataFrame(0, index=teams, columns=teams)

for _, row in data.iterrows():
    visitor = row[3]
    home = row[6]
    visitor_score = row[9]
    home_score = row[10]
    total_games.loc[visitor, home] += 1
    total_games.loc[home, visitor] += 1
    if visitor_score > home_score:
        wins.loc[visitor, home] += 1.0
    elif home_score > visitor_score:
        wins.loc[home, visitor] += 1.0
    else:
        wins.loc[visitor, home] += 0.5
        wins.loc[home, visitor] += 0.5

win_pct = pd.DataFrame(0.0, index=teams, columns=teams)

for team1 in teams:
    for team2 in teams:
        if team1 == team2:
            win_pct.loc[team1, team2] = np.nan
        elif total_games.loc[team1, team2] > 0:
            win_pct.loc[team1, team2] = wins.loc[team1, team2] / total_games.loc[team1, team2]
        else:
            win_pct.loc[team1, team2] = np.nan
win_pct.to_csv('win_rate_matrix.csv')