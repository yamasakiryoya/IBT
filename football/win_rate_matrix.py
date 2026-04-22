import pandas as pd
import numpy as np

df = pd.read_csv('Season_2425_PremierLeague.csv', usecols=['HomeTeam', 'AwayTeam', 'FTR'])
df = df.dropna(subset=['FTR'])

teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
n = len(teams)
team_idx = {t: i for i, t in enumerate(teams)}

wins = np.zeros((n, n))
games = np.zeros((n, n))

for _, row in df.iterrows():
    h = team_idx[row['HomeTeam']]
    a = team_idx[row['AwayTeam']]
    games[h][a] += 1
    games[a][h] += 1
    if row['FTR'] == 'H':
        print(h,a,'H')
        wins[h][a] += 1.
    elif row['FTR'] == 'A':
        print(h,a,'A')
        wins[a][h] += 1.
    elif row['FTR'] == 'D':
        print(h,a,'D')
        wins[h][a] += 0.5
        wins[a][h] += 0.5

win_rate = np.full((n, n), np.nan)
np.fill_diagonal(win_rate, np.nan)  # 自己対戦はNaN

for i in range(n):
    for j in range(n):
        if i != j and games[i][j] > 0:
            win_rate[i][j] = wins[i][j] / games[i][j]
win_df = pd.DataFrame(win_rate, index=teams, columns=teams)
win_df.to_csv('win_rate_matrix.csv')
