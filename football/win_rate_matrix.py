"""
Season 2024/25 Premier League 勝率行列生成スクリプト
win_rate[i][j] = チームiがチームjと対戦したときの勝率
"""

import pandas as pd
import numpy as np

# データ読み込み
df = pd.read_csv('Season_2425_PremierLeague.csv', usecols=['HomeTeam', 'AwayTeam', 'FTR'])
df = df.dropna(subset=['FTR'])

# チームリスト（ソート済み）
teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
n = len(teams)
team_idx = {t: i for i, t in enumerate(teams)}

# 対戦ごとの勝利数・試合数カウント
wins = np.zeros((n, n))    # wins[i][j]: チームiがチームjに勝った回数
games = np.zeros((n, n))   # games[i][j]: チームiとチームjの対戦数

for _, row in df.iterrows():
    h = team_idx[row['HomeTeam']]
    a = team_idx[row['AwayTeam']]
    games[h][a] += 1
    games[a][h] += 1
    if row['FTR'] == 'H':      # ホーム勝ち
        print(h,a,'H')
        wins[h][a] += 1.
    elif row['FTR'] == 'A':    # アウェイ勝ち
        print(h,a,'A')
        wins[a][h] += 1.
    elif row['FTR'] == 'D':    # 引き分け
        print(h,a,'D')
        wins[h][a] += 0.5
        wins[a][h] += 0.5
    # 引き分けは勝利なし

# 勝率行列 (0.5 on diagonal where no games played, NaN is not needed)
win_rate = np.full((n, n), np.nan)
np.fill_diagonal(win_rate, np.nan)  # 自己対戦はNaN

for i in range(n):
    for j in range(n):
        if i != j and games[i][j] > 0:
            win_rate[i][j] = wins[i][j] / games[i][j]

# DataFrameとして出力
win_df = pd.DataFrame(win_rate, index=teams, columns=teams)

# CSV保存（行=チームi、列=チームj、セル=チームiのチームjへの勝率）
win_df.to_csv('win_rate_matrix.csv')
# print("win_rate_matrix.csv を保存しました。")
# print()
# print(f"チーム数: {n}")
# print(f"試合数: {len(df)}")
# print()

# # 整形表示
# pd.set_option('display.max_columns', 21)
# pd.set_option('display.width', 200)
# pd.set_option('display.float_format', '{:.3f}'.format)
# print("20×20 勝率行列（行=チームi、列=チームj、値=チームiのチームjへの勝率）:")
# print(win_df.to_string())
