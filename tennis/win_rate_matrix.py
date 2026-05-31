import pandas as pd
import numpy as np
import io
import requests

Y=2025

def get_max_player_count(years=[2025]):
    all_names = set()
    for year in years:
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        print(f"Scanning {year} data...")
        try:
            s = requests.get(url).content
            df = pd.read_csv(io.StringIO(s.decode('utf-8')))
            current_players = pd.concat([df['winner_name'], df['loser_name']]).dropna().unique()
            all_names.update(current_players)
        except Exception as e:
            print(f"Skipping {year} due to error: {e}")
    return len(all_names), sorted(list(all_names))

max_count, player_list = get_max_player_count(years=[Y])
print(max_count, player_list)
N = max_count

def generate_tennis_win_matrix(year=2025, top_n=100):
    url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
    print(f"Downloading ATP matches for {year}...")
    try:
        s = requests.get(url).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None, None
    matches = df[['winner_name', 'loser_name']].dropna()
    all_players = pd.concat([matches['winner_name'], matches['loser_name']])
    player_counts = all_players.value_counts()
    top_players = player_counts.head(top_n).index.tolist()
    top_players.sort()
    player_to_idx = {name: i for i, name in enumerate(top_players)}
    n = len(top_players)
    match_counts = np.zeros((n, n))
    win_counts = np.zeros((n, n))
    print(f"Processing {len(matches)} matches for top {n} players...")
    for _, row in matches.iterrows():
        w = row['winner_name']
        l = row['loser_name']
        if w in player_to_idx and l in player_to_idx:
            idx_w = player_to_idx[w]
            idx_l = player_to_idx[l]
            match_counts[idx_w, idx_l] += 1
            match_counts[idx_l, idx_w] += 1
            win_counts[idx_w, idx_l] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        win_rate_matrix = np.where(match_counts > 0, win_counts / match_counts, -1.0)
    return win_rate_matrix, top_players

if __name__ == "__main__":
    win_matrix, player_names = generate_tennis_win_matrix(year=Y, top_n=N)
    if win_matrix is not None:
        print(f"Success! Matrix shape: {win_matrix.shape}")
        np.savetxt(f"tennis_win_rate_{Y}_{N}.csv", win_matrix, delimiter=",")
        i, j = 0, 1
        while win_matrix[i, j] == -1:
            i += 1
        if win_matrix[i, j] != -1:
            s = win_matrix[i, j] + win_matrix[j, i]
            print(f"Verification for {player_names[i]} vs {player_names[j]}: Sum = {s}")