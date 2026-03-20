import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

INDIA_DATA_PATH = r"g:\ML\Cricket\Data\India_Specific_Combined.csv"

def load_data():
    print("Loading India Specialized data (IPL + WPL + India T20Is)...")
    df = pd.read_csv(INDIA_DATA_PATH, low_memory=False)
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'match_id', 'innings', 'over', 'ball'])
    print(f"Total Combined Indian Rows: {len(df)}")
    return df


def feature_engineer_match_winner(df):
    print("\n--- 1. MEGA XGBoost Match Winner Prediction ---")
    start_time = time.time()
    
    match_level = df.drop_duplicates(subset=['match_id']).copy()
    match_level = match_level.dropna(subset=['match_won_by'])
    
    match_level['team1'] = match_level['batting_team']
    match_level['team2'] = match_level['bowling_team']
    
    # 1. First Innings Runs
    first_innings = df[df['innings'] == 1].groupby('match_id')['runs_total'].sum().reset_index()
    first_innings.columns = ['match_id', 'first_innings_runs']
    match_level = match_level.merge(first_innings, on='match_id', how='left')
    
    # 2. Venues Average First Innings
    venue_avg_score = match_level.groupby('venue')['first_innings_runs'].mean().reset_index()
    venue_avg_score.columns = ['venue', 'venue_avg_first_innings']
    match_level = match_level.merge(venue_avg_score, on='venue', how='left')
    
    # 3. Global Win Rates
    team_wins = match_level['match_won_by'].value_counts()
    team_matches = match_level['team1'].value_counts() + match_level['team2'].value_counts()
    team_win_rates = (team_wins / team_matches).fillna(0.5).to_dict()
    
    match_level['team1_win_rate'] = match_level['team1'].map(team_win_rates)
    match_level['team2_win_rate'] = match_level['team2'].map(team_win_rates)
    
    # 4. Head to Head Win Rates (NEW)
    print("Calculating Head-to-Head win rates...")
    matches = match_level[['team1', 'team2', 'match_won_by']].dropna().copy()
    
    def get_matchup(t1, t2):
        return tuple(sorted([str(t1), str(t2)]))
        
    matches['matchup'] = matches.apply(lambda x: get_matchup(x['team1'], x['team2']), axis=1)
    
    h2h_wins = matches.groupby(['matchup', 'match_won_by']).size().reset_index(name='wins')
    h2h_dict = {}
    for _, row in h2h_wins.iterrows():
        matchup = row['matchup']
        winner = str(row['match_won_by'])
        wins = row['wins']
        if matchup not in h2h_dict:
            h2h_dict[matchup] = {}
        h2h_dict[matchup][winner] = wins
        
    def get_h2h_win_rate(t1, t2):
        t1, t2 = str(t1), str(t2)
        matchup = get_matchup(t1, t2)
        match_stats = h2h_dict.get(matchup, {})
        t1_wins = match_stats.get(t1, 0)
        t2_wins = match_stats.get(t2, 0)
        total = t1_wins + t2_wins
        return t1_wins / total if total > 0 else 0.5

    match_level['team1_h2h_win_rate'] = match_level.apply(lambda x: get_h2h_win_rate(x['team1'], x['team2']), axis=1)
    
    match_level['toss_winner_is_team1'] = (match_level['toss_winner'] == match_level['team1']).astype(int)
    match_level['toss_decision_bat'] = (match_level['toss_decision'] == 'bat').astype(int)
    
    features = match_level[['venue', 'team1', 'team2', 'toss_winner_is_team1', 'toss_decision_bat', 
                            'team1_win_rate', 'team2_win_rate', 'team1_h2h_win_rate', 'venue_avg_first_innings', 'match_won_by']]
    features = features.dropna()
    
    le_venue = LabelEncoder()
    le_team = LabelEncoder()
    
    all_teams = pd.concat([features['team1'], features['team2'], features['match_won_by']]).unique()
    le_team.fit(all_teams)
    
    features['venue'] = le_venue.fit_transform(features['venue'])
    features['team1'] = le_team.transform(features['team1'])
    features['team2'] = le_team.transform(features['team2'])
    features['target'] = le_team.transform(features['match_won_by'])
    
    X = features[['venue', 'team1', 'team2', 'toss_winner_is_team1', 'toss_decision_bat', 
                  'team1_win_rate', 'team2_win_rate', 'team1_h2h_win_rate', 'venue_avg_first_innings']]
    y = features['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    unique_classes = np.sort(np.unique(y_train))
    mapping = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
    y_train_mapped = y_train.map(mapping)
    y_test_mapped = y_test.map(mapping).fillna(-1)
    
    X_test = X_test[y_test_mapped != -1]
    y_test_mapped = y_test_mapped[y_test_mapped != -1]
    
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(n_estimators=700, max_depth=8, learning_rate=0.01, random_state=42, verbose=-1)
    clf.fit(X_train, y_train_mapped)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test_mapped, preds)
    print(f"-> MEGA Match Winner Accuracy: {acc*100:.2f}% (Time: {time.time()-start_time:.2f}s)")
    return clf

def extract_player_features(df):
    """Common feature extraction for individual player predictions"""
    print("Extracting detailed player features...")
    
    # 1. Determine aggregate match performance
    player_match = df.groupby(['match_id', 'date', 'batter', 'batting_team', 'bowling_team', 'venue']).agg(
        runs_scored=('runs_batter', 'sum'),
        balls_faced=('ball', 'count')
    ).reset_index()
    
    # 2. Extract Approximate Batting Position (HUGE predictor for runs)
    print("Calculating batting positions...")
    batting_order = df.groupby(['match_id', 'innings', 'batter'], as_index=False)['over'].min()
    batting_order = batting_order.sort_values(['match_id', 'innings', 'over'])
    batting_order['batting_position'] = batting_order.groupby(['match_id', 'innings']).cumcount() + 1
    
    player_match = player_match.merge(batting_order[['match_id', 'batter', 'batting_position']], on=['match_id', 'batter'], how='left')
    player_match['batting_position'] = player_match['batting_position'].fillna(7) # default to tail-end
    
    # 3. Compute temporal (career) features
    print("Calculating career & temporal features...")
    player_match = player_match.sort_values(by=['batter', 'date'])
    
    # Career Averages
    player_match['career_runs_avg'] = player_match.groupby('batter')['runs_scored'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    
    # Career Strike Rate
    player_match['career_balls_faced'] = player_match.groupby('batter')['balls_faced'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
    player_match['career_total_runs'] = player_match.groupby('batter')['runs_scored'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
    player_match['career_strike_rate'] = np.where(player_match['career_balls_faced'] > 0, 
                                                 (player_match['career_total_runs'] / player_match['career_balls_faced']) * 100, 
                                                 0)
    
    # Recent Form (last 5 matches)
    player_match['recent_form_5'] = player_match.groupby('batter')['runs_scored'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
    
    # Quick Out specific features
    player_match['quick_out'] = ((player_match['runs_scored'] < 10) & (player_match['balls_faced'] <= 10)).astype(int)
    player_match['career_quick_out_rate'] = player_match.groupby('batter')['quick_out'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    
    # Venue Specific features
    venue_avg = player_match.groupby(['batter', 'venue'])['runs_scored'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    player_match['venue_avg_runs'] = venue_avg
    
    return player_match.dropna()

def feature_engineer_player_runs(player_match):
    print("\n--- 2. MEGA XGBoost Player Runs Prediction ---")
    start_time = time.time()
    
    le_batter = LabelEncoder()
    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    
    player_match['batter_encoded'] = le_batter.fit_transform(player_match['batter'])
    
    all_teams = pd.concat([player_match['batting_team'], player_match['bowling_team']]).unique()
    le_team.fit(all_teams)
    player_match['batting_team_encoded'] = le_team.transform(player_match['batting_team'])
    player_match['bowling_team_encoded'] = le_team.transform(player_match['bowling_team'])
    
    player_match['venue_encoded'] = le_venue.fit_transform(player_match['venue'])
    
    X = player_match[['batter_encoded', 'batting_team_encoded', 'bowling_team_encoded', 'venue_encoded', 
                      'career_runs_avg', 'recent_form_5', 'batting_position', 'career_strike_rate', 'venue_avg_runs']]
    y = player_match['runs_scored']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    from lightgbm import LGBMRegressor
    reg = LGBMRegressor(n_estimators=1000, max_depth=8, learning_rate=0.01, random_state=42, verbose=-1)
    reg.fit(X_train, y_train)
    
    preds = reg.predict(X_test)
    preds = np.maximum(0, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"-> MEGA Player Runs Predictor MAE: {mae:.2f} runs (Time: {time.time()-start_time:.2f}s)")
    return reg

def feature_engineer_quick_out(player_match):
    print("\n--- 3. MEGA XGBoost Quick Out Prediction ---")
    start_time = time.time()
    
    X = player_match[['batting_team_encoded', 'bowling_team_encoded', 'venue_encoded', 
                      'career_quick_out_rate', 'recent_form_5', 'batting_position', 'career_strike_rate']]
    y = player_match['quick_out']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(n_estimators=700, max_depth=8, learning_rate=0.01, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"-> MEGA Quick Out Accuracy: {acc*100:.2f}% (Time: {time.time()-start_time:.2f}s)")
    return clf

if __name__ == "__main__":
    df = load_data()
    feature_engineer_match_winner(df)
    
    player_match = extract_player_features(df)
    feature_engineer_player_runs(player_match)
    feature_engineer_quick_out(player_match)
    print("\n[Done executing MEGA XGBoost pipelines]")
