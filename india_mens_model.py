import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
INDIA_DATA_PATH = r"g:\ML\Cricket\Data\India_Mens_Combined.csv"

def load_data():
    print("Loading India Mens Specialized data...")
    df = pd.read_csv(INDIA_DATA_PATH, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values(by=['date', 'match_id', 'innings', 'over', 'ball'])
    return df

def extract_batter_bowler_h2h(df):
    bb = df.groupby(['match_id', 'date', 'batter', 'bowler']).agg(
        runs_scored=('runs_batter', 'sum'),
        balls_faced=('ball', 'count'),
        dismissals=('player_out', lambda x: x.notna().sum())
    ).reset_index()
    bb = bb.sort_values(by=['date'])
    bb['hist_runs'] = bb.groupby(['batter', 'bowler'])['runs_scored'].transform(lambda x: x.shift().expanding().sum()).fillna(0)
    bb['hist_balls'] = bb.groupby(['batter', 'bowler'])['balls_faced'].transform(lambda x: x.shift().expanding().sum()).fillna(0)
    bb['hist_dismiss'] = bb.groupby(['batter', 'bowler'])['dismissals'].transform(lambda x: x.shift().expanding().sum()).fillna(0)
    
    batter_attack = bb.groupby(['match_id', 'batter']).agg(
        opp_attack_hist_runs=('hist_runs', 'sum'),
        opp_attack_hist_balls=('hist_balls', 'sum'),
        opp_attack_hist_dismiss=('hist_dismiss', 'sum')
    ).reset_index()
    batter_attack['opp_attack_hist_sr'] = np.where(batter_attack['opp_attack_hist_balls'] > 0, 
                                                   (batter_attack['opp_attack_hist_runs'] / batter_attack['opp_attack_hist_balls']) * 100, 0)
    return batter_attack

def extract_player_features(df):
    print("Extracting ultimate player structural features...")
    player_match = df.groupby(['match_id', 'date', 'match_format', 'batter', 'batting_team', 'bowling_team', 'venue']).agg(
        runs_scored=('runs_batter', 'sum'),
        balls_faced=('ball', 'count')
    ).reset_index()
    
    batting_order = df.groupby(['match_id', 'innings', 'batter'], as_index=False)['over'].min()
    batting_order = batting_order.sort_values(['match_id', 'innings', 'over'])
    batting_order['batting_position'] = batting_order.groupby(['match_id', 'innings']).cumcount() + 1
    
    player_match = player_match.merge(batting_order[['match_id', 'batter', 'batting_position']], on=['match_id', 'batter'], how='left')
    player_match['batting_position'] = player_match['batting_position'].fillna(7)
    player_match = player_match.sort_values(by=['batter', 'date'])
    
    player_match['career_runs_avg'] = player_match.groupby(['batter', 'match_format'])['runs_scored'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    player_match['career_balls_faced'] = player_match.groupby(['batter', 'match_format'])['balls_faced'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
    player_match['career_total_runs'] = player_match.groupby(['batter', 'match_format'])['runs_scored'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
    player_match['career_strike_rate'] = np.where(player_match['career_balls_faced'] > 0, 
                                                 (player_match['career_total_runs'] / player_match['career_balls_faced']) * 100, 0)
    player_match['recent_form_5'] = player_match.groupby(['batter', 'match_format'])['runs_scored'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
    player_match['quick_out'] = ((player_match['runs_scored'] < 10) & (player_match['balls_faced'] <= 10)).astype(int)
    player_match['career_quick_out_rate'] = player_match.groupby(['batter', 'match_format'])['quick_out'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    player_match['venue_avg_runs'] = player_match.groupby(['batter', 'venue'])['runs_scored'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    
    player_match['is_t20'] = player_match['match_format'].apply(lambda x: 1 if 'T20' in x or 'IPL' in x else 0)
    player_match['is_odi'] = player_match['match_format'].apply(lambda x: 1 if 'ODI' in x else 0)
    
    batter_bb_attack_metrics = extract_batter_bowler_h2h(df)
    player_match = player_match.merge(batter_bb_attack_metrics, on=['match_id', 'batter'], how='left').fillna(0)
    
    # Save the absolute latest generic stats per player for the Live Predictor UI
    print("Saving live player form database...")
    latest_player_stats = player_match.groupby('batter').last().reset_index()
    latest_player_stats.to_csv('models/latest_player_stats.csv', index=False)
    
    return player_match

def feature_engineer_match_winner(df, player_match):
    print("\n--- MEGA ENSEMBLE Match Winner Prediction ---")
    start_time = time.time()
    
    roster_strength = player_match.groupby(['match_id', 'batting_team']).agg(
        team_career_runs=('career_runs_avg', 'sum'),
        team_recent_form=('recent_form_5', 'sum'),
        team_career_sr=('career_strike_rate', 'mean')
    ).reset_index()
    
    match_level = df.drop_duplicates(subset=['match_id']).copy()
    match_level = match_level.dropna(subset=['match_won_by'])
    match_level['team1'] = match_level['batting_team']
    match_level['team2'] = match_level['bowling_team']
    
    t1_strength = roster_strength.rename(columns={'batting_team': 'team1', 'team_career_runs':'t1_runs', 'team_recent_form':'t1_form', 'team_career_sr':'t1_sr'})
    match_level = match_level.merge(t1_strength, on=['match_id', 'team1'], how='left')
    
    t2_strength = roster_strength.rename(columns={'batting_team': 'team2', 'team_career_runs':'t2_runs', 'team_recent_form':'t2_form', 'team_career_sr':'t2_sr'})
    match_level = match_level.merge(t2_strength, on=['match_id', 'team2'], how='left').fillna(0)

    first_innings = df[df['innings'] == 1].groupby('match_id').agg(
        first_innings_runs=('runs_total', 'sum'),
        first_innings_wickets=('player_out', lambda x: x.notna().sum()),
        first_innings_balls=('ball', 'count')
    ).reset_index()
    match_level = match_level.merge(first_innings, on='match_id', how='left').fillna(0)
    
    match_level['first_innings_rr'] = (match_level['first_innings_runs'] / match_level['first_innings_balls']) * 6
    
    second_innings = df[df['innings'] == 2].groupby('match_id').agg(
        second_innings_wickets=('player_out', lambda x: x.notna().sum())
    ).reset_index()
    match_level = match_level.merge(second_innings, on='match_id', how='left').fillna(0)
    
    pp_wickets = df[(df['innings'] == 1) & (df['over'] < 6)].groupby('match_id').agg(
        powerplay_wickets=('player_out', lambda x: x.notna().sum())
    ).reset_index()
    match_level = match_level.merge(pp_wickets, on='match_id', how='left').fillna(0)
    
    venue_avg_score = match_level.groupby('venue')['first_innings_runs'].mean().reset_index()
    venue_avg_score.columns = ['venue', 'venue_avg_first_innings']
    match_level = match_level.merge(venue_avg_score, on='venue', how='left')
    
    # Save the venue first innings averages!
    venue_avg_score.to_csv('models/venue_avg_score.csv', index=False)
    
    match_level['score_above_venue_avg'] = match_level['first_innings_runs'] - match_level['venue_avg_first_innings']
    match_level['toss_winner_is_team1'] = (match_level['toss_winner'] == match_level['team1']).astype(int)
    match_level['toss_decision_bat'] = (match_level['toss_decision'] == 'bat').astype(int)
    match_level['toss_winner_won_match'] = (match_level['toss_winner'] == match_level['match_won_by']).astype(int)
    
    venue_toss_win_rate_df = match_level.groupby('venue')['toss_winner_won_match'].mean().reset_index(name='venue_toss_win_rate_final')
    venue_toss_win_rate_df.to_csv('models/venue_toss_win_rate.csv', index=False)
    
    match_level['venue_toss_win_rate'] = match_level.groupby('venue')['toss_winner_won_match'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    
    match_level['chasing_team_won'] = (match_level['match_won_by'] == match_level['team2']).astype(int)
    
    venue_chase_win_rate_df = match_level.groupby('venue')['chasing_team_won'].mean().reset_index(name='venue_chase_win_rate_final')
    venue_chase_win_rate_df.to_csv('models/venue_chase_win_rate.csv', index=False)
    
    match_level['venue_chase_win_rate'] = match_level.groupby('venue')['chasing_team_won'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    
    team_wins = match_level['match_won_by'].value_counts()
    team_matches = match_level['team1'].value_counts() + match_level['team2'].value_counts()
    team_win_rates = (team_wins / team_matches).fillna(0.5).to_dict()
    joblib.dump(team_win_rates, 'models/team_win_rates.pkl')
    
    match_level['team1_win_rate'] = match_level['team1'].map(team_win_rates).fillna(0.5)
    match_level['team2_win_rate'] = match_level['team2'].map(team_win_rates).fillna(0.5)
    
    matches = match_level[['team1', 'team2', 'match_won_by']].dropna().copy()
    def get_matchup(t1, t2): return tuple(sorted([str(t1), str(t2)]))
    matches['matchup'] = matches.apply(lambda x: get_matchup(x['team1'], x['team2']), axis=1)
    
    h2h_wins = matches.groupby(['matchup', 'match_won_by']).size().reset_index(name='wins')
    h2h_dict = {}
    for _, row in h2h_wins.iterrows():
        matchup = row['matchup']
        winner = str(row['match_won_by'])
        if matchup not in h2h_dict: h2h_dict[matchup] = {}
        h2h_dict[matchup][winner] = row['wins']
    joblib.dump(h2h_dict, 'models/h2h_dict.pkl')
    
    def get_h2h_win_rate(t1, t2):
        t1, t2 = str(t1), str(t2)
        matchup = get_matchup(t1, t2)
        match_stats = h2h_dict.get(matchup, {})
        total = match_stats.get(t1, 0) + match_stats.get(t2, 0)
        return match_stats.get(t1, 0) / total if total > 0 else 0.5

    match_level['team1_h2h_win_rate'] = match_level.apply(lambda x: get_h2h_win_rate(x['team1'], x['team2']), axis=1)
    match_level['is_t20'] = match_level['match_format'].apply(lambda x: 1 if 'T20' in x or 'IPL' in x else 0)
    match_level['is_odi'] = match_level['match_format'].apply(lambda x: 1 if 'ODI' in x else 0)
    
    features = match_level[['venue', 'team1', 'team2', 'toss_winner_is_team1', 'toss_decision_bat', 'venue_toss_win_rate', 'venue_chase_win_rate',
                            'team1_win_rate', 'team2_win_rate', 'team1_h2h_win_rate', 'venue_avg_first_innings', 'score_above_venue_avg',
                            't1_runs', 't1_form', 't1_sr', 't2_runs', 't2_form', 't2_sr',
                            'first_innings_runs', 'first_innings_wickets', 'first_innings_rr', 'powerplay_wickets',
                            'second_innings_wickets', 'is_t20', 'is_odi', 'match_won_by']]
    features = features.dropna()
    
    le_venue = LabelEncoder()
    le_team = LabelEncoder()
    all_teams = pd.concat([features['team1'], features['team2'], features['match_won_by']]).unique()
    le_team.fit(all_teams)
    
    features['venue_encoded'] = le_venue.fit_transform(features['venue'])
    features['team1_encoded'] = le_team.transform(features['team1'])
    features['team2_encoded'] = le_team.transform(features['team2'])
    features['target'] = le_team.transform(features['match_won_by'])
    
    joblib.dump(le_venue, 'models/le_venue.pkl')
    joblib.dump(le_team, 'models/le_team.pkl')
    
    features['roster_form_diff'] = features['t1_form'] - features['t2_form']
    features['roster_runs_diff'] = features['t1_runs'] - features['t2_runs']
    features['sr_diff'] = features['t1_sr'] - features['t2_sr']
    
    X = features[['venue_encoded', 'team1_encoded', 'team2_encoded', 'toss_winner_is_team1', 'toss_decision_bat', 
                  'venue_toss_win_rate', 'venue_chase_win_rate', 'team1_win_rate', 'team2_win_rate', 'team1_h2h_win_rate', 'venue_avg_first_innings',
                  'score_above_venue_avg', 't1_runs', 't1_form', 't1_sr', 't2_runs', 't2_form', 't2_sr',
                  'roster_form_diff', 'roster_runs_diff', 'sr_diff',
                  'first_innings_runs', 'first_innings_wickets', 'first_innings_rr', 'powerplay_wickets',
                  'second_innings_wickets', 'is_t20', 'is_odi']]
    y = features['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    print("Training XGBoost + LightGBM + Random Forest + CatBoost Hybrid Hive...")
    lgbm = LGBMClassifier(n_estimators=1000, max_depth=9, learning_rate=0.015, random_state=42, verbose=-1)
    xgb = XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.03, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    cb = CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.02, random_seed=42, verbose=0)
    
    clf = VotingClassifier(estimators=[('lgbm', lgbm), ('xgb', xgb), ('rf', rf), ('cb', cb)], voting='soft')
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"-> GIGANTIC BOOST MATCH WINNER ACCURACY: {acc*100:.2f}% (Time: {time.time()-start_time:.2f}s)")
    
    joblib.dump(clf, 'models/match_winner_ensemble.pkl')
    return clf

def feature_engineer_player_runs(player_match):
    print("\n--- ENSEMBLE Player Runs Prediction ---")
    start_time = time.time()
    
    le_batter = LabelEncoder()
    player_match['batter_encoded'] = le_batter.fit_transform(player_match['batter'])
    joblib.dump(le_batter, 'models/le_batter.pkl')
    
    le_team = joblib.load('models/le_team.pkl')
    le_venue = joblib.load('models/le_venue.pkl')
    
    # Filter out unknowns safely
    player_match = player_match[player_match['batting_team'].isin(le_team.classes_)]
    player_match = player_match[player_match['bowling_team'].isin(le_team.classes_)]
    player_match = player_match[player_match['venue'].isin(le_venue.classes_)]
    
    player_match['batting_team_encoded'] = le_team.transform(player_match['batting_team'])
    player_match['bowling_team_encoded'] = le_team.transform(player_match['bowling_team'])
    player_match['venue_encoded'] = le_venue.transform(player_match['venue'])
    
    X = player_match[[
        'batter_encoded', 'batting_team_encoded', 'bowling_team_encoded', 'venue_encoded', 
        'career_runs_avg', 'recent_form_5', 'batting_position', 'career_strike_rate', 'venue_avg_runs',
        'opp_attack_hist_dismiss', 'opp_attack_hist_sr', 'opp_attack_hist_runs', 'is_t20', 'is_odi'
    ]]
    y = player_match['runs_scored']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    reg = LGBMRegressor(n_estimators=1500, max_depth=10, learning_rate=0.01, random_state=42, verbose=-1)
    reg.fit(X_train, y_train)
    
    preds = np.maximum(0, reg.predict(X_test))
    mae = mean_absolute_error(y_test, preds)
    print(f"-> Player Runs Predictor MAE: {mae:.2f} runs (Time: {time.time()-start_time:.2f}s)")
    
    joblib.dump(reg, 'models/player_runs_model.pkl')
    return reg

def feature_engineer_quick_out(player_match):
    print("\n--- ENSEMBLE Quick Out Prediction ---")
    start_time = time.time()
    
    le_batter = joblib.load('models/le_batter.pkl')
    le_team = joblib.load('models/le_team.pkl')
    le_venue = joblib.load('models/le_venue.pkl')
    
    # Filter safely
    pm_filtered = player_match[player_match['batter'].isin(le_batter.classes_)].copy()
    pm_filtered = pm_filtered[pm_filtered['batting_team'].isin(le_team.classes_)]
    pm_filtered = pm_filtered[pm_filtered['bowling_team'].isin(le_team.classes_)]
    pm_filtered = pm_filtered[pm_filtered['venue'].isin(le_venue.classes_)]
    
    pm_filtered['batter_encoded'] = le_batter.transform(pm_filtered['batter'])
    pm_filtered['batting_team_encoded'] = le_team.transform(pm_filtered['batting_team'])
    pm_filtered['bowling_team_encoded'] = le_team.transform(pm_filtered['bowling_team'])
    pm_filtered['venue_encoded'] = le_venue.transform(pm_filtered['venue'])
    
    X = pm_filtered[[
        'batter_encoded', 'batting_team_encoded', 'bowling_team_encoded', 'venue_encoded', 
        'career_quick_out_rate', 'recent_form_5', 'batting_position', 'career_strike_rate',
        'opp_attack_hist_dismiss', 'opp_attack_hist_sr', 'is_t20', 'is_odi'
    ]]
    y = pm_filtered['quick_out']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    clf = LGBMClassifier(n_estimators=1000, max_depth=9, learning_rate=0.015, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    print(f"-> Quick Out Accuracy: {accuracy_score(y_test, preds)*100:.2f}% (Time: {time.time()-start_time:.2f}s)")
    joblib.dump(clf, 'models/quick_out_model.pkl')
    return clf

if __name__ == "__main__":
    df = load_data()
    if len(df) > 0:
        player_match = extract_player_features(df)
        feature_engineer_match_winner(df, player_match)
        feature_engineer_player_runs(player_match)
        feature_engineer_quick_out(player_match)
        print("\n[Done executing and SAVING all pipelines]")
