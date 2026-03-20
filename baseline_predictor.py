import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import warnings
warnings.filterwarnings('ignore')

MATCH_DATA_PATH = r"g:\ML\Cricket\Data\IPL.csv"

def load_data():
    print("Loading data...")
    df = pd.read_csv(MATCH_DATA_PATH, low_memory=False)
    return df

def build_match_winner_model(df):
    print("--- 1. Building Match Winner Prediction Model ---")
    
    # Extract match level data taking first ball of each match
    match_level = df.drop_duplicates(subset=['match_id'])
    
    # Filter out matches with no clear winner (washouts etc.)
    match_level = match_level.dropna(subset=['match_won_by'])
    
    match_level['team1'] = match_level['batting_team']
    match_level['team2'] = match_level['bowling_team']
    
    features = match_level[['venue', 'team1', 'team2', 'toss_winner', 'toss_decision', 'match_won_by']]
    features = features.dropna()
    
    le_venue = LabelEncoder()
    le_team = LabelEncoder()
    le_toss_dec = LabelEncoder()
    
    all_teams = pd.concat([features['team1'], features['team2'], features['toss_winner'], features['match_won_by']]).unique()
    le_team.fit(all_teams)
    
    features['venue'] = le_venue.fit_transform(features['venue'])
    features['team1'] = le_team.transform(features['team1'])
    features['team2'] = le_team.transform(features['team2'])
    features['toss_winner'] = le_team.transform(features['toss_winner'])
    features['toss_decision'] = le_toss_dec.fit_transform(features['toss_decision'])
    features['target'] = le_team.transform(features['match_won_by'])
    
    X = features[['venue', 'team1', 'team2', 'toss_winner', 'toss_decision']]
    y = features['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    print(f"Match Winner Accuracy: {accuracy_score(y_test, preds)*100:.2f}%\n")

def build_player_runs_model(df):
    print("--- 2. Building Player Runs Prediction Model ---")
    
    player_match = df.groupby(['match_id', 'batter', 'batting_team', 'venue']).agg(
        runs_scored=('runs_batter', 'sum')
    ).reset_index()
    
    player_match = player_match.dropna()
    
    le_batter = LabelEncoder()
    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    
    player_match['batter_encoded'] = le_batter.fit_transform(player_match['batter'])
    player_match['team_encoded'] = le_team.fit_transform(player_match['batting_team'])
    player_match['venue_encoded'] = le_venue.fit_transform(player_match['venue'])
    
    X = player_match[['batter_encoded', 'team_encoded', 'venue_encoded']]
    y = player_match['runs_scored']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg = RandomForestRegressor(n_estimators=50, random_state=42)
    reg.fit(X_train, y_train)
    
    preds = reg.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Player Runs Predictor MAE: {mae:.2f} runs\n")

def build_quick_out_model(df):
    print("--- 3. Predicting Who Will Get Out Quickly ---")
    
    player_match = df.groupby(['match_id', 'batter', 'batting_team', 'venue']).agg(
        runs_scored=('runs_batter', 'sum'),
        balls_faced=('ball', 'count') 
    ).reset_index()
    
    player_match['quick_out'] = ((player_match['runs_scored'] < 10) & (player_match['balls_faced'] <= 10)).astype(int)
    
    le_batter = LabelEncoder()
    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    
    player_match['batter_encoded'] = le_batter.fit_transform(player_match['batter'])
    player_match['team_encoded'] = le_team.fit_transform(player_match['batting_team'])
    player_match['venue_encoded'] = le_venue.fit_transform(player_match['venue'])
    
    X = player_match[['batter_encoded', 'team_encoded', 'venue_encoded']]
    y = player_match['quick_out']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    print(f"Quick Out Classification Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
    
if __name__ == "__main__":
    df = load_data()
    build_match_winner_model(df)
    build_player_runs_model(df)
    build_quick_out_model(df)
    print("\n[Done executing model pipelines]")
