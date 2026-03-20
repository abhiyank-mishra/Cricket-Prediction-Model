import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import india_mens_model as imm

df = imm.load_data()
player_match = imm.extract_player_features(df)

le_batter = joblib.load('models/le_batter.pkl')
le_team = joblib.load('models/le_team.pkl')
le_venue = joblib.load('models/le_venue.pkl')

player_match = player_match[player_match['batting_team'].isin(le_team.classes_)]
player_match = player_match[player_match['bowling_team'].isin(le_team.classes_)]
player_match = player_match[player_match['venue'].isin(le_venue.classes_)]

player_match['batter_encoded'] = le_batter.transform(player_match['batter'])
player_match['batting_team_encoded'] = le_team.transform(player_match['batting_team'])
player_match['bowling_team_encoded'] = le_team.transform(player_match['bowling_team'])
player_match['venue_encoded'] = le_venue.transform(player_match['venue'])

X = player_match[[
    'batter_encoded', 'batting_team_encoded', 'bowling_team_encoded', 'venue_encoded', 
    'career_quick_out_rate', 'recent_form_5', 'batting_position', 'career_strike_rate',
    'opp_attack_hist_dismiss', 'opp_attack_hist_sr', 'is_t20', 'is_odi'
]]
y = player_match['quick_out']

print("Training Quick Out Model Bypass...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
clf = LGBMClassifier(n_estimators=1000, max_depth=9, learning_rate=0.015, random_state=42, verbose=-1)
clf.fit(X_train, y_train)

joblib.dump(clf, 'models/quick_out_model.pkl')
print("Quick Out Model Successfully Saved to Disk!")
