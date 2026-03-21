import os
import zipfile
import json
import pandas as pd
import time
import requests
import io
import joblib
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import StackingClassifier, StackingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('Data', exist_ok=True)

CRICSHEET_URLS = [
    "https://cricsheet.org/downloads/tests_male_json.zip",
    "https://cricsheet.org/downloads/odis_male_json.zip",
    "https://cricsheet.org/downloads/t20s_male_json.zip",
    "https://cricsheet.org/downloads/ipl_json.zip"
]

OUTPUT_CSV = r"Data\India_Mens_Combined.csv"

# ---------------------------------------------------------------------------
# 1. BATTER vs BOWLER H2H
# ---------------------------------------------------------------------------
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
    batter_attack['opp_attack_hist_sr'] = np.where(
        batter_attack['opp_attack_hist_balls'] > 0,
        (batter_attack['opp_attack_hist_runs'] / batter_attack['opp_attack_hist_balls']) * 100, 0)
    return batter_attack

# ---------------------------------------------------------------------------
# 2. ENHANCED PLAYER FEATURES
# ---------------------------------------------------------------------------
def extract_player_features(df):
    print("Extracting enhanced player features (EMA + consistency + momentum)...")

    player_match = df.groupby(
        ['match_id', 'date', 'match_format', 'batter', 'batting_team', 'bowling_team', 'venue']
    ).agg(
        runs_scored=('runs_batter', 'sum'),
        balls_faced=('ball', 'count'),
        boundaries=('runs_batter', lambda x: ((x == 4) | (x == 6)).sum()),
    ).reset_index()

    # Batting order / position
    batting_order = df.groupby(['match_id', 'innings', 'batter'], as_index=False)['over'].min()
    batting_order = batting_order.sort_values(['match_id', 'innings', 'over'])
    batting_order['batting_position'] = batting_order.groupby(['match_id', 'innings']).cumcount() + 1
    player_match = player_match.merge(
        batting_order[['match_id', 'batter', 'batting_position']], on=['match_id', 'batter'], how='left')
    player_match['batting_position'] = player_match['batting_position'].fillna(7)
    player_match = player_match.sort_values(by=['batter', 'date'])

    grp = player_match.groupby(['batter', 'match_format'])

    # Career aggregates (shifted to avoid leakage)
    player_match['career_runs_avg'] = grp['runs_scored'].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0)
    player_match['career_balls_faced'] = grp['balls_faced'].transform(
        lambda x: x.shift(1).expanding().sum()).fillna(0)
    player_match['career_total_runs'] = grp['runs_scored'].transform(
        lambda x: x.shift(1).expanding().sum()).fillna(0)
    player_match['career_strike_rate'] = np.where(
        player_match['career_balls_faced'] > 0,
        (player_match['career_total_runs'] / player_match['career_balls_faced']) * 100, 0)

    # Experience
    player_match['career_matches'] = grp.cumcount()

    # EMA form (alpha=0.3)
    player_match['ema_form'] = grp['runs_scored'].transform(
        lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean()).fillna(0)

    # Simple rolling form (kept for backward compat)
    player_match['recent_form_5'] = grp['runs_scored'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)

    # Form momentum  (short EMA − long EMA: positive = improving)
    ema_short = grp['runs_scored'].transform(
        lambda x: x.shift(1).ewm(alpha=0.5, min_periods=1).mean()).fillna(0)
    ema_long = grp['runs_scored'].transform(
        lambda x: x.shift(1).ewm(alpha=0.1, min_periods=1).mean()).fillna(0)
    player_match['form_momentum'] = ema_short - ema_long

    # Consistency = 1 / (1 + CV)
    roll_std = grp['runs_scored'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).std()).fillna(30)
    roll_mean = grp['runs_scored'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()).fillna(1)
    player_match['consistency'] = 1.0 / (1.0 + np.where(roll_mean > 0, roll_std / roll_mean, 2.0))

    # Career 50+ rate
    player_match['scored_50'] = (player_match['runs_scored'] >= 50).astype(int)
    player_match['career_50_rate'] = grp['scored_50'].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0)

    # Career boundary rate (boundaries per ball)
    cum_bounds = grp['boundaries'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
    cum_balls = grp['balls_faced'].transform(lambda x: x.shift(1).expanding().sum()).fillna(1)
    player_match['career_boundary_rate'] = np.where(cum_balls > 0, cum_bounds / cum_balls, 0)

    # Quick-out metrics
    player_match['quick_out'] = ((player_match['runs_scored'] < 10) & (player_match['balls_faced'] <= 10)).astype(int)
    player_match['career_quick_out_rate'] = grp['quick_out'].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0)

    # Venue-specific average
    player_match['venue_avg_runs'] = player_match.groupby(['batter', 'venue'])['runs_scored'].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0)

    # Format flags
    player_match['is_t20'] = player_match['match_format'].apply(lambda x: 1 if 'T20' in x or 'IPL' in x else 0)
    player_match['is_odi'] = player_match['match_format'].apply(lambda x: 1 if 'ODI' in x else 0)

    # Batter vs bowler H2H
    batter_bb = extract_batter_bowler_h2h(df)
    player_match = player_match.merge(batter_bb, on=['match_id', 'batter'], how='left').fillna(0)

    # Save latest player stats
    print("Saving enhanced player form database...")
    latest = player_match.groupby('batter').last().reset_index()
    latest.to_csv('models/latest_player_stats.csv', index=False)

    return player_match

# ---------------------------------------------------------------------------
# 3. MATCH WINNER  — Binary Stacking Ensemble
# ---------------------------------------------------------------------------
def feature_engineer_match_winner(df, player_match):
    print("\n--- STACKING ENSEMBLE Match Winner (Binary) ---")
    start = time.time()

    # ---- roster strength ----
    roster = player_match.groupby(['match_id', 'batting_team']).agg(
        team_career=('career_runs_avg', 'sum'),
        team_form=('ema_form', 'sum'),
        team_sr=('career_strike_rate', 'mean'),
        team_consistency=('consistency', 'mean'),
        team_momentum=('form_momentum', 'sum'),
        team_50_rate=('career_50_rate', 'mean'),
        team_experience=('career_matches', 'sum'),
    ).reset_index()

    # ---- match level base ----
    match_level = df.drop_duplicates(subset=['match_id']).copy()
    match_level = match_level.dropna(subset=['match_won_by'])
    match_level['team1'] = match_level['batting_team']
    match_level['team2'] = match_level['bowling_team']

    # merge team1 roster
    r1 = roster.rename(columns={
        'batting_team': 'team1', 'team_career': 't1_career', 'team_form': 't1_form',
        'team_sr': 't1_sr', 'team_consistency': 't1_consistency',
        'team_momentum': 't1_momentum', 'team_50_rate': 't1_50_rate', 'team_experience': 't1_exp'})
    match_level = match_level.merge(r1, on=['match_id', 'team1'], how='left')

    r2 = roster.rename(columns={
        'batting_team': 'team2', 'team_career': 't2_career', 'team_form': 't2_form',
        'team_sr': 't2_sr', 'team_consistency': 't2_consistency',
        'team_momentum': 't2_momentum', 'team_50_rate': 't2_50_rate', 'team_experience': 't2_exp'})
    match_level = match_level.merge(r2, on=['match_id', 'team2'], how='left').fillna(0)

    # ---- first innings aggregates ----
    fi = df[df['innings'] == 1]
    fi_agg = fi.groupby('match_id').agg(
        fi_runs=('runs_total', 'sum'),
        fi_wkts=('player_out', lambda x: x.notna().sum()),
        fi_balls=('ball', 'count'),
        fi_bounds=('runs_batter', lambda x: ((x == 4) | (x == 6)).sum()),
        fi_dots=('runs_batter', lambda x: (x == 0).sum()),
    ).reset_index()
    match_level = match_level.merge(fi_agg, on='match_id', how='left').fillna(0)
    match_level['fi_rr'] = np.where(match_level['fi_balls'] > 0,
                                    (match_level['fi_runs'] / match_level['fi_balls']) * 6, 0)
    match_level['boundary_pct'] = np.where(match_level['fi_runs'] > 0,
                                           match_level['fi_bounds'] / (match_level['fi_runs'] / 4), 0)
    match_level['dot_pct'] = np.where(match_level['fi_balls'] > 0,
                                      match_level['fi_dots'] / match_level['fi_balls'], 0)

    # ---- phase stats (first innings) ----
    def phase_agg(sub):
        return sub.groupby('match_id').agg(
            p_runs=('runs_total', 'sum'),
            p_wkts=('player_out', lambda x: x.notna().sum()),
            p_balls=('ball', 'count'),
        ).reset_index()

    pp = phase_agg(fi[fi['over'] < 6]).rename(columns={'p_runs': 'pp_runs', 'p_wkts': 'pp_wkts', 'p_balls': 'pp_balls'})
    mo = phase_agg(fi[(fi['over'] >= 6) & (fi['over'] < 15)]).rename(
        columns={'p_runs': 'mo_runs', 'p_wkts': 'mo_wkts', 'p_balls': 'mo_balls'})
    do = phase_agg(fi[fi['over'] >= 15]).rename(
        columns={'p_runs': 'do_runs', 'p_wkts': 'do_wkts', 'p_balls': 'do_balls'})

    for phase_df in [pp, mo, do]:
        match_level = match_level.merge(phase_df, on='match_id', how='left').fillna(0)

    match_level['pp_rr'] = np.where(match_level['pp_balls'] > 0, (match_level['pp_runs'] / match_level['pp_balls']) * 6, 0)
    match_level['mo_rr'] = np.where(match_level['mo_balls'] > 0, (match_level['mo_runs'] / match_level['mo_balls']) * 6, 0)
    match_level['do_rr'] = np.where(match_level['do_balls'] > 0, (match_level['do_runs'] / match_level['do_balls']) * 6, 0)

    # ---- format flags (MUST be computed before venue stats) ----
    match_level['is_t20'] = match_level['match_format'].apply(lambda x: 1 if 'T20' in x or 'IPL' in x else 0)
    match_level['is_odi'] = match_level['match_format'].apply(lambda x: 1 if 'ODI' in x else 0)

    # ---- venue stats (FORMAT-SPECIFIC to avoid mixing Test/T20/ODI averages) ----
    venue_avg = match_level.groupby(['venue', 'is_t20', 'is_odi'])['fi_runs'].mean().reset_index()
    venue_avg.columns = ['venue', 'is_t20', 'is_odi', 'venue_avg_first_innings']
    match_level = match_level.merge(venue_avg, on=['venue', 'is_t20', 'is_odi'], how='left')
    # Fallback: global format average for venues with no format-specific data
    global_format_avg = match_level.groupby(['is_t20', 'is_odi'])['fi_runs'].transform('mean')
    match_level['venue_avg_first_innings'] = match_level['venue_avg_first_innings'].fillna(global_format_avg)
    venue_avg.to_csv('models/venue_avg_score.csv', index=False)

    # venue phase defaults (for app.py) — also format-specific
    venue_phase = match_level.groupby(['venue', 'is_t20', 'is_odi']).agg(
        v_pp_runs=('pp_runs', 'mean'), v_pp_wkts=('pp_wkts', 'mean'), v_pp_rr=('pp_rr', 'mean'),
        v_mo_runs=('mo_runs', 'mean'), v_mo_wkts=('mo_wkts', 'mean'), v_mo_rr=('mo_rr', 'mean'),
        v_do_runs=('do_runs', 'mean'), v_do_wkts=('do_wkts', 'mean'), v_do_rr=('do_rr', 'mean'),
        v_boundary_pct=('boundary_pct', 'mean'), v_dot_pct=('dot_pct', 'mean'),
    ).reset_index()
    venue_phase.to_csv('models/venue_phase_stats.csv', index=False)

    match_level['score_above_avg'] = match_level['fi_runs'] - match_level['venue_avg_first_innings']

    # ---- toss ----
    match_level['toss_is_t1'] = (match_level['toss_winner'] == match_level['team1']).astype(int)
    match_level['toss_bat'] = (match_level['toss_decision'] == 'bat').astype(int)
    match_level['toss_won_match'] = (match_level['toss_winner'] == match_level['match_won_by']).astype(int)

    vtw = match_level.groupby('venue')['toss_won_match'].mean().reset_index(name='venue_toss_win_rate_final')
    vtw.to_csv('models/venue_toss_win_rate.csv', index=False)
    match_level['venue_toss_wr'] = match_level.groupby('venue')['toss_won_match'].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0.5)

    match_level['chased_won'] = (match_level['match_won_by'] == match_level['team2']).astype(int)
    vcw = match_level.groupby('venue')['chased_won'].mean().reset_index(name='venue_chase_win_rate_final')
    vcw.to_csv('models/venue_chase_win_rate.csv', index=False)
    match_level['venue_chase_wr'] = match_level.groupby('venue')['chased_won'].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0.5)

    # ---- team win rates & H2H ----
    team_wins = match_level['match_won_by'].value_counts()
    team_matches = match_level['team1'].value_counts() + match_level['team2'].value_counts()
    team_wr = (team_wins / team_matches).fillna(0.5).to_dict()
    joblib.dump(team_wr, 'models/team_win_rates.pkl')

    match_level['t1_wr'] = match_level['team1'].map(team_wr).fillna(0.5)
    match_level['t2_wr'] = match_level['team2'].map(team_wr).fillna(0.5)
    match_level['wr_diff'] = match_level['t1_wr'] - match_level['t2_wr']

    def get_matchup(a, b): return tuple(sorted([str(a), str(b)]))
    matches_h2h = match_level[['team1', 'team2', 'match_won_by']].dropna().copy()
    matches_h2h['matchup'] = matches_h2h.apply(lambda r: get_matchup(r['team1'], r['team2']), axis=1)
    h2h_wins = matches_h2h.groupby(['matchup', 'match_won_by']).size().reset_index(name='wins')
    h2h_dict = {}
    for _, row in h2h_wins.iterrows():
        mu = row['matchup']
        if mu not in h2h_dict: h2h_dict[mu] = {}
        h2h_dict[mu][str(row['match_won_by'])] = row['wins']
    joblib.dump(h2h_dict, 'models/h2h_dict.pkl')

    def h2h_rate(t1, t2):
        mu = get_matchup(t1, t2)
        s = h2h_dict.get(mu, {})
        tot = s.get(str(t1), 0) + s.get(str(t2), 0)
        return s.get(str(t1), 0) / tot if tot > 0 else 0.5

    match_level['t1_h2h_wr'] = match_level.apply(lambda r: h2h_rate(r['team1'], r['team2']), axis=1)

    # (format flags already computed above before venue stats)

    # ---- diffs ----
    match_level['form_diff'] = match_level['t1_form'] - match_level['t2_form']
    match_level['career_diff'] = match_level['t1_career'] - match_level['t2_career']
    match_level['sr_diff'] = match_level['t1_sr'] - match_level['t2_sr']
    match_level['consistency_diff'] = match_level['t1_consistency'] - match_level['t2_consistency']
    match_level['momentum_diff'] = match_level['t1_momentum'] - match_level['t2_momentum']
    match_level['exp_diff'] = match_level['t1_exp'] - match_level['t2_exp']

    # ---- encoding ----
    le_venue = LabelEncoder()
    le_team = LabelEncoder()
    all_teams = pd.concat([match_level['team1'], match_level['team2'], match_level['match_won_by']]).unique()
    le_team.fit(all_teams)

    match_level['venue_enc'] = le_venue.fit_transform(match_level['venue'])
    match_level['t1_enc'] = le_team.transform(match_level['team1'])
    match_level['t2_enc'] = le_team.transform(match_level['team2'])

    joblib.dump(le_venue, 'models/le_venue.pkl')
    joblib.dump(le_team, 'models/le_team.pkl')

    # ---- BINARY TARGET: does the batting-first team win? ----
    match_level['target'] = (match_level['match_won_by'] == match_level['team1']).astype(int)

    FEATURE_COLS = [
        'venue_enc', 't1_enc', 't2_enc',
        'toss_is_t1', 'toss_bat', 'venue_toss_wr', 'venue_chase_wr',
        't1_wr', 't2_wr', 't1_h2h_wr', 'wr_diff',
        'venue_avg_first_innings', 'score_above_avg',
        't1_career', 't1_form', 't1_sr', 't1_consistency', 't1_momentum', 't1_50_rate', 't1_exp',
        't2_career', 't2_form', 't2_sr', 't2_consistency', 't2_momentum', 't2_50_rate', 't2_exp',
        'form_diff', 'career_diff', 'sr_diff', 'consistency_diff', 'momentum_diff', 'exp_diff',
        'fi_runs', 'fi_wkts', 'fi_rr', 'boundary_pct', 'dot_pct',
        'pp_runs', 'pp_wkts', 'pp_rr',
        'mo_runs', 'mo_wkts', 'mo_rr',
        'do_runs', 'do_wkts', 'do_rr',
        'is_t20', 'is_odi',
    ]
    joblib.dump(FEATURE_COLS, 'models/match_feature_cols.pkl')

    features = match_level[FEATURE_COLS + ['target']].dropna()

    X = features[FEATURE_COLS]
    y = features['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

    print(f"Training set: {len(X_train)} | Test set: {len(X_test)} | Features: {len(FEATURE_COLS)}")
    print("Training Stacking Ensemble (LightGBM + XGBoost + CatBoost → LogisticRegression)...")

    lgbm = LGBMClassifier(
        n_estimators=2000, max_depth=8, learning_rate=0.01, num_leaves=63,
        reg_alpha=0.1, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=30, random_state=42, verbose=-1, n_jobs=-1)

    xgb = XGBClassifier(
        n_estimators=1200, max_depth=7, learning_rate=0.015,
        reg_alpha=0.1, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=30, random_state=42, eval_metric='logloss', n_jobs=-1)

    cb = CatBoostClassifier(
        iterations=1500, depth=7, learning_rate=0.015,
        l2_leaf_reg=3.0, random_seed=42, verbose=0)

    meta = LogisticRegression(C=0.1, max_iter=2000, solver='lbfgs')

    stack = StackingClassifier(
        estimators=[('lgbm', lgbm), ('xgb', xgb), ('cb', cb)],
        final_estimator=meta, cv=5, n_jobs=-1, passthrough=False)
    stack.fit(X_train, y_train)

    preds = stack.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"-> STACKING MATCH WINNER ACCURACY: {acc*100:.2f}% ({time.time()-start:.1f}s)")

    joblib.dump(stack, 'models/match_winner_stacking.pkl')
    return stack

# ---------------------------------------------------------------------------
# 4. PLAYER RUNS — Stacking Regressor with log-target
# ---------------------------------------------------------------------------
def feature_engineer_player_runs(player_match):
    print("\n--- STACKING ENSEMBLE Player Runs Prediction ---")
    start = time.time()

    le_batter = LabelEncoder()
    player_match['batter_enc'] = le_batter.fit_transform(player_match['batter'])
    joblib.dump(le_batter, 'models/le_batter.pkl')

    le_team = joblib.load('models/le_team.pkl')
    le_venue = joblib.load('models/le_venue.pkl')

    pm = player_match[player_match['batting_team'].isin(le_team.classes_)].copy()
    pm = pm[pm['bowling_team'].isin(le_team.classes_)]
    pm = pm[pm['venue'].isin(le_venue.classes_)]

    pm['bat_enc'] = le_team.transform(pm['batting_team'])
    pm['bowl_enc'] = le_team.transform(pm['bowling_team'])
    pm['ven_enc'] = le_venue.transform(pm['venue'])

    PLAYER_FEATURES = [
        'batter_enc', 'bat_enc', 'bowl_enc', 'ven_enc',
        'career_runs_avg', 'career_strike_rate', 'career_matches',
        'ema_form', 'recent_form_5', 'form_momentum', 'consistency',
        'batting_position', 'venue_avg_runs',
        'career_50_rate', 'career_boundary_rate', 'career_quick_out_rate',
        'opp_attack_hist_dismiss', 'opp_attack_hist_sr', 'opp_attack_hist_runs',
        'is_t20', 'is_odi',
    ]
    joblib.dump(PLAYER_FEATURES, 'models/player_feature_cols.pkl')

    X = pm[PLAYER_FEATURES]
    y = pm['runs_scored'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    print(f"Training set: {len(X_train)} | Test set: {len(X_test)} | Features: {len(PLAYER_FEATURES)}")
    print("Training Stacking Regressor (Huber + Quantile → Ridge)...")

    lgbm_r = LGBMRegressor(
        n_estimators=2500, max_depth=10, learning_rate=0.008, num_leaves=127,
        reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, objective='huber', random_state=42, verbose=-1, n_jobs=-1)

    xgb_r = XGBRegressor(
        n_estimators=1500, max_depth=8, learning_rate=0.01,
        reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, objective='reg:pseudohubererror', random_state=42, n_jobs=-1)

    cb_r = CatBoostRegressor(
        iterations=2000, depth=8, learning_rate=0.01,
        l2_leaf_reg=3.0, loss_function='MAE', random_seed=42, verbose=0)

    meta_r = Ridge(alpha=1.0)

    stack_r = StackingRegressor(
        estimators=[('lgbm', lgbm_r), ('xgb', xgb_r), ('cb', cb_r)],
        final_estimator=meta_r, cv=5, n_jobs=-1, passthrough=False)
    stack_r.fit(X_train, y_train)

    preds = np.maximum(0, stack_r.predict(X_test))
    mae = mean_absolute_error(y_test, preds)
    print(f"-> STACKING PLAYER RUNS MAE: {mae:.2f} runs ({time.time()-start:.1f}s)")

    joblib.dump(stack_r, 'models/player_runs_stacking.pkl')
    return stack_r

# ---------------------------------------------------------------------------
# 5. QUICK OUT — Enhanced LightGBM + XGBoost
# ---------------------------------------------------------------------------
def feature_engineer_quick_out(player_match):
    print("\n--- ENHANCED Quick Out Prediction ---")
    start = time.time()

    le_batter = joblib.load('models/le_batter.pkl')
    le_team = joblib.load('models/le_team.pkl')
    le_venue = joblib.load('models/le_venue.pkl')

    pm = player_match[player_match['batter'].isin(le_batter.classes_)].copy()
    pm = pm[pm['batting_team'].isin(le_team.classes_)]
    pm = pm[pm['bowling_team'].isin(le_team.classes_)]
    pm = pm[pm['venue'].isin(le_venue.classes_)]

    pm['batter_enc'] = le_batter.transform(pm['batter'])
    pm['bat_enc'] = le_team.transform(pm['batting_team'])
    pm['bowl_enc'] = le_team.transform(pm['bowling_team'])
    pm['ven_enc'] = le_venue.transform(pm['venue'])

    QO_FEATURES = [
        'batter_enc', 'bat_enc', 'bowl_enc', 'ven_enc',
        'career_quick_out_rate', 'ema_form', 'recent_form_5', 'form_momentum', 'consistency',
        'batting_position', 'career_strike_rate', 'career_runs_avg',
        'opp_attack_hist_dismiss', 'opp_attack_hist_sr',
        'is_t20', 'is_odi',
    ]
    joblib.dump(QO_FEATURES, 'models/qo_feature_cols.pkl')

    X = pm[QO_FEATURES]
    y = pm['quick_out']

    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos = neg_count / pos_count if pos_count > 0 else 1.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

    print(f"Training set: {len(X_train)} | Test: {len(X_test)} | scale_pos_weight: {scale_pos:.2f}")

    lgbm_q = LGBMClassifier(
        n_estimators=1500, max_depth=9, learning_rate=0.015, num_leaves=63,
        scale_pos_weight=scale_pos, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1)
    lgbm_q.fit(X_train, y_train)

    preds = lgbm_q.predict(X_test)
    print(f"-> Quick Out Accuracy: {accuracy_score(y_test, preds)*100:.2f}% ({time.time()-start:.1f}s)")
    joblib.dump(lgbm_q, 'models/quick_out_model.pkl')
    return lgbm_q

# ---------------------------------------------------------------------------
# 6. DATA DOWNLOAD & PIPELINE
# ---------------------------------------------------------------------------
def perform_end_to_end_pipeline():
    print("🚀 STARTED REDESIGNED CRICKET ORACLE PIPELINE 🚀")
    final_df = None

    if os.path.exists(OUTPUT_CSV):
        print(f"✅ Existing dataset found at {OUTPUT_CSV}! Skipping download...")
        final_df = pd.read_csv(OUTPUT_CSV, low_memory=False)
    else:
        dfs = []
        for url in CRICSHEET_URLS:
            print(f"Downloading from: {url}")
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(resp.content), 'r') as zf:
                        json_files = [f for f in zf.namelist() if f.endswith('.json')]
                        print(f"Extracting {len(json_files)} matches...")

                        for jf in json_files:
                            with zf.open(jf) as f:
                                data = json.loads(f.read().decode('utf-8'))

                            info = data.get('info', {})
                            if info.get('gender') != 'male': continue

                            teams = info.get('teams', [])
                            match_type = str(info.get('match_type', ''))

                            is_indian = False
                            if 'IPL' in str(info.get('competition', '')) or match_type == 'IPL':
                                is_indian = True
                            elif 'India' in teams:
                                is_indian = True
                            if not is_indian: continue

                            match_id = jf.replace('.json', '')
                            date = info.get('dates', [''])[0]
                            city = info.get('city', '')
                            venue = info.get('venue', '')
                            toss_winner = info.get('toss', {}).get('winner', '')
                            toss_decision = info.get('toss', {}).get('decision', '')
                            match_won_by = info.get('outcome', {}).get('winner', '')

                            rows = []
                            for inn_idx, inning in enumerate(data.get('innings', [])):
                                batting_team = inning.get('team', '')
                                bowling_team = [t for t in teams if t != batting_team]
                                bowling_team = bowling_team[0] if bowling_team else ''

                                for over in inning.get('overs', []):
                                    over_num = over.get('over')
                                    for ball_idx, delivery in enumerate(over.get('deliveries', [])):
                                        runs = delivery.get('runs', {})
                                        wks = delivery.get('wickets', [])
                                        player_out = wks[0].get('player_out') if wks else None

                                        rows.append({
                                            'match_id': match_id, 'date': date, 'match_format': match_type,
                                            'batting_team': batting_team, 'bowling_team': bowling_team,
                                            'innings': inn_idx + 1, 'over': over_num, 'ball': ball_idx + 1,
                                            'batter': delivery.get('batter'), 'bowler': delivery.get('bowler'),
                                            'runs_batter': runs.get('batter', 0), 'runs_total': runs.get('total', 0),
                                            'player_out': player_out, 'venue': venue, 'city': city,
                                            'match_won_by': match_won_by, 'toss_winner': toss_winner,
                                            'toss_decision': toss_decision
                                        })
                            if rows: dfs.append(pd.DataFrame(rows))
            except Exception as e:
                print(f"Failed handling {url}: {e}")

        if len(dfs) > 0:
            final_df = pd.concat(dfs, ignore_index=True)
            print(f"✅ Dataset size: {len(final_df)} rows. Saving CSV...")
            final_df.to_csv(OUTPUT_CSV, index=False)
        else:
            print("Failed to build dataset.")

    if final_df is not None:
        final_df['date'] = pd.to_datetime(final_df['date'], format='mixed', errors='coerce')
        final_df = final_df.dropna(subset=['date']).sort_values(by=['date', 'match_id', 'innings', 'over', 'ball'])

        player_match = extract_player_features(final_df)
        feature_engineer_match_winner(final_df, player_match)
        feature_engineer_player_runs(player_match)
        feature_engineer_quick_out(player_match)
        print("\n🎉 ALL PIPELINES INSTALLED AND READY! Run `streamlit run app.py` 🎉")


if __name__ == "__main__":
    tt = time.time()
    perform_end_to_end_pipeline()
    print(f"Complete pipeline installed in {time.time() - tt:.2f} seconds!")
