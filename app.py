import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import plotly.express as px
import plotly.graph_objects as go

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Indian AI Predictor Core", page_icon="🏏", layout="wide")

# ---- CUSTOM CSS & STYLING ----
st.markdown("""
<style>
/* Gradient Text for Title */
.gradient-text {
    background: -webkit-linear-gradient(45deg, #0cebeb, #20e3b2, #29ffc6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem !important;
    font-weight: 900 !important;
    margin-bottom: 0px !important;
}

/* Glassmorphism Metric Cards */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 15px !important;
    transition: all 0.3s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(32, 227, 178, 0.3);
    border-color: rgba(32, 227, 178, 0.5);
}

/* Animated Live Pulse */
.live-pulse {
    animation: pulse-animation 2s infinite;
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #ff3b3b;
    margin-right: 8px;
    box-shadow: 0 0 10px #ff3b3b;
}

@keyframes pulse-animation {
  0% { box-shadow: 0 0 0 0px rgba(255, 59, 59, 0.8); }
  100% { box-shadow: 0 0 0 15px rgba(255, 59, 59, 0); }
}

/* Beautiful Tabs */
div[data-testid="stTabs"] button {
    font-size: 1.2rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="gradient-text">🏏 Ultimate Indian Men\'s Cricket AI Oracle</h1>', unsafe_allow_html=True)
st.markdown("---")

# ---- CHECK FOR STACKING MODEL (NEW) OR FALLBACK TO OLD ----
STACKING_MODE = os.path.exists('models/match_winner_stacking.pkl')

if not STACKING_MODE and not os.path.exists('models/match_winner_ensemble.pkl'):
    st.error("🚨 Models are currently training! Please wait and refresh.")
    st.stop()

@st.cache_resource
def load_assets():
    if STACKING_MODE:
        mdls = {
            'match_winner': joblib.load('models/match_winner_stacking.pkl'),
            'player_runs': joblib.load('models/player_runs_stacking.pkl'),
            'quick_out': joblib.load('models/quick_out_model.pkl'),
        }
    else:
        mdls = {
            'match_winner': joblib.load('models/match_winner_ensemble.pkl'),
            'player_runs': joblib.load('models/player_runs_model.pkl'),
            'quick_out': joblib.load('models/quick_out_model.pkl'),
        }
    encs = {
        'team': joblib.load('models/le_team.pkl'),
        'venue': joblib.load('models/le_venue.pkl'),
        'batter': joblib.load('models/le_batter.pkl'),
    }
    stat = {
        'player': pd.read_csv('models/latest_player_stats.csv'),
        'venue_avg_first': pd.read_csv('models/venue_avg_score.csv'),
        'venue_toss_win': pd.read_csv('models/venue_toss_win_rate.csv'),
        'venue_chase_win': pd.read_csv('models/venue_chase_win_rate.csv'),
        'team_win_rates': joblib.load('models/team_win_rates.pkl'),
        'h2h_dict': joblib.load('models/h2h_dict.pkl'),
    }
    # Load venue phase stats if available
    if os.path.exists('models/venue_phase_stats.csv'):
        stat['venue_phase'] = pd.read_csv('models/venue_phase_stats.csv')
    else:
        stat['venue_phase'] = None
    return mdls, encs, stat

models, encoders, stats = load_assets()
available_batters = sorted(stats['player']['batter'].unique())


def get_best_match(name, choices, threshold=3):
    import difflib
    name_clean = str(name).lower().strip()
    best_score = -1
    best_match = None
    for c in choices:
        c_clean = str(c).lower()
        ratio = difflib.SequenceMatcher(None, name_clean, c_clean).ratio()
        parts = name_clean.split()
        part_score = 0
        for p in parts:
            if len(p) > 2 and p in c_clean:
                part_score += len(p)
                if c_clean.endswith(p):
                    part_score += 5
        score = (ratio * 10) + part_score
        if score > best_score:
            best_score = score
            best_match = c
    return best_match if best_score > threshold else None


def _get_venue_phase(venue, is_t20=1, is_odi=0):
    """Get venue phase defaults from saved stats (format-specific)."""
    vp = stats.get('venue_phase')
    if vp is not None:
        row = vp[(vp['venue'] == venue) & (vp['is_t20'] == is_t20) & (vp['is_odi'] == is_odi)]
        if len(row) > 0:
            return row.iloc[0]
        # Fallback to any format at this venue
        row = vp[vp['venue'] == venue]
        if len(row) > 0:
            return row.iloc[0]
    # Defaults
    return pd.Series({
        'v_pp_runs': 45, 'v_pp_wkts': 1.5, 'v_pp_rr': 7.5,
        'v_mo_runs': 65, 'v_mo_wkts': 2.5, 'v_mo_rr': 7.2,
        'v_do_runs': 50, 'v_do_wkts': 2.0, 'v_do_rr': 10.0,
        'v_boundary_pct': 0.35, 'v_dot_pct': 0.40,
    })


def execute_model_prediction(venue, team1, team2, t1_roster, t2_roster, is_t20, is_odi, first_runs, first_wks):
    pdf = stats['player']
    t1_df = pdf[pdf['batter'].isin(t1_roster)]
    t2_df = pdf[pdf['batter'].isin(t2_roster)]

    # Venue stats (format-specific lookup)
    vdf = stats['venue_avg_first']
    if 'is_t20' in vdf.columns:
        venue_avg_row = vdf[(vdf['venue'] == venue) & (vdf['is_t20'] == is_t20) & (vdf['is_odi'] == is_odi)]
    else:
        venue_avg_row = vdf[vdf['venue'] == venue]
    venue_avg = venue_avg_row['venue_avg_first_innings'].values[0] if len(venue_avg_row) > 0 else (160.0 if is_t20 else 250.0)

    toss_win_row = stats['venue_toss_win'][stats['venue_toss_win']['venue'] == venue]
    venue_toss_wr = toss_win_row['venue_toss_win_rate_final'].values[0] if len(toss_win_row) > 0 else 0.5

    chase_win_row = stats['venue_chase_win'][stats['venue_chase_win']['venue'] == venue]
    venue_chase_wr = chase_win_row['venue_chase_win_rate_final'].values[0] if len(chase_win_row) > 0 else 0.5

    t1_wr = stats['team_win_rates'].get(team1, 0.5)
    t2_wr = stats['team_win_rates'].get(team2, 0.5)

    matchup = tuple(sorted([str(team1), str(team2)]))
    h2h = stats['h2h_dict'].get(matchup, {})
    total_h2h = h2h.get(team1, 0) + h2h.get(team2, 0)
    t1_h2h = h2h.get(team1, 0) / total_h2h if total_h2h > 0 else 0.5

    fi_rr = (first_runs / 120.0) * 6 if is_t20 else (first_runs / 300.0) * 6
    vp = _get_venue_phase(venue, is_t20, is_odi)

    if STACKING_MODE:
        # ---- STACKING MODEL (new binary architecture) ----
        def _safe_mean(s, col, default=0):
            return s[col].mean() if col in s.columns and len(s) > 0 else default
        def _safe_sum(s, col, default=0):
            return s[col].sum() if col in s.columns and len(s) > 0 else default

        input_data = pd.DataFrame([{
            'venue_enc': encoders['venue'].transform([venue])[0],
            't1_enc': encoders['team'].transform([team1])[0],
            't2_enc': encoders['team'].transform([team2])[0],
            'toss_is_t1': 1, 'toss_bat': 1,
            'venue_toss_wr': venue_toss_wr, 'venue_chase_wr': venue_chase_wr,
            't1_wr': t1_wr, 't2_wr': t2_wr, 't1_h2h_wr': t1_h2h, 'wr_diff': t1_wr - t2_wr,
            'venue_avg_first_innings': venue_avg, 'score_above_avg': first_runs - venue_avg,
            't1_career': _safe_sum(t1_df, 'career_runs_avg'),
            't1_form': _safe_sum(t1_df, 'ema_form'),
            't1_sr': _safe_mean(t1_df, 'career_strike_rate'),
            't1_consistency': _safe_mean(t1_df, 'consistency', 0.5),
            't1_momentum': _safe_sum(t1_df, 'form_momentum'),
            't1_50_rate': _safe_mean(t1_df, 'career_50_rate'),
            't1_exp': _safe_sum(t1_df, 'career_matches'),
            't2_career': _safe_sum(t2_df, 'career_runs_avg'),
            't2_form': _safe_sum(t2_df, 'ema_form'),
            't2_sr': _safe_mean(t2_df, 'career_strike_rate'),
            't2_consistency': _safe_mean(t2_df, 'consistency', 0.5),
            't2_momentum': _safe_sum(t2_df, 'form_momentum'),
            't2_50_rate': _safe_mean(t2_df, 'career_50_rate'),
            't2_exp': _safe_sum(t2_df, 'career_matches'),
            'form_diff': _safe_sum(t1_df, 'ema_form') - _safe_sum(t2_df, 'ema_form'),
            'career_diff': _safe_sum(t1_df, 'career_runs_avg') - _safe_sum(t2_df, 'career_runs_avg'),
            'sr_diff': _safe_mean(t1_df, 'career_strike_rate') - _safe_mean(t2_df, 'career_strike_rate'),
            'consistency_diff': _safe_mean(t1_df, 'consistency', 0.5) - _safe_mean(t2_df, 'consistency', 0.5),
            'momentum_diff': _safe_sum(t1_df, 'form_momentum') - _safe_sum(t2_df, 'form_momentum'),
            'exp_diff': _safe_sum(t1_df, 'career_matches') - _safe_sum(t2_df, 'career_matches'),
            'fi_runs': first_runs, 'fi_wkts': first_wks, 'fi_rr': fi_rr,
            'boundary_pct': vp.get('v_boundary_pct', 0.35),
            'dot_pct': vp.get('v_dot_pct', 0.40),
            'pp_runs': vp.get('v_pp_runs', 45), 'pp_wkts': vp.get('v_pp_wkts', 1.5), 'pp_rr': vp.get('v_pp_rr', 7.5),
            'mo_runs': vp.get('v_mo_runs', 65), 'mo_wkts': vp.get('v_mo_wkts', 2.5), 'mo_rr': vp.get('v_mo_rr', 7.2),
            'do_runs': vp.get('v_do_runs', 50), 'do_wkts': vp.get('v_do_wkts', 2.0), 'do_rr': vp.get('v_do_rr', 10.0),
            'is_t20': is_t20, 'is_odi': is_odi,
        }])

        # Binary: prob of team1 winning
        prob_t1 = models['match_winner'].predict_proba(input_data)[0][1] * 100
        prob_t2 = 100 - prob_t1

    else:
        # ---- LEGACY MODEL (multi-class voting) ----
        score_above_venue_avg = first_runs - venue_avg
        t1_runs_s = t1_df['career_runs_avg'].sum()
        t1_form_s = t1_df['recent_form_5'].sum()
        t1_sr_s = t1_df['career_strike_rate'].mean()
        t2_runs_s = t2_df['career_runs_avg'].sum()
        t2_form_s = t2_df['recent_form_5'].sum()
        t2_sr_s = t2_df['career_strike_rate'].mean()
        pp_w = first_wks / 3.0
        second_wks = 4.0

        input_data = pd.DataFrame([{
            'venue_encoded': encoders['venue'].transform([venue])[0],
            'team1_encoded': encoders['team'].transform([team1])[0],
            'team2_encoded': encoders['team'].transform([team2])[0],
            'toss_winner_is_team1': 1, 'toss_decision_bat': 1,
            'venue_toss_win_rate': venue_toss_wr, 'venue_chase_win_rate': venue_chase_wr,
            'team1_win_rate': t1_wr, 'team2_win_rate': t2_wr, 'team1_h2h_win_rate': t1_h2h,
            'venue_avg_first_innings': venue_avg, 'score_above_venue_avg': score_above_venue_avg,
            't1_runs': t1_runs_s, 't1_form': t1_form_s, 't1_sr': t1_sr_s,
            't2_runs': t2_runs_s, 't2_form': t2_form_s, 't2_sr': t2_sr_s,
            'roster_form_diff': t1_form_s - t2_form_s, 'roster_runs_diff': t1_runs_s - t2_runs_s,
            'sr_diff': t1_sr_s - t2_sr_s,
            'first_innings_runs': first_runs, 'first_innings_wickets': first_wks, 'first_innings_rr': fi_rr,
            'powerplay_wickets': pp_w, 'second_innings_wickets': second_wks,
            'is_t20': is_t20, 'is_odi': is_odi,
        }])

        pred_probs = models['match_winner'].predict_proba(input_data)[0]
        classes = models['match_winner'].classes_
        t_t1 = encoders['team'].transform([team1])[0]
        t_t2 = encoders['team'].transform([team2])[0]
        p1_idx = np.where(classes == t_t1)[0][0] if t_t1 in classes else -1
        p2_idx = np.where(classes == t_t2)[0][0] if t_t2 in classes else -1
        prob_t1 = pred_probs[p1_idx] * 100 if p1_idx != -1 else 0
        prob_t2 = pred_probs[p2_idx] * 100 if p2_idx != -1 else 0

    return prob_t1, prob_t2, venue_avg, t1_df


def predict_batter_runs(row, team1, team2, venue, is_t20, is_odi):
    """Predict expected runs for a single batter."""
    try:
        b_enc = encoders['batter'].transform([row['batter']])[0]
        bat_enc = encoders['team'].transform([team1])[0]
        bowl_enc = encoders['team'].transform([team2])[0]
        ven_enc = encoders['venue'].transform([venue])[0]
    except Exception:
        return None, None

    if STACKING_MODE:
        b_in = pd.DataFrame([{
            'batter_enc': b_enc, 'bat_enc': bat_enc, 'bowl_enc': bowl_enc, 'ven_enc': ven_enc,
            'career_runs_avg': row.get('career_runs_avg', 0),
            'career_strike_rate': row.get('career_strike_rate', 0),
            'career_matches': row.get('career_matches', 0),
            'ema_form': row.get('ema_form', 0),
            'recent_form_5': row.get('recent_form_5', 0),
            'form_momentum': row.get('form_momentum', 0),
            'consistency': row.get('consistency', 0.5),
            'batting_position': row.get('batting_position', 5),
            'venue_avg_runs': row.get('venue_avg_runs', 0),
            'career_50_rate': row.get('career_50_rate', 0),
            'career_boundary_rate': row.get('career_boundary_rate', 0),
            'career_quick_out_rate': row.get('career_quick_out_rate', 0),
            'opp_attack_hist_dismiss': row.get('opp_attack_hist_dismiss', 0),
            'opp_attack_hist_sr': row.get('opp_attack_hist_sr', 0),
            'opp_attack_hist_runs': row.get('opp_attack_hist_runs', 0),
            'is_t20': is_t20, 'is_odi': is_odi,
        }])
        # Model trained on raw runs (Huber loss)
        exp_runs = max(0, models['player_runs'].predict(b_in)[0])

        qo_in = pd.DataFrame([{
            'batter_enc': b_enc, 'bat_enc': bat_enc, 'bowl_enc': bowl_enc, 'ven_enc': ven_enc,
            'career_quick_out_rate': row.get('career_quick_out_rate', 0),
            'ema_form': row.get('ema_form', 0),
            'recent_form_5': row.get('recent_form_5', 0),
            'form_momentum': row.get('form_momentum', 0),
            'consistency': row.get('consistency', 0.5),
            'batting_position': row.get('batting_position', 5),
            'career_strike_rate': row.get('career_strike_rate', 0),
            'career_runs_avg': row.get('career_runs_avg', 0),
            'opp_attack_hist_dismiss': row.get('opp_attack_hist_dismiss', 0),
            'opp_attack_hist_sr': row.get('opp_attack_hist_sr', 0),
            'is_t20': is_t20, 'is_odi': is_odi,
        }])
        quick_out = models['quick_out'].predict(qo_in)[0]
    else:
        b_in = pd.DataFrame([{
            'batter_encoded': b_enc, 'batting_team_encoded': bat_enc,
            'bowling_team_encoded': bowl_enc, 'venue_encoded': ven_enc,
            'career_runs_avg': row.get('career_runs_avg', 0),
            'recent_form_5': row.get('recent_form_5', 0),
            'batting_position': row.get('batting_position', 5),
            'career_strike_rate': row.get('career_strike_rate', 0),
            'venue_avg_runs': row.get('venue_avg_runs', 0),
            'opp_attack_hist_dismiss': row.get('opp_attack_hist_dismiss', 0),
            'opp_attack_hist_sr': row.get('opp_attack_hist_sr', 0),
            'opp_attack_hist_runs': row.get('opp_attack_hist_runs', 0),
            'is_t20': is_t20, 'is_odi': is_odi,
        }])
        exp_runs = max(0, models['player_runs'].predict(b_in)[0])

        qo_in = pd.DataFrame([{
            'batter_encoded': b_enc, 'batting_team_encoded': bat_enc,
            'bowling_team_encoded': bowl_enc, 'venue_encoded': ven_enc,
            'career_quick_out_rate': row.get('career_quick_out_rate', 0),
            'recent_form_5': row.get('recent_form_5', 0),
            'batting_position': row.get('batting_position', 5),
            'career_strike_rate': row.get('career_strike_rate', 0),
            'opp_attack_hist_dismiss': row.get('opp_attack_hist_dismiss', 0),
            'opp_attack_hist_sr': row.get('opp_attack_hist_sr', 0),
            'is_t20': is_t20, 'is_odi': is_odi,
        }])
        quick_out = models['quick_out'].predict(qo_in)[0]

    return exp_runs, quick_out


def render_batter_chart(t1_df, team1, team2, venue, is_t20, is_odi):
    """Predict and render the batter impact chart."""
    t1_preds = []
    for _, row in t1_df.iterrows():
        exp_runs, quick_out = predict_batter_runs(row, team1, team2, venue, is_t20, is_odi)
        if exp_runs is not None:
            t1_preds.append((row['batter'], exp_runs, quick_out))

    if not t1_preds:
        st.info("No batter predictions available.")
        return

    t1_preds.sort(key=lambda x: x[1], reverse=True)
    plot_df = pd.DataFrame(t1_preds[:7], columns=["Batter", "Expected Runs", "Threat"])
    color_safe = '#29ffc6' if STACKING_MODE else '#20e3b2'
    color_danger = '#ff416c' if STACKING_MODE else '#ff4b2b'
    plot_df["Color"] = plot_df["Threat"].apply(lambda x: color_danger if x == 1 else color_safe)

    bar_fig = go.Figure(data=[go.Bar(
        x=plot_df["Expected Runs"], y=plot_df["Batter"], orientation='h',
        marker_color=plot_df["Color"], text=plot_df["Expected Runs"].round(1), textposition='auto'
    )])
    title_txt = "Top 7 Expected Run Scorers (Red = Quick Out Threat)"
    bar_fig.update_layout(title=title_txt, margin=dict(t=40, b=0, l=0, r=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    bar_fig.update_yaxes(autorange="reversed")
    st.plotly_chart(bar_fig, use_container_width=True)


# ===================== TABS =====================
tab_live, tab_manual = st.tabs(["🔴 Live Auto-Tracker (API)", "🔮 Manual Pre-Match Setup"])

with tab_live:
    st.markdown("<h3><span class='live-pulse'></span>Real-Time ESPN Match Matrix</h3>", unsafe_allow_html=True)
    st.write("Fetching continuous game states from the Global ESPN Network, autonomously paired with our prediction suite.")

    if st_autorefresh:
        st_autorefresh(interval=60000, key="api_refresh")

    with st.spinner("Synchronizing with Satellite Feeds..."):
        try:
            espn_url = "https://site.web.api.espn.com/apis/site/v2/sports/cricket/scorepanel"
            res = requests.get(espn_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            data = res.json()

            scores_array = data.get('scores', [{}])
            live_matches = scores_array[0].get('events', []) if len(scores_array) > 0 else []

            ipl_url = "https://site.api.espn.com/apis/site/v2/sports/cricket/8048/scoreboard"
            try:
                ipl_res = requests.get(ipl_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                ipl_events = ipl_res.json().get('events', [])
                existing_ids = {str(m.get('id')) for m in live_matches if m.get('id')}
                for ipl_match in ipl_events:
                    if str(ipl_match.get('id')) not in existing_ids:
                        live_matches.append(ipl_match)
            except Exception:
                pass

            match_options = {m['name']: m for m in live_matches}

            if len(match_options) == 0:
                st.warning("No matches are currently broadcasting on the ESPN Global Matrix right now!")
            else:
                selected_match = st.selectbox("🎯 Target Active Match", list(match_options.keys()))
                if selected_match:
                    m_obj = match_options[selected_match]
                    c_obj = m_obj['competitions'][0]
                    teams_arr = c_obj.get('competitors', [])

                    t1_name = teams_arr[0]['team']['displayName'] if len(teams_arr) > 0 else "Unknown"
                    t2_name = teams_arr[1]['team']['displayName'] if len(teams_arr) > 1 else "Unknown"
                    venue_raw = c_obj.get('venue', {}).get('fullName', '')

                    def get_idx(val, choices):
                        return list(choices).index(val) if val in choices else 0

                    mapped_t1 = get_best_match(t1_name, encoders['team'].classes_, threshold=1) or 'India'
                    mapped_t2 = get_best_match(t2_name, encoders['team'].classes_, threshold=1) or 'Australia'
                    mapped_venue = get_best_match(venue_raw, encoders['venue'].classes_, threshold=1) or encoders['venue'].classes_[0]

                    live_runs, live_wickets, match_overs = 0, 0, 0
                    for t in teams_arr:
                        if t.get('score') and float(t.get('overs', 0)) > 0:
                            live_runs = int(t['score'])
                            live_wickets = int(t.get('wickets', 0))
                            match_overs = float(t.get('overs', 0))
                            break

                    crr = live_runs / match_overs if match_overs > 0 else 0
                    assumed_overs = 50 if match_overs > 20 else 20
                    projected_score = int(crr * assumed_overs) if match_overs > 0 else 0

                    # Venue stats (format-specific lookup)
                    vdf = stats['venue_avg_first']
                    if 'is_t20' in vdf.columns:
                        venue_avg_row = vdf[(vdf['venue'] == mapped_venue) & (vdf['is_t20'] == 1) & (vdf['is_odi'] == 0)]
                    else:
                        venue_avg_row = vdf[vdf['venue'] == mapped_venue]
                    v_avg = venue_avg_row['venue_avg_first_innings'].values[0] if len(venue_avg_row) > 0 else 160.0

                    st.markdown("<br/>", unsafe_allow_html=True)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("🏟️ Match Venue", venue_raw[:15] + "..")
                    m2.metric("⚡ Current Score", f"{live_runs}/{live_wickets}", f"{match_overs} Overs")
                    m3.metric("📈 Current Run Rate", f"{crr:.2f}")
                    m4.metric("🎯 Projected Total", f"{projected_score}", f"{projected_score - v_avg:+.0f} vs Par")

                    st.markdown("### Match Context (Auto-Matched by AI)")
                    col_v, col_t1, col_t2 = st.columns(3)
                    live_venue = col_v.selectbox("AI Mapped Stadium", encoders['venue'].classes_, index=get_idx(mapped_venue, encoders['venue'].classes_), key="live_venue")
                    live_t1 = col_t1.selectbox("AI Mapped Batting Team 1", encoders['team'].classes_, index=get_idx(mapped_t1, encoders['team'].classes_), key="live_t1")
                    live_t2 = col_t2.selectbox("AI Mapped Chasing Team 2", encoders['team'].classes_, index=get_idx(mapped_t2, encoders['team'].classes_), key="live_t2")

                    st.markdown("### 🧑‍🤝‍🧑 Squad Alignment")
                    col1, col2 = st.columns(2)
                    with col1:
                        txt1 = st.text_area(f"{live_t1} Squad (Paste comma separated)", key="live_txt1")
                        if txt1:
                            parsed1 = [m for n in txt1.split(',') if (m := get_best_match(n, available_batters))]
                            if st.button("Link Roster 1", use_container_width=True): st.session_state['live_ms1'] = parsed1[:11]; st.rerun()
                        t1_roster = st.multiselect(f"Active {live_t1} Roster", available_batters, key="live_ms1")

                    with col2:
                        txt2 = st.text_area(f"{live_t2} Squad (Paste comma separated)", key="live_txt2")
                        if txt2:
                            parsed2 = [m for n in txt2.split(',') if (m := get_best_match(n, available_batters))]
                            if st.button("Link Roster 2", use_container_width=True): st.session_state['live_ms2'] = parsed2[:11]; st.rerun()
                        t2_roster = st.multiselect(f"Active {live_t2} Roster", available_batters, key="live_ms2")

                    if st.button("🔥 EXECUTE NEURAL PREDICTION MATRIX", type="primary", use_container_width=True):
                        if len(t1_roster) > 0 and len(t2_roster) > 0:
                            input_runs = projected_score if projected_score > 0 else live_runs
                            prob1, prob2, _, t1_df = execute_model_prediction(live_venue, live_t1, live_t2, t1_roster, t2_roster, 1, 0, input_runs, live_wickets)

                            st.markdown("---")
                            st.markdown("### 🔮 Real-Time Quantum Forecast")
                            res1, res2 = st.columns((1, 1.5))

                            with res1:
                                fig = go.Figure(data=[go.Pie(labels=[live_t1, live_t2],
                                                             values=[prob1, prob2], hole=.5,
                                                             marker_colors=['#0cebeb', '#ff4b2b'],
                                                             textinfo='label+percent')])
                                fig.update_layout(title_text="Win Probability Distribution", showlegend=False,
                                                  margin=dict(t=40, b=0, l=0, r=0),
                                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                                st.plotly_chart(fig, use_container_width=True)

                            with res2:
                                st.markdown("#### 🏏 Batter Dynamic Impact (Live Expected Runs)")
                                render_batter_chart(t1_df, live_t1, live_t2, live_venue, 1, 0)
                        else:
                            st.error("Please assign some players to the active rosters!")
        except Exception as e:
            st.error(f"External API Limit hit. The endpoint is throttling connections: {e}")

with tab_manual:
    col_menu, col_main = st.columns([1, 3])
    with col_menu:
        st.markdown("### 🎛️ Simulation Parameters")
        match_format = st.selectbox("Match Format", ["T20 / IPL", "ODI"])
        is_t20 = 1 if match_format == "T20 / IPL" else 0
        is_odi = 1 if match_format == "ODI" else 0
        venue = st.selectbox("Select Pitch / Venue", np.sort(encoders['venue'].classes_))

        # Venue stats (format-specific lookup)
        vdf = stats['venue_avg_first']
        if 'is_t20' in vdf.columns:
            venue_avg_row = vdf[(vdf['venue'] == venue) & (vdf['is_t20'] == is_t20) & (vdf['is_odi'] == is_odi)]
        else:
            venue_avg_row = vdf[vdf['venue'] == venue]
        v_avg = venue_avg_row['venue_avg_first_innings'].values[0] if len(venue_avg_row) > 0 else (160.0 if is_t20 else 250.0)
        st.metric(label="📊 Historical Venue Avg", value=f"{v_avg:.1f} Runs")
        st.info("Input player rosters on the right to simulate exactly what will happen on this pitch.")

    with col_main:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🏏 Team BATTING 1st")
            team1 = st.selectbox("Home / Batting Team", np.sort(encoders['team'].classes_),
                                 index=list(np.sort(encoders['team'].classes_)).index('India') if 'India' in encoders['team'].classes_ else 0)

            demo_t1 = [b for b in ['V Kohli', 'RG Sharma', 'SA Yadav', 'RR Pant', 'HH Pandya', 'RA Jadeja', 'KL Rahul', 'S Dhawan', 'SS Iyer', 'B Kumar', 'JJ Bumrah'] if b in available_batters]
            with st.expander("🛠️ Smart Importer (e.g. 'kohli, rohit')"):
                txt_m1 = st.text_area("Commas separated", key="t1_txt_m")
                if txt_m1:
                    matches1 = [m for n in txt_m1.split(',') if (m := get_best_match(n, available_batters))]
                    if matches1:
                        if st.button("Apply to Team 1", key="btn_m1"): st.session_state['t1_roster_m'] = matches1; st.rerun()
            t1_roster_m = st.multiselect("Playing XI", available_batters, default=demo_t1[:11], key="t1_roster_m")

        with c2:
            st.markdown("#### 🎯 Team CHASING 2nd")
            team2 = st.selectbox("Away / Chasing Team", np.sort(encoders['team'].classes_),
                                 index=list(np.sort(encoders['team'].classes_)).index('Australia') if 'Australia' in encoders['team'].classes_ else 1)

            demo_t2 = [b for b in ['DA Warner', 'GJ Maxwell', 'SPD Smith', 'AJ Finch', 'MP Stoinis', 'PJ Cummins', 'MA Starc', 'A Zampa', 'MR Marsh', 'TM Head', 'JP Inglis'] if b in available_batters]
            with st.expander("🛠️ Smart Importer (e.g. 'smith, maxwell')"):
                txt_m2 = st.text_area("Commas separated", key="t2_txt_m")
                if txt_m2:
                    matches2 = [m for n in txt_m2.split(',') if (m := get_best_match(n, available_batters))]
                    if matches2:
                        if st.button("Apply to Team 2", key="btn_m2"): st.session_state['t2_roster_m'] = matches2; st.rerun()
            t2_roster_m = st.multiselect("Playing XI", available_batters, default=demo_t2[:11], key="t2_roster_m")

        if st.button("🌌 RUN DEEP PRE-SIMULATION", type="primary", use_container_width=True):
            if len(t1_roster_m) == 0 or len(t2_roster_m) == 0:
                st.error("Please assign some players to the rosters!")
                st.stop()

            st.markdown("---")
            prob1, prob2, _, t1_df = execute_model_prediction(venue, team1, team2, t1_roster_m, t2_roster_m, is_t20, is_odi, v_avg, 6.0)

            r_c1, r_c2 = st.columns((1, 1.5))
            with r_c1:
                st.markdown("#### 🏆 Win Probability (Pre-Toss)")
                fig = go.Figure(data=[go.Pie(labels=[team1, team2], values=[prob1, prob2], hole=.6,
                                             marker_colors=['#29ffc6', '#ff416c'], textinfo='label+percent')])
                fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0),
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with r_c2:
                st.markdown("#### 🏏 Batter Dynamic Impact")
                render_batter_chart(t1_df, team1, team2, venue, is_t20, is_odi)
