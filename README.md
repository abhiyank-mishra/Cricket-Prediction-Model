# 🏏 Indian Men's Cricket AI Oracle Architecture

This document provides a comprehensive overview of the **Indian Men's Cricket AI Oracle**, a state-of-the-art predictive machine learning application designed to forecast cricket match outcomes, player expected runs, and quick-out threats either statically (pre-match) or dynamically (live-tracking).

---

## 🏗️ System Architecture & Components

The application is structured into three primary architectural pillars: **Data Ingestion**, **Machine Learning Pipelines**, and the **Interactive Dashboard**.

### 1. Data Ingestion & Processing Pipeline
Scripts responsible for fetching, cleaning, and structuring massive amounts of raw ball-by-ball data.
*   **`download_cricsheet.py`**: Fetches the latest JSON/CSV archives from the open-source Cricsheet repository.
*   **`parse_cricsheet_data.py`**: Converts nested match data into flat, manageable CSV structures.
*   **`build_india_dataset.py` & `build_india_mega_dataset.py`**: Complex filtering ETL scripts that extract only relevant matches (e.g., India Men\'s matches across IPL, ODIs, and T20Is). It processes the metadata (`_info.csv`) and ball-by-ball telemetry, calculates over/ball formatting, and produces the monolithic `India_Mens_Combined.csv` dataset.

### 2. Feature Engineering & ML Models (`india_mens_model.py`)
This script acts as the core brain. It calculates historically intensive rolling averages, head-to-head batter vs. attack metrics (strike rates, dismissals), and venue chase/toss parities.

Three primary models are produced and serialized into the `/models/` directory:
1.  **Match Winner Ensemble (`match_winner_ensemble.pkl`)**: 
    A colossal **Hybrid Hive Model** (Soft Voting Classifier) combining the strengths of four cutting-edge algorithms:
    *   **LightGBM** (`LGBMClassifier`)
    *   **XGBoost** (`XGBClassifier`)
    *   **Random Forest** 
    *   **CatBoost** (`CatBoostClassifier`)
    *   *Features*: Venue averages, Toss parity, Head-to-Head win rates, Roster form, and live-innings telemetry (e.g. powerplay wickets).
2.  **Player Runs Predictor (`player_runs_model.pkl`)**: 
    A specialized `LGBMRegressor` that predicts precisely how many runs a specific batter will score based on form, their historic performance against the opposition\'s bowling attack, batting position, and pitch analytics.
3.  **Quick Out Predictor (`quick_out_model.pkl`)**: 
    An `LGBMClassifier` trained to predict the probability of a batter getting dismissed cheaply (under 10 runs within 10 balls).

### 3. Real-Time Application UI (`app.py`)
A gorgeous, asynchronous web application developed using **Streamlit** encompassing premium UX characteristics:
*   **Global Live Matrix**: Pulls raw real-time scoreboard data from the hidden **ESPN GraphQL/REST API**.
*   **Dynamic Projected Score Engine**: Employs live Current Run Rate (CRR) parsing to calculate a target *Projected Score* for incomplete first innings. This bridges the gap between pre-match trained ML bounds and live-match scenarios seamlessly.
*   **Glassmorphism & Aesthetics**: Injects raw CSS payload for custom gradient headers, neon glowing metric cards, and a pulsating live status tracker.
*   **Plotly Integration**: Renders highly interactive donut charts (`go.Pie`) for Win Probability dispersion and conditionally color-coded horizontal bar charts (`go.Bar`) warning the user about potential "Quick Out" threats in real-time.

---

## 💾 Data Scope and Volumes

The computational scale of this oracle relies heavily on the open-source ball-by-ball telemetry provided by Cricsheet. 

### Core Datasets Involved
- **`India_Mens_Combined.csv` (~155 MB)**: The primary training dataset containing heavily filtered and joined metadata specific strictly to Indian Men's cricket (International fixtures and IPL).
- **`T20I_combined.csv` (~161 MB)**: Broader T20I universe dataset.
- **`IPL.csv` (~107 MB)**: Domestic Indian Premier League data boundary.
- **`models/latest_player_stats.csv`**: A dense, localized tabular cache generated after model training. Contains rolling 5-window forms, lifetime strike rates, and career run metrics for every recognized batsman. Ensures the streamlit UI can execute $O(1)$ lookups without querying the 155 MB database.

> [!NOTE] 
> **Ball-by-Ball Grandularly:** 155 MB of pure tabulated numerical/string data represents data spanning **millions of individual deliveries** meticulously tracked across decades of cricket, allowing the XGBoost/CatBoost algorithms to recognize subtle micro-patterns (e.g., how Virat Kohli performs at Wankhede Stadium against a right-arm pace attack in the powerplay).

---

## 🚀 Execution Flow Summary

1. User queries the **Streamlit Web UI** (`app.py`).
2. Streamlit auto-pings **ESPN API** and parses Live `t1_score` and `t2_score`.
3. Background algorithms extrapolate the projected score based on RR.
4. Extrapolated stats are fed into the **XGBoost/LightGBM Pipeline** alongside `joblib` loaded context parameters (H2H rates, toss stats).
5. Plotly dynamically re-renders probability Donuts and Threat Bar Charts instantly.
