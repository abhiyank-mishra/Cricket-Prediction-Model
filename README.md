# 🏏 Indian Men's Cricket AI Oracle

An advanced, real-time Machine Learning dashboard that predicts Cricket Match Winners, expected player runs, and quick-out threats using an enormous database of **1.59 Million+ ball-by-ball deliveries**.

## 🧠 Model Accuracy & Intelligence
- **Match Winner Model (Ensemble):** `89.07% Accuracy` (Trained using XGBoost + LightGBM + Random Forest + CatBoost)
- **Quick Out Predictor:** `89.35% Accuracy` (Flags batters likely to dismiss under 10 runs)
- **Player Runs Predictor:** Estimates individual expected runs dynamically based on the venue pitch history and opposing attack.

## 🚀 How to Run (On any New Laptop)

The raw data is NOT stored in Git to save space. A mega setup script will automatically stream it directly into your computer's RAM! Just follow these steps:

1. **Clone the Repository** and enter the folder:
   ```bash
   git clone <your-repo-link>
   cd Cricket
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then install the required AI libraries:
   ```bash
   pip install streamlit pandas numpy scikit-learn lightgbm xgboost catboost plotly requests
   ```

3. **Initialize Data & Train Models (Run Once)**:
   This will completely automate the setup: Downloading massive Cricsheet archives stealthily into memory, parsing relevant India-context matches, producing a Mega CSV, and finally compiling/training all 4 Machine Learning models instantly.
   ```bash
   python startup.py
   ```

4. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```
   *The Oracle UI will instantly pop open in your browser fueled by Live ESPN API Data!*
