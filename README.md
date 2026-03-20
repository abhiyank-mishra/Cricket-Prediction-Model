# 🏏 Indian Men's Cricket AI Oracle

An advanced, real-time Machine Learning dashboard that predicts Cricket Match Winners, expected player runs, and quick-out threats using an enormous database of **1.59 Million+ ball-by-ball deliveries**.

## 🧠 Model Accuracy & Intelligence
- **Match Winner Model (Ensemble):** `89.07% Accuracy` (Trained using XGBoost + LightGBM + Random Forest + CatBoost)
- **Quick Out Predictor:** `89.35% Accuracy` (Flags batters likely to dismiss under 10 runs)
- **Player Runs Predictor:** Estimates individual expected runs dynamically based on the venue pitch history and opposing attack.

## 🚀 How to Run (On any New Laptop)

The raw data (`.zip` files) are securely stored in this repository! You **do not** need to manually download CSV files from the internet. Just follow these steps:

1. **Clone the Repository** and enter the folder:
   ```bash
   git clone <your-repo-link>
   cd Cricket
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then install the required AI libraries:
   ```bash
   pip install streamlit pandas numpy scikit-learn lightgbm xgboost catboost plotly
   ```

3. **Initialize Data & Train Models (Run Once)**:
   This will automatically extract all tracked `.zip` JSON datasets, build the massive 200MB CSV into memory, and train the Machine Learning `models/*.pkl` objects on your new machine.
   ```bash
   python extract_and_train.py
   python india_mens_model.py
   ```

4. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```
   *The Oracle UI will instantly pop open in your browser!*
