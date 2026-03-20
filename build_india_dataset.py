import os
import pandas as pd
import glob
import time

DOMESTIC_DIRS = [
    r"g:\ML\Cricket\Data\India_Specific\IPL",
    r"g:\ML\Cricket\Data\India_Specific\WPL"
]
T20I_CSV = r"g:\ML\Cricket\Data\T20I_combined.csv"
OUTPUT_FILE = r"g:\ML\Cricket\Data\India_Specific_Combined.csv"

def extract_metadata(info_files):
    match_metadata = {}
    for info_file in info_files:
        try:
            match_id = os.path.basename(info_file).replace('_info.csv', '')
            with open(info_file, 'r', encoding='utf-8') as f:
                winner, toss_winner, toss_decision, city = None, None, None, None
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        if parts[1] == 'winner': winner = parts[2]
                        elif parts[1] == 'toss_winner': toss_winner = parts[2]
                        elif parts[1] == 'toss_decision': toss_decision = parts[2]
                        elif parts[1] == 'city': city = parts[2]
            
            match_metadata[match_id] = {
                'match_won_by': winner, 'toss_winner': toss_winner,
                'toss_decision': toss_decision, 'city': city
            }
        except:
            pass
    return match_metadata

def process_india_data():
    dfs = []
    print("Parsing Domestic India Data...")
    
    needed_cols = [
        'match_id', 'date', 'batting_team', 'bowling_team', 'innings', 'over', 'ball',
        'batter', 'bowler', 'runs_batter', 'runs_total', 'player_out', 'venue', 'city',
        'match_won_by', 'toss_winner', 'toss_decision'
    ]
    
    for directory in DOMESTIC_DIRS:
        all_files = glob.glob(os.path.join(directory, "*.csv"))
        info_files = [f for f in all_files if f.endswith("_info.csv")]
        data_files = [f for f in all_files if not f.endswith("_info.csv")]
        
        print(f"[{os.path.basename(directory)}] Found {len(data_files)} matches.")
        match_metadata = extract_metadata(info_files)
        
        for idx, f in enumerate(data_files):
            try:
                df = pd.read_csv(f, low_memory=False)
                match_id = str(df['match_id'].iloc[0])
                
                df = df.rename(columns={'start_date': 'date', 'striker': 'batter', 'runs_off_bat': 'runs_batter'})
                if 'player_dismissed' in df.columns: df['player_out'] = df['player_dismissed']
                else: df['player_out'] = None
                
                df['over'] = df['ball'].apply(lambda x: int(float(x)))
                
                if 'extras' in df.columns: df['runs_total'] = df['runs_batter'] + df['extras']
                else: df['runs_total'] = df['runs_batter']
                
                meta = match_metadata.get(match_id, {})
                df['match_won_by'] = meta.get('match_won_by')
                df['toss_winner'] = meta.get('toss_winner')
                df['toss_decision'] = meta.get('toss_decision')
                df['city'] = meta.get('city')
                
                df['match_id'] = f"{os.path.basename(directory)}_{match_id}"
                
                df = df[[c for c in needed_cols if c in df.columns]]
                dfs.append(df)
            except:
                continue

    # Now load international matches specifically played by India
    print("Loading International match data to extract India matches...")
    if os.path.exists(T20I_CSV):
        t20_df = pd.read_csv(T20I_CSV, low_memory=False)
        ind_international = t20_df[(t20_df['batting_team'] == 'India') | (t20_df['bowling_team'] == 'India') | 
                                   (t20_df['batting_team'] == 'India Women') | (t20_df['bowling_team'] == 'India Women')]
        print(f"[T20 Internationals] Found {len(ind_international['match_id'].unique())} matches played by India.")
        if len(ind_international) > 0:
            ind_international['match_id'] = 'T20I_' + ind_international['match_id'].astype(str)
            dfs.append(ind_international)
            
    print("Concatenating all Indian data frames...")
    final_df = pd.concat(dfs, ignore_index=True)
    # Filter out empty winner if predicting Match Winner purely
    print(f"Saved {len(final_df)} super specific Indian ball-by-ball deliveries to disk!")
    final_df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    process_india_data()
