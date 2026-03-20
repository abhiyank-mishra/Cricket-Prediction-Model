import os
import pandas as pd
import glob
import time

DOMESTIC_DIRS = [
    r"g:\ML\Cricket\Data\India_Specific\IPL",
    r"g:\ML\Cricket\Data\India_Specific\ODI",
    r"g:\ML\Cricket\Data\India_Specific\TEST"
]
T20I_DIR = r"g:\ML\Cricket\Data\T20I"

OUTPUT_FILE = r"g:\ML\Cricket\Data\India_Mens_Combined.csv"

def extract_metadata(info_files):
    match_metadata = {}
    for info_file in info_files:
        try:
            match_id = os.path.basename(info_file).replace('_info.csv', '')
            with open(info_file, 'r', encoding='utf-8') as f:
                winner, toss_winner, toss_decision, city = None, None, None, None
                gender = None
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        if parts[1] == 'gender': gender = parts[2]
                        elif parts[1] == 'winner': winner = parts[2]
                        elif parts[1] == 'toss_winner': toss_winner = parts[2]
                        elif parts[1] == 'toss_decision': toss_decision = parts[2]
                        elif parts[1] == 'city': city = parts[2]
            
            # We ONLY want male matches.
            if gender == 'male':
                match_metadata[match_id] = {
                    'match_won_by': winner, 'toss_winner': toss_winner,
                    'toss_decision': toss_decision, 'city': city
                }
        except Exception as e:
            pass
    return match_metadata

def process_india_mens_data():
    dfs = []
    print("Parsing All India Mens Data (IPL, ODIs, Tests)...")
    
    needed_cols = [
        'match_id', 'date', 'match_format', 'batting_team', 'bowling_team', 'innings', 'over', 'ball',
        'batter', 'bowler', 'runs_batter', 'runs_total', 'player_out', 'venue', 'city',
        'match_won_by', 'toss_winner', 'toss_decision'
    ]
    
    # Process all directories
    all_directories = DOMESTIC_DIRS + [T20I_DIR]
    
    for directory in all_directories:
        format_name = os.path.basename(directory)
        print(f"Reading directory: {format_name}")
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} doesn't exist yet.")
            continue
            
        all_files = glob.glob(os.path.join(directory, "*.csv"))
        info_files = [f for f in all_files if f.endswith("_info.csv")]
        data_files = [f for f in all_files if not f.endswith("_info.csv")]
        
        print(f"[{format_name}] Found {len(data_files)} matches. Extracting metadata...")
        match_metadata = extract_metadata(info_files)
        
        valid_matches = 0
        for idx, f in enumerate(data_files):
            try:
                match_id = str(os.path.basename(f).replace('.csv', ''))
                
                # If it's not a male match recorded in info.csv, skip it forever.
                if match_id not in match_metadata:
                    continue
                
                df = pd.read_csv(f, low_memory=False)
                
                teams_involved = set(df['batting_team'].dropna().unique()).union(set(df['bowling_team'].dropna().unique()))
                
                is_indian_context = False
                if format_name == 'IPL':
                    is_indian_context = True # All IPL is Indian Men's
                elif any('India' in t for t in teams_involved):
                    is_indian_context = True # If it's an International match involving India
                    
                if not is_indian_context:
                    continue
                    
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
                
                df['match_id'] = f"{format_name}_{match_id}"
                df['match_format'] = format_name
                
                df = df[[c for c in needed_cols if c in df.columns]]
                dfs.append(df)
                valid_matches += 1
            except Exception as e:
                continue
                
        print(f"-> Extracted {valid_matches} confirmed Indian Mens Matches from {format_name}.")

    print("Concatenating all Indian data frames...")
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        print(f"Saved {len(final_df)} super specific Mens Indian ball-by-ball deliveries to disk!")
        final_df.to_csv(OUTPUT_FILE, index=False)
    else:
        print("No valid matches found.")

if __name__ == "__main__":
    process_india_mens_data()
