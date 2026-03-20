import os
import pandas as pd
import glob
import time

# Script to assemble all individual T20I CSVs and Info CSVs into a format matching the IPL dataset.
T20I_DIR = r"g:\ML\Cricket\Data\T20I"
OUTPUT_FILE = r"g:\ML\Cricket\Data\T20I_combined.csv"

def process_cricsheet_data():
    print("Finding all Cricsheet CSVs...")
    all_files = glob.glob(os.path.join(T20I_DIR, "*.csv"))
    
    # Separate info files from data files
    info_files = [f for f in all_files if f.endswith("_info.csv")]
    data_files = [f for f in all_files if not f.endswith("_info.csv")]
    
    print(f"Found {len(data_files)} match files and {len(info_files)} info files.")
    
    if len(data_files) == 0:
        print("No data found! Double check the directory.")
        return

    # First, let's parse the info files to get match_level metadata quickly
    match_metadata = {}
    print("Parsing info files for metadata (match_won_by, toss_winner, toss_decision)...")
    
    for info_file in info_files:
        try:
            match_id = os.path.basename(info_file).replace('_info.csv', '')
            with open(info_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                winner = None
                toss_winner = None
                toss_decision = None
                city = None
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        if parts[1] == 'winner':
                            winner = parts[2]
                        elif parts[1] == 'toss_winner':
                            toss_winner = parts[2]
                        elif parts[1] == 'toss_decision':
                            toss_decision = parts[2]
                        elif parts[1] == 'city':
                            city = parts[2]
            
            match_metadata[match_id] = {
                'match_won_by': winner,
                'toss_winner': toss_winner,
                'toss_decision': toss_decision,
                'city': city
            }
        except Exception as e:
            continue

    print(f"Extracted metadata for {len(match_metadata)} matches.")
    
    # Process the ball-by-ball files
    dfs = []
    print("Processing ball-by-ball match files...")
    start = time.time()
    
    for idx, f in enumerate(data_files):
        try:
            df = pd.read_csv(f, low_memory=False)
            match_id = str(df['match_id'].iloc[0])
            
            # Match schema to IPL script dependencies
            # IPL columns expected: match_id, date, batting_team, bowling_team, innings, over, ball, batter, runs_batter, bowler, runs_total, player_out, venue, toss_winner, toss_decision, match_won_by, extra_type
            
            df = df.rename(columns={
                'start_date': 'date',
                'striker': 'batter',
                'runs_off_bat': 'runs_batter'
            })
            
            if 'player_dismissed' in df.columns:
                df['player_out'] = df['player_dismissed']
            else:
                df['player_out'] = None
                
            # Compute over and ball (from ball float: e.g. 0.1 -> over 0, ball 1)
            # wait, cricsheet gives it as 0.1, 0.2, etc. in 'ball' column
            df['over'] = df['ball'].apply(lambda x: int(float(x)))
            # Keep original ball for sorting but we can use it as is
            
            # Compute runs_total = runs_batter + extras
            if 'extras' in df.columns:
                df['runs_total'] = df['runs_batter'] + df['extras']
            else:
                df['runs_total'] = df['runs_batter']
            
            # Append match metadata
            meta = match_metadata.get(match_id, {})
            df['match_won_by'] = meta.get('match_won_by', None)
            df['toss_winner'] = meta.get('toss_winner', None)
            df['toss_decision'] = meta.get('toss_decision', None)
            df['city'] = meta.get('city', None)
            
            # Select only needed columns to save RAM and disk space
            needed_cols = [
                'match_id', 'date', 'batting_team', 'bowling_team', 'innings', 'over', 'ball',
                'batter', 'bowler', 'runs_batter', 'runs_total', 'player_out', 'venue', 'city',
                'match_won_by', 'toss_winner', 'toss_decision'
            ]
            
            # Include only columns that exist
            df = df[[c for c in needed_cols if c in df.columns]]
            dfs.append(df)
            
            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1} files...")
        except Exception as e:
            continue

    if dfs:
        print("Concatenating all dataframes...")
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined DataFrame shape: {combined_df.shape}")
        
        # Save to CSV
        combined_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully saved to {OUTPUT_FILE}")
        print(f"Total time: {time.time() - start:.1f}s")
    else:
        print("No valid dataframes to combine.")

if __name__ == "__main__":
    process_cricsheet_data()
