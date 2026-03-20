import os
import zipfile
import json
import pandas as pd
import glob
import time

DATA_DIR = r"g:\ML\Cricket\Data"
OUTPUT_FILE = r"g:\ML\Cricket\Data\India_Mens_Mega_Combined_JSON.csv"

def process_all_zips():
    dfs = []
    zip_files = glob.glob(os.path.join(DATA_DIR, "*.zip"))
    print(f"Found {len(zip_files)} zip archives.")
    
    total_matches = 0
    valid_india_matches = 0
    
    for zf_path in zip_files:
        if '_csv' in zf_path:
            continue
            
        print(f"Processing archive: {os.path.basename(zf_path)}")
        try:
            with zipfile.ZipFile(zf_path, 'r') as zf:
                json_files = [f for f in zf.namelist() if f.endswith('.json')]
                for jf in json_files:
                    total_matches += 1
                    with zf.open(jf) as f:
                        data = json.loads(f.read().decode('utf-8'))
                        
                    info = data.get('info', {})
                    if info.get('gender') != 'male':
                        continue
                        
                    teams = info.get('teams', [])
                    match_type = str(info.get('match_type', ''))
                    
                    # Filtering: Is it India Men's or IPL?
                    is_indian_context = False
                    if 'IPL' in str(info.get('competition', '')) or match_type == 'IPL':
                        is_indian_context = True
                    elif 'India' in teams:
                        is_indian_context = True
                        
                    if not is_indian_context:
                        continue
                        
                    # Extract Match Level Data
                    match_id = jf.replace('.json', '')
                    date = info.get('dates', [''])[0]
                    city = info.get('city', '')
                    venue = info.get('venue', '')
                    toss = info.get('toss', {})
                    toss_winner = toss.get('winner', '')
                    toss_decision = toss.get('decision', '')
                    
                    outcome = info.get('outcome', {})
                    match_won_by = outcome.get('winner', '')
                    
                    rows = []
                    innings = data.get('innings', [])
                    for inn_idx, inning in enumerate(innings):
                        batting_team = inning.get('team', '')
                        bowling_team = [t for t in teams if t != batting_team]
                        bowling_team = bowling_team[0] if bowling_team else ''
                        
                        overs = inning.get('overs', [])
                        for over in overs:
                            over_num = over.get('over')
                            for ball_idx, delivery in enumerate(over.get('deliveries', [])):
                                runs = delivery.get('runs', {})
                                wks = delivery.get('wickets', [])
                                
                                player_out = None
                                if wks:
                                    player_out = wks[0].get('player_out')
                                    
                                row = {
                                    'match_id': match_id,
                                    'date': date,
                                    'match_format': match_type,
                                    'batting_team': batting_team,
                                    'bowling_team': bowling_team,
                                    'innings': inn_idx + 1,
                                    'over': over_num,
                                    'ball': ball_idx + 1,
                                    'batter': delivery.get('batter'),
                                    'bowler': delivery.get('bowler'),
                                    'runs_batter': runs.get('batter', 0),
                                    'runs_total': runs.get('total', 0),
                                    'player_out': player_out,
                                    'venue': venue,
                                    'city': city,
                                    'match_won_by': match_won_by,
                                    'toss_winner': toss_winner,
                                    'toss_decision': toss_decision
                                }
                                rows.append(row)
                    
                    if rows:
                        dfs.append(pd.DataFrame(rows))
                        valid_india_matches += 1
                        
                    if valid_india_matches % 500 == 0 and valid_india_matches > 0:
                        print(f"Extracted {valid_india_matches} Indian matches so far...")
                        
        except Exception as e:
            print(f"Error processing {zf_path}: {e}")
            
    print(f"\nScanning Initialized: Scanned {total_matches} Total JSON balls across the entire globe.")
    print(f"Successfully extracted {valid_india_matches} India/IPL Context Matches!")
    
    if dfs:
        print("Concatenating into a Giant Dataframe...")
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved MEGA DATASET to {OUTPUT_FILE} completely. Rows: {len(final_df)}")
        
        # We must overwrite the input to the original model trainer
        import shutil
        shutil.copy(OUTPUT_FILE, r"g:\ML\Cricket\Data\India_Mens_Combined.csv")
        print("Overwrote old baseline dataset with the new MEGA Universal JSON Extract.")
    else:
        print("No valid data extracted.")

if __name__ == "__main__":
    s = time.time()
    process_all_zips()
    print(f"Full JSON Pipeline Execution Time: {time.time() - s:.2f}s")
