import pandas as pd
df = pd.read_csv('Data/India_Mens_Combined.csv', low_memory=False)
q = df[(df['match_id'] == 386494) & (df['innings'] == 1)]
print(q[['over', 'ball', 'batter', 'bowler', 'runs_total']].head(15))
