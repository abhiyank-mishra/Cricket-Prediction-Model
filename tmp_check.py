import pandas as pd
df = pd.read_csv('Data/India_Mens_Combined.csv', low_memory=False)
ami_t20 = df[(df['venue'] == 'AMI Stadium') & (df['match_format'].str.contains('T20')) & (df['innings'] == 1)]
print("Total rows:", len(ami_t20))
print("Match IDs:", ami_t20['match_id'].unique())
for match in ami_t20['match_id'].unique():
    print(f"Match {match} fi_runs:", ami_t20[ami_t20['match_id'] == match]['runs_total'].sum())
