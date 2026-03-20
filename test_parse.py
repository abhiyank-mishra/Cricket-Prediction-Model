import requests
import json

api_key = "7e4a2dca-94d6-4232-b163-cafd682ec2ef"
m_id = "2e7fce75-8ebb-43e1-b81e-1724d6ade08d"
sq_res = requests.get(f"https://api.cricapi.com/v1/match_squad?apikey={api_key}&id={m_id}").json()

if sq_res.get('data'):
    squad_data = sq_res['data']
    print("SQUAD DATA TEAM 1:", json.dumps(squad_data[0], indent=2)[:500])
    
    # Try finding players
    players = squad_data[0].get('players', [])
    if len(players) > 0:
        print("Player example keys:", players[0].keys())
        print("First player:", players[0])
    else:
        print("Players array is empty.")
else:
    print("No data in sq_res")
