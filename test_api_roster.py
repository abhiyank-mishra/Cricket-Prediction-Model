import requests
import json

api_key = "7e4a2dca-94d6-4232-b163-cafd682ec2ef"
try:
    matches = requests.get(f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}&offset=0").json()
    if matches.get('data'):
        m_id = matches['data'][0]['id']
        print("Match ID:", m_id)
        
        info = requests.get(f"https://api.cricapi.com/v1/match_info?apikey={api_key}&offset=0&id={m_id}").json()
        print("MATCH INFO KEYS:", info.get('data', {}).keys())
        
        squad = requests.get(f"https://api.cricapi.com/v1/match_squad?apikey={api_key}&id={m_id}").json()
        print("SQUAD STATUS:", squad.get('status'))
        if squad.get('data'):
            import pprint
            pprint.pprint(squad['data'][0])
except Exception as e:
    print(e)
