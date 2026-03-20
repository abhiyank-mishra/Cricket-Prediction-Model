import requests
import json

url = "http://mapps.cricbuzz.com/cbzios/match/livematches"
try:
    response = requests.get(url, headers={'User-Agent':'Mozilla/5.0'})
    data = response.json()
    print("KEYS:", data.keys())
    if 'matches' in data:
        print(f"Found {len(data['matches'])} matches.")
        if len(data['matches']) > 0:
            print("First match:", json.dumps(data['matches'][0], indent=2))
except Exception as e:
    print("Error:", e)
