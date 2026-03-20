import requests
import json

api_key = "7e4a2dca-94d6-4232-b163-cafd682ec2ef"
try:
    res = requests.get(f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}&offset=0")
    print(res.json())
except Exception as e:
    print(e)
