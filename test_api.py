import requests
try:
    url = "https://api.cricapi.com/v1/currentMatches?apikey=demo&offset=0"
    res = requests.get(url)
    print(res.json())
except Exception as e:
    print(e)
