import requests
from bs4 import BeautifulSoup

url = "https://www.cricbuzz.com/cricket-match/live-scores"
headers = {"User-Agent": "Mozilla/5.0"}
try:
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    matches = soup.find_all('div', class_='cb-mtch-lst')
    print(f"Found {len(matches)} matches on cricbuzz!")
    
    for match in matches[:3]:
        teams = match.find('h3', class_='cb-lv-scr-mtch-hdr')
        if teams:
            title = teams.text.strip()
            print("MATCH:", title)
            
            venue = match.find('div', class_='text-gray')
            if venue: print("VENUE:", venue.text.strip())
            
            bat_team = match.find('div', class_='cb-hm-rght')
            if bat_team: 
                print("BATTING TEAM/SCORE:", bat_team.text.strip())
                
        print("---")
except Exception as e:
    print(e)
