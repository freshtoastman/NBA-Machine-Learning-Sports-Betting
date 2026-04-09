"""NBA team metadata: official CDN team IDs and Traditional Chinese names.

Used by Flask templates to render team logos and Chinese display names.
"""

# Real NBA stats team IDs (used by cdn.nba.com logo URLs).
NBA_TEAM_IDS = {
    "Atlanta Hawks": 1610612737,
    "Boston Celtics": 1610612738,
    "Brooklyn Nets": 1610612751,
    "Charlotte Hornets": 1610612766,
    "Chicago Bulls": 1610612741,
    "Cleveland Cavaliers": 1610612739,
    "Dallas Mavericks": 1610612742,
    "Denver Nuggets": 1610612743,
    "Detroit Pistons": 1610612765,
    "Golden State Warriors": 1610612744,
    "Houston Rockets": 1610612745,
    "Indiana Pacers": 1610612754,
    "Los Angeles Clippers": 1610612746,
    "LA Clippers": 1610612746,
    "Los Angeles Lakers": 1610612747,
    "Memphis Grizzlies": 1610612763,
    "Miami Heat": 1610612748,
    "Milwaukee Bucks": 1610612749,
    "Minnesota Timberwolves": 1610612750,
    "New Orleans Pelicans": 1610612740,
    "New York Knicks": 1610612752,
    "Oklahoma City Thunder": 1610612760,
    "Orlando Magic": 1610612753,
    "Philadelphia 76ers": 1610612755,
    "Phoenix Suns": 1610612756,
    "Portland Trail Blazers": 1610612757,
    "Sacramento Kings": 1610612758,
    "San Antonio Spurs": 1610612759,
    "Toronto Raptors": 1610612761,
    "Utah Jazz": 1610612762,
    "Washington Wizards": 1610612764,
}

TEAM_NAMES_ZH = {
    "Atlanta Hawks": "亞特蘭大老鷹",
    "Boston Celtics": "波士頓塞爾提克",
    "Brooklyn Nets": "布魯克林籃網",
    "Charlotte Hornets": "夏洛特黃蜂",
    "Chicago Bulls": "芝加哥公牛",
    "Cleveland Cavaliers": "克里夫蘭騎士",
    "Dallas Mavericks": "達拉斯獨行俠",
    "Denver Nuggets": "丹佛金塊",
    "Detroit Pistons": "底特律活塞",
    "Golden State Warriors": "金州勇士",
    "Houston Rockets": "休士頓火箭",
    "Indiana Pacers": "印第安納溜馬",
    "Los Angeles Clippers": "洛杉磯快艇",
    "LA Clippers": "洛杉磯快艇",
    "Los Angeles Lakers": "洛杉磯湖人",
    "Memphis Grizzlies": "曼菲斯灰熊",
    "Miami Heat": "邁阿密熱火",
    "Milwaukee Bucks": "密爾瓦基公鹿",
    "Minnesota Timberwolves": "明尼蘇達灰狼",
    "New Orleans Pelicans": "紐奧良鵜鶘",
    "New York Knicks": "紐約尼克",
    "Oklahoma City Thunder": "奧克拉荷馬雷霆",
    "Orlando Magic": "奧蘭多魔術",
    "Philadelphia 76ers": "費城七六人",
    "Phoenix Suns": "鳳凰城太陽",
    "Portland Trail Blazers": "波特蘭拓荒者",
    "Sacramento Kings": "沙加緬度國王",
    "San Antonio Spurs": "聖安東尼奧馬刺",
    "Toronto Raptors": "多倫多暴龍",
    "Utah Jazz": "猶他爵士",
    "Washington Wizards": "華盛頓巫師",
}


def team_logo_url(team_name):
    """Return official NBA CDN logo URL for a team display name, or None."""
    team_id = NBA_TEAM_IDS.get(team_name)
    if team_id is None:
        return None
    return f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"


def team_name_zh(team_name):
    """Return Traditional Chinese name; falls back to original if missing."""
    return TEAM_NAMES_ZH.get(team_name, team_name)
