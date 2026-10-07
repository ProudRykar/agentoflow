import json

data = {
  "findTags": {
    "count": 624,
    "tags": [
      {"id": "2000", "name": "Anal Penetration", "scene_count": 2},
      {"id": "2003", "name": "BBC Worship", "scene_count": 1},
      {"id": "1997", "name": "Birthday", "scene_count": 1},
      {"id": "1994", "name": "Black Stockings", "scene_count": 0},
      {"id": "750", "name": "Blond Hair (Male)", "scene_count": 1},
      {"id": "686", "name": "Blue Hair (Female)", "scene_count": 1},
      {"id": "1402", "name": "bob cut", "scene_count": 1},
      {"id": "884", "name": "Bodyguard", "scene_count": 2},
      {"id": "925", "name": "Bottomless", "scene_count": 1},
      {"id": "323", "name": "Bouncing Tits", "scene_count": 3},
      {"id": "728", "name": "Bound Wrists", "scene_count": 1},
      {"id": "947", "name": "Boyshorts", "scene_count": 1},
      {"id": "2004", "name": "Braids", "scene_count": 1},
      {"id": "931", "name": "Brazilian", "scene_count": 1},
      {"id": "1254", "name": "Breast Holding", "scene_count": 1},
      {"id": "1255", "name": "Breast Squeezing", "scene_count": 1},
      {"id": "930", "name": "Cameltoe", "scene_count": 1},
      {"id": "869", "name": "Canadian", "scene_count": 1},
      {"id": "771", "name": "Cape", "scene_count": 1},
      {"id": "1155", "name": "Cardigan", "scene_count": 1},
      {"id": "1485", "name": "casero", "scene_count": 1},
      {"id": "298", "name": "Caught", "scene_count": 1},
      {"id": "1756", "name": "caught jerking off", "scene_count": 1},
      {"id": "1794", "name": "celeb", "scene_count": 1},
      {"id": "914", "name": "CFNM", "scene_count": 1},
      {"id": "904", "name": "CGI", "scene_count": 1},
      {"id": "624", "name": "Chilean", "scene_count": 1},
      {"id": "150", "name": "Climax", "scene_count": 0},
      {"id": "1314", "name": "closed caption", "scene_count": 1},
      {"id": "672", "name": "Clothed Sex", "scene_count": 10},
      {"id": "629", "name": "Cock Slapping", "scene_count": 3},
      {"id": "704", "name": "Coerced", "scene_count": 1},
      {"id": "217", "name": "Collar", "scene_count": 1},
      {"id": "1899", "name": "colleague", "scene_count": 1},
      {"id": "181", "name": "Colombian", "scene_count": 7},
      {"id": "555", "name": "Colonial Era", "scene_count": 1},
      {"id": "956", "name": "Colored Contacts", "scene_count": 3},
      {"id": "366", "name": "Colored Hair (Female)", "scene_count": 5},
      {"id": "722", "name": "Colored Stockings", "scene_count": 2},
      {"id": "1272", "name": "compulsion", "scene_count": 1},
      {"id": "333", "name": "Condom", "scene_count": 4},
      {"id": "726", "name": "Condom Removal", "scene_count": 1},
      {"id": "1500", "name": "cornudos", "scene_count": 1},
      {"id": "814", "name": "Corporal Punishment", "scene_count": 10},
      {"id": "741", "name": "Corruption", "scene_count": 2},
      {"id": "772", "name": "Corset", "scene_count": 1},
      {"id": "194", "name": "Cougar", "scene_count": 2},
      {"id": "1010", "name": "Countertop", "scene_count": 1},
      {"id": "798", "name": "Couple", "scene_count": 5},
      {"id": "749", "name": "Cowboy Hat", "scene_count": 1}
    ]
  }
}

tag_ids = [tag['id'] for tag in data['findTags']['tags']]
print(f"Found {len(tag_ids)} tags with missing descriptions.")