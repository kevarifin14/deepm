import json

from markethistory import MarketHistory
from constants import * 

with open('config.json') as file:
    config = json.load(file)
markethistory = MarketHistory(config)

