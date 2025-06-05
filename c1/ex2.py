import json
import requests
from textblob import TextBlob

# Definim URL-ul API-ului CoinDesk (sau CryptoCompare în acest caz) pentru a obține ultimele știri despre Bitcoin
url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

# Efectuăm o cerere GET către API
response = requests.get(url)
# Parsăm răspunsul din format JSON într-o structură de date Python (dicționare și liste)
data = response.json()

# Iterăm prin fiecare articol din lista de știri primită
for article in data["Data"]:
  # Extragem titlul știrii
  headline = article["title"]

  # Creeăm un obiect TextBlob pe baza titlului; TextBlob oferă metode de analiză de text
  analysis = TextBlob(headline)

  # Calculăm scorul de polaritate al titlului:
    # - Valoare între -1.0 și 1.0
    # - <0 → sentiment negativ
    # - =0 → sentiment neutru
    # - >0 → sentiment pozitiv
  sentiment = analysis.sentiment.polarity

  # Afișăm titlul și scorul de sentiment în consolă
  print(f"Headline: {headline}")
  print(f"Sentiment Score: {sentiment}\n")
