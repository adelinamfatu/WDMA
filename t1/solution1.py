import requests
import pandas as pd

# 1: Definim orașele pentru care vrem să obținem date meteo
cities = ["New York", "London", "Tokyo", "Paris", "Berlin"]

# 2: Parcurgem fiecare oraș și extragem condiția și temperatura curentă
weather_data = [] # Listă în care vom adăuga subliste [Oraș, Condiție, Temperatura]

for city in cities:
    # Construim URL-ul folosind formatul specific wttr.in:
    # ?format=%C+%t → ne aduce textul condiției (de ex. "Sunny") și + temperatura (ex. "15°C")
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        # Dacă cererea a avut succes, răspunsul e un text simplu de forma: "Condiție Temperatura°C"
        data = response.text.strip()
        # Exemplu: data = "Sunny 15°C" sau "Light rain 8°C"

        condition, temperature = data.rsplit(" ", 1)  # Extract weather condition and temperature
        # - condition = tot ce e înainte de ultimul spațiu, ex. "Sunny" sau "Light rain"
        # - temperature = ultima bucată, ex. "15°C" sau "8°C"
        
        # Eliminăm sufixul "°C" din string-ul temperaturii, păstrăm doar cifrele
        temperature = temperature.replace("°C", "")

        # Adăugăm în listă un sub-list cu: [Oraș, Condiție, Temperatura fără "°C"]
        weather_data.append([city, condition, temperature])
    else:
        print(f"Failed to retrieve data for {city}")

# 3: Cream un DataFrame pandas din lista de date obținute
df = pd.DataFrame(weather_data, columns=["City", "Weather Condition", "Temperature (°C)"])

# 4: Salvăm DataFrame-ul rezultat într-un fișier CSV, fără coloana index
df.to_csv("cleaned_weather_data.csv", index=False)

# Afișăm rezultatul final în consolă
print("Cleaned Weather Data:\n")
print(df)
