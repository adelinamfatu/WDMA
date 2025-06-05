import requests
from bs4 import BeautifulSoup

# URL-ul paginii de unde vom extrage informații despre produse și prețuri
URL = "https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops"

# 1. Efectuăm o cerere GET pentru a obține conținutul paginii
response = requests.get(URL)
# - response.status_code ne-ar arăta codul HTTP (200 înseamnă OK)
# - response.text conține HTML-ul paginii

# 2. Parsăm conținutul HTML cu BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")
# - "html.parser" este parser-ul încorporat în Python.
# - Obiectul 'soup' ne permite să navigăm și să căutăm elemente în arborele DOM.

# 3. Găsim toate "thumbnail"-urile de produs (fiecare produs stă în div.thumbnail)
products = soup.find_all("div", class_="thumbnail")
# - find_all returnează o listă de obiecte BeautifulSoup care corespund tag-ului <div> cu class="thumbnail"

print("Scraped Product Prices:\n")

# 4. Iterăm prin primele 10 produse (pentru exemplu)
for product in products[:10]:  # Limit to first 10 products
    # 4a. Extragem numele produsului
    # În interiorul fiecărui <div class="thumbnail"> există un <a class="title"> cu titlul produsului
    name = product.find("a", class_="title").text.strip()
    # - find(...).text preia textul brut din interiorul tag-ului <a>
    # - .strip() elimină spațiile libere de la început și sfârșit

    # 4b. Extragem prețul produsului
    # În interiorul aceluiași div, prețul este într-un <h4 class="price">
    price = product.find("h4", class_="price").text.strip()
    # - find(...).text preia textul brut, inclusiv simbolul valutei
    # - .strip() elimină spațiile inutile    

    print(f"Product: {name}")
    print(f"Price: {price}\n")
