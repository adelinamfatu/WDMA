import requests
from bs4 import BeautifulSoup
import pandas as pd

# 1: Definim URL-ul paginii de unde vom prelua lista de produse
URL = "https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops"

# 2: Efectuăm o cerere HTTP GET către URL
response = requests.get(URL)
# - response.status_code conține codul HTTP (200 = OK, 404 = Not Found etc.)
# - response.text conține întregul HTML al paginii ca string

# 3: Parsăm textul HTML cu BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")
# - "html.parser" este parser-ul încorporat în Python
# - Obiectul 'soup' devine rădăcina arborelui DOM, pe care putem face find/find_all

# 4: Extragem numele și prețurile produselor
products = [] # Listă în care vom adăuga câte [nume_produs, preț_produs] pentru fiecare item
for product in soup.find_all("div", class_="thumbnail"): 
    # find_all("div", class_="thumbnail") returnează o listă cu toate <div class="thumbnail">
    # fiecărui element i se spune 'product'
    
    # 4a: Extragem numele produsului
    # În interiorul fiecărui <div class="thumbnail"> există un <a class="title">
    name = product.find("a", class_="title").text.strip()
    # - product.find("a", class_="title") găsește primul tag <a> cu class="title"
    # - .text preia textul interior al acelui tag (numele produsului)
    # - .strip() elimină spațiile / newline-urile de la început și sfârșit

    # 4b: Extragem prețul produsului
    # În interior există și un tag <h4 class="price"> cu prețul
    price = product.find("h4", class_="price").text.strip()
    # - .text preia textul (de ex. "$1,200.00"), .strip() curăță spații

    # 4c: Adăugăm într-o listă sub-lista [nume, preț]
    products.append([name, price])

# 5: Convertim lista 'products' într-un DataFrame pandas
df = pd.DataFrame(products, columns=["Product Name", "Price"])
# - Rezultă un tabel cu două coloane: "Product Name" și "Price"
# - Fiecare rând corespunde unui produs extras

# 6: Eliminăm eventualele duplicate din DataFrame
df_cleaned = df.drop_duplicates()
# - drop_duplicates() verifică toate coloanele; dacă două rânduri sunt identice, păstrează doar primul

# 7: Afișăm primele 10 produse din DataFrame-ul curățat
print("Scraped Product Listings:\n")
print(df_cleaned.head(10))  # Show the first 10 products
# 8: Salvăm DataFrame-ul curățat într-un fișier CSV

df_cleaned.to_csv("scraped_products.csv", index=False)
