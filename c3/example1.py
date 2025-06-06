import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Creăm un dataset simplu (toy dataset) cu date despre case:
# - 'sqft': suprafața casei în picioare pătrate
# - 'bedrooms': numărul de dormitoare
# - 'location': un cod de locație (categoric: 'cityA' sau 'cityB')
# - 'price': prețul casei
data = {
    'sqft':      [1500, 2000, 1100, 2500, 1400, 2300],
    'bedrooms':  [3,    4,    2,    5,    3,    4],
    'location':  ['cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB'],
    'price':     [300000, 400000, 200000, 500000, 280000, 450000]
}
df = pd.DataFrame(data)
# - df arată astfel:
#   sqft  bedrooms location  price
# 0 1500        3    cityA  300000
# 1 2000        4    cityB  400000
# 2 1100        2    cityA  200000
# 3 2500        5    cityB  500000
# 4 1400        3    cityA  280000
# 5 2300        4    cityB  450000

# 2. Separăm caracteristicile (X) de țintă (y):
X = df[['sqft', 'bedrooms', 'location']]
y = df['price']
# - X conține coloanele numerice și categorice
# - y conține prețurile de antrenament

# 3. Convertim coloana 'location' (categorică) în variabile dummy (one-hot encoding)
# - drop_first=True elimină prima coloană dummy pentru a evita multicolinearitatea
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)
# - Rezultat:
#  sqft  bedrooms  location_cityB
# 0 1500         3               0
# 1 2000         4               1
# 2 1100         2               0
# 3 2500         5               1
# 4 1400         3               0
# 5 2300         4               1
# - 'location_cityB' = 1 dacă locația e 'cityB'; 0 în caz contrar (oraș implicit 'cityA')

# 4. Antrenăm un model de regresie liniară
model = LinearRegression()
model.fit(X_encoded, y)
# - model.coeef_ conține coeficienții asociați fiecărei coloane din X_encoded
# - model.intercept_ este interceptul (b) al ecuației liniare

# 5. Afișăm coeficienții și interceptul
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
# - De exemplu, coeficienții pot arăta asemănător cu [ 100,000, 20,000, 50,000 ],
# ceea ce înseamnă:
# * pentru fiecare 1 sqft adițional, prețul crește cu ~100.000 (dacă unitățile ar fi scalate);
# * pentru fiecare dormitor în plus, +20.000; 
# * dacă 'location_cityB'=1, adaugă ~50.000 față de 'cityA'.

# 6. Facem o predicție pentru o casă nouă:
# - 1600 sqft, 3 dormitoare, locație 'cityB'
new_house = pd.DataFrame({
    'sqft': [1600],
    'bedrooms': [3],
    'location': ['cityB']
})

# Transformăm noul exemplu la același format ca datele de antrenament
new_house_encoded = pd.get_dummies(new_house, columns=['location'], drop_first=True)
# - Aici apare doar 'location_cityB', pentru că 'drop_first=True'
# - new_house_encoded arată astfel:
#  sqft  bedrooms  location_cityB
# 0 1600         3               1

# Asigurăm că noul DataFrame are aceleași coloane exact în aceeași ordine ca X_encoded
# (dacă în antrenament ar fi existat și 'location_cityA', acum s-ar completa cu 0)
new_house_encoded = new_house_encoded.reindex(columns=X_encoded.columns, fill_value=0)
# - X_encoded.columns este ['sqft', 'bedrooms', 'location_cityB']
# - reindex se ocupă să adauge coloane lipsă cu valoarea 0, dacă e cazul

# Realizăm predicția
predicted_price = model.predict(new_house_encoded)
print("Predicted price for the new house:", predicted_price[0])
# - Afișează prețul estimat de model pentru casa cu specificațiile date
