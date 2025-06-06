import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# 1. Creăm un dataset sintetic cu câteva case
# - 'location': orașul (categoric)
# - 'sqft': suprafața casei
# - 'bedrooms': numărul de dormitoare
# - 'price': prețul casei (ținta)
data = {
    'location': ['cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB'],
    'sqft':     [1000,    2000,    1200,    1800,   900,     2200   ],
    'bedrooms': [2,       4,       3,       4,      2,       5      ],
    'price':    [200000,  400000,  260000,  350000, 180000,   450000]
}
df = pd.DataFrame(data)
# - df arată astfel:
#    location  sqft  bedrooms   price
# 0    cityA  1000         2  200000
# 1    cityB  2000         4  400000
# 2    cityA  1200         3  260000
# 3    cityB  1800         4  350000
# 4    cityA   900         2  180000
# 5    cityB  2200         5  450000

# 2. Separăm caracteristicile (X) de țintă (y)
X = df[['location', 'sqft', 'bedrooms']]
y = df['price']

# 3. Codificăm coloana categorică 'location' în variabile dummy (one-hot encoding)
# - drop_first=True elimină prima categorie ('cityA') pentru a evita multicolinearitatea
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)
# - După codificare, X_encoded va avea coloanele: ['sqft', 'bedrooms', 'location_cityB']
# - 'location_cityB' = 1 dacă locația este cityB, 0 în caz contrar (cityA)

# 4. (Opțional) Scalăm caracteristicile numerice pentru a îmbunătăți calculul distanțelor
# - KNN folosește distanțe euclidiene, așa că scale diferite pot domina măsurile
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
# - fit_transform calculează media și deviația standard pe X_encoded și transformă fiecare coloană
# - Rezultatul X_scaled este un array NumPy cu aceleași dimensiuni ca X_encoded

# 5. Împărțim datele în set de antrenament (70%) și test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 6. Creăm și antrenăm regressorul KNN
# - n_neighbors=3 înseamnă că vom lua media prețurilor celor 3 cele mai apropiate case
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)
# - .fit() memorează punctele de antrenament; nu există „model” parametric, ci doar datele în formă scalată

# 7. Evaluăm pe setul de test
y_pred_test = knn_reg.predict(X_test)
print("Test Set Predictions:", y_pred_test)
print("True Values:", y_test.values)
# - .predict() calculează pentru fiecare exemplu din X_test media prețurilor celor 3 vecini cei mai apropiați

# 8. Prezicem prețul unei case noi
# - Exemplu: casă cu 1500 sqft, 3 dormitoare, în 'cityB'
new_house = pd.DataFrame({
    'sqft': [1500],
    'bedrooms': [3],
    'location_cityB': [1] # 1 dacă e cityB (cityA e reprezentat de 0 implicit)
})

# Trebuie să scalăm noul exemplu cu același scaler folosit la antrenament
new_house_scaled = scaler.transform(new_house)
# - scaler.transform folosește media și deviația standard calculate anterior
# - Rezultatul e un array 2D, compatibil cu X_train/X_test

predicted_price = knn_reg.predict(new_house_scaled)
print("Predicted price for the new house:", predicted_price[0])
# - Modelul va găsi cei trei vecini cei mai apropiați în spațiul caracteristicilor scalate
# - Returnează media prețurilor acelorași trei vecini
