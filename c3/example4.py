import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# 1. Creăm un dataset simplu cu caracteristicile unei case și prețul acesteia
data = {
    'location': ['cityA', 'cityA', 'cityB', 'cityB', 'cityA', 'cityB'],
    'rooms':    [2,      3,      2,      4,      3,      5     ],
    'sqft':     [800,   1200,    900,   1800,   1100,   2200   ],
    'price':    [100000,180000,160000,290000,200000,360000]
}
df = pd.DataFrame(data)
# - 'location' este categorică (cityA sau cityB)
# - 'rooms' și 'sqft' sunt caracteristici numerice
# - 'price' e ținta (valoarea casei în dolari)

# 2. Separăm caracteristicile (X) de țintă (y)
X = df[['location', 'rooms', 'sqft']]
y = df['price']

# 3. Codificăm coloana categorică 'location' în variabile dummy
# - drop_first=True: păstrăm doar 'location_cityB'; dacă e 0 înseamnă cityA
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)
# - X_encoded arată astfel:
#     rooms  sqft  location_cityB
# 0     2    800               0
# 1     3   1200               0
# 2     2    900               1
# 3     4   1800               1
# 4     3   1100               0
# 5     5   2200               1

# 4. Creăm și antrenăm un DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_encoded, y)
# - Modelul va învăța reguli de tip:
# * „dacă location_cityB = 1, merge în ramura B; altfel merge în ramura A”
# * apoi va verifica numărul de camere pentru a doua despărțire
# * apoi va folosi sqft pentru ultima despărțire, până când frunzele conțin prețul mediu al subsetului

# 5. Prezicem prețul unei case noi:
# - ex: casă în 'cityB', 4 camere, 2000 sqft
new_house = pd.DataFrame({
    'rooms': [4],
    'sqft':  [2000],
    'location_cityB': [1] # 1 dacă e cityB, 0 dacă ar fi fost cityA
})
predicted_price = tree_reg.predict(new_house)
print("Predicted price for new house:", predicted_price[0])
# - Modelul va parcurge arborele:
# 1) Verifică location_cityB (1 → cityB), deci se duce spre ramura cityB
# 2) Compară rooms (4) cu pragul în nodul cityB
# 3) Compară sqft (2000) cu pragul corespunzător
# 4) Ajunge într-o frunză și returnează prețul mediu al antrenamentului din acel nod
