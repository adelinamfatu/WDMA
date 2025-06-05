import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1: Creăm un DataFrame simplu care imită un set de date cu prețuri de locuințe
data = {
    "HouseID": [1, 2, 3, 4, 5],
    "SquareFeet": [1500, 1800, 1200, 2000, 1600],
    "NumRooms": [3, 4, 2, 5, 3],
    "Price": [250000, 300000, 200000, 350000, 280000]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# 2: Creăm instanțele scalatoarelor
# - MinMaxScaler: scalează valorile între 0 și 1, pe baza min și max ale fiecărei caracteristici
minmax_scaler = MinMaxScaler()
# - StandardScaler: scalează valorile astfel încât media = 0 și deviația standard = 1
standard_scaler = StandardScaler()

# 3: Aplicăm Min-Max Scaling pe coloanele numerice
df_minmax_scaled = df.copy()
#  Copiem DataFrame-ul original pentru a nu suprascrie valorile inițiale
#  Apoi selectăm coloanele numerice și aplicăm fit_transform:
df_minmax_scaled[["SquareFeet", "NumRooms", "Price"]] = minmax_scaler.fit_transform(df[["SquareFeet", "NumRooms", "Price"]])
#   HouseID  SquareFeet  NumRooms     Price
# 0       1    0.230769  0.250000  0.333333
# 1       2    0.615385  0.500000  0.666667
# 2       3    0.000000  0.000000  0.000000
# 3       4    1.000000  1.000000  1.000000
# 4       5    0.384615  0.250000  0.466667
#  - SquareFeet: (val - min)/(max - min) → 1500→0.2308, 1800→0.6154, etc.
#  - NumRooms:   (val - min)/(max - min) → 3→0.25, 4→0.5, etc.
#  - Price:      (val - min)/(max - min) → 250k→0.3333, etc.

print("\nDataset After Min-Max Scaling:\n")
print(df_minmax_scaled)

# 4: Aplicăm Standard Scaling pe aceleași coloane numerice
df_standard_scaled = df.copy()
#  Copiem din nou pentru a păstra DataFrame-ul original intact
df_standard_scaled[["SquareFeet", "NumRooms", "Price"]] = standard_scaler.fit_transform(df[["SquareFeet", "NumRooms", "Price"]])

print("\nDataset After Standard Scaling:\n")
print(df_standard_scaled)
