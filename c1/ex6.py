import pandas as pd
import numpy as np

# 1: Creăm un DataFrame simplu care imită setul Titanic, cu valori lipsă la coloana "Age"
data = {
    "PassengerID": [1, 2, 3, 4, 5],
    "Name": ["John", "Emma", "Liam", "Sophia", "Noah"],
    "Age": [22, np.nan, 24, np.nan, 30], # Missing values in Age column
    "Survived": [1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

print("Original Dataset with Missing Age Values:\n")
print(df)

# 2: Imputare cu media vârstelor (mean imputation)
# - calculează media valorilor existente din coloana "Age"
# - înlocuiește fiecare np.nan cu acea valoare medie
df["Age_mean"] = df["Age"].fillna(df["Age"].mean())
# df["Age"].mean() → (22 + 24 + 30) / 3 = 25.3333...
# df["Age_mean"] conține vârstele originale, dar unde era NaN apare 25.3333

# 3: Imputare cu mediana vârstelor (median imputation)
# - calculează mediana valorilor existente din coloana "Age"
# - înlocuiește fiecare np.nan cu valoarea mediană
df["Age_median"] = df["Age"].fillna(df["Age"].median())
# Valorile existente ordonate: [22, 24, 30] → mediana = 24
# df["Age_median"] va avea 24 în loc de np.nan

# 4: Imputare cu modul vârstelor (mode imputation)
# - găsește valoarea cea mai frecventă (modă) din coloana "Age"
# - mode()[0] extrage prima valoare de modă (dacă sunt mai multe, ia prima)
df["Age_mode"] = df["Age"].fillna(df["Age"].mode()[0])
# În cazul ăsta, fiecare vârstă apare o singură dată, deci .mode()[0] poate fi 22 (prima)
# sau 24 sau 30 – pandas returnează mărimea cea mai frecventă; aici oricare, dar de obicei 22

print("\nDataset After Handling Missing Age Values (Mean, Median, Mode):\n")
print(df[["PassengerID", "Name", "Age", "Age_mean", "Age_median", "Age_mode"]])
