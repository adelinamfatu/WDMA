import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 1: Încărcăm setul de date "titanic" direct din seaborn
df = sns.load_dataset("titanic")
# - Acesta este un DataFrame pandas ce conține informații despre pasagerii Titanicului

# 2: Cream o nouă caracteristică "family_size"
# care reprezintă mărimea familiei fiecărui pasager (soț/soție + părinți/copii + el însuși)
df["family_size"] = df["sibsp"] + df["parch"] + 1
# - sibsp = numărul de frați/soți la bord
# - parch = numărul de părinți/copii la bord
# - adăugăm 1 pentru a include pasagerul însuși în calcul

# 3: Tratăm valorile lipsă pentru coloanele "age" și "embarked"
# 3a: Înlocuim valorile NaN din "age" cu mediana vârstelor existente
df["age"].fillna(df["age"].median(), inplace=True)
# - df["age"].median() calculează mediana vârstelor
# - inplace=True înseamnă că modificăm direct coloana originală

# 3b: Înlocuim valorile NaN din "embarked" cu modul (cea mai frecventă valoare)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
# - df["embarked"].mode()[0] returnează prima valoare de modă (cel mai comun port de îmbarcare)

# 4: Codificăm variabilele categorice "sex" și "embarked" prin one-hot encoding
df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)
# - Înlocuim coloana "sex" cu două coloane: "sex_male" (1 dacă bărbat, 0 dacă femeie)
# (drop_first=True elimină una dintre coloanele dummy pentru a evita colinearitatea)
# - Înlocuim coloana "embarked" cu două coloane: "embarked_Q", "embarked_S"
# (orașele de îmbarcare: C, Q, S; cu drop_first=True se păstrează doar Q și S)

# 5: Selectăm caracteristicile numerice pentru scalare: "age", "fare", "family_size"
scaler = MinMaxScaler()
df[["age", "fare", "family_size"]] = scaler.fit_transform(df[["age", "fare", "family_size"]])
# - MinMaxScaler scalează valorile fiecărei coloane într-un interval [0, 1]
# - fit_transform() învață min și max pe datele din DataFrame și apoi aplică transformarea

# 6: Afișăm primele 5 rânduri din DataFrame-ul procesat, doar coloanele cheie
print("Processed Titanic Dataset (First 5 Rows):\n")
print(df[["age", "fare", "family_size", "sex_male", "embarked_Q", "embarked_S"]].head())
# - "age", "fare", "family_size" (toate scalate între 0 și 1)
# - "sex_male" (1 dacă pasagerul e bărbat, 0 dacă femeie)
# - "embarked_Q", "embarked_S" (dummy variables pentru portul de îmbarcare)
