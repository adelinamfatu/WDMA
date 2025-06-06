import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Încărcăm setul de date Iris din scikit-learn
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#  - df conține 150 de rânduri și 4 coloane: sepal length, sepal width, petal length, petal width

# 2. Introducem valori lipsă artificiale (opțional, pentru demonstrație)
# - Setăm valorile de la rândurile 5–9 la NaN în coloana 'petal length (cm)'
df.iloc[5:10, 2] = np.nan

# 3. Tratăm valorile lipsă
# - Folosim SimpleImputer pentru a înlocui NaN cu media coloanei respective
imputer = SimpleImputer(strategy='mean')
# - strategy='mean' → înlocuiește cu media coloanei
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
# - fit_transform învață media fiecărei coloane și apoi înlocuiește NaN cu aceste valori

# 4. Scalăm datele
# - StandardScaler transformă fiecare caracteristică astfel încât să aibă media=0 și deviația standard=1
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_imputed),
    columns=df_imputed.columns
)
# - fit_transform calculează media și deviația standard pentru fiecare coloană pe df_imputed,
# apoi transformă valorile inițiale în scoruri standardizate

# 5. Verificăm rezultatele
# - Putem verifica statisticile descriptive înainte și după scalare
print("După imputation (statistici descriptive):")
print(df_imputed.describe())

print("\nDupă scalare (statistici descriptive):")
print(df_scaled.describe())

# 6. (Opțional) Afișăm primele rânduri pentru a confirma preprocesarea
print("\nPrimele 5 rânduri după scalare:")
print(df_scaled.head())