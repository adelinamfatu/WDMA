import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1: Încărcăm setul de date "titanic" direct din seaborn
df = sns.load_dataset("titanic")
# - Conține informații despre pasageri: vârstă, clasă, sex, tarife, supraviețuire etc.

# 2: Inginerie de caracteristici (prelucrată în Exercise 4)

# 2a: Cream o nouă caracteristică "family_size" care reprezintă numărul total de membri de familie la bord
df["family_size"] = df["sibsp"] + df["parch"] + 1
# - sibsp = numărul de soți/soți și frați/surori la bord
# - parch = numărul de părinți/copii la bord
# - +1 pentru a include pasagerul însuși

# 2b: Înlocuim valorile lipsă din "age" cu mediana vârstei
df["age"].fillna(df["age"].median(), inplace=True)
# - df["age"].median() calculează mediana vârstelor existente
# - inplace=True modifică direct coloana din DataFrame

# 2c: Înlocuim valorile lipsă din "embarked" (portul de îmbarcare) cu valoarea cea mai frecventă (moda)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
# - df["embarked"].mode()[0] returnează prima valoare de modă din coloana "embarked"

# 2d: Eliminăm rândurile care au NaN în coloana "fare" (tarif)
df.dropna(subset=["fare"], inplace=True)
# - dropna(subset=["fare"]) elimină doar rândurile unde "fare" este NaN

# 2e: Aplicăm one-hot encoding pe variabilele categorice "sex" și "embarked"
# - drop_first=True elimină prima coloană dummy pentru fiecare variabilă pentru a evita colinearitatea
df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)
# - După aceasta, vom avea coloanele "sex_male", "embarked_Q" și "embarked_S" (porturile: C, Q, S)

# 3: Selectăm caracteristicile (features) și scalăm numeric valorile
features = ["age", "fare", "family_size", "sex_male", "embarked_Q", "embarked_S"]
# - age, fare, family_size (toate numerice, vor fi scalate)
# - sex_male, embarked_Q, embarked_S (dummy variables, deja 0/1)

# Creăm un scaler MinMax pentru a aduce valorile numerice între 0 și 1
scaler = MinMaxScaler()
# Aplicăm scalarea doar pe coloanele numerice listate în "features"
df[features] = scaler.fit_transform(df[features])
# - fit_transform calculează min și max pentru fiecare coloană pe tot DataFrame-ul
# și apoi transformă valorile astfel încât să fie în intervalul [0, 1]

# PANA AICI ERA EX 4

# Step 4: Definim variabila țintă și împărțim datele în setul de antrenament și setul de test
X = df[features] # Caracteristicile de intrare
y = df["survived"] # Variabila țintă: 1 dacă a supraviețuit, 0 dacă nu

# Împărțim setul în proporție 80% pentru antrenament, 20% pentru test, cu seed fix
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# - test_size=0.2 înseamnă 20% din rânduri vor fi în setul de test
# - random_state=42 asigură reproducibilitatea împărțirii

# 5: Antrenăm modelul de regresie logistică
model = LogisticRegression()
model.fit(X_train, y_train)
# - fit antrenează modelul pe X_train și y_train

# 6: Realizăm predicții pe setul de test
y_pred = model.predict(X_test)
# - predict returnează eticheta prezisă (0 sau 1) pentru fiecare eșantion din X_test

# 7: Evaluăm performanța modelului
print("Model Accuracy:", accuracy_score(y_test, y_pred))
# - accuracy_score calculează proporția predicțiilor corecte (număr corecte / număr total)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
# - classification_report afișează:
# * precision (câte predicții pozitive au fost corecte)
# * recall (câte exemple pozitive reale au fost detectate)
# * f1-score (media armonică a precision și recall)
# * support (numărul de exemple reale din fiecare clasă)
