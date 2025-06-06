import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Creăm un dataset mic cu caracteristici demografice și eticheta "purchased"
# Fiecare tuplu reprezintă: (age, gender, income, purchased)
data = [
    (25, "Male",   50000, 0),
    (40, "Female", 70000, 1),
    (35, "Female", 60000, 0),
    (50, "Male",   80000, 1),
    (28, "Male",   52000, 0),
    (60, "Female",100000, 1),
    (45, "Male",   75000, 1),
    (22, "Female", 48000, 0),
    (39, "Female", 68000, 1)
]
df = pd.DataFrame(data, columns=["age", "gender", "income", "purchased"])
# - "age": vârsta clientului
# - "gender": sexul clientului ("Male"/"Female")
# - "income": venitul anual al clientului
# - "purchased": 1 dacă clientul a cumpărat, 0 dacă nu

# 2. Codificăm variabila categorică "gender" folosind one-hot encoding
df_encoded = pd.get_dummies(df, columns=["gender"], drop_first=True)
# - drop_first=True elimină prima coloană dummy ("gender_Female") și păstrează doar "gender_Male"
# - Acum "gender_Male" = 1 dacă clientul e bărbat, 0 dacă femeie

# 3. Separăm caracteristicile (X) și eticheta țintă (y)
X = df_encoded[["age", "income", "gender_Male"]]
y = df_encoded["purchased"]
# - X conține coloanele numerice și coloana dummy "gender_Male"
# - y conține eticheta 0/1

# 4. Împărțim datele în set de antrenament (80%) și test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% merg în setul de test
    random_state=42 # seed pentru reproducibilitate
)

# 5. Antrenăm un clasificator SVM cu kernel liniar
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)
# - fit învață modelul SVM pe datele de antrenament

# 6. Facem predicții pe setul de test
y_pred = model.predict(X_test)
# - predict returnează 0 sau 1 pentru fiecare exemplu din X_test

# 7. Evaluăm performanța folosind acuratețea simplă
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
# - accuracy_score compară predicțiile cu valorile reale și calculează proporția de corecte

# (Opțional) Afișăm coeficienții (greutățile) fiecărei caracteristici în SVM-ul liniar
coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]
})
print("\nCoefficients (Linear SVM):")
print(coefficients)
# - coef_ arată cât de importantă e fiecare caracteristică pentru decizia modelului
# - semnul indică direcția: coef pozitiv înseamnă că valoarea mare a caracteristicii crește probabilitatea de 1

# (Opțional) Afișăm și interceptul (b)
print("\nIntercept (bias):", model.intercept_[0])

# (Opțional) Afișăm DataFrame-ul cu coloanele codificate pentru verificare
print("\nEncoded Dataset:")
print(df_encoded)
