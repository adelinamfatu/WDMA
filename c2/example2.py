import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Creăm un dataset mic de recenzii de film și etichete (1 = pozitiv, 0 = negativ)
data = [
    ("I absolutely loved this movie, it was fantastic!", 1),
    ("Horrible plot and terrible acting, wasted my time.", 0),
    ("An instant classic, superb in every aspect!", 1),
    ("I wouldn't recommend this film to anyone.", 0),
    ("It was just okay, nothing special or groundbreaking.", 0),
    ("Brilliant! I enjoyed every minute of it!", 1)
]
df = pd.DataFrame(data, columns=["text", "label"])
# - df conține două coloane:
# * "text": recenzia sub formă de șir de caractere
# * "label": 1 dacă recenzia e pozitivă, 0 dacă e negativă

# 2. Transformăm textele în caracteristici numerice folosind bag-of-words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
# - fit_transform construiește vocabularul (toate cuvintele unice) și returnează o matrice sparse
# unde fiecare rând corespunde unei recenzii, iar fiecare coloană unui cuvânt.
# Valoarea din matrice = numărul de apariții ale cuvântului respectiv.
y = df["label"]
# - y este seria etichetelor (0 sau 1)

# 3. Împărțim datele în set de antrenament (70%) și set de test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# - test_size=0.3 → 30% din exemple merg în setul de test
# - random_state=42 pentru reproducibilitate

# 4. Antrenăm un clasificator Naïve Bayes (MultinomialNB e potrivit pentru date cu contori)
model = MultinomialNB()
model.fit(X_train, y_train)
# - fit antrenează modelul pe matricea X_train și etichetele y_train

# 5. Facem predicții pe setul de test
y_pred = model.predict(X_test)
# - predict returnează un array cu etichete prezise pentru fiecare rând din X_test

# 6. Calculăm și afișăm acuratețea simplă
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.2f}")
# - (y_pred == y_test) creează un array boolean în care True = predicție corectă
# - .mean() transformă boolean în numeric (True→1, False→0) și calculează media

# Opțional: Afișăm recenzia, eticheta reală și eticheta prezisă într-un DataFrame
comparison = pd.DataFrame({
    "Review": df["text"].iloc[y_test.index], # textele corespunzătoare rândurilor din y_test
    "Actual Label": y_test, # etichetele reale
    "Predicted Label": y_pred # etichetele prezise
})
print("\nPredictions vs. Actual:")
print(comparison)
