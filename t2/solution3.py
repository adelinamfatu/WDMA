import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Încărcăm setul de date Wine (conține caracteristici și etichete pentru 3 clase de vin)
wine = load_wine()
X = wine.data # Matricea de caracteristici (13 coloane: aciditate, alcool etc.)
y = wine.target # Vectorul țintă (valori 0, 1 sau 2 pentru cele 3 clase)

# (Opțional) Conversia într-un DataFrame pentru explorare:
# df = pd.DataFrame(X, columns=wine.feature_names)
# df['target'] = y
# print(df.head())  # Inspectăm primele rânduri

# 2. Împărțim datele în seturi de antrenament (80%) și test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% din date merg la test
    random_state=42 # seed fix pentru reproducibilitate
)

# 3. Antrenăm un Decision Tree cu adâncime maximă 3 (limităm pentru a preveni overfitting)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
# - fit() construiește arborele prin împărțirea nodurilor până la nivelul 3 sau până la omogenitate.

# 4. Verificăm acuratețea pe setul de test
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
# - accuracy_score calculează proporția de observații pe care modelul le-a prezis corect.

# (Opțional) Vizualizăm structura arborelui
plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=wine.feature_names, # Numele caracteristicilor afișate în noduri
    class_names=wine.target_names, # Numele claselor de vin
    filled=True # Colorează nodurile după clasă
)
plt.show()
#   - plot_tree afișează arborele: fiecare nod arată condiția de split și proporția fiecărei clase

# (Opțional) Afișăm importanța fiecărei caracteristici în decizia arborelui
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': wine.feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importance_df)
# - feature_importances_ indică cât de mult contribuie fiecare caracteristică la reducerea impurității
