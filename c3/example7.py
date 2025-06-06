# Importăm librăriile necesare
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Definim seturi mici de date pentru prețurile reale și cele prezise
# - actual_prices: vector cu valorile reale ale prețurilor caselor
# - predicted_prices: vector cu valorile prezise de model
actual_prices = np.array([300000, 450000, 250000, 400000, 320000])
predicted_prices = np.array([280000, 480000, 230000, 420000, 310000])

# 2. Calculăm metricile de evaluare
# - r2_score: coeficientul de determinare R², indică cât de bine se potrivește modelul 
# (1.0 înseamnă potrivire perfectă, 0.0 înseamnă că modelul explică la fel de bine ca media, negativ înseamnă mai rău)
# - mean_squared_error: eroarea pătratică medie (MSE), media pătratelor diferențelor 
# (penalizează mai mult abaterile mari)
# - mean_absolute_error: eroarea absolută medie (MAE), media valorii absolute a diferențelor 
# (oferă o măsură directă a erorii medii, în aceleași unități ca ținta)
r2 = r2_score(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)

# 3. Afișăm rezultatele metricilor
print("R² Score:", r2) # Exemplu de calcul: R² = 1 - (SS_res / SS_tot)
print("MSE:", mse) # Exemplu: MSE = mean((actual - predicted)^2)
print("MAE:", mae) # Exemplu: MAE = mean(|actual - predicted|)
