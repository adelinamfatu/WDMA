import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# 1: Creăm un DataFrame de exemplu cu distribuție dezechilibrată
np.random.seed(42)
data = {
    "TransactionID": range(1, 21),
    "Amount": np.random.randint(10, 1000, 20), # sume aleatorii între 10 și 999
    "IsFraud": [0] * 17 + [1] * 3 # 17 non-fraud (0), 3 fraud (1) → dezechilibru 17:3
}

df = pd.DataFrame(data)

# 2: Separăm caracteristicile (X) și ținta (y)
X = df[["Amount"]] # X este DataFrame cu coloana "Amount"
y = df["IsFraud"] # y este Series cu valorile binare 0/1

# Afișăm distribuția inițială a claselor
print("Original Class Distribution:", Counter(y))

# 3: Undersampling (reducerea numărului de exemple din clasa majoritară)
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_under, y_under = undersampler.fit_resample(X, y)

print("Class Distribution After Undersampling:", Counter(y_under))

# 4: Oversampling cu SMOTE (Synthetic Minority Over-sampling Technique)
# SMOTE(sampling_strategy, k_neighbors): creează exemple sintetice ale clasei minoritare
# sampling_strategy=0.8 → (număr_fraud nou)/(număr_nonfraud inițial) = 0.8 → 3 fraud → creează 3*0.8 = 2.4 → rotunjit 2 exemple noi pentru fraud
# k_neighbors=1 → folosește doar 1 vecin în timpul generării pentru a evita erorile când sunt puține exemples
smote = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=1)  # Reduce k_neighbors
X_smote, y_smote = smote.fit_resample(X, y)

print("Class Distribution After Oversampling (SMOTE):", Counter(y_smote))
