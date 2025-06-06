import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 1. Generăm date sintetice despre clienți
# - purchase_frequency: de câte ori cumpără pe lună (1–19)
# - avg_spent: cheltuială medie per achiziție (10–299)
# - loyalty_score: scor de loialitate (1–9)
np.random.seed(42)  # pentru reproducibilitate
num_customers = 20
customer_data = pd.DataFrame({
    'purchase_frequency': np.random.randint(1, 20, num_customers),
    'avg_spent':           np.random.randint(10, 300, num_customers),
    'loyalty_score':       np.random.randint(1, 10, num_customers)
})
# - customer_data este un DataFrame cu 20 rânduri și 3 coloane reprezentând clienți

# 2. Facem segmentare cu K-Means (3 clustere)
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['segment'] = kmeans.fit_predict(customer_data)
# - fit_predict antrenează modelul și returnează eticheta de segment (0, 1 sau 2) pentru fiecare client
# - Adăugăm o nouă coloană 'segment' în customer_data

# 3. Definim mesaje de marketing diferite pentru fiecare segment
marketing_strategies = {
    0: {
        'subject': "Welcome to the Community!",
        'offer':   "10% discount on your next purchase"
    },
    1: {
        'subject': "VIP Customer Appreciation",
        'offer':   "Exclusive access to new products"
    },
    2: {
        'subject': "Loyalty Program Boost",
        'offer':   "Double loyalty points for a limited time"
    },
}
# - Dicționar cu subiect și ofertă pentru fiecare segment (0, 1, 2)

# 4. Atribuim mesajele personalizate în funcție de segment
customer_data['email_subject'] = customer_data['segment'].apply(
    lambda seg: marketing_strategies[seg]['subject']
)
# - Pentru fiecare rând, luăm segmentul și preluăm 'subject' corespunzător

customer_data['special_offer'] = customer_data['segment'].apply(
    lambda seg: marketing_strategies[seg]['offer']
)
# - Similar, atribuim oferta specială pe baza segmentului

# 5. Afișăm primele 10 rânduri pentru a verifica strategiile personalizate
print(customer_data.head(10))