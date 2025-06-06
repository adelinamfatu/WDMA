import torch
import torch.nn as nn
import torch.optim as optim

# Definim clasa rețelei neuronale simplă
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Strat liniar (fully connected) de la 2 caracteristici de intrare la 4 neuroni
        self.fc1 = nn.Linear(2, 4)  
        # Funcție de activare ReLU
        self.relu = nn.ReLU()
        # Strat liniar de la cele 4 neuroni la 1 neuron de ieșire
        self.fc2 = nn.Linear(4, 1)  
        # Funcție de activare Sigmoid (pentru ieșire între 0 și 1, clasificare binară)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Pasăm datele prin primul strat liniar și apoi prin ReLU
        x = self.relu(self.fc1(x))
        # Pasăm rezultatul prin al doilea strat liniar și apoi prin Sigmoid
        x = self.sigmoid(self.fc2(x))
        return x
        # Returnăm ieșirea rețelei (proba între 0 și 1)

# Instanțiem modelul
model = SimpleNN()

# Definim funcția de pierdere (Binary Cross-Entropy) pentru clasificare binară
criterion = nn.BCELoss()

# Instanțiem optimizer-ul Adam, care va actualiza parametrii rețelei
optimizer = optim.Adam(model.parameters(), lr=0.01)
# - model.parameters(): listează toți parametrii (greutățile și bias-urile) care vor fi antrenați
# - lr=0.01: rata de învățare

# Definim tensori de intrare și țintă
X = torch.tensor(
    [
        [0.1, 0.2],
        [0.4, 0.3],
        [0.6, 0.8],
        [0.9, 0.5]
    ], 
    dtype=torch.float32
)
#   - X.shape = (4, 2): 4 mostre, fiecare cu 2 caracteristici

y = torch.tensor(
    [
        [1],
        [0],
        [1],
        [0]
    ], 
    dtype=torch.float32
)
# - y.shape = (4, 1): etichete binare (1 sau 0) corespunzătoare fiecărui rând din X

# Bucla de antrenare (10 epoci)
for epoch in range(10):
    # Resetăm gradientele la zero înainte de backpropagation
    optimizer.zero_grad()
    
    # Propagare înainte: calculăm predicțiile rețelei pentru toate mostrele din X
    outputs = model(X)
    
    # Calculăm pierderea între predicții și etichete
    loss = criterion(outputs, y)
    
    # Propagare înapoi: calculăm gradientele funcției de pierdere față de parametrii rețelei
    loss.backward()
    
    # Actualizăm parametrii rețelei pe baza gradientelor (Adam update)
    optimizer.step()
    
    # Afișăm valoarea pierderii la fiecare epocă
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
