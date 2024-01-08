import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Utilizar un conjunto de datos más grande (MovieLens 100K o MovieLens 1M)
# Cargar el conjunto de datos adecuado aquí

# Crear tensores de PyTorch a partir de los datos
ratings = torch.FloatTensor(ratings)

# Utilizar otras funciones de pérdida (entropía cruzada o Kullback-Leibler)
criterion = nn.CrossEntropyLoss()

# Utilizar otros optimizadores (Adam en lugar de SGD)
optimizer = optim.Adam(model.parameters(), lr=lr)  # Cambio al optimizador Adam

# Crear el modelo (utilizar otras técnicas de recomendación avanzadas si es necesario)
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        return torch.sum(user_embedding * item_embedding, dim=1)

# Hiperparámetros
num_users, num_items = ratings.shape
embedding_dim = 2
lr = 0.01
epochs = 1000

# Crear el modelo
model = MatrixFactorization(num_users, num_items, embedding_dim)

# Entrenamiento del modelo
for epoch in range(epochs):
    optimizer.zero_grad()
    user_ids, item_ids = torch.nonzero(ratings)
    predictions = model(user_ids, item_ids)
    loss = criterion(predictions, ratings[user_ids, item_ids])
    loss.backward()
    optimizer.step()

# Hacer recomendaciones para un usuario específico (usuario 0)
user_id = 0
user_embedding = model.user_embeddings(torch.LongTensor([user_id]))
item_embeddings = model.item_embeddings.weight
scores = torch.mm(user_embedding, item_embeddings.t())
recommended_item_id = scores.argmax().item()

print(f"Usuario {user_id} podría estar interesado en la película {recommended_item_id}")

# Explorar técnicas de recomendación avanzadas (por implementar)
# Puedes explorar técnicas adicionales aquí para mejorar la personalización y precisión de las recomendaciones
