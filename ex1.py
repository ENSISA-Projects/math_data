# Importation des bibliothèques nécessaires
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from descente_stochastique import GradientDescent
from sklearn.linear_model import LogisticRegression

# --------- Chargement et prétraitement des données ---------

# Chargement des données digits
digits = load_digits()
X, y = digits.data, digits.target

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Transformation du problème en classification binaire (chiffre 0 ou pas 0)
y_binary = (y == 0).astype(int)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# --------- Définition des fonctions nécessaires ---------

def sigmoid(z):
    """
    Fonction sigmoïde.
    """
    return 1 / (1 + np.exp(-z))

def log_loss(theta, X, y):
    """
    Fonction de coût log-loss.
    """
    m = len(y)
    h = sigmoid(X @ theta)
    return -(1/m) * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))

def log_loss_gradient(theta, X, y):
    """
    Gradient de la fonction de coût log-loss.
    """
    m = len(y)
    h = sigmoid(X @ theta)
    return (1/m) * (X.T @ (h - y))

# --------- Initialisation et entraînement ---------

# Initialisation des paramètres
n_features = X_train.shape[1]
initial_theta = np.zeros(n_features)

# Test des fonctions (facultatif : pour validation intermédiaire)
loss_initial = log_loss(initial_theta, X_train, y_train)
grad_initial = log_loss_gradient(initial_theta, X_train, y_train)

print(f"Coût initial : {loss_initial}")
print(f"Gradient initial (premières valeurs) : {grad_initial[:5]}")

# Descente de gradient personnalisée
gd = GradientDescent(
    gradient=lambda theta: log_loss_gradient(theta, X_train, y_train),
    learning_rate=0.1,
    max_iterations=10000,
    epsilon=1e-6
)

# Entraînement
optimal_theta = gd.descent(initial_theta, data=None)

# Coût final après optimisation
final_loss = log_loss(optimal_theta, X_train, y_train)
print(f"Coût final après descente de gradient : {final_loss}")

# --------- Évaluation des modèles ---------

# Prédictions et précision avec descente de gradient personnalisée
y_pred_gd = (sigmoid(X_test @ optimal_theta) >= 0.5).astype(int)
accuracy_gd = accuracy_score(y_test, y_pred_gd)

# Modèle de régression logistique avec Scikit-learn
logreg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
logreg.fit(X_train, y_train)

# Prédictions et précision avec LogisticRegression
y_pred_sklearn = logreg.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

# Affichage des résultats
print("\nRésultats :")
print(f"Point optimal :\n {optimal_theta}")
print(f"Itérations effectuées : {gd.num_iterations}")
print(f"Précision avec descente de gradient personnalisée : {accuracy_gd * 100:.2f}%")
print(f"Précision avec LogisticRegression de Scikit-learn : {accuracy_sklearn * 100:.2f}%")