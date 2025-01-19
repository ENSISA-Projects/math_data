# Importation des bibliothèques nécessaires
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from gradient_descent import GradientDescent
from sklearn.linear_model import LogisticRegression

# --------- Chargement et prétraitement des données ---------

# Chargement des données digits
digits = load_digits()
X, y = digits.data, digits.target

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --------- Définition des fonctions nécessaires ---------

def softmax(z):
    """
    Fonction softmax pour calculer les probabilités d'appartenance à chaque classe.
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Pour stabiliser les calculs
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(theta, X, y):
    """
    Fonction de coût multiclasse (cross-entropy).
    """
    m = len(y)
    h = softmax(X @ theta.T)  # Activation softmax
    log_loss = -np.sum(np.log(h[np.arange(m), y])) / m
    return log_loss

def cross_entropy_loss_gradient(theta, X, y):
    """
    Gradient de la fonction de coût multiclasse (cross-entropy).
    """
    m = len(y)
    h = softmax(X @ theta.T)
    h[np.arange(m), y] -= 1
    gradient = (X.T @ h) / m
    return gradient.T  # On retourne un tableau de taille (n_classes, n_features)

# --------- Initialisation et entraînement ---------

# Initialisation des paramètres
n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))  # Nombre de classes
initial_theta = np.zeros((n_classes, n_features))  # Initialisation des poids pour chaque classe

# Descente de gradient
gd = GradientDescent(
    gradient=lambda theta, data: cross_entropy_loss_gradient(theta, *data),
    learning_rate=1e-3,
    max_iterations=5000,
    epsilon=1e-3
)

# Entraînement
optimal_theta = gd.descent(initial_theta, data=(X_train, y_train), loss_function=cross_entropy_loss)

# Tracé de la courbe de perte
plt.figure(figsize=(8, 6))
plt.plot(gd.loss_history, label="Cross-Entropy Loss")
plt.xlabel("Itération")
plt.ylabel("Loss")
plt.title("Évolution de la Loss au fil des itérations")
plt.legend()
plt.grid()
plt.show()

# --------- Évaluation des modèles ---------

# Prédictions et précision avec descente de gradient personnalisée
y_pred_gd = np.argmax(softmax(X_test @ optimal_theta.T), axis=1)
accuracy_gd = accuracy_score(y_test, y_pred_gd)

# Modèle de régression logistique avec Scikit-learn (multiclasse)
logreg = LogisticRegression(max_iter=10000, multi_class='ovr', solver='lbfgs')
logreg.fit(X_train, y_train)

# Prédictions et précision avec LogisticRegression
y_pred_sklearn = logreg.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

# Nombre d'itérations pour la convergence de Scikit-learn
iterations_sklearn = logreg.n_iter_[0]  # Prend la valeur de la première classe (pour régression multiclasse)

# --------- Comparaison des résultats ---------

# Création d'un tableau des résultats
results = {
    "Méthode": ["Descente de gradient", "LogisticRegression (Scikit-learn)"],
    "Précision (%)": [accuracy_gd * 100, accuracy_sklearn * 100],
    "Nombre d'itérations": [gd.num_iterations, iterations_sklearn]
}

results_df = pd.DataFrame(results)

# Affichage du tableau
print("\nComparaison des résultats :")
print(results_df)
