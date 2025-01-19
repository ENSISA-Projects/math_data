import numpy as np

class GradientDescent:
    """
    Implémente un algorithme générique de descente de gradient pour un problème multiclasse.
    
    Paramètres initiaux :
    - gradient (callable) : La fonction qui calcule le gradient de la fonction de coût.
    - learning_rate (float) : Le taux d'apprentissage (par défaut à 0.01).
    - max_iterations (int) : Nombre maximal d'itérations (par défaut à 1000).
    - epsilon (float) : Critère d'arrêt pour la norme du gradient (par défaut à 1e-6).
    - batch_size (int) : Taille des mini-lots pour SGD (par défaut à 1, équivalent au pur SGD).
    """
    
    def __init__(self, gradient, learning_rate=0.01, max_iterations=1000, epsilon=1e-6, batch_size=1):
        self.gradient = gradient
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.num_iterations = 0
        self.batch_size = batch_size
        self.loss_history = []  # Historique des pertes pour le traçage ou l'analyse

    def descent(self, initial_point, data=None, loss_function=None):
        """
        Effectue l'algorithme de descente de gradient pour un problème multiclasse.
        
        Paramètres :
        - initial_point (array-like) : Point initial pour l'algorithme (initialisation des poids).
        - data (tuple ou None) : Les données nécessaires pour SGD, sous la forme (X, y).
        - loss_function (callable ou None) : Fonction pour calculer la perte (ex. cross-entropy).
        
        Retourne :
        - array-like : Le point optimal trouvé par l'algorithme.
        """
        current_point = np.array(initial_point, dtype=float)
        self.loss_history = []  # Réinitialisation à chaque appel
        stopped = False  # Flag pour indiquer un arrêt anticipé

        for epoch in range(self.max_iterations):
            if data is not None:
                # Si des données sont fournies, on exécute un SGD
                X, y = data
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                
                for start in range(0, len(X), self.batch_size):
                    batch_indices = indices[start:start + self.batch_size]
                    X_batch, y_batch = X[batch_indices], y[batch_indices]
                    current_gradient = self.gradient(current_point, (X_batch, y_batch))
                    
                    if np.linalg.norm(current_gradient) < self.epsilon:
                        stopped = True
                        break
                    
                    current_point = self.update(current_point, current_gradient)
            else:
                # Descente classique (pas de données nécessaires)
                current_gradient = self.gradient(current_point)
                
                if np.linalg.norm(current_gradient) < self.epsilon:
                    stopped = True
                    break
                
                current_point = self.update(current_point, current_gradient)

            # Calcul de la perte si une fonction de perte est fournie
            if loss_function is not None and data is not None:
                X, y = data
                loss = loss_function(current_point, X, y)
                self.loss_history.append(loss)

            if stopped:
                break

        self.num_iterations = epoch + 1
        return current_point

    def update(self, point, gradient_value):
        """
        Met à jour un point en utilisant le gradient et le taux d'apprentissage.
        
        Paramètres :
        - point (array-like) : Point actuel.
        - gradient_value (array-like) : Gradient de la fonction de coût au point actuel.
        
        Retourne :
        - array-like : Nouveau point mis à jour.
        """
        return point - self.learning_rate * gradient_value
