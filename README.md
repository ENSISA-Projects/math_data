# Projet Mathématiques pour les data sciences : Comparaison des Approches de Régression Logistique

## Description
Ce projet compare deux approches pour implémenter la régression logistique multiclasses appliquée au jeu de données "digits" de Scikit-learn :

1. **Descente de gradient** : implémentation d'une méthode de descente de gradient pour optimiser les poids d'un modèle de régression logistique en utilisant la fonction de coût "cross-entropy".
2. **LogisticRegression de Scikit-learn** : solution standard utilisant le module `LogisticRegression` pour effectuer une régression logistique multiclasses.

Le projet met en évidence les précisions obtenues et le nombre d'itérations nécessaires pour chaque approche.

---

## Prérequis

Assurez-vous que les bibliothèques suivantes sont installées :
- `numpy`
- `scikit-learn`
- `matplotlib`
- `pandas`

Pour installer les dépendances manquantes, utilisez la commande suivante :

```bash
pip install -r requirements.txt
```

---

## Fonctionnalités principales

1. **Chargement et prétraitement des données** :
   - Chargement du jeu de données "digits".
   - Normalisation des données avec `StandardScaler`.
   - Division en ensembles d'entraînement et de test.

2. **Descente de gradient** :
   - Calcul des probabilités d'appartenance à chaque classe via la fonction softmax.
   - Implémentation de la fonction de coût (cross-entropy) et de son gradient.
   - Optimisation des poids via une classe `GradientDescent`.

3. **Modèle Scikit-learn** :
   - Entraînement et évaluation d'un modèle de régression logistique utilisant Scikit-learn.

4. **Comparaison des résultats** :
   - Comparaison des précisions obtenues par les deux approches.
   - Comparaison du nombre d'itérations pour chaque méthode.
   - Affichage des résultats dans un tableau.

---

## Utilisation

### Structure du code

1. **Chargement et prétraitement des données** :
   Le jeu de données est chargé depuis `sklearn.datasets.load_digits`, normalisé et divisé en ensembles d'entraînement et de test.

2. **Descente de gradient** :
   La classe `GradientDescent` optimise les poids du modèle à l'aide de la fonction de coût "cross-entropy" et de son gradient.

3. **Modèle Scikit-learn** :
   Le modèle `LogisticRegression` est entraîné pour comparer les performances.

4. **Affichage des résultats** :
   Les précisions et les nombres d'itérations des deux approches sont présentés dans un tableau et un graphique.

### Exécution

Exécutez le script principal pour voir les résultats :

```bash
python main.py
```

### Exemple de sortie (tableau des résultats)

| Méthode                          | Précision (%) | Nombre d'itérations |
|----------------------------------|---------------|----------------------|
| Descente de gradient | 95.93         | 30                   |
| LogisticRegression (Scikit-learn) | 96.30         | 19                  |

---

## Améliorations possibles

1. Ajouter des méthodes de régularisation (L1, L2) à la descente de gradient.
2. Implémenter d'autres solveurs pour la régression logistique.
3. Tester le modèle sur d'autres jeux de données multiclasses.

---

## Auteurs

Ce projet a été développé pour comparer une approche de descente de gradient avec une implémentation standard en utilisant Scikit-learn.
Quentin GIRARDAT, Romain BOMBA

---

## Licence

Ce projet est sous licence MIT. Vous êtes libre de l'utiliser, de le modifier et de le distribuer avec attribution.

