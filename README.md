# math_data

## Description

**math_data** est un projet visant à reconnaître des chiffres manuscrits à l'aide de deux approches différentes :
1. Une implémentation personnalisée de la descente de gradient.
2. L'utilisation de l'algorithme de régression logistique fourni par Scikit-learn.

Le jeu de données utilisé provient de l'archive [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), qui contient des images numériques des chiffres manuscrits.

---

## Fonctionnalités principales

- **Descente de gradient personnalisée** : Une implémentation de la descente de gradient, permettant d'entraîner un modèle de régression logistique binaire.
- **Comparaison avec Scikit-learn** : Évaluation des performances d'un modèle similaire basé sur la régression logistique intégrée de Scikit-learn.

---

## Installation et Prérequis

### Prérequis
- Python 3.7+
- Bibliothèques nécessaires :
  - `numpy`
  - `scikit-learn`

### Installation
1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/votre-repo/math_data.git
   cd math_data
   pip install -r requirements.txt
2. Utilisation :
   python ex1.py

## Sortie
Coût initial : 0.6931471805599435
Gradient initial (premières valeurs) : [ 0.          0.03586964  0.02460265 -0.03241423  0.01146664]
Coût final après descente de gradient : 0.4797149271911441

Résultats :
Point optimal :
 [ 0.         -0.0299818  -0.18499728  0.28873865 -0.20854736 -0.33127344
  0.20955025 -0.02535302  0.05000968 -0.18136344  0.09528485 -0.08454168
  0.46636106  0.20532618 -0.14887793 -0.04725453 -0.0883459   0.17269816
 -0.19425659  0.00526788 -0.26094714  0.66531083  0.01882807  0.02217531
 -0.00337092  0.41777481 -0.04221921 -0.11526247 -0.65593076 -0.46660245
  0.3005251   0.01525709  0.          0.09547375  0.19696295 -0.60239612
 -0.03595058 -0.52391096  0.54128673  0.         -0.02508164 -0.14915807
  0.63261601 -0.63001966 -0.61144796 -0.03230864  0.05421166 -0.09310198
 -0.05819206 -0.23031167  0.09958355  0.30109335  0.41393298  0.06481136
 -0.4403271   0.20263607  0.02476945  0.07067217  0.02004412 -0.00382742
 -0.10783367 -0.05120038 -0.00123005 -0.08043362]

Itérations effectuées : 9806
Précision avec descente de gradient personnalisée : 71.39%
Précision avec LogisticRegression de Scikit-learn : 99.44%

## Comparaison des Méthodes

| Méthode                           | Coût Final | Précision sur les données de test |
|-----------------------------------|------------|------------------------------------|
| Descente de Gradient Personnalisée | 0.4797     | 71.39%                            |
| LogisticRegression (Scikit-learn) | N/A        | 99.44%                            |

