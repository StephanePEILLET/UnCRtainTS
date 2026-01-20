# 📝 TODO List
Revoir les shapes dans le cas du training

## 🌍 Général
- [X] 📊 Métriques par bandes
- [X] 🎯 Vérifier que la target n'est pas dans l'input :
    - [X] 🎲 Regarder dans le cas du sampler random
    - [X] 🔒 Regarder dans le cas du sampler fixed

## ⚙️ Code fonction `iterate`
- [X] 🔪 Mieux découper la fonction `iterate` de base afin d'éviter qu'il y ait trop d'erreurs ou de dépendances ?
- [X] 🔄 Modifier la fonction `iterate` afin de pouvoir retourner toutes les inférences d'une TS :
    - [X] 📦 Reprendre la classe Imputation d'U-TILISE
    - [X] 1️⃣ Faire en sorte de ne sortir qu'une inférence de l'imputation (cas nominal)
    - [X] 🚧 Modifier le code afin de prévoir l'utilisation du modèle sur les bords de la TS
    - [X] 📏 Adapter le code des métriques afin de pouvoir les calculer sur une/plusieurs dates de la TS

## 💾 Code class `dataloader`
- [x] 👁️ Faire une passe sur le code pour mieux comprendre le fonctionnement du chargement des données
- [X] 📏 Regarder les inputs/outputs du modèle afin de comprendre le rapport de longueurs entre i/o du modèle
- [x] 🎲 Si besoin, voir si la sélection aléatoire de sous-parties de la TS doit être revue
- [X] 🔄 Implémenter la possibilité de retourner une TS entière
- [X] 🎭 Faire en sorte d'avoir les masques de données pour les tests sets aléatoires et consécutifs

## 🚀 Amélioration Training
- [X] 📉 Changer le val pour avoir une variation dans la sélection des obs de val


Faire les inférences pour Michael

Avancer sur la rédaction