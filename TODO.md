# 📝 TODO List

## 🌍 Général
- [ ] 📊 Métriques par bandes
- [ ] 🎯 Vérifier que la target n'est pas dans l'input :
    - [ ] 🎲 Regarder dans le cas du sampler random
    - [ ] 🔒 Regarder dans le cas du sampler fixed
- [ ] 🔄 Faire la distinction des modifications à faire dans le code du `iterate` et ce qu'il y a à faire dans le `dataloader` !

## ⚙️ Code fonction `iterate`
- [ ] 🔪 Mieux découper la fonction `iterate` de base afin d'éviter qu'il y ait trop d'erreurs ou de dépendances ?
- [ ] 🔄 Modifier la fonction `iterate` afin de pouvoir retourner toutes les inférences d'une TS :
    - [ ] 📦 Reprendre la classe Imputation d'U-TILISE
    - [ ] 1️⃣ Faire en sorte de ne sortir qu'une inférence de l'imputation (cas nominal)
    - [ ] 🚧 Modifier le code afin de prévoir l'utilisation du modèle sur les bords de la TS
    - [ ] 📏 Adapter le code des métriques afin de pouvoir les calculer sur une/plusieurs dates de la TS

## 💾 Code class `dataloader`
- [x] 👁️ Faire une passe sur le code pour mieux comprendre le fonctionnement du chargement des données
- [ ] 📏 Regarder les inputs/outputs du modèle afin de comprendre le rapport de longueurs entre i/o du modèle
- [x] 🎲 Si besoin, voir si la sélection aléatoire de sous-parties de la TS doit être revue
- [ ] 🔄 Implémenter la possibilité de retourner une TS entière
- [ ] 🎭 Faire en sorte d'avoir les masques de données pour les tests sets aléatoires et consécutifs

## 🚀 Amélioration Training
- [ ] 🎭 Induire du masquage dans la série temporelle 
- [ ] 📉 Changer le val pour avoir une variation dans la sélection des obs de val
