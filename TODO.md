TODO:
----
-[ ] Faire la distinction des modifications a faire dans le code du iterate et ce qu'il y a faire dans le dataloader!

    * Code fonction iterate:
    ========================
        - [ ] mieux découper la fonction iterate de base afin d'éviter qu'il y ait trop d'erreurs ou de dépendances?
        - [ ] modifier la fonction iterate afin de pouvoir retourner toutes les inférences d'une TS:
            - [ ] reprendre la classe Imputation d'U-TILISE
            - [ ] faire en sorte de ne sortir qu'une inférence de l'imputation, cas nominale
            - [ ] modifier le code afin de prévoir l'utilisation du modèle sur les bords de la TS.
            - [ ] adapter le code des métriques afin de pouvoir les calculer sur une/plusieurs dates de la TS.

    * Code class dataloader:
    ========================
        - [ ] faire une passe sur le code pour mieux comprendre le fonctionnement du chargement des données.
        - [ ] regarder les inputs/outputs du modèles afin de comprendre le rapport de longeurs entre i/o du modèle.
        - [ ] si besoin voir si la sélection aléatoire de sous-parties de la TS doit être revu.
        - [ ] implementer la possibilité de retourner une TS entière.
        - [ ] faire en sorte d'avoir les masques de données pour les tests sets aléatoires et consécutives.  



Futurs Prompts Copilot:
-----------------------

Peux-tu t'aider à te reperer dans le repo, en créant un ficher TREE.md où tu pourras te regrouper les liaisons des différents imports entre les fichiers présents dans le repo

peux tu lire tous les  fichiers dans le repo afin d'enrichir ta compréhension du code de manière générale et les dépendances et les intéractions entre les différents éléments du code permettant de faire tourner le repo (aussi bien pour le train_reconstruct que pour le test reconstruct.py)

Fait la modification de code permettant de prendre en compte les modifications de code faites dans iterate_proposal.py afin d'utiliser la nouvelle fonction iterate_v2 à la place de l'ancienne version du code iterate se trouvant dans le fichier train_reconstruct.py 

tu peux tester les modifications que tu as faites directement en lançant le debugger avec la configuration "Train Reconstruct" (la commande si besoin se trouve dans le fichier.vscode/launch.json)
