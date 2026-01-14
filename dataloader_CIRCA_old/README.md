README.md 

Le but de ce répertoire est de regrouper les fonctions et les classes permettant de faire
le chargement de données du dataset CIRCA et cela dans le but de faire de la reconstruction
de données Sentinel-S2 sans nuages. 

TODO:
-----

Ajout d'un bon datalinter comme ruff
Avant tout, commencer à bien regarder ce que fait produit le modèle en sortie des fonctions de U-TILISE
(ensuite relister les étapes de préprocessing)
-comprendre exactement ce qu'est passé en entrée à la loss et 
=> commencer à faire une fiche permettant de lister

dans le __get_item__:
    - finir un premier jet de la fonction (finir generate_mask, le trimming de TS, voir si le sampling est pertinent, ajout détection de sequence non nuageuse consécutive)
    - penser à la sélection des masques dans la série temporelle (cette sélection doit être faite en amont du
    filtrage => prendre tous les masques nuages T * 1 * H * W et faire une sélection aléatoire, comment faire pour sélectionner des masques nuages pertinents le modèle doit voir )
    - penser au filtrage, finalement notre filtrage étant efficace, est-ce que le nombre de dates non-nuageuses consécutives est bon pour le modèle. 
    => faire des stats pour regarder les longeurs de dates nuagueuses consécutives
    est-ce que ça serait intéressant de faire des stats de manière générale sur le dataset CIRCA? 
    (stats vis-à-vis des nuages et aussi regarder si des fonctions de préprocessing seraient intéressantes, 
    genre du clipping et/ou du rescaling) 
    - rendre le dataset fonctionnel pour faire de l'inférence (non génération de masks et non modification des inputs plus insert d'infos en plus comme l'output filename)

- continuer de faire le dataloader pour le modèle U-TILISE. 
    => adapter les sorties du dataloader CIRCA à ce qu'attend le modèle U-TILISE
    => regarder dans le code du modèle comment est gérer les masks et le filtrage
    => réfléchir à comment faire pour gérer les longeurs de TS
    => remettre au propre les typings, commentaires et les docstrings
    => faire une passe avec isort et black
    - déterminer comment est fait le distingo entre image input / target
    - bien regarder la gestion et la création des masks nuages
    - sortir dans le code la longeur minimale d'une TS

Pour U-TILISE, liste des futurs trainings faire:
    - faire avec et sans data augmentation
    - faire varier les fonctions de filtrages
    - faire varier les différentes longueurs de séquences.
    - intégrer et tester les différentes façons de paire du PE
    - faire varier les différents types de masks disponibles (dilatation, longeur min etc .. )
    - tester avec et sans SAR pour voir s'il y a une réelle différence
    - le modèle fonctionne-t-il mieux en voyant davantage de dates nuageuses que sans?

Faire les scripts permettant de stocker les datasets au format hdF5 sur le store-DAI ou jzay
version hdF5 des datasets en mode preload? 

- on part avec nos propres fonctions de filtrages de nuages, avec le temps on cherchera à les enrichir avec les fonctionnalités provenant des différents repos.
- faire un script permettant de passer de donner découper à la volée à un fichier hdf5. 
- finir de regarder comment fonctionne le dataloader de UnCRtainTS et apporter les modifications nécessaires dans mon dataloader CIRCA dédié.
- refaire rapidement des liens vers jzay sur Nautilus (revoir comment faire la connection ssh via Nautilus)
- regarder les données de SEN12MSCRTSDataset sur jzay

- faire un script permettant de calculer les métriques de reconstruction en prenant en entrée une donnée sans nuage, son mask et la vérité terrain correspondante. 
    => (réfléchir à faire des métriques pour les pixels masqués et ceux non masqués)

- commencer à faire une doc sur le masquage des nuages possibles et intéressant à couvrir (s'inspirer de U-TILISE)

- faire des fonctions de timeout pour savoir que prends le temps du dataloader et voir quels sont les améliorations que l'on peut y faire pour l'accélérer.

- enrichier le README

python tif2hdf5.py /lustre/fsn1/projects/rech/tel/uug84ql/cloud_reconstruction/africa all africa /lustre/fsn1/projects/rech/tel/uug84ql/DATA-SEN12MS-CR-TS