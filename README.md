# Projet 6 OC — Déploiement et monitoring d'un modèle de scoring

## Contexte

Ce projet s'inscrit dans le parcours OpenClassrooms AI Engineer. Il fait suite au projet "Initiez-vous au MLOps", dans lequel un modèle de scoring crédit a été développé, versionné et évalué.

L'entreprise fictive **"Prêt à Dépenser"** souhaite mettre ce modèle en production pour le département "Crédit Express", afin de traiter les nouvelles demandes de crédit en quasi temps réel. L'objectif de ce projet est donc de :

- exposer le modèle de scoring via une **API fonctionnelle et déployable** (conteneurisée avec Docker),
- mettre en place un **pipeline CI/CD** pour automatiser les tests et le déploiement,
- préparer le **monitoring** de la solution une fois en production (stockage des données, suivi du data drift).

## Fonctionnement de l'API

L'API est développée avec **FastAPI** et expose deux routes principales :

### `POST /Request`
Reçoit les caractéristiques (features) d'un client (situation de crédit, revenus, emploi, type de logement, historique de paiement, etc.). Ces données sont stockées dans une base **PostgreSQL** (table `Project6_Features`), associées à un identifiant unique (UUID) généré à la volée. L'endpoint retourne cet identifiant.

### `POST /Predict`
Prend en paramètre l'identifiant retourné par `/Request`, récupère les features correspondantes en base, puis les passe au modèle de scoring (préalablement entraîné et chargé au démarrage de l'API via `joblib`). Le modèle renvoie une probabilité d'acceptation et de refus du crédit.

Cette séparation en deux étapes (enregistrement des données, puis prédiction à partir d'un id) permet de conserver une trace de chaque demande traitée par l'API, réutilisable ensuite pour l'analyse du data drift.

Une route `GET /` sert par ailleurs une page d'accueil statique (`index.html`).

## Structure du dépôt

```
.
├── main.py               # Code de l'API FastAPI (routes /Request et /Predict)
├── test_main.py          # Tests automatisés (flux Request → Predict)
├── index.html            # Page d'accueil de l'API
├── model_trained          # Modèle de scoring entraîné (sérialisé avec joblib)
├── Dockerfile             # Image Docker de l'API
├── requirements.txt       # Dépendances Python
├── .dockerignore
├── .gitignore
└── .github/workflows/      # Pipeline CI/CD (GitHub Actions)
```

## Installation et lancement en local

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/tmininihub/Projet-6-OC.git
   cd Projet-6-OC
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Configurer les variables d'environnement (fichier `local.env`), notamment l'URL de connexion à la base de données :
   ```
   URLBDD=postgresql://<user>:<password>@<host>:<port>/<database>
   ```

4. Lancer l'API :
   ```bash
   uvicorn main:app --reload
   ```

L'API est alors accessible sur `http://localhost:8000`, avec la documentation interactive Swagger sur `http://localhost:8000/docs`.

## Lancement avec Docker

```bash
docker build -t projet6-oc .
docker run -p 8000:8000 --env-file docker.env projet6-oc
```

## Tests

Les tests automatisés vérifient le bon fonctionnement du flux complet (envoi des features, récupération de l'id, puis prédiction) :

```bash
pytest test_main.py
```

## Intégration continue

Un pipeline CI/CD (GitHub Actions, dans `.github/workflows/`) exécute automatiquement les tests et gère le déploiement à chaque push sur la branche principale.

## Modèle utilisé

Le modèle de scoring est un modèle **LightGBM**, entraîné dans le cadre du projet précédent (MLOps) et sérialisé au format `joblib`. Il est chargé une seule fois au démarrage de l'API afin d'éviter tout rechargement inutile à chaque requête.
