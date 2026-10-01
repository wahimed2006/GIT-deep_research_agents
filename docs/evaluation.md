# Evaluation des agents

## Lancer le benchmark

Le fichier `test/evaluation_questions.json` contient exactement 100 questions. Une exécution complète appelle Ollama et le web:

```bash
./.venv/bin/python scripts/run_evaluation.py
```

Pendant le développement, limiter le nombre de requêtes:

```bash
./.venv/bin/python scripts/run_evaluation.py --limit 1 --db /tmp/evaluation.sqlite3 --export /tmp/evaluation-results.json
```

Chaque question est exécutée indépendamment. Une exception est enregistrée dans l'export et les questions suivantes continuent.

## Base de données et identifiants

La configuration par défaut est une base SQLite locale `evaluation.sqlite3`; SQLite n'a pas de compte ni de mot de passe. Le fichier est exclu de Git par `.gitignore`. La protection est celle du système de fichiers:

```bash
chmod 600 evaluation.sqlite3
```

Il n'y a donc aucun mot de passe à fournir ou à stocker. Le schéma exécutable est dans `docs/evaluation-schema.sql`, et le module Python qui écrit les lignes est `src/evaluation/database.py`.

Pour une base PostgreSQL partagée, utiliser un secret hors dépôt, par exemple `DATABASE_URL` dans l'environnement CI, puis migrer les colonnes JSON texte vers `JSONB`; aucun mot de passe ne doit être commité.

## Rapport JavaScript

Après une exécution:

```bash
node scripts/report.js evaluation-results.json evaluation-report.html
```

Le fichier HTML généré contient un tableau par question et une courbe de durée totale. Il est autonome et n'a pas de dépendance npm.

## Tests unitaires

```bash
./.venv/bin/python -m pytest -q test/test_evaluation.py
```

Les tests remplacent les appels Ollama, recherche web et scraping par des doublures; ils ne nécessitent donc ni modèle téléchargé ni réseau.
