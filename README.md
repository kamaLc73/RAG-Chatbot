# RAG Chatbot RCAR/CNRA

Assistant conversationnel RAG pour interroger des contenus documentaires RCAR/CNRA en francais, avec separation des organismes, recherche hybride Vespa, API FastAPI et interface React.

## Prerequis

- Docker Desktop lance.
- Donnees preparees placees dans `data/supportstagerag/`, `data/forms/` et `data/youtube/`.
- Un fichier `.env` cree depuis `.env.example`.
- Acces Ollama Cloud ou session Ollama deja authentifiee pour `ministral-3:14b-cloud`.
- `HF_TOKEN` si Hugging Face le demande pour telecharger les modeles BGE-M3 et reranker.
- `MISTRAL_API_KEY` uniquement si la preparation OCR doit etre rejouee.
- `CEREBRAS_API_KEY` uniquement pour rejouer les evaluations RAGAS.

Les donnees ne sont pas incluses dans le depot. Pour les obtenir, demander le jeu de donnees prepare au responsable du projet ou a l'encadrant autorise.

## Configuration

Copier le modele d'environnement puis renseigner les valeurs necessaires :

```powershell
Copy-Item .env.example .env
```

Pour Ollama Cloud, connecter le conteneur Ollama avant de preparer le modele :

```powershell
docker compose up -d ollama
docker compose exec ollama ollama signin
```

## Installation automatisee

Le script `setup.py` orchestre les prerequis locaux :

- construction des images Docker ;
- demarrage de Vespa, Ollama, PostgreSQL et Redis ;
- preparation du modele Ollama ;
- deploiement des schemas Vespa ;
- telechargement des modeles d'embedding et de reranking ;
- indexation des documents, formulaires et videos ;
- lancement final de l'API et de l'interface.

Commande principale :

```powershell
python setup.py
```

L'indexation Vespa peut prendre plusieurs minutes au premier lancement, car les embeddings sont calcules puis inseres dans les schemas `doc`, `form` et `video`.

Options utiles :

```powershell
python setup.py --skip-index
python setup.py --skip-app-start
python setup.py --strict-keys
```

## Commandes manuelles

Construire et lancer les services :

```powershell
docker compose up --build
```

Indexer separement les donnees :

```powershell
docker compose --profile index run --rm index-docs
docker compose --profile index run --rm index-forms
docker compose --profile index run --rm index-videos
```

Precharger les modeles d'embedding et de reranking :

```powershell
docker compose --profile index run --rm download-embeddings
docker compose --profile index run --rm download-reranker
```

## URLs locales

- UI React : [http://localhost:5173](http://localhost:5173)
- API FastAPI : [http://localhost:8000](http://localhost:8000)
- Health API : [http://localhost:8000/health](http://localhost:8000/health)
- Vespa : [http://localhost:8080](http://localhost:8080)
- Vespa config : [http://localhost:19071](http://localhost:19071)
- Ollama : [http://localhost:11434](http://localhost:11434)
- PostgreSQL : `localhost:5433`
- Redis : `localhost:6380`

## Structure principale

```text
src/chatbot/       Pipeline RAG, retrieval et reranking
src/store/vespa/   Schemas et configuration Vespa
src/evaluation/    Scripts d'evaluation
UI/                Interface React
data/              Donnees preparees et jeux d'evaluation
logs/              Journaux applicatifs
```
