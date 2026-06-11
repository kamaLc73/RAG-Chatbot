# RAG Chatbot RCAR/CNRA

Assistant conversationnel RAG (Retrieval-Augmented Generation) pour interroger des contenus RCAR/CNRA.

## Démarrage Docker

Prérequis :

- Docker Desktop lancé.
- Ollama lancé via Docker Compose ou sur la machine hôte, avec le modèle utilisé par `OLLAMA_MODEL`.
- Le schéma Vespa doit être déployé avant une première indexation.

Construire et lancer l'application :

```powershell
docker compose up --build
```

Connecter Ollama Cloud puis preparer le modele cloud utilise par le chatbot :

```powershell
docker compose up -d ollama
docker compose exec ollama ollama signin
docker compose --profile models run --rm prepare-ollama-cloud-model
```

URLs :

- UI React : http://localhost:5173
- API FastAPI : http://localhost:8000
- Health API : http://localhost:8000/health
- Vespa : http://localhost:8080
- Vespa config : http://localhost:19071
- Ollama : http://localhost:11434
- PostgreSQL hôte : `localhost:5433`
- Redis hôte : `localhost:6380`

Variables utiles :

```powershell
$env:API_BOOTSTRAP_SUPERUSER_EMAIL="admin@cdg.dev"
$env:API_BOOTSTRAP_SUPERUSER_USERNAME="admin_local"
$env:API_BOOTSTRAP_SUPERUSER_FULL_NAME="admin local"
$env:API_BOOTSTRAP_SUPERUSER_PASSWORD="<mot-de-passe-local>"
$env:DOCKER_OLLAMA_BASE_URL="http://ollama:11434"
$env:OLLAMA_MODEL="ministral-3:14b-cloud"
docker compose up --build
```

Le frontend est compilé avec `DOCKER_VITE_API_URL`, par défaut `http://localhost:8000`.

## Vespa et indexation

Si le volume Vespa est vide, déployer d'abord le schéma Vespa depuis l'hôte :

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; import docker; from vespa.deployment import VespaDocker; client=docker.from_env(); container=client.containers.get('vespa'); VespaDocker(url='http://localhost', port=8080, container=container).deploy_from_disk(application_name='rcarcnra', application_root=Path('src/store/vespa'), max_wait_configserver=120, max_wait_application=120, docker_timeout=120); print('vespa deploy OK')"
```

Indexer les données depuis les conteneurs :

```powershell
docker compose run --rm index-docs
docker compose run --rm index-forms
docker compose run --rm index-videos
```

Les services d'indexation utilisent les mêmes volumes `data`, `logs` et caches modèles que l'API.
