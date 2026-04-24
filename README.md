# RAG Chatbot RCAR/CNRA

Assistant conversationnel RAG (Retrieval-Augmented Generation) pour interroger des contenus RCAR/CNRA, avec un pipeline complet:

- crawl web multi-sources (CNRA + RCAR)
- extraction de texte pages/PDF
- OCR en fallback sur PDF scannes
- preparation des corpus
- indexation vectorielle (Chroma)
- interface utilisateur Streamlit

## Objectif

Ce projet permet de construire un chatbot metier autour de contenus retraite/prevoyance RCAR/CNRA.

Le but est de:

- centraliser les contenus utiles
- les rendre interrogeables en langage naturel
- fournir des reponses contextualisees depuis la base documentaire

## Fonctionnalites principales

- Crawler HTTP base sur Crawlee (mode FR/AR, gestion des retries et limitations)
- Extraction de texte nettoye depuis les pages HTML
- Detection et traitement des liens PDF
- Pipeline de preparation avec OCR optionnel/automatique
- Indexation vectorielle via Chroma + embeddings HuggingFace
- Generation de reponses via Ollama
- Interface Streamlit avec historique et logs centralises

## Structure du projet

```text
src/
  app.py                    # Interface Streamlit
  chatbot/
    rag_pipeline.py         # Retrieval + generation
    index_data.py           # Construction de la base vectorielle
  crawler/
    main.py                 # Point d'entree crawl
    crawler.py              # Logique de crawl
    data_handler.py         # Extraction/sauvegarde pages/PDF
  preparation/
    main.py                 # Point d'entree preparation
    pipeline.py             # Preparation + OCR
  config/
    settings.py             # Parametres globaux
    logger.py               # Configuration logs

data/
  raw/                      # Donnees brutes du crawl
  processed/                # Donnees preparees
  supportstagerag/          # Corpus de support (demo/indexation)
  vectorstore/              # Base Chroma persistente

logs/                       # Logs applicatifs et rapports
```

## Prerequis

- Python 3.11+ (3.12 recommande)
- pip
- Ollama installe et accessible
- Connexion internet pour les embeddings/modeles si non caches

OCR (si besoin):

- Tesseract est supporte via un executable embarque dans `src/preparation/Tesseract-OCR` (ou via installation systeme)

## Installation

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Creer un fichier `.env` a la racine (ne pas commiter de secrets):

```env
OLLAMA_MODEL=mistral:latest
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=
HF_TOKEN=
```

Notes:

- `HF_TOKEN` est recommande pour eviter des limites de telechargement Hugging Face.
- `OLLAMA_API_KEY` est utile si vous ciblez un endpoint Ollama distant/cloud.
- Le fichier `.env` est deja ignore par git.

## Workflow recommande (de bout en bout)

### 1) Crawl des sources

```powershell
python src/crawler/main.py --source all
```

Exemples:

```powershell
python src/crawler/main.py --source cnra
python src/crawler/main.py --source rcar --no-pdfs
python src/crawler/main.py --source all --max-pages 20 --max-depth 2
```

### 2) Preparation des donnees

```powershell
python src/preparation/main.py --source all --output-format md
```

Exemples:

```powershell
python src/preparation/main.py --source cnra --output-format md
python src/preparation/main.py --source all --min-native-chars 300
```

### 3) Indexation vectorielle

```powershell
python src/chatbot/index_data.py
```

Par defaut, l'indexation cible:

- `data/supportstagerag/rcar/faq`
- `data/supportstagerag/cnra/faq`

Collection Chroma par defaut:

- `rcar_cnra_fr`

### 4) Lancer le chatbot

```powershell
streamlit run src/app.py
```

## Commandes utiles

- Reindexer proprement la base vectorielle: relancer `python src/chatbot/index_data.py`
- Regenerer rapidement un petit jeu de donnees: utiliser `--max-pages` et `--max-pdfs` au crawl
- Changer la source unique: `--source cnra` ou `--source rcar`

## Sorties et logs

Fichiers utiles:

- `logs/crawl.log`
- `logs/preparation.log`
- `logs/index_data.log`
- `logs/chatbot.log`
- `logs/failed_urls.json`
- `data/raw/crawl_report.json`

## Depannage

### "Base vectorielle vide"

Executer:

1. `python src/chatbot/index_data.py`
2. verifier que `data/vectorstore/chroma_db` existe et contient la collection attendue

### "Erreur de generation" / Ollama indisponible

- verifier que Ollama tourne
- verifier `OLLAMA_BASE_URL`
- verifier le modele (`OLLAMA_MODEL`) present

### Premiere reponse lente

- normal au premier chargement (warmup embeddings + modele)
- les reponses suivantes sont plus rapides

### OCR non fonctionnel

- verifier la presence de Tesseract
- verifier les dependances PIL/numpy/pytesseract
- verifier les options OCR du script preparation

## Securite

- Ne jamais versionner des tokens/API keys dans `.env`.
- Si une cle a ete exposee, la revoquer et en generer une nouvelle.
- Eviter de partager les logs contenant des informations sensibles.

## Pistes d'amelioration

- Ajouter une suite de tests automatises (unitaires + integration)
- Ajouter des scripts `make`/`task` pour standardiser les commandes
- Ajouter un pipeline CI pour lint/tests
- Etendre l'indexation a d'autres sous-corpus que `faq`
