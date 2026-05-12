"""
src/chatbot/intent_classifier.py
==================================
Classification d'intent par similarité cosinus sur embeddings BGE-M3.

Inspiré du IntentSimilarityClassifier du projet shipping, adapté pour
fonctionner sans base de données ni LLM supplémentaire :
    - Les exemples sont chargés depuis des fichiers texte (data/intents/*/suggk.txt)
  - Les embeddings sont calculés une seule fois au démarrage, puis mis en cache
  - La classification est synchrone et rapide (~5-15ms sur GPU)

Architecture scalable : ajouter un nouveau intent = créer un dossier
    data/intents/<nom_intent>/suggk.txt
et définir son tier dans INTENT_TIERS.

Intents disponibles (classés par tier) :

  Tier 1 — Basiques :
    greeting              Salutations et demandes d'aide initiales
    retrieval             Questions documentaires générales (fallback par défaut)
    meta                  Questions sur le chatbot lui-même
    chitchat              Réponses courtes, remerciements, bavardage
    out_of_scope          Sujets hors périmètre RCAR/CNRA
    feedback              Retours sur la qualité des réponses

  Tier 2 — Contextuels :
    clarification         Question ambiguë ou incomplète
    comparison            Demandes de comparaison RCAR vs CNRA ou entre prestations
    procedural            Demandes de procédures étape par étape
    navigation            Références au contexte conversationnel précédent
    confirmation          Demandes de vérification d'une information
    reformulation_request Demandes de reformulation ou simplification
    negation              Corrections ou désaccords avec une réponse précédente
    multi_part            Plusieurs questions dans une seule requête
    conversational_fallback Réactions conversationnelles sans contenu informatif

  Tier 3 — Avancés :
    calculation           Calculs numériques (montants, dates, taux)
    administrative_declaration Déclarations de situation (attestation de vie, etc.)
    personal_info_request Demandes d'informations personnelles de l'affilié
    insistence_personal_data Insistance pour obtenir des données personnelles
    prompt_injection      Tentatives de manipulation ou d'injection de prompt

  Tier 4 — Supplémentaires (gate retrievers) :
    needs_form            Demandes explicites de formulaires ou documents à remplir
    needs_video           Demandes explicites de vidéos explicatives

Utilisation normale (pipeline complet) :
    classifier = IntentClassifier(shared_embeddings=pipeline.embeddings)
    result = classifier.classify("je veux télécharger le formulaire de pension")
    # → {"intent": "needs_form", "confidence": 0.87, "tier": 4, "reasoning": "..."}

Utilisation standalone (tests, debug) :
    classifier = IntentClassifier()  # charge BGE-M3 seul, log d'avertissement
    result = classifier.classify("bonjour")

Gate vidéo/formulaire dans rag_pipeline.py :
    RETRIEVER_INTENTS = {
        "video": {"needs_video", "procedural", "retrieval", "comparison"},
        "form":  {"needs_form", "procedural", "administrative_declaration"},
    }
"""
# ── Compatibilité Python 3.8+ pour les annotations de type ───────────────────
from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────

INTENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "intents"

# Modèle de fallback standalone (même modèle que le pipeline principal)
_STANDALONE_EMBEDDING_MODEL = "BAAI/bge-m3"

# ─────────────────────────────────────────────────────────────────────────────
# Définition des tiers — SEUL endroit à modifier pour changer un tier
# ─────────────────────────────────────────────────────────────────────────────

INTENT_TIERS: Dict[str, int] = {
    # Tier 1 — Basiques
    "greeting":               1,
    "retrieval":              1,
    "meta":                   1,
    "chitchat":               1,
    "out_of_scope":           1,
    "feedback":               1,
    # Tier 2 — Contextuels
    "clarification":          2,
    "comparison":             2,
    "procedural":             2,
    "navigation":             2,
    "confirmation":           2,
    "reformulation_request":  2,
    "negation":               2,
    "multi_part":             2,
    "conversational_fallback": 2,
    # Tier 3 — Avancés
    "calculation":            3,
    "administrative_declaration": 3,
    "personal_info_request":  3,
    "insistence_personal_data": 3,
    "prompt_injection":       3,
    # Tier 4 — Gate retrievers
    "needs_form":             4,
    "needs_video":            4,
}

# Intents qui autorisent le retriever vidéo
VIDEO_RETRIEVER_INTENTS = {"needs_video", "procedural", "retrieval", "comparison"}

# Intents qui autorisent le retriever formulaires
FORM_RETRIEVER_INTENTS = {"needs_form", "procedural", "administrative_declaration"}

# Seuil de confiance minimum pour appliquer le gate
# En dessous, on laisse passer (comportement conservateur)
GATE_CONFIDENCE_THRESHOLD = 0.60

# Nombre de voisins pour le vote majoritaire
TOP_K = 7


class IntentClassifier:
    """
    Classifie l'intent d'une requête utilisateur par similarité cosinus
    sur les embeddings BGE-M3 des exemples d'entraînement.

    Chaque intent correspond à un dossier dans data/intents/ contenant
    un fichier suggk.txt avec un exemple par ligne.

    Scalabilité : ajouter un dossier data/intents/<nom>/suggk.txt et son tier
    dans INTENT_TIERS. Le classifier le détectera automatiquement au prochain
    chargement.

    Args:
        shared_embeddings: Instance HuggingFaceEmbeddings partagée avec RAGPipeline
                           (évite de recharger BGE-M3 en mémoire une 2e fois).
                           Si None, charge BGE-M3 de façon autonome (usage standalone,
                           tests, debug) — un avertissement est émis dans ce cas.
        intents_dir: Dossier racine des exemples d'intent (défaut: intents/)
        top_k: Nombre de voisins pour le vote majoritaire (défaut: 7)
        gate_confidence_threshold: Confiance min pour activer le gate (défaut: 0.45)
    """

    def __init__(
        self,
        shared_embeddings: Optional[Any] = None,
        intents_dir: Path = INTENTS_DIR,
        top_k: int = TOP_K,
        gate_confidence_threshold: float = GATE_CONFIDENCE_THRESHOLD,
    ) -> None:
        # Mémorise si les embeddings ont été fournis depuis l'extérieur.
        # Utilisé dans get_stats() pour distinguer mode partagé / standalone —
        # après _ensure_embeddings(), self.embeddings est toujours non-None et
        # ne permet plus de faire la distinction.
        self._shared_embeddings_provided: bool = shared_embeddings is not None

        self.embeddings = shared_embeddings
        self.intents_dir = intents_dir
        self.top_k = top_k
        self.gate_confidence_threshold = gate_confidence_threshold

        # Matrices chargées au warm-up
        self.example_embeddings: Optional[np.ndarray] = None
        self.example_labels: List[str] = []
        self.example_texts: List[str] = []
        self.intent_tier_map: Dict[str, int] = {}
        self._loaded = False

        # Charger les exemples au démarrage
        self._load_examples()

    # ── Chargement ────────────────────────────────────────────────────────────

    def _ensure_embeddings(self) -> bool:
        """
        Garantit que self.embeddings est disponible.

        Si shared_embeddings n'a pas été fourni au constructeur, charge BGE-M3
        de façon autonome (mode standalone). Émet un avertissement car ce mode
        consomme de la VRAM supplémentaire si le pipeline principal tourne en
        parallèle.

        Returns:
            True si les embeddings sont prêts, False en cas d'échec.
        """
        if self.embeddings is not None:
            return True

        logger.warning(
            "IntentClassifier: aucun shared_embeddings fourni — "
            "chargement standalone de {} (double VRAM si RAGPipeline actif). "
            "Passez shared_embeddings=pipeline.embeddings pour éviter ça.",
            _STANDALONE_EMBEDDING_MODEL,
        )

        try:
            import torch
            from langchain_huggingface import HuggingFaceEmbeddings

            device = "cuda" if torch.cuda.is_available() else "cpu"
            hf_token = os.getenv("HF_TOKEN", "").strip()
            model_kwargs: Dict[str, Any] = {"device": device}
            if hf_token:
                model_kwargs["token"] = hf_token

            self.embeddings = HuggingFaceEmbeddings(
                model_name=_STANDALONE_EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(
                "IntentClassifier: embeddings standalone chargés ({} sur {})",
                _STANDALONE_EMBEDDING_MODEL, device,
            )
            return True

        except Exception as exc:
            logger.error(
                "IntentClassifier: impossible de charger les embeddings standalone: {}. "
                "Classifier désactivé.",
                exc,
            )
            return False

    def _load_examples(self) -> None:
        """
        Lit tous les fichiers data/intents/*/suggk.txt, calcule les embeddings
        et construit la matrice de similarité en RAM.
        """
        start = time.perf_counter()

        if not self.intents_dir.exists():
            logger.warning(
                "IntentClassifier: dossier intents introuvable: {}. "
                "Classifier désactivé — vérifiez INTENTS_DIR ou la structure du projet.",
                self.intents_dir,
            )
            return

        texts: List[str] = []
        labels: List[str] = []

        # Parcourir tous les sous-dossiers d'intents
        intent_dirs = sorted([d for d in self.intents_dir.iterdir() if d.is_dir()])

        if not intent_dirs:
            logger.warning(
                "IntentClassifier: aucun dossier d'intent trouvé dans {}",
                self.intents_dir,
            )
            return

        for intent_dir in intent_dirs:
            intent_name = intent_dir.name
            suggk_file = intent_dir / "suggk.txt"

            if not suggk_file.exists():
                logger.debug(
                    "IntentClassifier: pas de suggk.txt dans {}, ignoré", intent_dir
                )
                continue

            examples = [
                line.strip()
                for line in suggk_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            if not examples:
                logger.debug(
                    "IntentClassifier: suggk.txt vide pour {}, ignoré", intent_name
                )
                continue

            tier = INTENT_TIERS.get(intent_name)
            if tier is None:
                logger.warning(
                    "IntentClassifier: intent '{}' non défini dans INTENT_TIERS — "
                    "ajoutez-le avec son tier. Intent ignoré.",
                    intent_name,
                )
                continue

            texts.extend(examples)
            labels.extend([intent_name] * len(examples))
            self.intent_tier_map[intent_name] = tier
            logger.debug(
                "IntentClassifier: {} exemples chargés pour '{}'",
                len(examples), intent_name,
            )

        if not texts:
            logger.error(
                "IntentClassifier: aucun exemple valide chargé. Classifier désactivé."
            )
            return

        # Garantir que les embeddings sont disponibles avant de les appeler
        if not self._ensure_embeddings():
            return

        logger.info(
            "IntentClassifier: calcul embeddings pour {} exemples ({} intents)...",
            len(texts), len(self.intent_tier_map),
        )

        try:
            raw_embeddings = self.embeddings.embed_documents(texts)
            embeddings_np = np.array(raw_embeddings, dtype=np.float32)

            # Normalisation L2 pour la similarité cosinus via produit scalaire
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings_np = embeddings_np / norms

            self.example_embeddings = embeddings_np
            self.example_labels = labels
            self.example_texts = texts
            self._loaded = True

            elapsed = (time.perf_counter() - start) * 1000
            counts = Counter(labels)
            logger.info(
                "IntentClassifier prêt: {} exemples, {} intents, {:.0f}ms",
                len(texts), len(self.intent_tier_map), elapsed,
            )
            for intent in sorted(counts):
                tier = self.intent_tier_map.get(intent, "?")
                logger.debug(
                    "  [tier {}] {}: {} exemples", tier, intent, counts[intent]
                )

        except Exception as exc:
            logger.error(
                "IntentClassifier: erreur calcul embeddings: {}. Classifier désactivé.",
                exc,
            )

    # ── Classification ────────────────────────────────────────────────────────

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classifie l'intent d'une requête par vote majoritaire cosinus.

        Pipeline :
          1. Embed la query (BGE-M3 partagé ou standalone)
          2. Cosine similarity contre tous les exemples
          3. Top-K voisins → vote majoritaire
          4. Confiance = score cosinus moyen des voisins de l'intent gagnant

        Returns:
            {
                "intent":     str,    # nom de l'intent classifié
                "confidence": float,  # score cosinus moyen (0.0 – 1.0)
                "tier":       int,    # tier de l'intent (1–4)
                "reasoning":  str,    # explication lisible du vote
                "loaded":     bool,   # False si le classifier n'est pas prêt
            }
        """
        if not self._loaded or self.example_embeddings is None:
            logger.warning("IntentClassifier: non chargé, retour fallback 'retrieval'")
            return self._fallback("Classifier non chargé")

        query = (query or "").strip()
        if not query:
            return self._fallback("Query vide")

        start = time.perf_counter()

        try:
            # 1. Embed la query
            query_embedding = np.array(
                self.embeddings.embed_query(query), dtype=np.float32
            )
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

            # 2. Cosine similarity (produit scalaire sur vecteurs normalisés)
            similarities = query_embedding @ self.example_embeddings.T  # shape: (N,)

            # 3. Top-K voisins
            k = min(self.top_k, len(similarities))
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_indices = top_k_indices[
                np.argsort(similarities[top_k_indices])[::-1]
            ]

            top_labels = [self.example_labels[i] for i in top_k_indices]
            top_scores = [float(similarities[i]) for i in top_k_indices]

            # 4. Vote majoritaire
            vote_counts = Counter(top_labels)
            winner, winner_votes = vote_counts.most_common(1)[0]

            # Confiance = score cosinus moyen des voisins du winner
            winner_scores = [
                top_scores[j] for j, lbl in enumerate(top_labels) if lbl == winner
            ]
            confidence = float(np.mean(winner_scores)) if winner_scores else 0.0
            tier = self.intent_tier_map.get(winner, 1)

            # ── Fallback retrieval sur faible confiance ───────────────────────
            # Si l'intent gagnant n'est pas "retrieval" mais sa confiance est
            # insuffisante (< 0.60), on retombe sur "retrieval" pour ne pas
            # bloquer une vraie question documentaire.
            # Ex : "C'est quoi un RECORE ?" classé "greeting" à 0.49 → retrieval
            RETRIEVAL_FALLBACK_THRESHOLD = 0.60
            NON_DOC_INTENTS = {
                "greeting", "chitchat", "out_of_scope",
                "conversational_fallback", "feedback",
            }
            if winner in NON_DOC_INTENTS and confidence < RETRIEVAL_FALLBACK_THRESHOLD:
                logger.info(
                    "IntentClassifier fallback retrieval : '{}' conf={:.3f} < {:.2f} → retrieval",
                    winner, confidence, RETRIEVAL_FALLBACK_THRESHOLD,
                )
                winner = "retrieval"
                tier = self.intent_tier_map.get("retrieval", 1)

            elapsed_ms = (time.perf_counter() - start) * 1000

            top3_str = ", ".join(
                f"{self.example_labels[int(i)]}({float(similarities[int(i)]):.3f})"
                for i in top_k_indices[:3]
            )
            reasoning = (
                f"top-{k}: [{top3_str}] → {winner} "
                f"(confidence={confidence:.3f}, votes={winner_votes}/{k})"
            )

            logger.info(
                "IntentClassifier: '{}' → {} [tier {}] conf={:.3f} ({:.0f}ms)",
                query[:60], winner, tier, confidence, elapsed_ms,
            )

            return {
                "intent":     winner,
                "confidence": confidence,
                "tier":       tier,
                "reasoning":  reasoning,
                "loaded":     True,
            }

        except Exception as exc:
            logger.error("IntentClassifier: erreur classify: {}", exc)
            return self._fallback(str(exc))

    # ── Gate helpers ──────────────────────────────────────────────────────────

    def should_retrieve_videos(self, classification: Dict[str, Any]) -> bool:
        """
        Retourne True si le retriever vidéo doit être activé pour cette requête.

        Règle : intent dans VIDEO_RETRIEVER_INTENTS ET confidence >= seuil.
        En dessous du seuil, on laisse passer (comportement conservateur).
        """
        if not classification.get("loaded", True):
            return True  # Classifier non disponible → laisser passer

        intent = classification.get("intent", "retrieval")
        confidence = classification.get("confidence", 0.0)

        if confidence < self.gate_confidence_threshold:
            logger.debug(
                "IntentClassifier gate vidéo: confiance {} < seuil {} → pass-through",
                confidence, self.gate_confidence_threshold,
            )
            return True  # Confiance trop faible → comportement conservateur

        result = intent in VIDEO_RETRIEVER_INTENTS
        logger.debug(
            "IntentClassifier gate vidéo: intent='{}' → {}",
            intent, "ACTIF" if result else "BLOQUÉ",
        )
        return result

    def should_retrieve_forms(self, classification: Dict[str, Any]) -> bool:
        """
        Retourne True si le retriever formulaires doit être activé pour cette requête.

        Règle : intent dans FORM_RETRIEVER_INTENTS ET confidence >= seuil.
        En dessous du seuil, on laisse passer (comportement conservateur).
        """
        if not classification.get("loaded", True):
            return True

        intent = classification.get("intent", "retrieval")
        confidence = classification.get("confidence", 0.0)

        if confidence < self.gate_confidence_threshold:
            logger.debug(
                "IntentClassifier gate form: confiance {} < seuil {} → pass-through",
                confidence, self.gate_confidence_threshold,
            )
            return True

        result = intent in FORM_RETRIEVER_INTENTS
        logger.debug(
            "IntentClassifier gate form: intent='{}' → {}",
            intent, "ACTIF" if result else "BLOQUÉ",
        )
        return result

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def _fallback(self, reason: str) -> Dict[str, Any]:
        """Résultat par défaut quand la classification échoue."""
        return {
            "intent":     "retrieval",
            "confidence": 0.0,
            "tier":       1,
            "reasoning":  f"Fallback: {reason}",
            "loaded":     False,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_intents_by_tier(self) -> Dict[int, List[str]]:
        """Retourne les intents groupés par tier (utile pour debug/admin)."""
        by_tier: Dict[int, List[str]] = {}
        for intent, tier in sorted(self.intent_tier_map.items()):
            by_tier.setdefault(tier, []).append(intent)
        return {t: sorted(intents) for t, intents in sorted(by_tier.items())}

    def get_stats(self) -> Dict[str, Any]:
        """Stats de chargement pour les logs de démarrage."""
        if not self._loaded:
            return {"loaded": False}
        counts = Counter(self.example_labels)
        return {
            "loaded":           True,
            "total_examples":   len(self.example_labels),
            "total_intents":    len(self.intent_tier_map),
            "top_k":            self.top_k,
            "gate_threshold":   self.gate_confidence_threshold,
            # True = mode standalone (BGE-M3 chargé ici), False = embeddings partagés.
            # Mémorisé au __init__ car après _ensure_embeddings() self.embeddings
            # est toujours non-None et ne permet plus de distinguer les deux cas.
            "standalone_mode":  not self._shared_embeddings_provided,
            "intents_by_tier":  self.get_intents_by_tier(),
            "examples_per_intent": dict(counts),
        }