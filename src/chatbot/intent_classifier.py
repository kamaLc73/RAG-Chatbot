"""
src/chatbot/intent_classifier.py
================================

Intent classifier for the RCAR/CNRA chatbot.

The classifier is intentionally small and strict:
- examples are loaded from data/intents/<intent>/suggk.txt;
- BGE-M3 embeddings are reused from RAGPipeline when possible;
- resource intents require explicit wording from the user;
- low-confidence classifications fall back to retrieval instead of opening
  noisy resource gates.
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

INTENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "intents"
_STANDALONE_EMBEDDING_MODEL = "BAAI/bge-m3"

# Only keep intents that are actually trained and consumed by the pipeline.
INTENT_TIERS: Dict[str, int] = {
    "retrieval": 1,
    "greeting": 1,
    "out_of_scope": 1,
    "negation": 2,
    "prompt_injection": 3,
    "needs_form": 4,
    "needs_video": 4,
}

VIDEO_RETRIEVER_INTENTS = {"needs_video"}
FORM_RETRIEVER_INTENTS = {"needs_form"}

GATE_CONFIDENCE_THRESHOLD = 0.68
CLASSIFICATION_FALLBACK_THRESHOLD = 0.62
TOP_K = 7

DOMAIN_TERMS = {
    " rcar ",
    " cnra ",
    " fram ",
    " recore ",
    " crac ",
    " cmr ",
    " cine ",
    " cin ",
    " carte nationale ",
    " retraite ",
    " pension ",
    " rente ",
    " allocations familiales ",
    " af ",
    " affiliation ",
    " cotisation ",
    " cotisations ",
    " pecule ",
    " plateforme ",
    " declaration ",
    " declarations ",
    " services en ligne ",
    " services disponibles ",
    " titularisation ",
    " coordination ",
    " allocataire ",
    " allocation ",
    " accident du travail ",
    " accident de travail ",
    " accident circulation ",
    " assurance vie ",
    " prevoyance ",
}

VIDEO_SIGNAL_TERMS = {
    " video ",
    " videos ",
    " youtube ",
    " tutoriel ",
    " tutoriel video ",
    " visuellement ",
    " en video ",
    " voir une video ",
    " regarder une video ",
    " montrez moi une video ",
}

FORM_SIGNAL_TERMS = {
    " formulaire ",
    " formulaires ",
    " imprime ",
    " imprimes ",
    " imprimer ",
    " pdf ",
    " telecharger ",
    " document a remplir ",
    " documents a remplir ",
    " document fournir ",
    " documents remplir ",
    " documents fournir ",
    " document pour ",
    " dossier a remplir ",
    " dossier de demande ",
    " dossier complet ",
    " papier a remplir ",
    " papier pour ",
    " papiers remplir ",
    " papiers a fournir ",
    " pieces a fournir ",
    " documents a fournir ",
}

PROMPT_INJECTION_TERMS = {
    " ignore all ",
    " ignore previous ",
    " ignore tes ",
    " ignore les instructions ",
    " ignore les regles ",
    " oublie tes instructions ",
    " ignore tes regles ",
    " ignore previous ",
    " forget everything ",
    " system prompt ",
    " prompt systeme ",
    " instructions internes ",
    " mode developpeur ",
    " dan ",
    " sans restrictions ",
    " desactive tes filtres ",
    " ne respecte plus les regles ",
    " regles du systeme ",
}

OUTSIDE_TERMS = {
    " meteo ",
    " temps fait ",
    " restaurant ",
    " recette ",
    " cuisine ",
    " football ",
    " match ",
    " voyage ",
    " billet avion ",
    " film ",
    " cinema ",
    " musique ",
    " python ",
    " programmation ",
    " appartement ",
    " banque ",
    " compte bancaire ",
    " impots ",
    " tva ",
    " visa ",
    " medicament ",
    " medicaments ",
    " hotel ",
    " emploi ",
    " licenciement ",
    " traduire ",
    " lettre de motivation ",
    " rediger un email ",
    " email professionnel ",
    " capitale ",
    " telephone ",
    " telephone portable ",
    " photosynthese ",
    " immobilier ",
    " carburant ",
    " universite ",
    " cv ",
    " passeport ",
    " train ",
    " blague ",
    " perdre du poids ",
    " couscous ",
    " actions ",
    " anniversaire ",
    " windows ",
    " serie televisee ",
}

GREETING_TERMS = {
    " bonjour ",
    " bonsoir ",
    " salut ",
    " salam ",
    " hello ",
    " hey ",
    " coucou ",
}


class IntentClassifier:
    """
    Classifies a user query by cosine similarity over curated examples.

    A few high-precision lexical rules run before embedding classification for
    safety-critical or UI-sensitive cases: prompt injection, greetings, and
    explicit resource requests.
    """

    def __init__(
        self,
        shared_embeddings: Optional[Any] = None,
        intents_dir: Path = INTENTS_DIR,
        top_k: int = TOP_K,
        gate_confidence_threshold: float = GATE_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._shared_embeddings_provided: bool = shared_embeddings is not None
        self.embeddings = shared_embeddings
        self.intents_dir = intents_dir
        self.top_k = top_k
        self.gate_confidence_threshold = gate_confidence_threshold

        self.example_embeddings: Optional[np.ndarray] = None
        self.example_labels: List[str] = []
        self.example_texts: List[str] = []
        self.intent_tier_map: Dict[str, int] = {}
        self._loaded = False

        self._load_examples()

    @staticmethod
    def _normalize_query(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return f" {normalized.strip()} "

    @staticmethod
    def _contains_any(normalized: str, terms: set[str]) -> bool:
        return any(term in normalized for term in terms)

    def has_video_signal(self, query: str) -> bool:
        return self._contains_any(self._normalize_query(query), VIDEO_SIGNAL_TERMS)

    def has_form_signal(self, query: str) -> bool:
        return self._contains_any(self._normalize_query(query), FORM_SIGNAL_TERMS)

    def has_domain_signal(self, query: str) -> bool:
        return self._contains_any(self._normalize_query(query), DOMAIN_TERMS)

    def _rule_result(self, intent: str, confidence: float, reason: str) -> Dict[str, Any]:
        return {
            "intent": intent,
            "confidence": confidence,
            "tier": self.intent_tier_map.get(intent, INTENT_TIERS.get(intent, 1)),
            "reasoning": reason,
            "loaded": True,
        }

    def _classify_by_rules(self, query: str) -> Optional[Dict[str, Any]]:
        normalized = self._normalize_query(query)
        words = normalized.strip().split()

        if self._contains_any(normalized, PROMPT_INJECTION_TERMS):
            return self._rule_result("prompt_injection", 0.99, "rule: prompt injection phrase")

        if self.has_video_signal(query):
            return self._rule_result("needs_video", 0.98, "rule: explicit video request")

        if self.has_form_signal(query):
            return self._rule_result("needs_form", 0.98, "rule: explicit form/document request")

        if (
            len(words) <= 4
            and self._contains_any(normalized, GREETING_TERMS)
            and not self.has_domain_signal(query)
        ):
            return self._rule_result("greeting", 0.95, "rule: short greeting")

        if self._contains_any(normalized, OUTSIDE_TERMS) and not self.has_domain_signal(query):
            return self._rule_result("out_of_scope", 0.94, "rule: obvious outside-domain request")

        if self.has_domain_signal(query):
            return self._rule_result("retrieval", 0.96, "rule: known RCAR/CNRA domain term")

        return None

    def _ensure_embeddings(self) -> bool:
        if self.embeddings is not None:
            return True

        logger.warning(
            "IntentClassifier: aucun shared_embeddings fourni - chargement standalone de {}.",
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
            logger.info("IntentClassifier: embeddings standalone charges sur {}", device)
            return True
        except Exception as exc:
            logger.error("IntentClassifier: impossible de charger les embeddings: {}", exc)
            return False

    def _load_examples(self) -> None:
        start = time.perf_counter()

        if not self.intents_dir.exists():
            logger.warning("IntentClassifier: dossier intents introuvable: {}", self.intents_dir)
            return

        texts: List[str] = []
        labels: List[str] = []

        for intent_dir in sorted([d for d in self.intents_dir.iterdir() if d.is_dir()]):
            intent_name = intent_dir.name
            suggk_file = intent_dir / "suggk.txt"
            if not suggk_file.exists():
                logger.debug("IntentClassifier: pas de suggk.txt dans {}, ignore", intent_dir)
                continue

            if intent_name not in INTENT_TIERS:
                logger.warning("IntentClassifier: intent '{}' non supporte, ignore", intent_name)
                continue

            examples = [
                line.strip()
                for line in suggk_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not examples:
                logger.debug("IntentClassifier: suggk.txt vide pour {}, ignore", intent_name)
                continue

            texts.extend(examples)
            labels.extend([intent_name] * len(examples))
            self.intent_tier_map[intent_name] = INTENT_TIERS[intent_name]

        if not texts:
            logger.error("IntentClassifier: aucun exemple valide charge.")
            return

        if not self._ensure_embeddings():
            return

        try:
            raw_embeddings = self.embeddings.embed_documents(texts)
            embeddings_np = np.array(raw_embeddings, dtype=np.float32)
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
                "IntentClassifier pret: {} exemples, {} intents, {:.0f}ms",
                len(texts),
                len(self.intent_tier_map),
                elapsed,
            )
            for intent in sorted(counts):
                logger.debug("  {}: {} exemples", intent, counts[intent])
        except Exception as exc:
            logger.error("IntentClassifier: erreur calcul embeddings: {}", exc)

    def classify(self, query: str) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return self._fallback("Query vide")

        rule_based = self._classify_by_rules(query)
        if rule_based is not None:
            logger.info(
                "IntentClassifier rule: '{}' -> {} conf={:.3f}",
                query[:60],
                rule_based["intent"],
                rule_based["confidence"],
            )
            return rule_based

        if not self._loaded or self.example_embeddings is None:
            logger.warning("IntentClassifier: non charge, fallback retrieval")
            return self._fallback("Classifier non charge")

        start = time.perf_counter()
        try:
            query_embedding = np.array(self.embeddings.embed_query(query), dtype=np.float32)
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

            similarities = query_embedding @ self.example_embeddings.T
            k = min(self.top_k, len(similarities))
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

            top_labels = [self.example_labels[i] for i in top_k_indices]
            top_scores = [float(similarities[i]) for i in top_k_indices]
            vote_counts = Counter(top_labels)
            winner, winner_votes = vote_counts.most_common(1)[0]
            winner_scores = [
                top_scores[index]
                for index, label in enumerate(top_labels)
                if label == winner
            ]
            confidence = float(np.mean(winner_scores)) if winner_scores else 0.0
            original_winner = winner

            if winner in {"needs_video", "needs_form"}:
                winner = "retrieval"
            elif winner == "out_of_scope" and self.has_domain_signal(query):
                winner = "retrieval"
            elif winner in {"greeting", "negation"} and confidence < CLASSIFICATION_FALLBACK_THRESHOLD:
                winner = "retrieval"
            elif confidence < CLASSIFICATION_FALLBACK_THRESHOLD:
                winner = "retrieval"

            tier = self.intent_tier_map.get(winner, INTENT_TIERS.get(winner, 1))
            elapsed_ms = (time.perf_counter() - start) * 1000

            top3_str = ", ".join(
                f"{self.example_labels[int(i)]}({float(similarities[int(i)]):.3f})"
                for i in top_k_indices[:3]
            )
            if winner != original_winner:
                reasoning = (
                    f"top-{k}: [{top3_str}] -> {original_winner}; "
                    f"fallback -> {winner} (confidence={confidence:.3f}, votes={winner_votes}/{k})"
                )
            else:
                reasoning = (
                    f"top-{k}: [{top3_str}] -> {winner} "
                    f"(confidence={confidence:.3f}, votes={winner_votes}/{k})"
                )

            logger.info(
                "IntentClassifier: '{}' -> {} [tier {}] conf={:.3f} ({:.0f}ms)",
                query[:60],
                winner,
                tier,
                confidence,
                elapsed_ms,
            )

            return {
                "intent": winner,
                "confidence": confidence,
                "tier": tier,
                "reasoning": reasoning,
                "loaded": True,
            }
        except Exception as exc:
            logger.error("IntentClassifier: erreur classify: {}", exc)
            return self._fallback(str(exc))

    def should_retrieve_videos(self, classification: Dict[str, Any]) -> bool:
        if not classification.get("loaded", True):
            return False
        confidence = float(classification.get("confidence", 0.0) or 0.0)
        if confidence < self.gate_confidence_threshold:
            return False
        return classification.get("intent") in VIDEO_RETRIEVER_INTENTS

    def should_retrieve_forms(self, classification: Dict[str, Any]) -> bool:
        if not classification.get("loaded", True):
            return False
        confidence = float(classification.get("confidence", 0.0) or 0.0)
        if confidence < self.gate_confidence_threshold:
            return False
        return classification.get("intent") in FORM_RETRIEVER_INTENTS

    def _fallback(self, reason: str) -> Dict[str, Any]:
        return {
            "intent": "retrieval",
            "confidence": 0.0,
            "tier": INTENT_TIERS["retrieval"],
            "reasoning": f"Fallback: {reason}",
            "loaded": False,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_intents_by_tier(self) -> Dict[int, List[str]]:
        by_tier: Dict[int, List[str]] = {}
        for intent, tier in sorted(self.intent_tier_map.items()):
            by_tier.setdefault(tier, []).append(intent)
        return {tier: sorted(intents) for tier, intents in sorted(by_tier.items())}

    def get_stats(self) -> Dict[str, Any]:
        if not self._loaded:
            return {"loaded": False}
        counts = Counter(self.example_labels)
        return {
            "loaded": True,
            "total_examples": len(self.example_labels),
            "total_intents": len(self.intent_tier_map),
            "top_k": self.top_k,
            "gate_threshold": self.gate_confidence_threshold,
            "standalone_mode": not self._shared_embeddings_provided,
            "intents_by_tier": self.get_intents_by_tier(),
            "examples_per_intent": dict(counts),
        }
