from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.chatbot.intent_classifier import INTENTS_DIR, SUPPORTED_INTENTS

from ..models import IntentDefinition, IntentExample


INTENT_LABELS_FR = {
    "retrieval": "Recherche documentaire",
    "greeting": "Salutation",
    "out_of_scope": "Hors périmètre",
    "negation": "Négation",
    "prompt_injection": "Injection de prompt",
    "needs_form": "Demande de formulaire",
    "needs_video": "Demande de vidéo",
}

INTENT_DESCRIPTIONS = {
    "retrieval": "Question normale qui doit passer par la récupération documentaire.",
    "greeting": "Message court de salutation.",
    "out_of_scope": "Question hors domaine RCAR/CNRA.",
    "negation": "Réponse courte de refus ou négation.",
    "prompt_injection": "Tentative de contourner les consignes du système.",
    "needs_form": "Demande explicite de formulaire, PDF ou pièce à fournir.",
    "needs_video": "Demande explicite de vidéo ou tutoriel.",
}


def _read_seed_examples(intent_name: str, intents_dir: Path = INTENTS_DIR) -> list[str]:
    path = intents_dir / intent_name / "suggk.txt"
    if not path.exists():
        return []
    seen: set[str] = set()
    examples: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and text not in seen:
            seen.add(text)
            examples.append(text)
    return examples


def _read_disabled_seed_examples(intent_name: str, intents_dir: Path = INTENTS_DIR) -> list[str]:
    path = intents_dir / intent_name / "suggk.disabled.txt"
    if not path.exists():
        return []
    seen: set[str] = set()
    examples: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and text not in seen:
            seen.add(text)
            examples.append(text)
    return examples


def _read_intent_meta(intent_name: str, intents_dir: Path = INTENTS_DIR) -> dict[str, Any]:
    path = intents_dir / intent_name / "intent_meta.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


async def seed_intents_from_files(db: AsyncSession) -> None:
    for name in sorted(SUPPORTED_INTENTS):
        meta = _read_intent_meta(name)
        result = await db.execute(select(IntentDefinition).where(IntentDefinition.name == name))
        intent = result.scalar_one_or_none()
        if intent is None:
            intent = IntentDefinition(
                name=name,
                label_fr=INTENT_LABELS_FR.get(name, name),
                description=INTENT_DESCRIPTIONS.get(name),
                enabled=bool(meta.get("enabled", True)),
                builtin=True,
            )
            db.add(intent)
            await db.flush()
        else:
            intent.builtin = True
            if "enabled" in meta:
                intent.enabled = bool(meta["enabled"])
            if not intent.label_fr:
                intent.label_fr = INTENT_LABELS_FR.get(name, name)

        existing_result = await db.execute(select(IntentExample.text).where(IntentExample.intent_id == intent.id))
        existing = set(existing_result.scalars().all())
        for example in _read_seed_examples(name):
            if example not in existing:
                db.add(IntentExample(intent_id=intent.id, text=example, enabled=True, source="seed"))
        for example in _read_disabled_seed_examples(name):
            if example not in existing:
                db.add(IntentExample(intent_id=intent.id, text=example, enabled=False, source="seed"))

    await db.commit()


async def export_intents_to_files(db: AsyncSession, intents_dir: Path = INTENTS_DIR) -> None:
    result = await db.execute(
        select(IntentDefinition)
        .options(selectinload(IntentDefinition.examples))
        .where(IntentDefinition.name.in_(SUPPORTED_INTENTS))
        .order_by(IntentDefinition.name.asc())
    )
    for intent in result.scalars().all():
        intent_dir = intents_dir / intent.name
        intent_dir.mkdir(parents=True, exist_ok=True)

        active_examples = [
            example.text.strip()
            for example in sorted(intent.examples, key=lambda item: item.id)
            if intent.enabled and example.enabled and example.text.strip()
        ]
        disabled_examples = [
            example.text.strip()
            for example in sorted(intent.examples, key=lambda item: item.id)
            if (not intent.enabled or not example.enabled) and example.text.strip()
        ]

        (intent_dir / "suggk.txt").write_text(
            "\n".join(active_examples) + ("\n" if active_examples else ""),
            encoding="utf-8",
        )
        disabled_path = intent_dir / "suggk.disabled.txt"
        if disabled_examples:
            disabled_path.write_text("\n".join(disabled_examples) + "\n", encoding="utf-8")
        elif disabled_path.exists():
            disabled_path.unlink()

        (intent_dir / "intent_meta.json").write_text(
            json.dumps(
                {
                    "enabled": intent.enabled,
                    "label_fr": intent.label_fr,
                    "exported_from": "postgresql_admin",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


async def load_active_examples_by_intent(db: AsyncSession) -> dict[str, list[str]]:
    result = await db.execute(
        select(IntentDefinition)
        .options(selectinload(IntentDefinition.examples))
        .where(IntentDefinition.enabled.is_(True))
    )
    examples_by_intent: dict[str, list[str]] = {}
    for intent in result.scalars().all():
        if intent.name not in SUPPORTED_INTENTS:
            continue
        examples = [example.text for example in intent.examples if example.enabled and example.text.strip()]
        if examples:
            examples_by_intent[intent.name] = examples
    return examples_by_intent


def intent_to_dict(intent: IntentDefinition, examples_count: int | None = None, active_examples_count: int | None = None) -> dict[str, Any]:
    examples = list(getattr(intent, "examples", []) or [])
    total = len(examples) if examples_count is None else examples_count
    active = sum(1 for example in examples if example.enabled) if active_examples_count is None else active_examples_count
    if not intent.enabled:
        active = 0
    return {
        "id": str(intent.id),
        "name": intent.name,
        "label_fr": intent.label_fr,
        "description": intent.description,
        "enabled": intent.enabled,
        "builtin": intent.builtin,
        "supported": intent.name in SUPPORTED_INTENTS,
        "examples": total,
        "active_examples": active,
        "created_at": intent.created_at.isoformat(),
        "updated_at": intent.updated_at.isoformat(),
    }


def example_to_dict(example: IntentExample) -> dict[str, Any]:
    return {
        "id": str(example.id),
        "intent_id": str(example.intent_id),
        "text": example.text,
        "enabled": example.enabled,
        "source": example.source,
        "created_at": example.created_at.isoformat(),
        "updated_at": example.updated_at.isoformat(),
    }


async def count_intent_examples(db: AsyncSession) -> dict[str, int]:
    total_result = await db.execute(select(func.count()).select_from(IntentExample))
    active_result = await db.execute(
        select(func.count())
        .select_from(IntentExample)
        .join(IntentDefinition, IntentExample.intent_id == IntentDefinition.id)
        .where(IntentExample.enabled.is_(True), IntentDefinition.enabled.is_(True))
    )
    return {"total": int(total_result.scalar_one()), "active": int(active_result.scalar_one())}
