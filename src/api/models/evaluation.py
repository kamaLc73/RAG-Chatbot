from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (UniqueConstraint("run_dir", name="uq_evaluation_run_dir"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_dir: Mapped[str] = mapped_column(Text, nullable=False)
    run_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    task: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    backend: Mapped[str] = mapped_column(String(30), default="unknown", nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(30), default="completed", nullable=False, index=True)
    dataset: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    commit: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at_label: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    file_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
