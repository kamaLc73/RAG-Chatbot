from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class KbSource(Base):
    __tablename__ = "kb_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    org: Mapped[str] = mapped_column(String(20), default="both", nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(80), default="admin_upload", nullable=False)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="indexed", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    items = relationship("KbSourceItem", back_populates="source", cascade="all, delete-orphan")


class KbSourceItem(Base):
    __tablename__ = "kb_source_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("kb_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    schema: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    vespa_id: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    source = relationship("KbSource", back_populates="items")
