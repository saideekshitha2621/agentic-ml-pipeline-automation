from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import DATABASE_URL

_IS_SQLITE = DATABASE_URL.startswith("sqlite")
engine = (
    create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    if _IS_SQLITE
    else create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=10)
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


_SQLITE_COLUMN_TYPES = {
    "VARCHAR": "VARCHAR",
    "INTEGER": "INTEGER",
    "FLOAT": "FLOAT",
    "BOOLEAN": "BOOLEAN",
    "JSON": "JSON",
    "DATETIME": "DATETIME",
    "TEXT": "TEXT",
}


def _ensure_columns():
    """`Base.metadata.create_all` only creates missing tables, it never alters existing
    ones — there's no Alembic in this project. Since model changes add columns to
    already-created tables on a developer's local app.db, patch them in with a plain
    `ALTER TABLE ... ADD COLUMN`, which SQLite supports for simple column additions."""
    from sqlalchemy import inspect

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in inspector.get_table_names():
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                col_type = _SQLITE_COLUMN_TYPES.get(column.type.__class__.__name__.upper(), "TEXT")
                if not _IS_SQLITE and col_type == "DATETIME":
                    col_type = "TIMESTAMP"  # PostgreSQL has no DATETIME type
                default_sql = ""
                if column.default is not None and getattr(column.default, "is_scalar", False):
                    val = column.default.arg
                    if isinstance(val, bool):
                        default_sql = f" DEFAULT {int(val)}"
                    elif isinstance(val, (int, float)):
                        default_sql = f" DEFAULT {val}"
                    elif isinstance(val, str):
                        default_sql = f" DEFAULT '{val}'"
                conn.exec_driver_sql(
                    f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}{default_sql}'
                )


def init_db():
    from app.db import models  # noqa: F401  (register models on Base.metadata)

    Base.metadata.create_all(bind=engine)
    _ensure_columns()
