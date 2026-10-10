"""SQLite access layer: schema, thread-safe connections and typed queries.

The database is a single local file (no server). Every thread gets its own
connection (SQLite connections must not be shared across threads) and WAL mode
keeps the UI responsive while worker threads write.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config.constants import MAX_HISTORY_ITEMS
from app.database.models import (
    CommandRecord,
    OperationItemRecord,
    OperationRecord,
    SavedPrompt,
)
from app.utils.helpers import get_app_data_dir, now_iso
from app.utils.logger import get_logger

log = get_logger("db")

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS saved_prompts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    prompt     TEXT NOT NULL,
    provider   TEXT NOT NULL DEFAULT 'auto',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS command_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    command    TEXT NOT NULL,
    provider   TEXT NOT NULL DEFAULT 'auto',
    favorite   INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS operations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    workspace   TEXT NOT NULL,
    command     TEXT NOT NULL DEFAULT '',
    provider    TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL,
    summary     TEXT NOT NULL DEFAULT '',
    undone_at   TEXT
);
CREATE TABLE IF NOT EXISTS operation_items (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id INTEGER NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    type         TEXT NOT NULL,
    source       TEXT,
    target       TEXT,
    temp_path    TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    extra        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_items_operation ON operation_items(operation_id, seq);
CREATE TABLE IF NOT EXISTS model_configs (
    id       TEXT PRIMARY KEY,
    priority INTEGER NOT NULL,
    data     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_health (
    model_id TEXT PRIMARY KEY,
    data     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS title_presets (
    name       TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search_cache (
    query   TEXT PRIMARY KEY,
    created REAL NOT NULL,
    data    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metadata_cache (
    path     TEXT NOT NULL,
    size     INTEGER NOT NULL,
    mtime    REAL NOT NULL,
    payload  TEXT NOT NULL,
    PRIMARY KEY (path, size, mtime)
);
"""


class Database:
    """Thread-safe wrapper around one SQLite file."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else get_app_data_dir() / "app.db"
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialised = False
        self.initialise()

    # ------------------------------------------------------------ connection
    def _connection(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 15000")
            self._local.conn = conn
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside ``BEGIN IMMEDIATE ... COMMIT`` (rolled back on error)."""
        conn = self._connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def close(self) -> None:
        """Close the calling thread's connection."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def initialise(self) -> None:
        """Create tables if needed."""
        with self._init_lock:
            if self._initialised:
                return
            conn = self._connection()
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.DatabaseError:  # e.g. network drive: stay on default journal
                log.warning("WAL journal mode unavailable; using default")
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._initialised = True

    # -------------------------------------------------------------- settings
    def get_raw(self, key: str) -> str | None:
        row = self._connection().execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_raw(self, key: str, value: str) -> None:
        self._connection().execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def delete_raw(self, key: str) -> None:
        self._connection().execute("DELETE FROM settings WHERE key = ?", (key,))

    def all_settings(self, prefix_exclude: str = "secret.") -> dict[str, str]:
        rows = self._connection().execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows if not r["key"].startswith(prefix_exclude)}

    # --------------------------------------------------------- saved prompts
    def list_saved_prompts(self) -> list[SavedPrompt]:
        rows = self._connection().execute("SELECT * FROM saved_prompts ORDER BY name COLLATE NOCASE").fetchall()
        return [SavedPrompt(r["id"], r["name"], r["prompt"], r["provider"], r["created_at"]) for r in rows]

    def add_saved_prompt(self, name: str, prompt: str, provider: str = "auto") -> int:
        cur = self._connection().execute(
            "INSERT INTO saved_prompts(name, prompt, provider, created_at) VALUES(?,?,?,?)",
            (name.strip(), prompt.strip(), provider, now_iso()),
        )
        return int(cur.lastrowid or 0)

    def update_saved_prompt(self, prompt_id: int, name: str, prompt: str, provider: str) -> None:
        self._connection().execute(
            "UPDATE saved_prompts SET name = ?, prompt = ?, provider = ? WHERE id = ?",
            (name.strip(), prompt.strip(), provider, prompt_id),
        )

    def delete_saved_prompt(self, prompt_id: int) -> None:
        self._connection().execute("DELETE FROM saved_prompts WHERE id = ?", (prompt_id,))

    def count_saved_prompts(self) -> int:
        return int(self._connection().execute("SELECT COUNT(*) AS n FROM saved_prompts").fetchone()["n"])

    # ------------------------------------------------------- command history
    def add_command(self, command: str, provider: str) -> int:
        """Record a command, moving an identical earlier entry to the top instead of duplicating."""
        command = command.strip()
        conn = self._connection()
        existing = conn.execute(
            "SELECT id, favorite FROM command_history WHERE command = ?", (command,)
        ).fetchone()
        favorite = 0
        if existing is not None:
            favorite = int(existing["favorite"])
            conn.execute("DELETE FROM command_history WHERE id = ?", (existing["id"],))
        cur = conn.execute(
            "INSERT INTO command_history(command, provider, favorite, created_at) VALUES(?,?,?,?)",
            (command, provider, favorite, now_iso()),
        )
        # Keep history bounded, but never drop favourites.
        conn.execute(
            "DELETE FROM command_history WHERE favorite = 0 AND id NOT IN "
            "(SELECT id FROM command_history WHERE favorite = 0 ORDER BY id DESC LIMIT ?)",
            (MAX_HISTORY_ITEMS,),
        )
        return int(cur.lastrowid or 0)

    def list_commands(self, limit: int = 200) -> list[CommandRecord]:
        rows = self._connection().execute(
            "SELECT * FROM command_history ORDER BY favorite DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [CommandRecord(r["id"], r["command"], r["provider"], bool(r["favorite"]), r["created_at"]) for r in rows]

    def set_command_favorite(self, command_id: int, favorite: bool) -> None:
        self._connection().execute(
            "UPDATE command_history SET favorite = ? WHERE id = ?", (1 if favorite else 0, command_id)
        )

    def delete_command(self, command_id: int) -> None:
        self._connection().execute("DELETE FROM command_history WHERE id = ?", (command_id,))

    def clear_commands(self, keep_favorites: bool = True) -> None:
        sql = "DELETE FROM command_history WHERE favorite = 0" if keep_favorites else "DELETE FROM command_history"
        self._connection().execute(sql)

    # ------------------------------------------------------------ operations
    def create_operation(
        self,
        workspace: str,
        command: str,
        provider: str,
        summary: str,
        items: Iterable[OperationItemRecord],
    ) -> int:
        """Insert an operation with all its items (status ``running``) in one transaction."""
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO operations(created_at, workspace, command, provider, status, summary) "
                "VALUES(?,?,?,?,?,?)",
                (now_iso(), workspace, command, provider, "running", summary),
            )
            op_id = int(cur.lastrowid or 0)
            conn.executemany(
                "INSERT INTO operation_items(operation_id, seq, type, source, target, temp_path, status, extra) "
                "VALUES(?,?,?,?,?,?,?,?)",
                [
                    (op_id, it.seq, it.type, it.source, it.target, it.temp_path, it.status, json.dumps(it.extra))
                    for it in items
                ],
            )
        return op_id

    def set_operation_status(self, op_id: int, status: str, *, undone: bool = False) -> None:
        conn = self._connection()
        if undone:
            conn.execute("UPDATE operations SET status = ?, undone_at = ? WHERE id = ?", (status, now_iso(), op_id))
        else:
            conn.execute("UPDATE operations SET status = ? WHERE id = ?", (status, op_id))

    def set_operation_summary(self, op_id: int, summary: str) -> None:
        self._connection().execute("UPDATE operations SET summary = ? WHERE id = ?", (summary, op_id))

    def update_item_statuses(self, updates: Iterable[tuple[int, str]]) -> None:
        """Bulk-update ``(item_id, status)`` pairs in a single transaction."""
        pairs = [(status, item_id) for item_id, status in updates]
        if not pairs:
            return
        with self.transaction() as conn:
            conn.executemany("UPDATE operation_items SET status = ? WHERE id = ?", pairs)

    def get_operation(self, op_id: int) -> OperationRecord | None:
        row = self._connection().execute("SELECT * FROM operations WHERE id = ?", (op_id,)).fetchone()
        return None if row is None else self._operation_from_row(row)

    def list_operations(self, limit: int = 100) -> list[OperationRecord]:
        rows = self._connection().execute("SELECT * FROM operations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._operation_from_row(r) for r in rows]

    def last_undoable_operation(self) -> OperationRecord | None:
        row = self._connection().execute(
            "SELECT * FROM operations WHERE status IN ('applied','partial','interrupted') "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else self._operation_from_row(row)

    def get_items(self, op_id: int) -> list[OperationItemRecord]:
        rows = self._connection().execute(
            "SELECT * FROM operation_items WHERE operation_id = ? ORDER BY seq", (op_id,)
        ).fetchall()
        return [
            OperationItemRecord(
                seq=r["seq"], type=r["type"], source=r["source"], target=r["target"],
                temp_path=r["temp_path"], status=r["status"], extra=json.loads(r["extra"] or "{}"), id=r["id"],
            )
            for r in rows
        ]  # fmt: skip

    def delete_operation(self, op_id: int) -> None:
        self._connection().execute("DELETE FROM operations WHERE id = ?", (op_id,))

    def clear_operations(self) -> None:
        self._connection().execute("DELETE FROM operations")

    def mark_interrupted_operations(self) -> int:
        """Operations still ``running`` at startup were interrupted (crash / power loss)."""
        cur = self._connection().execute("UPDATE operations SET status = 'interrupted' WHERE status = 'running'")
        return cur.rowcount

    @staticmethod
    def _operation_from_row(row: sqlite3.Row) -> OperationRecord:
        return OperationRecord(
            id=row["id"], created_at=row["created_at"], workspace=row["workspace"], command=row["command"],
            provider=row["provider"], status=row["status"], summary=row["summary"], undone_at=row["undone_at"],
        )  # fmt: skip

    # --------------------------------------------------------- model registry
    def list_model_rows(self) -> list[dict[str, object]]:
        rows = self._connection().execute("SELECT data FROM model_configs ORDER BY priority, rowid").fetchall()
        return [json.loads(r["data"]) for r in rows]

    def replace_model_rows(self, rows: list[tuple[str, int, dict[str, object]]]) -> None:
        """Atomically store the whole registry (``(id, priority, data)`` rows)."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM model_configs")
            conn.executemany(
                "INSERT INTO model_configs(id, priority, data) VALUES(?,?,?)",
                [(model_id, priority, json.dumps(data, ensure_ascii=False)) for model_id, priority, data in rows],
            )

    def get_health_rows(self) -> dict[str, dict[str, object]]:
        rows = self._connection().execute("SELECT model_id, data FROM model_health").fetchall()
        return {r["model_id"]: json.loads(r["data"]) for r in rows}

    def put_health_row(self, model_id: str, data: dict[str, object]) -> None:
        self._connection().execute(
            "INSERT OR REPLACE INTO model_health(model_id, data) VALUES(?,?)", (model_id, json.dumps(data))
        )

    def delete_health_row(self, model_id: str | None = None) -> None:
        if model_id is None:
            self._connection().execute("DELETE FROM model_health")
        else:
            self._connection().execute("DELETE FROM model_health WHERE model_id = ?", (model_id,))

    # -------------------------------------------------------- metadata cache
    def get_cached_metadata(self, path: str, size: int, mtime: float) -> dict[str, object] | None:
        row = self._connection().execute(
            "SELECT payload FROM metadata_cache WHERE path = ? AND size = ? AND mtime = ?", (path, size, mtime)
        ).fetchone()
        return None if row is None else json.loads(row["payload"])

    def put_cached_metadata(self, path: str, size: int, mtime: float, payload: dict[str, object]) -> None:
        self._connection().execute(
            "INSERT OR REPLACE INTO metadata_cache(path, size, mtime, payload) VALUES(?,?,?,?)",
            (path, size, mtime, json.dumps(payload)),
        )

    # ---------------------------------------------------------- title presets
    def list_title_presets(self) -> list[str]:
        rows = self._connection().execute("SELECT name FROM title_presets ORDER BY name COLLATE NOCASE").fetchall()
        return [str(r["name"]) for r in rows]

    def get_title_preset(self, name: str) -> dict[str, object] | None:
        row = self._connection().execute("SELECT data FROM title_presets WHERE name = ?", (name,)).fetchone()
        return None if row is None else json.loads(row["data"])

    def save_title_preset(self, name: str, data: dict[str, object]) -> None:
        self._connection().execute(
            "INSERT INTO title_presets(name, data, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
            (name.strip(), json.dumps(data, ensure_ascii=False), now_iso()),
        )

    def delete_title_preset(self, name: str) -> None:
        self._connection().execute("DELETE FROM title_presets WHERE name = ?", (name,))

    # ----------------------------------------------------------- search cache
    def get_search_cache(self, query: str, min_created: float) -> dict[str, object] | None:
        row = self._connection().execute(
            "SELECT data FROM search_cache WHERE query = ? AND created >= ?", (query, min_created)
        ).fetchone()
        return None if row is None else json.loads(row["data"])

    def put_search_cache(self, query: str, created: float, data: dict[str, object]) -> None:
        self._connection().execute(
            "INSERT INTO search_cache(query, created, data) VALUES(?,?,?) "
            "ON CONFLICT(query) DO UPDATE SET created = excluded.created, data = excluded.data",
            (query, created, json.dumps(data, ensure_ascii=False)),
        )

    def purge_search_cache(self, older_than: float | None = None) -> int:
        """Delete expired entries (or everything when ``older_than`` is None); returns the count."""
        if older_than is None:
            cur = self._connection().execute("DELETE FROM search_cache")
        else:
            cur = self._connection().execute("DELETE FROM search_cache WHERE created < ?", (older_than,))
        return cur.rowcount or 0

    def count_search_cache(self) -> int:
        return int(self._connection().execute("SELECT COUNT(*) FROM search_cache").fetchone()[0])

