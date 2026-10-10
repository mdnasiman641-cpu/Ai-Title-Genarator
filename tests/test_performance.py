"""Responsiveness with thousands of files (generous bounds - these catch pathological slowdowns)."""

from __future__ import annotations

import time

import pytest

from app.ai.schemas import NumberingAction, RemoveTextAction
from app.config.constants import FileKind
from app.files.operation_manager import OperationManager
from app.files.planner import Planner
from app.files.scanner import FileEntry, scan_folder
from app.ui.file_table_model import FileFilterProxy, FileTableModel


def fake_entries(n: int) -> list[FileEntry]:
    return [
        FileEntry(f"Show S01E{i:05d} 1080p WEB-DL.mp4", f"Show S01E{i:05d} 1080p WEB-DL.mp4", ".mp4", i * 1000, 1_700_000_000 + i, FileKind.VIDEO)
        for i in range(n)
    ]


@pytest.mark.gui
def test_table_model_and_proxy_handle_20000_rows(qapp):
    entries = fake_entries(20_000)
    model, proxy = FileTableModel(), FileFilterProxy()
    proxy.setSourceModel(model)
    start = time.perf_counter()
    model.set_entries(entries)
    model.sort_by(1)
    load = time.perf_counter() - start
    start = time.perf_counter()
    proxy.set_search("e0199")
    search = time.perf_counter() - start
    proxy.set_search("")
    start = time.perf_counter()
    model.sort_by(3, descending=True)  # by size
    model.set_checked_where(lambda e: True)
    bulk = time.perf_counter() - start
    assert proxy.rowCount() == 20_000 and model.checked_count() == 20_000
    assert model.entries[0].size == max(e.size for e in entries)  # descending by size
    assert load < 1.5 and search < 0.5 and bulk < 1.5, (load, search, bulk)  # was ~1.2 s *per sort* via the proxy


def test_planning_3000_files_is_fast(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    for i in range(3000):
        (root / f"Show S01E{i:04d} 1080p.mp4").write_bytes(b"")
    entries = scan_folder(root).entries
    start = time.perf_counter()
    plan = Planner(root, entries).build([RemoveTextAction(("1080p",)), NumberingAction(position="suffix")])
    elapsed = time.perf_counter() - start
    assert len(plan.ops) == 3000 and plan.counts()["applicable"] == 3000 and elapsed < 8, elapsed


def test_applying_and_undoing_2000_renames(tmp_path):
    from app.database.database import Database

    root = tmp_path / "w"
    root.mkdir()
    for i in range(2000):
        (root / f"v{i:04d}.mp4").write_bytes(b"")
    entries = scan_folder(root).entries
    mgr = OperationManager(Database(tmp_path / "p.db"))
    plan = Planner(root, entries).build([NumberingAction(position="replace", template="Episode {n}")])
    start = time.perf_counter()
    result = mgr.execute(plan)
    applied = time.perf_counter() - start
    start = time.perf_counter()
    undone = mgr.undo(result.operation_id)
    elapsed_undo = time.perf_counter() - start
    assert result.ok and undone.ok and applied < 15 and elapsed_undo < 15, (applied, elapsed_undo)
    assert (root / "v0000.mp4").exists() and len(list(root.iterdir())) == 2000
