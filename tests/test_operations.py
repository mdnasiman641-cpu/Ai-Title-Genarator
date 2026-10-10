"""Execution, two-phase renames, rollback and undo - against the real file system."""

from __future__ import annotations

import errno
import os
import threading
from pathlib import Path

import pytest

from app.ai.schemas import (
    AddPrefixAction,
    CopyAction,
    CreateFolderAction,
    DeleteAction,
    MoveAction,
    NumberingAction,
    RemoveTextAction,
    RenameAction,
)
from app.config.constants import TEMP_PREFIX, TRASH_DIR_NAME
from app.database.database import Database
from app.files import move_engine
from app.files.move_engine import StepLog
from app.files.operation_manager import OperationManager, summarize_kinds
from app.files.plan import OpStatus
from app.files.planner import Planner, PlannerOptions
from app.files.rename_engine import needs_two_phase, relocate_batch
from app.files.scanner import scan_folder


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "ops.db")


@pytest.fixture
def mgr(db):
    return OperationManager(db)


def plan_for(workspace, actions, *, only=None, recursive=False, **opts):
    entries = scan_folder(workspace, recursive=recursive).entries
    if only is not None:
        entries = [e for e in entries if e.rel_path in only]
    return Planner(workspace, entries, PlannerOptions(**opts)).build(actions)


def snapshot(workspace: Path) -> dict[str, str]:
    """relative path -> content, ignoring internal trash bookkeeping."""
    out = {}
    for path in workspace.rglob("*"):
        if path.is_file() and TRASH_DIR_NAME not in path.parts:
            out[path.relative_to(workspace).as_posix()] = path.read_text(encoding="utf-8")
    return out


def listing(workspace: Path) -> list[str]:
    return sorted(snapshot(workspace))


# ------------------------------------------------------------------ basics
def test_rename_applies_and_keeps_content(workspace, make_files, mgr):
    make_files("a.mp4", "b.mp4", content="C:")
    before = snapshot(workspace)
    plan = plan_for(workspace, [AddPrefixAction("X - ")])
    result = mgr.execute(plan, command="prefix", provider="offline")
    assert result.ok and result.applied == 2 and result.summary == "2 rename operations"
    after = snapshot(workspace)
    assert after == {f"X - {k}": v for k, v in before.items()}  # same bytes under new names
    assert result.changes == {"X - a.mp4": "Renamed", "X - b.mp4": "Renamed"}


def test_spec_sample_scenario_end_to_end(workspace, make_files, mgr):
    names = [f"VID_00{i}_FINAL_1080P_WEB-DL.mp4" for i in (1, 2, 3)]
    make_files(*names)
    plan = plan_for(workspace, [NumberingAction(start=1, width=2, position="replace", template="Blue Bloods - Episode {n}")])
    assert mgr.execute(plan).ok
    assert listing(workspace) == [f"Blue Bloods - Episode 0{i}.mp4" for i in (1, 2, 3)]
    assert snapshot(workspace)["Blue Bloods - Episode 02.mp4"].endswith("VID_002_FINAL_1080P_WEB-DL.mp4")


def test_nothing_to_do(workspace, make_files, mgr):
    make_files("a.mp4")
    result = mgr.execute(plan_for(workspace, [RemoveTextAction(("zzz",))]))
    assert result.ok and result.applied == 0 and result.operation_id is None


def test_conflicting_ops_are_skipped_valid_ones_applied(workspace, make_files, mgr):
    make_files("a_1.mp4", "a_2.mp4", "z.mp4")
    plan = plan_for(workspace, [RemoveTextAction(("_1", "_2"))])
    result = mgr.execute(plan)
    assert result.ok and result.applied == 1
    assert "a.mp4" in listing(workspace) and len([n for n in listing(workspace) if n.startswith("a_")]) == 1


# --------------------------------------------------------- two-phase renames
def test_chain_rename_uses_temp_names_and_loses_nothing(workspace, make_files, mgr):
    make_files("A.mp4", "B.mp4", content="@")
    plan = plan_for(workspace, [RenameAction("A.mp4", "B.mp4"), RenameAction("B.mp4", "C.mp4")])
    assert mgr.execute(plan).ok
    snap = snapshot(workspace)
    assert sorted(snap) == ["B.mp4", "C.mp4"]
    assert snap["B.mp4"] == "@A.mp4" and snap["C.mp4"] == "@B.mp4"  # A's data is now B, B's data is now C
    assert not [n for n in os.listdir(workspace) if n.startswith(TEMP_PREFIX)]  # no temp files left behind


def test_swap_names(workspace, make_files, mgr):
    make_files("A.mp4", "B.mp4", content="@")
    assert mgr.execute(plan_for(workspace, [RenameAction("A.mp4", "B.mp4"), RenameAction("B.mp4", "A.mp4")])).ok
    snap = snapshot(workspace)
    assert snap == {"A.mp4": "@B.mp4", "B.mp4": "@A.mp4"}


def test_shift_numbering_in_place(workspace, make_files, mgr):
    """Renumbering 01..05 -> 02..06 style shifts would collide naively."""
    make_files(*(f"Ep {i:02d}.mp4" for i in range(1, 6)), content="#")
    actions = [RenameAction(f"Ep {i:02d}.mp4", f"Ep {i + 1:02d}.mp4") for i in range(1, 6)]
    assert mgr.execute(plan_for(workspace, actions)).ok
    snap = snapshot(workspace)
    assert sorted(snap) == [f"Ep {i:02d}.mp4" for i in range(2, 7)]
    assert all(snap[f"Ep {i + 1:02d}.mp4"] == f"#Ep {i:02d}.mp4" for i in range(1, 6))


def test_needs_two_phase_detection(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    assert not needs_two_phase([(a, b)])
    assert not needs_two_phase([(a, b), (c, tmp_path / "d")])
    assert needs_two_phase([(a, b), (b, c)])
    assert needs_two_phase([(a, b), (b, a)])


def test_relocate_batch_direct_mode_has_no_temp_files(tmp_path):
    (tmp_path / "a").write_text("1")
    steps = StepLog()
    used = relocate_batch([(tmp_path / "a", tmp_path / "b")], steps)
    assert used is False and len(steps.moves) == 1


def test_relocate_batch_two_phase_records_all_steps(tmp_path):
    (tmp_path / "a").write_text("A")
    (tmp_path / "b").write_text("B")
    steps, temps = StepLog(), []
    used = relocate_batch([(tmp_path / "a", tmp_path / "b"), (tmp_path / "b", tmp_path / "c")], steps, token="t0k", temp_names=temps)
    assert used is True and len(steps.moves) == 4
    assert temps == [f"{TEMP_PREFIX}t0k_001", f"{TEMP_PREFIX}t0k_002"]
    assert (tmp_path / "b").read_text() == "A" and (tmp_path / "c").read_text() == "B"
    assert move_engine.rollback(steps) == []  # reversing the 4 primitive steps restores the original state
    assert (tmp_path / "a").read_text() == "A" and (tmp_path / "b").read_text() == "B" and not (tmp_path / "c").exists()


def test_case_only_rename_on_a_real_fs(workspace, make_files, mgr):
    make_files("episode.mp4", content="e")
    plan = plan_for(workspace, [RenameAction("episode.mp4", "Episode.mp4")], case_insensitive=True)
    assert mgr.execute(plan).ok
    assert listing(workspace) == ["Episode.mp4"]


# ------------------------------------------------------------------- undo
def test_undo_restores_original_names(workspace, make_files, mgr, db):
    make_files("a.mp4", "b.mp4", content="u")
    before = snapshot(workspace)
    result = mgr.execute(plan_for(workspace, [AddPrefixAction("X ")]), command="c", provider="gemini")
    assert snapshot(workspace) != before
    record = db.get_operation(result.operation_id)
    assert record.status == "applied" and record.command == "c" and record.provider == "gemini" and record.can_undo
    check = mgr.check_undo(result.operation_id)
    assert check.possible
    undone = mgr.undo(result.operation_id)
    assert undone.ok and snapshot(workspace) == before
    assert db.get_operation(result.operation_id).status == "undone" and not db.get_operation(result.operation_id).can_undo
    assert all(it.status == "undone" for it in db.get_items(result.operation_id))


def test_undo_chain_and_swap(workspace, make_files, mgr):
    make_files("A.mp4", "B.mp4", "C.mp4", content="@")
    before = snapshot(workspace)
    actions = [RenameAction("A.mp4", "B.mp4"), RenameAction("B.mp4", "C.mp4"), RenameAction("C.mp4", "A.mp4")]  # 3-cycle
    result = mgr.execute(plan_for(workspace, actions))
    assert result.ok and snapshot(workspace) != before
    assert mgr.undo(result.operation_id).ok
    assert snapshot(workspace) == before


def test_undo_blocked_when_original_name_is_taken_and_nothing_changes(workspace, make_files, mgr):
    make_files("a.mp4", content="u")
    result = mgr.execute(plan_for(workspace, [RenameAction("a.mp4", "b.mp4")]))
    (workspace / "a.mp4").write_text("someone made a new file named a.mp4")
    before = snapshot(workspace)
    check = mgr.check_undo(result.operation_id)
    assert not check.possible and "already exists" in check.blockers[0]
    outcome = mgr.undo(result.operation_id)
    assert not outcome.ok and outcome.blockers and snapshot(workspace) == before  # nothing was touched


def test_undo_blocked_when_renamed_file_is_gone(workspace, make_files, mgr):
    make_files("a.mp4")
    result = mgr.execute(plan_for(workspace, [RenameAction("a.mp4", "b.mp4")]))
    (workspace / "b.mp4").unlink()
    outcome = mgr.undo(result.operation_id)
    assert not outcome.ok and "missing" in outcome.blockers[0]


def test_undo_twice_is_not_offered(workspace, make_files, mgr, db):
    make_files("a.mp4")
    result = mgr.execute(plan_for(workspace, [RenameAction("a.mp4", "b.mp4")]))
    assert mgr.undo(result.operation_id).ok
    assert db.last_undoable_operation() is None


def test_last_undoable_operation_is_the_newest(workspace, make_files, mgr, db):
    make_files("a.mp4")
    first = mgr.execute(plan_for(workspace, [RenameAction("a.mp4", "b.mp4")]))
    second = mgr.execute(plan_for(workspace, [RenameAction("b.mp4", "c.mp4")]))
    assert db.last_undoable_operation().id == second.operation_id
    assert mgr.undo(second.operation_id).ok and db.last_undoable_operation().id == first.operation_id


def test_undoing_an_older_operation_first_is_blocked_with_an_explanation(workspace, make_files, mgr):
    make_files("a.mp4")
    first = mgr.execute(plan_for(workspace, [RenameAction("a.mp4", "b.mp4")]))
    mgr.execute(plan_for(workspace, [RenameAction("b.mp4", "c.mp4")]))
    outcome = mgr.undo(first.operation_id)
    assert not outcome.ok and "missing" in outcome.blockers[0]


# ----------------------------------------------------- failure and rollback
def test_failure_midway_rolls_everything_back(workspace, make_files, mgr, monkeypatch, db):
    make_files("a.mp4", "b.mp4", "c.mp4", content="r")
    before = snapshot(workspace)
    real_rename = os.rename
    calls = {"n": 0}

    def flaky(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:  # the second rename fails (e.g. file locked)
            raise PermissionError(13, "locked")
        return real_rename(src, dst, *a, **k)

    monkeypatch.setattr(os, "rename", flaky)
    result = mgr.execute(plan_for(workspace, [AddPrefixAction("X ")]))
    monkeypatch.setattr(os, "rename", real_rename)
    assert not result.ok and result.rolled_back and "Permission denied" in result.message
    assert snapshot(workspace) == before  # all-or-nothing
    assert db.get_operation(result.operation_id).status == "rolled_back"


def test_failure_in_two_phase_second_stage_restores_all(workspace, make_files, mgr, monkeypatch):
    make_files("A.mp4", "B.mp4", content="@")
    before = snapshot(workspace)
    real_rename = os.rename
    calls = {"n": 0}

    def flaky(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 4:  # last step of phase two
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_rename(src, dst, *a, **k)

    monkeypatch.setattr(os, "rename", flaky)
    result = mgr.execute(plan_for(workspace, [RenameAction("A.mp4", "B.mp4"), RenameAction("B.mp4", "A.mp4")]))
    monkeypatch.setattr(os, "rename", real_rename)
    assert not result.ok and "disk is full" in result.message.lower()
    assert snapshot(workspace) == before
    assert not [n for n in os.listdir(workspace) if n.startswith(TEMP_PREFIX)]


def test_cancel_rolls_back(workspace, make_files, mgr):
    make_files(*(f"f{i}.mp4" for i in range(20)))
    before = snapshot(workspace)
    cancel = threading.Event()

    def progress(done, total, _msg):
        if done == 5:
            cancel.set()

    result = mgr.execute(plan_for(workspace, [AddPrefixAction("X ")]), cancel=cancel, progress=progress)
    assert not result.ok and "Cancelled" in result.message and snapshot(workspace) == before


def test_file_deleted_after_preview_aborts_cleanly(workspace, make_files, mgr):
    paths = make_files("a.mp4", "b.mp4")
    plan = plan_for(workspace, [AddPrefixAction("X ")])
    paths[1].unlink()
    before = snapshot(workspace)
    result = mgr.execute(plan)
    assert not result.ok and "folder changed" in result.message and snapshot(workspace) == before


def test_target_appearing_after_preview_aborts_cleanly(workspace, make_files, mgr):
    make_files("a.mp4")
    plan = plan_for(workspace, [RenameAction("a.mp4", "b.mp4")])
    (workspace / "b.mp4").write_text("surprise")
    result = mgr.execute(plan)
    assert not result.ok and snapshot(workspace)["b.mp4"] == "surprise" and "a.mp4" in snapshot(workspace)


def test_progress_reports_processing_x_of_y(workspace, make_files, mgr):
    make_files(*(f"f{i}.mp4" for i in range(6)))
    seen = []
    mgr.execute(plan_for(workspace, [AddPrefixAction("X ")]), progress=lambda d, t, m: seen.append((d, t, m)))
    assert seen[-1][:2] == (6, 6) and seen[0][2].startswith("Processing 1 / 6")


# ---------------------------------------------------------- folders / copies
def test_move_into_new_folder_and_undo_removes_only_that_folder(workspace, make_files, mgr):
    make_files("S02E01.mp4", "S01E01.mp4", content="m")
    (workspace / "Existing").mkdir()
    before = snapshot(workspace)
    result = mgr.execute(plan_for(workspace, [MoveAction(target_folder="Season 2", match=("S02",))]))
    assert result.ok and result.applied == 1  # the auto-created folder is not a "change"
    assert listing(workspace) == ["S01E01.mp4", "Season 2/S02E01.mp4"]
    assert mgr.undo(result.operation_id).ok
    assert snapshot(workspace) == before
    assert not (workspace / "Season 2").exists() and (workspace / "Existing").is_dir()


def test_undo_keeps_preexisting_empty_parent_folders(workspace, make_files, mgr):
    make_files("a.mp4")
    (workspace / "Parent").mkdir()  # already there (and empty)
    result = mgr.execute(plan_for(workspace, [MoveAction(source="a.mp4", target_folder="Parent/Child")]))
    assert (workspace / "Parent" / "Child" / "a.mp4").exists()
    assert mgr.undo(result.operation_id).ok
    assert (workspace / "Parent").is_dir() and not (workspace / "Parent" / "Child").exists()


def test_create_folder_only_action(workspace, make_files, mgr):
    make_files("a.mp4")
    result = mgr.execute(plan_for(workspace, [CreateFolderAction("Sorted")]))
    assert result.ok and (workspace / "Sorted").is_dir()
    assert mgr.undo(result.operation_id).ok and not (workspace / "Sorted").exists()


def test_undo_keeps_a_folder_that_is_no_longer_empty(workspace, make_files, mgr):
    make_files("a.mp4")
    result = mgr.execute(plan_for(workspace, [CreateFolderAction("Sorted")]))
    (workspace / "Sorted" / "mine.txt").write_text("keep me")
    outcome = mgr.undo(result.operation_id)
    assert outcome.ok and (workspace / "Sorted" / "mine.txt").exists()
    assert any("not empty" in n for n in outcome.notes)


def test_copy_and_undo(workspace, make_files, mgr):
    make_files("a.mp4", content="c")
    result = mgr.execute(plan_for(workspace, [CopyAction(source="a.mp4", target_folder="Backup")]))
    assert result.ok and snapshot(workspace) == {"a.mp4": "ca.mp4", "Backup/a.mp4": "ca.mp4"}
    assert mgr.undo(result.operation_id).ok
    assert listing(workspace) == ["a.mp4"] and not (workspace / "Backup").exists()


def test_undo_refuses_to_delete_a_modified_copy(workspace, make_files, mgr):
    make_files("a.mp4", content="c")
    result = mgr.execute(plan_for(workspace, [CopyAction(source="a.mp4", target_folder="Backup")]))
    (workspace / "Backup" / "a.mp4").write_text("I edited the backup, it is longer now")
    outcome = mgr.undo(result.operation_id)
    assert not outcome.ok and "modified" in outcome.blockers[0]
    assert (workspace / "Backup" / "a.mp4").exists()


def test_copy_failure_removes_partial_copy_and_rolls_back(workspace, make_files, mgr, monkeypatch):
    make_files("a.mp4", "b.mp4")
    import shutil

    real = shutil.copy2
    count = {"n": 0}

    def flaky(src, dst, *a, **k):
        count["n"] += 1
        if count["n"] == 2:
            Path(dst).write_text("partial")
            raise OSError(errno.ENOSPC, "disk full")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "copy2", flaky)
    result = mgr.execute(plan_for(workspace, [CopyAction(target_folder="Backup", match=("a", "b"))]))
    assert not result.ok and listing(workspace) == ["a.mp4", "b.mp4"] and not (workspace / "Backup").exists()


# --------------------------------------------------------------- delete/trash
def test_delete_goes_to_trash_and_undo_restores(workspace, make_files, mgr):
    make_files("junk.txt", "keep.mp4", content="d")
    result = mgr.execute(plan_for(workspace, [DeleteAction(source="junk.txt")]))
    assert result.ok and listing(workspace) == ["keep.mp4"]
    assert OperationManager.trash_size(workspace) == 1
    assert mgr.undo(result.operation_id).ok
    assert snapshot(workspace)["junk.txt"] == "djunk.txt"
    assert not (workspace / TRASH_DIR_NAME).exists()  # empty trash folders are cleaned up


def test_delete_in_subfolder_restores_to_the_right_place(workspace, make_files, mgr):
    make_files("sub/deep/x.txt", content="d")
    result = mgr.execute(plan_for(workspace, [DeleteAction(source="sub/deep/x.txt")], recursive=True))
    assert result.ok and mgr.undo(result.operation_id).ok
    assert snapshot(workspace) == {"sub/deep/x.txt": "dsub/deep/x.txt"}


def test_permanent_delete_cannot_be_undone(workspace, make_files, mgr, db):
    make_files("junk.txt", "a.mp4")
    result = mgr.execute(plan_for(workspace, [DeleteAction(source="junk.txt"), RenameAction("a.mp4", "b.mp4")], delete_mode="permanent"))
    assert result.ok and result.irreversible == 1 and listing(workspace) == ["b.mp4"]
    assert not (workspace / TRASH_DIR_NAME).exists()
    check = mgr.check_undo(result.operation_id)
    assert check.possible and any("permanently deleted" in n for n in check.notes)
    assert mgr.undo(result.operation_id).ok  # the rename is undone, the deleted file stays gone
    assert listing(workspace) == ["a.mp4"]


def test_emptying_trash_makes_delete_undo_impossible_with_explanation(workspace, make_files, mgr):
    make_files("junk.txt")
    result = mgr.execute(plan_for(workspace, [DeleteAction(source="junk.txt")]))
    assert OperationManager.empty_trash(workspace) == 1
    outcome = mgr.undo(result.operation_id)
    assert not outcome.ok and "no longer in the undo-trash" in outcome.blockers[0]


def test_delete_frees_name_for_rename_and_both_undo(workspace, make_files, mgr):
    make_files("a.mp4", "b.mp4", content="@")
    before = snapshot(workspace)
    result = mgr.execute(plan_for(workspace, [DeleteAction(source="b.mp4"), RenameAction("a.mp4", "b.mp4")]))
    assert result.ok and snapshot(workspace) == {"b.mp4": "@a.mp4"}
    assert mgr.undo(result.operation_id).ok and snapshot(workspace) == before


def test_overwrite_mode_replaces_via_trash_and_undo_brings_it_back(workspace, make_files, mgr):
    make_files("a.mp4", "b.mp4", content="@")
    before = snapshot(workspace)
    plan = plan_for(workspace, [RenameAction("a.mp4", "b.mp4")], only={"a.mp4"}, prevent_overwrite=False)
    result = mgr.execute(plan)
    assert result.ok and snapshot(workspace) == {"b.mp4": "@a.mp4"}
    assert OperationManager.trash_size(workspace) == 1  # the replaced file is recoverable
    assert mgr.undo(result.operation_id).ok and snapshot(workspace) == before


def test_internal_trash_is_never_scanned_or_selected(workspace, make_files, mgr):
    make_files("junk.txt", "a.mp4")
    mgr.execute(plan_for(workspace, [DeleteAction(source="junk.txt")]))
    assert [e.name for e in scan_folder(workspace, recursive=True).entries] == ["a.mp4"]


# ------------------------------------------------------------- history & misc
def test_history_disabled_keeps_no_record(workspace, make_files, mgr, db):
    make_files("a.mp4")
    result = mgr.execute(plan_for(workspace, [RenameAction("a.mp4", "b.mp4")]), keep_history=False)
    assert result.ok and result.operation_id is None and db.list_operations() == []


def test_history_is_kept_for_deletes_even_when_disabled(workspace, make_files, mgr, db):
    make_files("junk.txt")
    result = mgr.execute(plan_for(workspace, [DeleteAction(source="junk.txt")]), keep_history=False)
    assert result.operation_id is not None and db.get_operation(result.operation_id)


def test_journal_is_written_with_items(workspace, make_files, mgr, db):
    make_files("a.mp4", "b.mp4")
    result = mgr.execute(plan_for(workspace, [RenameAction("a.mp4", "b.mp4"), RenameAction("b.mp4", "c.mp4")]))
    items = db.get_items(result.operation_id)
    assert [(i.type, i.source, i.target, i.status) for i in items] == [
        ("rename", "a.mp4", "b.mp4", "done"), ("rename", "b.mp4", "c.mp4", "done"),
    ]
    assert all(i.temp_path and i.temp_path.startswith(TEMP_PREFIX) for i in items)  # two-phase temps are journaled


def test_interrupted_operation_can_be_recovered_with_undo(workspace, make_files, db, mgr):
    """Simulate a crash after phase one of a two-phase rename: files sit under temp names."""
    make_files("A.mp4", "B.mp4", content="@")
    before = snapshot(workspace)
    plan = plan_for(workspace, [RenameAction("A.mp4", "B.mp4"), RenameAction("B.mp4", "A.mp4")])
    result = mgr.execute(plan)
    items = db.get_items(result.operation_id)
    # roll the disk back to the "crashed after phase 1" state by hand
    for item in items:
        os.rename(workspace / item.target, workspace / item.temp_path)
    db.set_operation_status(result.operation_id, "running")
    assert db.mark_interrupted_operations() == 1
    assert db.get_operation(result.operation_id).status == "interrupted" and db.get_operation(result.operation_id).can_undo
    assert any(n.startswith(TEMP_PREFIX) for n in os.listdir(workspace))
    outcome = mgr.undo(result.operation_id)
    assert outcome.ok and snapshot(workspace) == before


def test_summaries():
    assert summarize_kinds({"rename": 126}) == "126 rename operations"
    assert summarize_kinds({"rename": 1}) == "1 rename operation"
    assert summarize_kinds({"move": 3, "delete": 2}) == "3 move, 2 delete operations"
    assert summarize_kinds({}) == "no operations"


def test_large_batch_with_chain_is_fast_and_correct(workspace, mgr):
    n = 600
    for i in range(n):
        (workspace / f"clip_{i:04d}.mp4").write_text(str(i))
    actions = [RenameAction(f"clip_{i:04d}.mp4", f"clip_{(i + 1) % n:04d}.mp4") for i in range(n)]  # one big cycle
    result = mgr.execute(plan_for(workspace, actions))
    assert result.ok and result.applied == n
    assert (workspace / "clip_0001.mp4").read_text() == "0" and (workspace / "clip_0000.mp4").read_text() == str(n - 1)
    assert mgr.undo(result.operation_id).ok
    assert (workspace / "clip_0000.mp4").read_text() == "0"


def test_unicode_and_bangla_roundtrip(workspace, make_files, mgr):
    names = ["ভিডিও ০১.mp4", "日本語 [HD].mkv", "Café (2020).mp4"]
    make_files(*names, content="é")
    before = snapshot(workspace)
    result = mgr.execute(plan_for(workspace, [AddPrefixAction("পর্ব - ")]))
    assert result.ok and listing(workspace) == sorted(f"পর্ব - {n}" for n in names)
    assert mgr.undo(result.operation_id).ok and snapshot(workspace) == before


def test_status_enum_helpers():
    assert OpStatus.OK.applicable and OpStatus.WARNING.applicable
    assert not OpStatus.CONFLICT.applicable and not OpStatus.INVALID.applicable


# ---------------------------------------------------- spec: error situations
def test_workspace_deleted_after_preview_is_reported_cleanly(workspace, make_files, mgr):
    import shutil

    make_files("a.mp4")
    plan = plan_for(workspace, [RenameAction("a.mp4", "b.mp4")])
    shutil.rmtree(workspace)
    result = mgr.execute(plan)
    assert not result.ok and "folder changed" in result.message  # no crash, no traceback


def test_workspace_deleted_makes_undo_impossible_with_explanation(workspace, make_files, mgr):
    import shutil

    make_files("a.mp4")
    result = mgr.execute(plan_for(workspace, [RenameAction("a.mp4", "b.mp4")]))
    shutil.rmtree(workspace)
    outcome = mgr.undo(result.operation_id)
    assert not outcome.ok and "no longer exists" in outcome.blockers[0]


def test_read_only_target_folder_gives_permission_message(workspace, make_files, mgr):
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("permission bits are not enforced for this user/platform")
    make_files("a.mp4")
    (workspace / "Locked").mkdir()
    os.chmod(workspace / "Locked", 0o500)
    try:
        result = mgr.execute(plan_for(workspace, [MoveAction(source="a.mp4", target_folder="Locked")]))
    finally:
        os.chmod(workspace / "Locked", 0o700)
    assert not result.ok and "Permission denied" in result.message and (workspace / "a.mp4").exists()


def test_sharing_violation_style_error_is_friendly(workspace, make_files, mgr, monkeypatch):
    make_files("a.mp4")
    real = os.rename

    def locked(src, dst, *a, **k):
        err = PermissionError(13, "The process cannot access the file because it is being used by another process")
        err.winerror = 32  # what Windows reports for a locked file
        raise err

    monkeypatch.setattr(os, "rename", locked)
    result = mgr.execute(plan_for(workspace, [RenameAction("a.mp4", "b.mp4")]))
    monkeypatch.setattr(os, "rename", real)
    assert not result.ok and "locked" in result.message and (workspace / "a.mp4").exists()
