"""Planner: actions -> validated plan (rules, explicit actions, conflicts, safety)."""

from __future__ import annotations

import pytest

from app.ai.response_parser import parse_response
from app.ai.schemas import (
    AddPrefixAction,
    AddSuffixAction,
    ChangeCaseAction,
    CopyAction,
    CreateFolderAction,
    DeleteAction,
    MoveAction,
    NumberingAction,
    RemoveTextAction,
    RenameAction,
    ReplaceTextAction,
    SortAction,
)
from app.files.plan import OpKind, OpStatus
from app.files.planner import Planner, PlannerOptions
from app.files.scanner import scan_folder


def build(workspace, actions, *, only=None, recursive=False, rejected=None, **opts):
    entries = scan_folder(workspace, recursive=recursive).entries
    if only is not None:
        entries = [e for e in entries if e.rel_path in only]
    return Planner(workspace, entries, PlannerOptions(**opts)).build(actions, rejected)


def mapping(plan, kind=None):
    return {op.source: op.target for op in plan.ops if op.source and (kind is None or op.kind == kind)}


def op_for(plan, source):
    return next(op for op in plan.ops if op.source == source)


# ------------------------------------------------------------- spec sample
def test_spec_sample_test_case(workspace, make_files):
    """Folder with VID_00x_FINAL_1080P_WEB-DL.mp4 -> Blue Bloods - Episode 0x.mp4"""
    make_files(*(f"VID_00{i}_FINAL_1080P_WEB-DL.mp4" for i in (1, 2, 3)))
    plan = build(workspace, [NumberingAction(start=1, width=2, position="replace", template="Blue Bloods - Episode {n}")])
    assert mapping(plan) == {
        "VID_001_FINAL_1080P_WEB-DL.mp4": "Blue Bloods - Episode 01.mp4",
        "VID_002_FINAL_1080P_WEB-DL.mp4": "Blue Bloods - Episode 02.mp4",
        "VID_003_FINAL_1080P_WEB-DL.mp4": "Blue Bloods - Episode 03.mp4",
    }
    assert all(op.status == OpStatus.OK for op in plan.ops)  # extension kept, no warnings


def test_same_result_from_explicit_ai_renames(workspace, make_files):
    make_files(*(f"VID_00{i}_FINAL_1080P_WEB-DL.mp4" for i in (1, 2, 3)))
    ai = parse_response(
        '{"actions":[' + ",".join(
            f'{{"type":"rename","source":"VID_00{i}_FINAL_1080P_WEB-DL.mp4","target":"Blue Bloods - Episode 0{i}.mp4"}}'
            for i in (1, 2, 3)
        ) + "]}"
    )
    plan = build(workspace, ai.actions)
    assert list(mapping(plan).values()) == [f"Blue Bloods - Episode 0{i}.mp4" for i in (1, 2, 3)]
    assert plan.counts()["applicable"] == 3


# ------------------------------------------------------------------- rules
def test_remove_text_rule_applies_to_every_file_locally(workspace, make_files):
    make_files("A_1080p_WEB-DL.mp4", "B_1080P_WEB-DL.mkv", "keep.mp4")
    plan = build(workspace, [RemoveTextAction(("1080p", "WEB-DL"))])
    assert mapping(plan) == {"A_1080p_WEB-DL.mp4": "A.mp4", "B_1080P_WEB-DL.mkv": "B.mkv"}
    assert plan.unchanged == 1


def test_extension_is_never_touched_by_rules(workspace, make_files):
    make_files("movie.mp4.mp4", "x.MP4")
    plan = build(workspace, [RemoveTextAction(("mp4",))])
    # removing "mp4" from the stem must not eat the real extension
    assert mapping(plan) == {"movie.mp4.mp4": "movie.mp4"}


def test_prefix_suffix_replace_case_chain_in_order(workspace, make_files):
    make_files("episode_one.mp4")
    plan = build(workspace, [
        ReplaceTextAction("_", " "), ChangeCaseAction("title"), AddPrefixAction("Blue Bloods - "), AddSuffixAction(" HD"),
    ])
    assert mapping(plan) == {"episode_one.mp4": "Blue Bloods - Episode One HD.mp4"}


def test_numbering_prefix_suffix_and_start(workspace, make_files):
    make_files("a.mp4", "b.mp4")
    suffix = build(workspace, [NumberingAction(start=5, width=2, position="suffix")])
    assert mapping(suffix) == {"a.mp4": "a 05.mp4", "b.mp4": "b 06.mp4"}
    prefix = build(workspace, [NumberingAction(start=1, width=3, position="prefix", separator=" - ")])
    assert mapping(prefix) == {"a.mp4": "001 - a.mp4", "b.mp4": "002 - b.mp4"}


def test_numbering_width_grows_for_big_folders(workspace):
    for i in range(120):
        (workspace / f"f{i:03d}.mp4").write_bytes(b"")
    plan = build(workspace, [NumberingAction(start=1, width=2, position="replace", template="Ep {n}")])
    targets = sorted(mapping(plan).values())
    assert targets[0] == "Ep 001.mp4" and targets[-1] == "Ep 120.mp4"  # consistent width keeps sorting correct


def test_numbering_follows_the_selection_order_and_sort_action(workspace):
    (workspace / "small.mp4").write_bytes(b"1")
    (workspace / "big.mp4").write_bytes(b"1" * 100)
    by_name = build(workspace, [NumberingAction(position="replace", template="E{n}")])
    assert mapping(by_name) == {"big.mp4": "E01.mp4", "small.mp4": "E02.mp4"}
    by_size = build(workspace, [SortAction("size", descending=True), NumberingAction(position="replace", template="E{n}")])
    assert mapping(by_size) == {"big.mp4": "E01.mp4", "small.mp4": "E02.mp4"}
    by_size_asc = build(workspace, [SortAction("size"), NumberingAction(position="replace", template="E{n}")])
    assert mapping(by_size_asc) == {"small.mp4": "E01.mp4", "big.mp4": "E02.mp4"}
    assert by_size_asc.sort == SortAction("size")


def test_natural_order_numbering(workspace, make_files):
    make_files("ep10.mp4", "ep2.mp4", "ep1.mp4")
    plan = build(workspace, [NumberingAction(position="replace", template="Show {n}")])
    assert mapping(plan) == {"ep1.mp4": "Show 01.mp4", "ep2.mp4": "Show 02.mp4", "ep10.mp4": "Show 03.mp4"}


def test_match_filter_limits_rules(workspace, make_files):
    make_files("S01E01.mp4", "S02E01.mp4", "S02E02.mp4")
    plan = build(workspace, [NumberingAction(position="replace", template="Season 2 Ep {n}", match=("s02",))])
    assert mapping(plan) == {"S02E01.mp4": "Season 2 Ep 01.mp4", "S02E02.mp4": "Season 2 Ep 02.mp4"}


def test_only_selected_files_are_touched(workspace, make_files):
    make_files("a_x.mp4", "b_x.mp4")
    plan = build(workspace, [RemoveTextAction(("_x",))], only={"a_x.mp4"})
    assert mapping(plan) == {"a_x.mp4": "a.mp4"}


def test_rule_that_changes_nothing_yields_empty_plan(workspace, make_files):
    make_files("a.mp4")
    plan = build(workspace, [RemoveTextAction(("zzz",))])
    assert plan.is_empty and plan.unchanged == 1


# ----------------------------------------------------------- explicit actions
def test_extension_preserved_when_ai_omits_it(workspace, make_files):
    make_files("a.mp4")
    plan = build(workspace, [RenameAction("a.mp4", "New Name")])
    op = op_for(plan, "a.mp4")
    assert op.target == "New Name.mp4" and op.status == OpStatus.WARNING and "kept" in op.messages[0]


def test_title_with_dot_still_gets_extension(workspace, make_files):
    make_files("a.mkv")
    plan = build(workspace, [RenameAction("a.mkv", "Episode 1.2")])
    assert op_for(plan, "a.mkv").target == "Episode 1.2.mkv"


def test_unrequested_extension_change_is_rejected(workspace, make_files):
    make_files("a.mp4")
    op = op_for(build(workspace, [RenameAction("a.mp4", "a.mkv")]), "a.mp4")
    assert op.status == OpStatus.INVALID and "Extension change" in op.messages[0]


def test_explicit_extension_change_is_allowed_with_warning(workspace, make_files):
    make_files("a.txt")
    op = op_for(build(workspace, [RenameAction("a.txt", "a.md", change_extension=True)]), "a.txt")
    assert op.target == "a.md" and op.status == OpStatus.WARNING


def test_extension_case_is_preserved_from_ai_target(workspace, make_files):
    make_files("a.MP4")
    assert op_for(build(workspace, [RenameAction("a.MP4", "b.mp4")]), "a.MP4").target == "b.mp4"


def test_unknown_or_unselected_source_is_invalid(workspace, make_files):
    make_files("a.mp4", "b.mp4")
    plan = build(workspace, [RenameAction("ghost.mp4", "x.mp4"), RenameAction("b.mp4", "y.mp4")], only={"a.mp4"})
    assert all(op.status == OpStatus.INVALID for op in plan.ops)
    assert "not one of the selected files" in op_for(plan, "ghost.mp4").messages[0]
    assert "not one of the selected files" in op_for(plan, "b.mp4").messages[0]  # exists, but was not selected


def test_rename_to_same_name_is_ignored(workspace, make_files):
    make_files("a.mp4")
    assert build(workspace, [RenameAction("a.mp4", "a.mp4")]).is_empty


def test_source_lookup_by_basename_and_unicode_normalisation(workspace, make_files):
    make_files("sub/ep1.mp4", "e\u0301.mp4")  # second file uses a decomposed 'é'
    plan = build(workspace, [RenameAction("ep1.mp4", "x.mp4"), RenameAction("\u00e9.mp4", "ok.mp4")], recursive=True)
    assert op_for(plan, "sub/ep1.mp4").target == "sub/x.mp4"  # stays in its own folder
    assert op_for(plan, "e\u0301.mp4").target == "ok.mp4"


@pytest.mark.parametrize(
    "target",
    ["../outside.mp4", "..\\outside.mp4", "sub/x.mp4", "C:\\x.mp4", "/abs.mp4", "\\\\srv\\sh\\x.mp4", "a:b.mp4", "bad|name.mp4", "bad?.mp4", "CON.mp4"],
)
def test_unsafe_rename_targets_are_invalid(workspace, make_files, target):
    make_files("a.mp4")
    op = op_for(build(workspace, [RenameAction("a.mp4", target)]), "a.mp4")
    assert op.status == OpStatus.INVALID and op.messages


def test_stray_space_before_the_extension_is_trimmed(workspace, make_files):
    make_files("a.mp4")
    op = op_for(build(workspace, [RenameAction("a.mp4", "name .mp4")]), "a.mp4")
    assert op.target == "name.mp4" and op.status == OpStatus.OK


def test_illegal_character_message_includes_a_suggestion(workspace, make_files):
    make_files("a.mp4")
    op = op_for(build(workspace, [RenameAction("a.mp4", "Ep 1: Pilot.mp4")]), "a.mp4")
    assert op.status == OpStatus.INVALID and "Illegal" in op.messages[0] and "Ep 1_ Pilot.mp4" in op.messages[0]


def test_rule_producing_illegal_name_is_invalid(workspace, make_files):
    make_files("a.mp4")
    op = op_for(build(workspace, [AddPrefixAction("Season 2: ")]), "a.mp4")
    assert op.status == OpStatus.INVALID and "Illegal" in op.messages[0]


def test_rule_that_empties_the_name_is_invalid(workspace, make_files):
    make_files("1080p.mp4")
    op = op_for(build(workspace, [RemoveTextAction(("1080p",))]), "1080p.mp4")
    assert op.status == OpStatus.INVALID and "empty" in op.messages[0]


def test_long_names_are_shortened_with_a_warning(workspace, make_files):
    make_files("a.mp4")
    op = op_for(build(workspace, [RenameAction("a.mp4", "x" * 400 + ".mp4")]), "a.mp4")
    # a 404-char name is not silently truncated for explicit targets: it is rejected clearly
    assert op.status == OpStatus.INVALID
    rule = op_for(build(workspace, [AddPrefixAction("y" * 300)]), "a.mp4")
    assert rule.status == OpStatus.WARNING and len(rule.target) <= 255 and rule.target.endswith(".mp4")
    assert "shortened" in " ".join(rule.messages)


def test_bangla_unicode_spaces_brackets_special_characters(workspace, make_files):
    names = ["ভিডিও ০১.mp4", "日本語 (2020) [HD] {x}.mkv", "it's #1 & 100% done!.mp4", "a  b.mp4"]
    make_files(*names)
    plan = build(workspace, [AddPrefixAction("Blue Bloods - ")])
    assert all(op.status == OpStatus.OK for op in plan.ops)
    assert {op.target for op in plan.ops} == {f"Blue Bloods - {n}" for n in names}


# --------------------------------------------------------- conflict detection
def test_duplicate_targets_are_conflicts(workspace, make_files):
    make_files("a_1080p.mp4", "a_720p.mp4")
    plan = build(workspace, [RemoveTextAction(("_1080p", "_720p"))])
    statuses = sorted(op.status.value for op in plan.ops)
    assert statuses == ["conflict", "ok"]  # first claims "a.mp4", second collides
    bad = next(op for op in plan.ops if op.status == OpStatus.CONFLICT)
    assert "also the new name" in bad.messages[0]
    assert plan.has_blockers and plan.counts()["applicable"] == 1


def test_existing_unselected_file_is_a_conflict(workspace, make_files):
    make_files("a.mp4", "b.mp4")
    plan = build(workspace, [RenameAction("a.mp4", "b.mp4")], only={"a.mp4"})
    op = op_for(plan, "a.mp4")
    assert op.status == OpStatus.CONFLICT and "already exists" in op.messages[0]


def test_existing_file_not_renamed_blocks_selected_rename(workspace, make_files):
    make_files("a.mp4", "b.mp4")
    assert op_for(build(workspace, [RenameAction("a.mp4", "b.mp4")]), "a.mp4").status == OpStatus.CONFLICT


def test_chain_is_valid_because_the_target_is_vacated(workspace, make_files):
    make_files("A.mp4", "B.mp4")
    plan = build(workspace, [RenameAction("A.mp4", "B.mp4"), RenameAction("B.mp4", "C.mp4")])
    assert [op.status for op in plan.ops] == [OpStatus.OK, OpStatus.OK]


def test_swap_is_valid(workspace, make_files):
    make_files("A.mp4", "B.mp4")
    plan = build(workspace, [RenameAction("A.mp4", "B.mp4"), RenameAction("B.mp4", "A.mp4")])
    assert all(op.status == OpStatus.OK for op in plan.ops)


def test_case_only_rename_is_not_a_conflict_on_case_insensitive_fs(workspace, make_files):
    make_files("episode.mp4")
    plan = build(workspace, [RenameAction("episode.mp4", "Episode.mp4")], case_insensitive=True)
    assert op_for(plan, "episode.mp4").status == OpStatus.OK


def test_names_differing_only_by_case_collide_on_windows(workspace, make_files):
    make_files("a.mp4", "b.mp4")
    plan = build(workspace, [RenameAction("a.mp4", "X.mp4"), RenameAction("b.mp4", "x.mp4")], case_insensitive=True)
    assert sorted(op.status.value for op in plan.ops) == ["conflict", "ok"]
    plan_linux = build(workspace, [RenameAction("a.mp4", "X.mp4"), RenameAction("b.mp4", "x.mp4")], case_insensitive=False)
    assert all(op.status == OpStatus.OK for op in plan_linux.ops)


def test_duplicate_policy_number_resolves_collisions(workspace, make_files):
    make_files("a_1080p.mp4", "a_720p.mp4", "a.mp4")
    plan = build(workspace, [RemoveTextAction(("_1080p", "_720p"))], duplicate_policy="number")
    targets = sorted(op.target for op in plan.ops)
    assert targets == ["a (2).mp4", "a (3).mp4"]
    assert all(op.status == OpStatus.WARNING for op in plan.ops) and not plan.has_blockers


def test_prevent_overwrite_off_replaces_existing_via_trash(workspace, make_files):
    make_files("a.mp4", "b.mp4")
    plan = build(workspace, [RenameAction("a.mp4", "b.mp4")], only={"a.mp4"}, prevent_overwrite=False)
    op = op_for(plan, "a.mp4")
    assert op.status == OpStatus.WARNING and op.overwrite and "replaced" in op.messages[0]


def test_existing_folder_with_target_name_is_always_a_conflict(workspace, make_files):
    make_files("a.mp4")
    (workspace / "b.mp4").mkdir()
    plan = build(workspace, [RenameAction("a.mp4", "b.mp4")], only={"a.mp4"}, prevent_overwrite=False)
    assert op_for(plan, "a.mp4").status == OpStatus.CONFLICT


def test_each_source_can_only_be_used_once(workspace, make_files):
    make_files("a.mp4")
    plan = build(workspace, [RenameAction("a.mp4", "x.mp4"), RenameAction("a.mp4", "y.mp4"), DeleteAction(source="a.mp4")])
    assert [op.status for op in plan.ops] == [OpStatus.OK, OpStatus.INVALID, OpStatus.INVALID]


def test_unticking_a_conflicting_op_revalidates(workspace, make_files):
    make_files("a_1.mp4", "a_2.mp4")
    plan = build(workspace, [RemoveTextAction(("_1", "_2"))])
    loser = next(op for op in plan.ops if op.status == OpStatus.CONFLICT)
    winner = next(op for op in plan.ops if op.status == OpStatus.OK)
    winner.included = False
    plan.revalidate()
    assert loser.status == OpStatus.OK and not plan.has_blockers  # no longer competes for the name
    winner.included = True
    plan.revalidate()
    assert loser.status == OpStatus.CONFLICT


def test_rejected_ai_actions_show_up_as_invalid_rows(workspace, make_files):
    make_files("a.mp4")
    parsed = parse_response('{"actions":[{"type":"organize"},{"type":"rename","source":"a.mp4","target":"b.mp4"}]}')
    plan = build(workspace, parsed.actions, rejected=parsed.rejected)
    assert [op.status for op in plan.ops].count(OpStatus.INVALID) == 1 and plan.counts()["applicable"] == 1


# --------------------------------------------------------- folders: move/copy
def test_move_rule_creates_folder_automatically(workspace, make_files):
    make_files("S02E01.mp4", "S01E01.mp4")
    plan = build(workspace, [MoveAction(target_folder="Season 2", match=("S02",))])
    kinds = [(op.kind, op.target, op.implicit) for op in plan.ops]
    assert (OpKind.CREATE_FOLDER, "Season 2", True) in kinds
    assert (OpKind.MOVE, "Season 2/S02E01.mp4", False) in kinds
    assert all(op.status == OpStatus.OK for op in plan.ops) and plan.unchanged == 1
    assert plan.kind_summary() == {"move": 1}  # the helper folder is not counted as a user-visible change


def test_explicit_create_folder_plus_move(workspace, make_files):
    make_files("a.mp4")
    plan = build(workspace, [CreateFolderAction("Season 2"), MoveAction(source="a.mp4", target_folder="Season 2")])
    assert [op.kind for op in plan.ops] == [OpKind.CREATE_FOLDER, OpKind.MOVE]
    assert not any(op.implicit for op in plan.ops)


def test_move_into_existing_folder_needs_no_creation(workspace, make_files):
    make_files("a.mp4")
    (workspace / "Done").mkdir()
    plan = build(workspace, [MoveAction(source="a.mp4", target_folder="Done")])
    assert [op.kind for op in plan.ops] == [OpKind.MOVE]


def test_move_with_new_name(workspace, make_files):
    make_files("a.mp4")
    plan = build(workspace, [MoveAction(source="a.mp4", target_folder="X", new_name="b")])
    assert op_for(plan, "a.mp4").target == "X/b.mp4"


def test_unticking_all_moves_hides_the_implicit_folder(workspace, make_files):
    make_files("a.mp4")
    plan = build(workspace, [MoveAction(target_folder="New", match=("a",))])
    move = next(op for op in plan.ops if op.kind == OpKind.MOVE)
    move.included = False
    plan.revalidate()
    folder = next(op for op in plan.ops if op.kind == OpKind.CREATE_FOLDER)
    assert folder.noop and plan.is_empty


def test_unticking_the_folder_op_invalidates_moves_into_it(workspace, make_files):
    make_files("a.mp4")
    plan = build(workspace, [CreateFolderAction("New"), MoveAction(source="a.mp4", target_folder="New")])
    plan.ops[0].included = False
    plan.revalidate()
    assert plan.ops[1].status == OpStatus.INVALID and "does not exist" in plan.ops[1].messages[0]


@pytest.mark.parametrize("folder", ["../out", "C:\\out", "/abs", "\\\\srv\\x", "a:b", "con"])
def test_unsafe_folders_are_invalid(workspace, make_files, folder):
    make_files("a.mp4")
    assert all(op.status == OpStatus.INVALID for op in build(workspace, [MoveAction(source="a.mp4", target_folder=folder)]).ops)
    assert all(op.status == OpStatus.INVALID for op in build(workspace, [CreateFolderAction(folder)]).ops)


def test_unsafe_folder_in_rule_move_is_reported(workspace, make_files):
    make_files("a.mp4")
    plan = build(workspace, [MoveAction(target_folder="../escape", match=("a",))])
    assert plan.ops and all(op.status == OpStatus.INVALID for op in plan.ops)


def test_move_to_existing_target_name_conflicts(workspace, make_files):
    make_files("a.mp4", "Done/a.mp4")
    plan = build(workspace, [MoveAction(source="a.mp4", target_folder="Done")], only={"a.mp4"})
    assert op_for(plan, "a.mp4").status == OpStatus.CONFLICT


def test_copy_rule_and_explicit_copy(workspace, make_files):
    make_files("a.mp4", "b.mp4")
    plan = build(workspace, [CopyAction(source="a.mp4", target_folder="Backup"), CopyAction(target_folder="Backup2", match=("b",))])
    assert mapping(plan, OpKind.COPY) == {"a.mp4": "Backup/a.mp4", "b.mp4": "Backup2/b.mp4"}
    assert all(op.status == OpStatus.OK for op in plan.ops)


def test_copy_and_rename_of_same_file_is_refused(workspace, make_files):
    make_files("a.mp4")
    plan = build(workspace, [CopyAction(source="a.mp4", target_folder="B"), RenameAction("a.mp4", "z.mp4")])
    copy = next(op for op in plan.ops if op.kind == OpKind.COPY)
    assert copy.status == OpStatus.INVALID
    plan2 = build(workspace, [AddPrefixAction("X "), CopyAction(target_folder="B", match=("a",))])
    assert next(op for op in plan2.ops if op.kind == OpKind.COPY).status == OpStatus.INVALID


def test_copy_onto_itself_is_invalid(workspace, make_files):
    make_files("a.mp4")
    assert build(workspace, [CopyAction(source="a.mp4", target_folder="")]).ops[0].status == OpStatus.INVALID


# ------------------------------------------------------------------- delete
def test_delete_is_flagged_and_reports_the_mode(workspace, make_files):
    make_files("junk.txt", "a.mp4")
    plan = build(workspace, [DeleteAction(source="junk.txt")])
    op = op_for(plan, "junk.txt")
    assert op.kind == OpKind.DELETE and op.status == OpStatus.WARNING and "undo-trash" in op.messages[0]
    assert plan.has_deletes
    permanent = build(workspace, [DeleteAction(source="junk.txt")], delete_mode="permanent")
    assert "cannot be undone" in op_for(permanent, "junk.txt").messages[0]


def test_delete_rule_by_match(workspace, make_files):
    make_files("sample.mp4", "movie.mp4", "movie-sample.mkv")
    plan = build(workspace, [DeleteAction(match=("sample",))])
    assert {op.source for op in plan.ops} == {"sample.mp4", "movie-sample.mkv"}
    assert len(plan.delete_ops) == 2


def test_delete_frees_the_name_for_a_rename_in_trash_mode_only(workspace, make_files):
    make_files("a.mp4", "b.mp4")
    actions = [DeleteAction(source="b.mp4"), RenameAction("a.mp4", "b.mp4")]
    assert op_for(build(workspace, actions), "a.mp4").status == OpStatus.OK
    assert op_for(build(workspace, actions, delete_mode="permanent"), "a.mp4").status == OpStatus.CONFLICT


# -------------------------------------------------------------------- safety
def test_symlinked_folder_cannot_be_a_target(workspace, make_files, tmp_path):
    make_files("a.mp4")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / "escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    plan = build(workspace, [MoveAction(source="a.mp4", target_folder="escape")])
    assert any(op.status == OpStatus.INVALID and "leaves the selected folder" in " ".join(op.messages) for op in plan.ops)


def test_missing_source_is_caught_on_revalidation(workspace, make_files):
    paths = make_files("a.mp4")
    plan = build(workspace, [RenameAction("a.mp4", "b.mp4")])
    paths[0].unlink()  # the file vanishes after the preview
    plan.revalidate()
    assert plan.ops[0].status == OpStatus.INVALID and "no longer exists" in plan.ops[0].messages[0]


def test_empty_actions_give_empty_plan(workspace, make_files):
    make_files("a.mp4")
    assert build(workspace, []).is_empty
