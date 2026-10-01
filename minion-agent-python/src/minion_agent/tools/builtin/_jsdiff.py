"""The pinned `diff` 8.0.4 package, ported exactly as pinned Pi calls it (spec/tools.md WP-13.2
"`edit` result details"): `diffLines(old, new)` with no options and `createTwoFilesPatch(path, path,
old, new, undefined, undefined, {context, headerOptions: FILE_HEADERS_ONLY})`.

Every string here is a JavaScript string in `_utf16` form (one character per UTF-16 code unit).
The Myers search keeps jsdiff's own shape -- diagonal pruning, the branch choice
`!canRemove || (canAdd && removePath.oldPos < addPath.oldPos)`, component merging -- because a
different but equally minimal edit script changes `details.diff`/`details.patch`. Sources:
`libesm/diff/base.js`, `libesm/diff/line.js`, `libesm/patch/create.js`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_LINE_SEPARATOR = re.compile("(\n|\r\n)")


@dataclass
class Component:
    """One change object: `count` tokens, `added`/`removed`, and (once built) its `value`."""

    count: int
    added: bool
    removed: bool
    previous: Component | None = None
    value: str = ""
    lines: list[str] = field(default_factory=list)


@dataclass
class _Path:
    old_pos: int
    last: Component | None


def tokenize_lines(value: str) -> list[str]:
    """`line.js` `tokenize` without options: split on `\\n`/`\\r\\n` keeping the separators, drop
    a trailing empty piece, merge each separator into the preceding line."""
    pieces = _LINE_SEPARATOR.split(value)
    if not pieces[-1]:
        pieces.pop()
    lines: list[str] = []
    for i, piece in enumerate(pieces):
        if i % 2:
            lines[-1] += piece
        else:
            lines.append(piece)
    return lines


def _add_to_path(path: _Path, added: bool, removed: bool, old_pos_inc: int) -> _Path:
    last = path.last
    if last is not None and last.added == added and last.removed == removed:
        return _Path(
            path.old_pos + old_pos_inc, Component(last.count + 1, added, removed, last.previous)
        )
    return _Path(path.old_pos + old_pos_inc, Component(1, added, removed, last))


def _extract_common(path: _Path, new: list[str], old: list[str], diagonal: int) -> int:
    old_pos = path.old_pos
    new_pos = old_pos - diagonal
    common = 0
    while (
        new_pos + 1 < len(new) and old_pos + 1 < len(old) and old[old_pos + 1] == new[new_pos + 1]
    ):
        new_pos += 1
        old_pos += 1
        common += 1
    if common:
        path.last = Component(common, False, False, path.last)
    path.old_pos = old_pos
    return new_pos


def _build_values(last: Component | None, new: list[str], old: list[str]) -> list[Component]:
    components: list[Component] = []
    while last is not None:
        components.append(last)
        last = last.previous
    components.reverse()
    new_pos = old_pos = 0
    for component in components:
        component.previous = None
        if not component.removed:
            component.value = "".join(new[new_pos : new_pos + component.count])
            new_pos += component.count
            if not component.added:
                old_pos += component.count
        else:
            component.value = "".join(old[old_pos : old_pos + component.count])
            old_pos += component.count
    return components


def diff_tokens(old: list[str], new: list[str]) -> list[Component]:
    """`Diff.diffWithOptionsObj` (base.js), synchronous, no `maxEditLength`, no timeout."""
    new_len, old_len = len(new), len(old)
    best: dict[int, _Path | None] = {0: _Path(-1, None)}
    seed = best[0]
    assert seed is not None
    new_pos = _extract_common(seed, new, old, 0)
    if seed.old_pos + 1 >= old_len and new_pos + 1 >= new_len:
        return _build_values(seed.last, new, old)
    min_diagonal: float = float("-inf")
    max_diagonal: float = float("inf")
    edit_length = 1
    while True:  # editLength never exceeds newLen + oldLen: a full delete+insert always finishes
        diagonal = int(max(min_diagonal, -edit_length))
        while diagonal <= min(max_diagonal, edit_length):
            remove_path = best.get(diagonal - 1)
            add_path = best.get(diagonal + 1)
            if remove_path is not None:
                best[diagonal - 1] = None
            can_add = False
            if add_path is not None:
                add_new_pos = add_path.old_pos - diagonal
                can_add = 0 <= add_new_pos < new_len
            can_remove = remove_path is not None and remove_path.old_pos + 1 < old_len
            if not can_add and not can_remove:
                best[diagonal] = None
                diagonal += 2
                continue
            if not can_remove or (can_add and remove_path.old_pos < add_path.old_pos):  # type: ignore[union-attr]
                base = _add_to_path(add_path, True, False, 0)  # type: ignore[arg-type]
            else:
                base = _add_to_path(remove_path, False, True, 1)  # type: ignore[arg-type]
            new_pos = _extract_common(base, new, old, diagonal)
            if base.old_pos + 1 >= old_len and new_pos + 1 >= new_len:
                return _build_values(base.last, new, old)
            best[diagonal] = base
            if base.old_pos + 1 >= old_len:
                max_diagonal = min(max_diagonal, diagonal - 1)
            if new_pos + 1 >= new_len:
                min_diagonal = max(min_diagonal, diagonal + 1)
            diagonal += 2
        edit_length += 1


def diff_lines(old: str, new: str) -> list[Component]:
    """`diffLines(old, new)` with no options (`removeEmpty` drops empty tokens)."""
    return diff_tokens([t for t in tokenize_lines(old) if t], [t for t in tokenize_lines(new) if t])


def _split_lines(text: str) -> list[str]:
    """create.js `splitLines`: lines WITH their trailing `\\n`, the last kept without one."""
    result = [line + "\n" for line in text.split("\n")]
    if text.endswith("\n"):
        result.pop()
    else:
        result[-1] = result[-1][:-1]
    return result


@dataclass
class _Hunk:
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    lines: list[str]


def _structured_hunks(old: str, new: str, context: int) -> list[_Hunk]:
    """`structuredPatch`'s `diffLinesResultToPatch`."""
    diff = diff_lines(old, new)
    diff.append(Component(0, False, False))  # `{value: '', lines: []}`: it keeps its ZERO lines
    sentinel = len(diff) - 1
    hunks: list[_Hunk] = []
    old_range_start = new_range_start = 0
    cur_range: list[str] = []
    old_line = new_line = 1
    for i, current in enumerate(diff):
        lines = [] if i == sentinel else _split_lines(current.value)
        current.lines = lines
        if current.added or current.removed:
            if not old_range_start:
                old_range_start, new_range_start = old_line, new_line
                if i > 0:
                    prev = diff[i - 1]
                    cur_range = [" " + e for e in prev.lines[-context:]] if context > 0 else []
                    old_range_start -= len(cur_range)
                    new_range_start -= len(cur_range)
            cur_range.extend(("+" if current.added else "-") + line for line in lines)
            if current.added:
                new_line += len(lines)
            else:
                old_line += len(lines)
        else:
            if old_range_start:
                if len(lines) <= context * 2 and i < len(diff) - 2:
                    cur_range.extend(" " + line for line in lines)
                else:
                    size = min(len(lines), context)
                    cur_range.extend(" " + line for line in lines[:size])
                    hunks.append(
                        _Hunk(
                            old_range_start,
                            old_line - old_range_start + size,
                            new_range_start,
                            new_line - new_range_start + size,
                            cur_range,
                        )
                    )
                    old_range_start = new_range_start = 0
                    cur_range = []
            old_line += len(lines)
            new_line += len(lines)
    for hunk in hunks:
        i = 0
        while i < len(hunk.lines):
            if hunk.lines[i].endswith("\n"):
                hunk.lines[i] = hunk.lines[i][:-1]
            else:
                hunk.lines.insert(i + 1, "\\ No newline at end of file")
                i += 1
            i += 1
    return hunks


def create_two_files_patch(old_name: str, new_name: str, old: str, new: str, context: int) -> str:
    """`createTwoFilesPatch(..., undefined, undefined, {context, headerOptions: FILE_HEADERS_ONLY})`
    then `formatPatch`: `---`/`+++` headers without timestamps, the zero-length-hunk start
    adjustment, and a final `\\n`."""
    out = ["--- " + old_name, "+++ " + new_name]
    for hunk in _structured_hunks(old, new, context):
        old_start = hunk.old_start - 1 if hunk.old_lines == 0 else hunk.old_start
        new_start = hunk.new_start - 1 if hunk.new_lines == 0 else hunk.new_start
        out.append(f"@@ -{old_start},{hunk.old_lines} +{new_start},{hunk.new_lines} @@")
        out.extend(hunk.lines)
    return "\n".join(out) + "\n"
