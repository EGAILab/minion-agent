"""`edit`'s matching, fuzzy normalization, unchanged-line preservation, BOM/line endings and result
details (`TOOL-030`/`TOOL-031`; pinned Pi `core/tools/edit-diff.ts` and `utils/text.ts`).

Every string argument and result here is a JavaScript string in `_utf16` form (one character per
UTF-16 code unit), so lengths, indices, `indexOf` and `split("")` are Pi's. Diagnostic messages
embed the tool's `path` argument as given, a Python string.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from ._jsdiff import create_two_files_patch, diff_lines
from ._utf16 import from_units, to_units
from .collation import pinned_collation
from .paths import BuiltinToolError

# ECMAScript WhiteSpace + LineTerminator, exactly (`String.prototype.trimEnd`). Not Python's own
# whitespace set, which also contains U+001C..U+001F and U+0085 and differs from it.
_JS_WHITESPACE = "\t\n\v\f\r " + "".join(
    map(
        chr,
        (0x00A0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF),
    )
)
_SINGLE_QUOTES = re.compile("[\\u2018\\u2019\\u201a\\u201b]")
_DOUBLE_QUOTES = re.compile("[\\u201c\\u201d\\u201e\\u201f]")
_DASHES = re.compile("[\\u2010\\u2011\\u2012\\u2013\\u2014\\u2015\\u2212]")
_SPACES = re.compile("[\\u00a0\\u2002-\\u200a\\u202f\\u205f\\u3000]")
_LONE_SURROGATE = re.compile("([\ud800-\udfff])")
_LINE_WITH_ENDING = re.compile("[^\n]*\n|[^\n]+")

BOM = chr(0xFEFF)

INTERNAL_RANGE = "Replacement range is outside the base content."
INTERNAL_LINE_COUNT = (
    "Cannot preserve unchanged lines because the base content has a different line count."
)


def split_bom(text: str) -> tuple[str, str]:
    """`splitBom`: `("\\ufeff", rest)` when the text starts with a BOM, else `("", text)`."""
    return (BOM, text[1:]) if text.startswith(BOM) else ("", text)


def detect_line_ending(content: str) -> str:
    crlf = content.find("\r\n")
    lf = content.find("\n")
    if lf == -1 or crlf == -1:
        return "\n"
    return "\r\n" if crlf < lf else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def _nfkc(units: str) -> str:
    """`String.prototype.normalize("NFKC")` at Unicode 16.0. An unpaired surrogate is a starter
    that nothing composes with, so the text is normalized piecewise around it."""
    collation = pinned_collation()
    pieces = _LONE_SURROGATE.split(from_units(units))
    return to_units(
        "".join(
            piece if i % 2 else collation.nfkc_unicode16(piece) for i, piece in enumerate(pieces)
        )
    )


def fuzzy_normalize(units: str) -> str:
    """`normalizeForFuzzyMatch`."""
    text = "\n".join(line.rstrip(_JS_WHITESPACE) for line in _nfkc(units).split("\n"))
    text = _SINGLE_QUOTES.sub("'", text)
    text = _DOUBLE_QUOTES.sub('"', text)
    text = _DASHES.sub("-", text)
    return _SPACES.sub(" ", text)


@dataclass(frozen=True)
class _Match:
    found: bool
    index: int
    length: int
    used_fuzzy: bool


def fuzzy_find(content: str, old: str) -> _Match:
    """`fuzzyFindText`: exact first; otherwise in fuzzy-normalized space."""
    exact = content.find(old)
    if exact != -1:
        return _Match(True, exact, len(old), False)
    fuzzy_old = fuzzy_normalize(old)
    index = fuzzy_normalize(content).find(fuzzy_old)
    if index == -1:
        return _Match(False, -1, 0, False)
    return _Match(True, index, len(fuzzy_old), True)


def count_occurrences(content: str, old: str) -> int:
    """`countOccurrences`: ALWAYS in fuzzy space. `split("")` yields one piece per code unit."""
    fuzzy_content = fuzzy_normalize(content)
    fuzzy_old = fuzzy_normalize(old)
    if not fuzzy_old:
        return len(fuzzy_content) - 1
    return fuzzy_content.count(fuzzy_old)


@dataclass(frozen=True)
class Replacement:
    edit_index: int
    index: int
    length: int
    new_text: str


def apply_replacements(content: str, replacements: list[Replacement], offset: int = 0) -> str:
    result = content
    for replacement in reversed(replacements):
        start = replacement.index - offset
        result = result[:start] + replacement.new_text + result[start + replacement.length :]
    return result


def _lines_with_endings(content: str) -> list[str]:
    return _LINE_WITH_ENDING.findall(content)


def _line_spans(content: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in _lines_with_endings(content):
        spans.append((offset, offset + len(line)))
        offset += len(line)
    return spans


def _replacement_line_range(
    spans: list[tuple[int, int]], replacement: Replacement
) -> tuple[int, int]:
    start = replacement.index
    end = replacement.index + replacement.length
    start_line = next((i for i, (s, e) in enumerate(spans) if s <= start < e), -1)
    if start_line == -1:
        raise BuiltinToolError(INTERNAL_RANGE)
    end_line = start_line
    while end_line < len(spans) and spans[end_line][1] < end:
        end_line += 1
    if end_line >= len(spans):
        raise BuiltinToolError(INTERNAL_RANGE)
    return start_line, end_line + 1


@dataclass
class _Group:
    start_line: int
    end_line: int  # exclusive
    replacements: list[Replacement]


def preserve_unchanged_lines(original: str, base: str, replacements: list[Replacement]) -> str:
    """`applyReplacementsPreservingUnchangedLines`."""
    original_lines = _lines_with_endings(original)
    spans = _line_spans(base)
    if len(original_lines) != len(spans):
        raise BuiltinToolError(INTERNAL_LINE_COUNT)
    groups: list[_Group] = []
    for replacement in sorted(replacements, key=lambda r: r.index):
        start_line, end_line = _replacement_line_range(spans, replacement)
        if groups and start_line < groups[-1].end_line:
            groups[-1].end_line = max(groups[-1].end_line, end_line)
            groups[-1].replacements.append(replacement)
            continue
        groups.append(_Group(start_line, end_line, [replacement]))
    result = ""
    index = 0
    for group in groups:
        result += "".join(original_lines[index : group.start_line])
        group_start = spans[group.start_line][0]
        group_end = spans[group.end_line - 1][1]
        result += apply_replacements(base[group_start:group_end], group.replacements, group_start)
        index = group.end_line
    return result + "".join(original_lines[index:])


def _not_found(path: str, i: int, n: int) -> BuiltinToolError:
    if n == 1:
        return BuiltinToolError(
            f"Could not find the exact text in {path}. The old text must match exactly "
            "including all whitespace and newlines."
        )
    return BuiltinToolError(
        f"Could not find edits[{i}] in {path}. The oldText must match exactly including all "
        "whitespace and newlines."
    )


def _duplicate(path: str, i: int, n: int, occurrences: int) -> BuiltinToolError:
    if n == 1:
        return BuiltinToolError(
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. "
            "Please provide more context to make it unique."
        )
    return BuiltinToolError(
        f"Found {occurrences} occurrences of edits[{i}] in {path}. Each oldText must be unique. "
        "Please provide more context to make it unique."
    )


def _no_change(path: str, n: int) -> BuiltinToolError:
    if n == 1:
        return BuiltinToolError(
            f"No changes made to {path}. The replacement produced identical content. This might "
            "indicate an issue with special characters or the text not existing as expected."
        )
    return BuiltinToolError(
        f"No changes made to {path}. The replacements produced identical content."
    )


def apply_edits(normalized: str, edits: list[tuple[str, str]], path: str) -> tuple[str, str]:
    """`applyEditsToNormalizedContent`: returns `(base, new)`. `edits` are `(oldText, newText)`
    pairs in `_utf16` form; every diagnostic is Pi's template."""
    n = len(edits)
    normalized_edits = [(normalize_to_lf(old), normalize_to_lf(new)) for old, new in edits]
    for i, (old, _) in enumerate(normalized_edits):
        if not old:
            if n == 1:
                raise BuiltinToolError(f"oldText must not be empty in {path}.")
            raise BuiltinToolError(f"edits[{i}].oldText must not be empty in {path}.")
    used_fuzzy = any(fuzzy_find(normalized, old).used_fuzzy for old, _ in normalized_edits)
    base = fuzzy_normalize(normalized) if used_fuzzy else normalized
    matched: list[Replacement] = []
    for i, (old, new) in enumerate(normalized_edits):
        match = fuzzy_find(base, old)
        if not match.found:
            raise _not_found(path, i, n)
        occurrences = count_occurrences(base, old)
        if occurrences > 1:
            raise _duplicate(path, i, n, occurrences)
        matched.append(Replacement(i, match.index, match.length, new))
    matched.sort(key=lambda r: r.index)
    for previous, current in itertools.pairwise(matched):
        if previous.index + previous.length > current.index:
            raise BuiltinToolError(
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )
    new_content = (
        preserve_unchanged_lines(normalized, base, matched)
        if used_fuzzy
        else apply_replacements(base, matched)
    )
    if new_content == normalized:
        raise _no_change(path, n)
    return normalized, new_content


def generate_unified_patch(path_units: str, old: str, new: str, context: int = 4) -> str:
    """`generateUnifiedPatch`."""
    return create_two_files_patch(path_units, path_units, old, new, context)


def generate_diff_string(old: str, new: str, context: int = 4) -> tuple[str, int | None]:
    """`generateDiffString`: `(diff, firstChangedLine)`."""
    parts = diff_lines(old, new)
    output: list[str] = []
    width = len(str(max(len(old.split("\n")), len(new.split("\n")))))
    old_num = new_num = 1
    last_was_change = False
    first_changed: int | None = None
    ellipsis = " " + "".rjust(width) + " ..."

    def context_line(line: str) -> None:
        nonlocal old_num, new_num
        output.append(f" {str(old_num).rjust(width)} {line}")
        old_num += 1
        new_num += 1

    for i, part in enumerate(parts):
        raw = part.value.split("\n")
        if raw[-1] == "":
            raw.pop()
        if part.added or part.removed:
            if first_changed is None:
                first_changed = new_num
            for line in raw:
                if part.added:
                    output.append(f"+{str(new_num).rjust(width)} {line}")
                    new_num += 1
                else:
                    output.append(f"-{str(old_num).rjust(width)} {line}")
                    old_num += 1
            last_was_change = True
            continue
        next_is_change = i < len(parts) - 1 and (parts[i + 1].added or parts[i + 1].removed)
        if last_was_change and next_is_change:
            if len(raw) <= context * 2:
                for line in raw:
                    context_line(line)
            else:
                leading = raw[:context]
                trailing = raw[len(raw) - context :]
                skipped = len(raw) - len(leading) - len(trailing)
                for line in leading:
                    context_line(line)
                output.append(ellipsis)
                old_num += skipped
                new_num += skipped
                for line in trailing:
                    context_line(line)
        elif last_was_change:
            shown = raw[:context]
            skipped = len(raw) - len(shown)
            for line in shown:
                context_line(line)
            if skipped > 0:
                output.append(ellipsis)
                old_num += skipped
                new_num += skipped
        elif next_is_change:
            skipped = max(0, len(raw) - context)
            if skipped > 0:
                output.append(ellipsis)
                old_num += skipped
                new_num += skipped
            for line in raw[skipped:]:
                context_line(line)
        else:
            old_num += len(raw)
            new_num += len(raw)
        last_was_change = False
    return "\n".join(output), first_changed
