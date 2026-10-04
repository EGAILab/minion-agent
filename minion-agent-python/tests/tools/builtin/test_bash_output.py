"""`TOOL-035`: the `bash` output accumulator and tail truncation (spec/tools.md WP-13.3 "Output"),
against the pinned-Pi authority rows in `data/wp133/` (see its README)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from minion_agent.tools.builtin.bash_output import (
    OutputAccumulator,
    _bytes_from_end,
    truncate_tail,
)

DATA = Path(__file__).parent / "data" / "wp133"


def _load(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _rep(text: str, times: int) -> str:
    return text * times


# The chunk sequences of boundary_probe.mjs's `rolling` rows (CON-R002, N001).
ROLLING: dict[str, list[str]] = {
    "singleLineOneChunk": [_rep("a", 250000)],
    "singleLineManyChunks": [_rep("a", 50000)] * 5,
    "singleLineMultibyte": [_rep("€", 70000)],
    "cutMidLineNewlineLater": [_rep("a", 150000) + "\n" + _rep("b", 60000)],
    "cutAfterNewline": [_rep("a", 110000) + "\n" + _rep("b", 102400)],
    "cutMidLineNewlineInLaterChunk": [_rep("a", 250000), "\n" + _rep("b", 1000)],
    "atTriggerNoTrim": [_rep("a", 204800)],
    "longLineThenNewline": [_rep("a", 250000) + "\n"],
    "longLineThenTenLines": [_rep("a", 250000) + _rep("x\n", 10)],
    "longLineThen2001Lines": [_rep("a", 250000) + _rep("x\n", 2001)],
}


def _accumulate(chunks: list[bytes]) -> OutputAccumulator:
    accumulator = OutputAccumulator()
    for chunk in chunks:
        accumulator.append(chunk)
    accumulator.finish()
    return accumulator


@pytest.mark.parametrize("row", sorted(ROLLING))
def test_rolling_tail_matches_pinned_accumulator(row: str) -> None:
    expected = _load("boundary-win32.json")["rolling"][row]
    accumulator = _accumulate([chunk.encode("utf-8") for chunk in ROLLING[row]])
    truncation = accumulator.snapshot()
    content = truncation.pop("content")
    assert len(content) == expected["content"]["length"]
    assert hashlib.sha256(content.encode("utf-8")).hexdigest() == expected["content"]["sha256"]
    assert truncation == expected["truncation"]
    assert accumulator.last_line_bytes == expected["lastLineBytes"]


BOM_CHUNKS = {
    "initial": ["efbbbf61"],
    "splitInitial": ["ef", "bbbf61"],
    "splitThree": ["ef", "bb", "bf", "61"],
    "nonInitial": ["78", "efbbbf61"],
    "doubleInitial": ["efbbbfefbbbf61"],
    "initialOnly": ["efbbbf"],
    "partialBomThenOther": ["efbb", "61"],
}


@pytest.mark.parametrize("row", sorted(BOM_CHUNKS))
def test_one_leading_bom_of_the_merged_stream_is_stripped(row: str) -> None:
    """`WP133-AUD-R002`: one streaming decoder over the merged chunks strips exactly one leading
    BOM, even split across chunks; a later BOM is kept."""
    expected = _load("projection-win32.json")["bom"][row]
    accumulator = _accumulate([bytes.fromhex(chunk) for chunk in BOM_CHUNKS[row]])
    truncation = accumulator.snapshot()
    assert [format(ord(char), "04X") for char in truncation["content"]] == expected["content"]
    assert truncation["totalBytes"] == expected["totalBytes"]
    assert truncation["outputBytes"] == expected["outputBytes"]


def test_raw_bytes_count_the_stripped_bom() -> None:
    accumulator = _accumulate([b"\xef\xbb\xbfa"])
    assert accumulator.total_raw_bytes == 4
    assert accumulator.total_decoded_bytes == 1


def test_one_decoder_across_both_streams_and_chunks() -> None:
    """A character interrupted by the other stream's chunk, and one split across two writes
    (characterization section 11: `\\ufffdx\\ufffd\\ufffd` and the euro sign)."""
    interrupted = _accumulate([b"\xe2", b"x", b"\x82\xac"]).snapshot()["content"]
    assert interrupted == "�x��"
    split = _accumulate([b"\xe2\x82", b"\xac"]).snapshot()["content"]
    assert split == "€"
    assert _accumulate([b"ok\xe2\x82"]).snapshot()["content"] == "ok�"


def test_temp_file_trigger_and_counters() -> None:
    accumulator = OutputAccumulator()
    accumulator.append(b"a" * 51200)
    assert not accumulator.should_use_temp_file
    accumulator.append(b"b")
    assert accumulator.should_use_temp_file
    lines = OutputAccumulator()
    lines.append(b"x\n" * 2001)
    assert lines.should_use_temp_file and lines.total_lines == 2001


def test_finished_accumulator_rejects_appends_and_finishes_once() -> None:
    accumulator = OutputAccumulator()
    accumulator.finish()
    accumulator.finish()
    with pytest.raises(RuntimeError):
        accumulator.append(b"x")


def test_truncate_tail_partial_line_and_line_limit() -> None:
    partial = truncate_tail("x" * 10 + "€" * 5, max_bytes=7)
    assert partial["content"] == "€" * 2 and partial["lastLinePartial"] is True
    assert partial["truncatedBy"] == "bytes" and partial["outputBytes"] == 6
    by_lines = truncate_tail("1\n2\n3\n", max_lines=2)
    assert by_lines["content"] == "2\n3" and by_lines["truncatedBy"] == "lines"
    assert truncate_tail("")["content"] == "" and truncate_tail("")["totalLines"] == 0


def test_trim_tail_short_buffer_resets_its_byte_count() -> None:
    """Pi's `trimTail` early return: a buffer already within the rolling bound only refreshes
    `tailBytes` (reachable when the running count overshoots the encoded length)."""
    accumulator = OutputAccumulator(max_bytes=4)
    accumulator._tail_text = "ab"
    accumulator._tail_bytes = 99
    accumulator._trim_tail()
    assert accumulator._tail_bytes == 2


def test_bytes_from_end_keeps_short_text_and_cuts_at_a_character_boundary() -> None:
    """Pi's `truncateStringToBytesFromEnd`: text within the limit is returned unchanged; a cut never
    starts inside a multi-byte character."""
    assert _bytes_from_end("abc", 3) == "abc"
    assert _bytes_from_end("aéb", 2) == "b"  # 'é' is 2 bytes: a cut at byte 2 skips its tail
