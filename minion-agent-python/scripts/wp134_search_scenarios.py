"""Generate the WP-13.4 `find`/`grep` canonical scenarios (`conformance/agent/builtin-search/`) from
the pinned-Pi authority outputs -- no expectation is hand-written.

    python scripts/wp134_search_scenarios.py <minion-agent-docs>/assurance/layers/data/13-wp134/out

Inputs are `search-win32.json` and `search-linux.json` from `harness/search_probe.mjs`, which runs
pinned Pi's own `find.ts`/`grep.ts` over the pinned engines. The corpus those runs used is written
alongside as `corpus.json` (transcribed from the harness's `build()`, `bulk()` and `edges()`), so
every binding builds the identical tree.

Comparison modes follow spec/tools.md WP-13.4 "Result order" (Owner Q2): cross-entry and cross-file
order is unspecified, so `find` results compare as a multiset (a limited result as a count plus a
sub-multiset of its unlimited reference), `grep` output compares per file (each file's lines exact,
in order, contiguous). The bulk cases use the same modes over members derived from the bulk
corpus and checked against the recorded observations.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "conformance" / "agent" / "builtin-search"
AUTHORITY = (
    "minion-agent-docs spec/tools.md WP-13.4 (master f3ae3ced); pinned-Pi search_probe outputs"
)
PI = "b7bb00b936dbe21b8e160b3e89efdec361846699"


def _text(content: str) -> dict[str, str]:
    return {"text": content}


def _hex(data: bytes) -> dict[str, str]:
    return {"hex": data.hex()}


def corpus() -> dict[str, Any]:
    ctx = "\n".join(f"L{i + 1} MATCH" if i in (1, 2) else f"L{i + 1}" for i in range(10)) + "\n"
    files: dict[str, dict[str, str]] = {
        ".gitignore": _text("ignored/\n*.log\nkeep.ts\n"),
        "src/a.ts": _text("alpha\nBeta\nfoo bar\nfoo.bar\nlast\n"),
        "src/b.spec.ts": _text("describe foo\n"),
        "src/sub/c.ts": _text("const foo = 1;\n"),
        "src/sub/d.spec.ts": _text("it foo\n"),
        "src/sub/deep/er/e.spec.ts": _text("deep it\n"),
        "src/.hidden.ts": _text("hidden foo\n"),
        ".hiddendir/e.ts": _text("hidden dir foo\n"),
        "data/x.json": _text('{"foo": 1}\n'),
        "data/y.JSON": _text('{"FOO": 2}\n'),
        "ünïcode/文件.ts": _text("naïve 日本語 foo\n"),
        "space dir/f.ts": _text("space foo\n"),
        "ignored/g.ts": _text("ignored foo\n"),
        "z.log": _text("log foo\n"),
        "keep.ts": _text("root keep foo\n"),
        "node_modules/pkg/index.ts": _text("module foo\n"),
        "nested/.gitignore": _text("inner.ts\n"),
        "nested/inner.ts": _text("inner foo\n"),
        "nested/keep.ts": _text("nested keep foo\n"),
        "text/crlf.txt": _text("one\r\ntwo match\r\nthree\r\n"),
        "text/cr.txt": _text("x\ry match\rz"),
        "text/long.txt": _text("a" * 600 + " needle " + "b" * 10 + "\nshort needle\n"),
        "text/unicode.txt": _text("café\nnaïve match\n日本語 match\nCAFÉ\n"),
        "text/bin.dat": _hex(b"match\n\x00\x00\x01binary match\n"),
        "text/many.txt": _text("".join(f"hit {i + 1}\n" for i in range(10))),
        "text/ctx.txt": _text(ctx),
        "text/edge.txt": _text("MATCH first\nmiddle\n\nlast MATCH\n"),
        "text/dash.txt": _text("-v option\nplain\n"),
        "text/multi.txt": _text("x x x\n"),
        "text/bom.txt": _hex(b"\xef\xbb\xbffirst match\nsecond\n"),
        "text/badutf8.txt": _hex(b"ok line\nbad \xff\xfe match\nafter\n"),
        " lead.ts": _text("leading space name\n"),
    }
    base = {
        "files": files,
        "directories": ["nested/.git"],
        "raw_name_files": {
            "linux": {
                "text/raw-ff.txt": {"name_hex": "7261772dff2e747874", "text": "rawname match\n"}
            }
        },
        "links": [
            {"path": "link-file.ts", "target": "src/a.ts", "kind": "file"},
            {"path": "link-dir", "target": "src/sub", "kind": "dir", "windows": "junction"},
        ],
    }
    pad = "p" * 48
    wide = "\n".join(f"hit {i:03d} " + "w" * 490 for i in range(120)) + "\n"
    return {
        "$comment": (
            "WP-13.4 canonical corpora, transcribed from minion-agent-docs "
            "assurance/layers/data/13-wp134/harness/search_probe.mjs build()/bulk()/edges(). "
            "raw_name_files are POSIX-only file names given as hex bytes, placed under their "
            "directory."
        ),
        "plain": base,
        "repo": {**base, "directories": ["nested/.git", ".git"]},
        "bulk": {
            "files": {
                **{f"many/f{i:04d}-{pad}.txt": _text("x\n") for i in range(1200)},
                "wide/w.txt": _text(wide),
            },
            "directories": [],
            "links": [],
        },
        "junction": {
            "$comment": (
                "win32 only: a .git junction whose target directory is then removed (dangling)"
            ),
            "files": {"src/same.ts": _text("x\n"), ".gitignore": _text("src/same*\n")},
            "directories": [],
            "links": [],
            "dangling_git_junction": True,
        },
        "duplicate": {
            "$comment": "linux only: two distinct names that Pi's trim() makes identical",
            "files": {"src/same.ts": _text("x\n"), "src/same.ts ": _text("x\n")},
            "directories": [],
            "links": [],
        },
    }


_GREP_LINE = re.compile(r"^(.*?)(?::(-?\d+(?:\.\d+)?): |-(-?\d+(?:\.\d+)?)- )")
_REFERENCE = {
    "limit-2": "ts-basename",
    "limit-zero": "ts-basename",
    "limit-exact": "case-json-lower",
}


def _split_notice(text: str) -> tuple[str, str | None]:
    body, sep, notice = text.partition("\n\n[")
    return body, ("[" + notice) if sep else None


def grep_files(body: str) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    for line in body.split("\n"):
        match = _GREP_LINE.match(line)
        key = match.group(1) if match else ""
        files.setdefault(key, []).append(line)
    return files


def _details(details: dict[str, Any] | None) -> dict[str, Any]:
    """Expected details: exact, except `truncation.content`, which depends on which entries an
    unspecified traversal kept; runners check it equals the returned body instead."""
    result = dict(details or {})
    if "truncation" in result:
        result["truncation"] = {k: v for k, v in result["truncation"].items() if k != "content"}
    return result


def expectation(
    tool: str, case: str, observed: dict[str, Any], plat: dict[str, Any], kind: str
) -> dict[str, Any]:
    if not observed["ok"]:
        return {"is_error": True, "mode": "exact", "text": observed["error"], "details": {}}
    text, details = observed["text"], _details(observed["details"])
    if text in ("No files found matching pattern", "No matches found", ""):
        return {"is_error": False, "mode": "exact", "text": text, "details": details}
    body, notice = _split_notice(text)
    if tool == "find":
        if "resultLimitReached" in details and case in _REFERENCE:
            reference = plat["find"][f"{kind}/{_REFERENCE[case]}"]["text"].split("\n")
            return {
                "is_error": False,
                "mode": "find_subset",
                "count": len(body.split("\n")),
                "of": sorted(reference),
                "notice": notice,
                "details": details,
            }
        return {
            "is_error": False,
            "mode": "find_multiset",
            "entries": sorted(body.split("\n")),
            "notice": notice,
            "details": details,
        }
    return {
        "is_error": False,
        "mode": "grep_by_file",
        "files": grep_files(body),
        "notice": notice,
        "details": details,
    }


def bulk_expectation(tool: str, case: str, observed: dict[str, Any]) -> dict[str, Any]:
    """The bulk cases (WP134-IMPL-R003). The authority run records a summary of each: its line
    count, body bytes, last line, notice and details (for `find`, `truncation.content` is the body
    itself). The allowed members are derived from the bulk corpus, formatted as Pi formats them,
    and every recorded observation is checked against that derivation before it is used."""
    corpus_files = corpus()["bulk"]["files"]
    if tool == "find":
        allowed = sorted(
            name.removeprefix("many/") for name in corpus_files if name.startswith("many/")
        )
        body = observed["details"]["truncation"]["content"]
        entries = body.split("\n")
        assert len(entries) == observed["lines"] and entries[-1] == observed["lastLine"], case
        assert len(body.encode("utf-8")) == observed["bodyBytes"], case
        assert not Counter(entries) - Counter(allowed), case
        return {
            "is_error": False,
            "mode": "find_subset",
            "count": observed["lines"],
            "of": allowed,
            "notice": observed["notice"],
            "details": _details(observed["details"]),
        }
    source = corpus_files["wide/w.txt"]["text"].split("\n")[:-1]
    lines = [f"w.txt:{n}: {text}" for n, text in enumerate(source, 1)][: observed["lines"]]
    assert lines[-1] == observed["lastLine"], case
    assert len("\n".join(lines).encode("utf-8")) == observed["bodyBytes"], case
    return {
        "is_error": False,
        "mode": "grep_by_file",
        "files": {"w.txt": lines},
        "notice": observed["notice"],
        "details": _details(observed["details"]),
    }


def main(out_dir: Path) -> None:
    platforms = {
        name: json.loads((out_dir / f"search-{name}.json").read_text(encoding="utf-8"))
        for name in ("win32", "linux")
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "corpus.json").write_text(
        json.dumps(corpus(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    scenarios: dict[str, dict[str, Any]] = {}

    def add(
        name: str,
        tool: str,
        kind: str,
        arguments: dict[str, Any],
        plat_name: str,
        expect: dict[str, Any],
        signal: str | None = None,
    ) -> None:
        entry = scenarios.setdefault(
            name,
            {
                "name": name,
                "family": "agent",
                "authority": AUTHORITY,
                "pi_revision": PI,
                "requirements": ["TOOL-036" if tool == "find" else "TOOL-037"],
                "witnesses": [name.replace("-", "_")],
                "builtin_search": {
                    "tool": tool,
                    "corpus": kind,
                    "arguments": arguments,
                    "expect": {},
                },
            },
        )
        if signal:
            entry["builtin_search"]["signal"] = signal
        entry["builtin_search"]["expect"][plat_name] = expect

    for plat_name, plat in platforms.items():
        for tool in ("find", "grep"):
            for key, observed in plat[tool].items():
                kind, case = key.split("/", 1)
                signal = "pre_aborted" if case == "pre-aborted" else None
                add(
                    f"builtin-search-{tool}-{kind}-{case}",
                    tool,
                    kind,
                    observed["args"],
                    plat_name,
                    expectation(tool, case, observed, plat, kind),
                    signal,
                )
        for key, observed in plat["bulk"].items():
            tool, case = key.split("/", 1)
            arguments = (
                {"pattern": "*.txt", "path": "many"}
                if tool == "find"
                else {"pattern": "hit", "path": "wide/w.txt"}
            )
            if case == "limit-1200-bytes-only":
                arguments["limit"] = 1200
            add(
                f"builtin-search-{tool}-bulk-{case}",
                tool,
                "bulk",
                arguments,
                plat_name,
                bulk_expectation(tool, case, observed),
            )
        # WP134-IMPL-R002: genuine `**/` components adjacent to another or inside braces
        # (harness `components` mode, its own output beside the default run's).
        components = out_dir / f"components-{plat_name}.json"
        for key, observed in json.loads(components.read_text(encoding="utf-8"))["find"].items():
            kind, case = key.split("/", 1)
            add(
                f"builtin-search-find-{kind}-{case}",
                "find",
                kind,
                observed["args"],
                plat_name,
                expectation("find", case, observed, plat, kind),
            )
        for key, observed in plat.get("edges", {}).items():
            tool, case = key.split("/", 1)
            kind = "junction" if case == "dangling-git-junction" else "duplicate"
            arguments = (
                {"pattern": "same*", "path": "src"}
                if kind == "junction"
                else {"pattern": "same.ts*", "path": "src"}
            )
            add(
                f"builtin-search-{tool}-{kind}-{case}",
                tool,
                kind,
                arguments,
                plat_name,
                expectation(tool, case, observed, plat, kind),
            )

    # DIV-002 (Owner Q1): a Windows full-path pattern with a "**" component keeps the zero-directory
    # meaning, so Minion's win32 expectation is the corrected (Linux) result; Pi's own Windows
    # output stays recorded as the reference side of the divergence.
    for scenario in scenarios.values():
        search = scenario["builtin_search"]
        pattern = search["arguments"].get("pattern", "")
        expect = search["expect"]
        if search["tool"] != "find" or "/" not in pattern or "**" not in pattern:
            continue
        if {"win32", "linux"} <= expect.keys() and expect["win32"] != expect["linux"]:
            pi_win32 = expect["win32"]
            expect["win32"] = expect["linux"]
            scenario["requirements"] = ["TOOL-036", "TOOL-036-DIV-002"]
            scenario["notes"] = (
                "DIV-002: the win32 expectation is the corrected result (Windows equals Linux). "
                "Pinned Pi's own Windows result, the divergence's reference side: "
                + json.dumps(pi_win32, ensure_ascii=False)
            )

    for old in OUT_DIR.glob("builtin-search-*.yaml"):
        old.unlink()
    for name, scenario in sorted(scenarios.items()):
        (OUT_DIR / f"{name}.yaml").write_text(
            yaml.safe_dump(scenario, allow_unicode=True, sort_keys=False, width=1000),
            encoding="utf-8",
        )
    print(f"wrote {len(scenarios)} scenarios and corpus.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
