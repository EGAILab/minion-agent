"""Node's `readline` over a child's stdout, as `find`/`grep` consume it (spec/tools.md WP-13.4,
`find` step 7): the default UTF-8 `StringDecoder` (WHATWG replacement, a character split across
chunks kept whole), and a line ends at LF, CRLF **or CR** -- a CR that ends one chunk followed by an
LF that starts the next is one break, as Node's `readline` treats it. A final unterminated line is
emitted at end of input; a trailing break emits no empty line.
"""

from __future__ import annotations

import codecs


class LineSplitter:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""
        self._skip_lf = False

    def feed(self, data: bytes) -> list[str]:
        return self._split(self._decoder.decode(data), final=False)

    def finish(self) -> list[str]:
        return self._split(self._decoder.decode(b"", final=True), final=True)

    def _split(self, text: str, *, final: bool) -> list[str]:
        if self._skip_lf and text:
            if text[0] == "\n":
                text = text[1:]
            self._skip_lf = False
        buffer = self._pending + text
        lines: list[str] = []
        start = 0
        index = 0
        while index < len(buffer):
            char = buffer[index]
            if char == "\n" or char == "\r":
                lines.append(buffer[start:index])
                if char == "\r":
                    if index + 1 < len(buffer):
                        if buffer[index + 1] == "\n":
                            index += 1
                    else:
                        self._skip_lf = True
                start = index + 1
            index += 1
        self._pending = buffer[start:]
        if final and self._pending:
            lines.append(self._pending)
            self._pending = ""
        return lines
