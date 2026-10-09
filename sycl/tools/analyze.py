#!/usr/bin/env python3
"""Splits an xe device coredump on its '**** Section ****' markers and prints the sections a fault hunt needs, with
the huge binary ones (GuC log, buffers) truncated rather than skipped entirely."""
import re
import sys

path = sys.argv[1]
want = sys.argv[2:] or ["Job", "HW Engines", "VM state", "Contexts", "GuC CT"]
raw = open(path, "rb").read().decode("utf-8", errors="replace")

marks = [(m.start(), m.group(1).strip()) for m in re.finditer(r"^\*\*\*\* (.+?) \*\*\*\*$", raw, re.M)]
print(f"{path}: {len(raw)} chars, sections: {[n for _, n in marks]}\n")
for i, (pos, name) in enumerate(marks):
    if not any(w.lower() in name.lower() for w in want):
        continue
    end = marks[i + 1][0] if i + 1 < len(marks) else len(raw)
    body = raw[pos:end].rstrip()
    lines = body.splitlines()
    print(f"===== {name} ({len(body)} chars, {len(lines)} lines) =====")
    if len(lines) <= 90:
        print(body)
    else:
        print("\n".join(lines[:60]))
        print(f"     ... [{len(lines) - 80} lines elided: a run of repeated/numeric state] ...")
        print("\n".join(lines[-20:]))
    print()
