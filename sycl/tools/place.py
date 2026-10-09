#!/usr/bin/env python3
import re
import sys

log = sys.argv[1]
addrs = [int(a, 16) for a in sys.argv[2:]]
text = open(log, errors="replace").read()
alloc = []
for m in re.finditer(r"strata alloc: ([a-z ]+?) (0x[0-9a-fA-F]+) (\d+) bytes ra=(0x[0-9a-fA-F]+)", text):
    alloc.append((int(m.group(2), 16), int(m.group(3)), m.group(4), "engine"))
for m in re.finditer(r"ur_alloc: (\w+) (0x[0-9a-fA-F]+) (\d+) bytes", text):
    alloc.append((int(m.group(2), 16), int(m.group(3)), "ur", "ur"))

uniq = {}
for p, n, ra, k in alloc:
    uniq[(p, n, k)] = ra
items = sorted((p, n, k, ra) for (p, n, k), ra in uniq.items())
print("distinct allocations:", len(items))

for f in addrs:
    ins = [(p, n, k, ra) for (p, n, k, ra) in items if p <= f < p + n]
    print("\n" + hex(f) + ": inside " + str(len(ins)) + " allocations")
    for p, n, k, ra in ins[:4]:
        print("   " + hex(p) + " +" + str(n) + " off=" + hex(f - p) + " " + k + " ra=" + ra)
    near = [(p, n, k, ra) for (p, n, k, ra) in items if abs(p - f) < 0x4000000]
    for p, n, k, ra in near[:8]:
        print("   near " + hex(p) + " +" + str(n) + " " + k + " ra=" + ra)
    ends = [(p, n, k, ra) for (p, n, k, ra) in items if abs((p + n) - f) < 0x4000000]
    for p, n, k, ra in ends[:8]:
        print("   endsnear " + hex(p) + " end=" + hex(p + n) + " d=" + str(f - (p + n)) + " " + k)
