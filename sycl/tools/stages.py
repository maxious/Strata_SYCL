import re
import sys

NAMES = ["pre", "hc-read0", "qkv gemv", "conv", "ab", "z", "rec", "kv-idx", "k/v rope", "kv append",
         "q", "scores+topk", "kv-resolve", "attention", "gate", "", "out-proj", "router+ring", "shared",
         "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "combine", "", "head"]

log = sys.argv[1] if len(sys.argv) > 1 else None
if log:
    text = open(log, errors='replace').read()
    events = re.findall(r'^\s*\d[\d.]* (STAGE)\s+v\S+ window (\d+) step (\d+) layer (\d+) aux (\d+)',
                        text, re.M)
    print('stage events:', len(events))
    for w in events[-8:]:
        _, win, i, l, aux = w
        i = int(i)
        name = NAMES[i] if 0 <= i < len(NAMES) else '?'
        print('last stages: layer %-3s step %-3s grp %s -> %s' % (l, i, aux, name))
