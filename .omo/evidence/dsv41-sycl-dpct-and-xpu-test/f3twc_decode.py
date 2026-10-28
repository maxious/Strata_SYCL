#!/usr/bin/env python
"""F3 two-card QA decode helper (task st_01a178ea). Usage:
  f3twc_decode.py --log <engine log>   extract comma-joined id batches, decode each with the pack tokenizer
  f3twc_decode.py --ids <file>         decode an ids file
Golden-checkable: same vocab/inv/token_type mapping as the pack tokenizer files.
"""
import json, re, sys, pathlib

PACK = pathlib.Path("/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4")
vocab = json.load(open(PACK / "tokenizer/vocab.json"))
inv = {i: t for t, i in vocab.items()}
types = json.load(open(PACK / "tokenizer/token_type.json"))

BATCH_RE = re.compile(r"(?<![\d.])\d+(?:,\d+)+,?(?![\d.])")


def batches_from_log(text):
    return [[int(x) for x in m.group(0).rstrip(",").split(",")] for m in BATCH_RE.finditer(text)]


def ids_from_file(path):
    txt = open(path).read()
    return [int(t) for t in re.split(r"[,\s]+", txt.strip()) if t.isdigit()]


def decode(ids):
    s = "".join(inv.get(i, "<unk:%d>" % i) for i in ids)
    return s.replace("\u0120", " ").replace("\u010a", "\n")


def report(ids_list):
    for n, ids in enumerate(ids_list, 1):
        print("batch%d ids: %s" % (n, " ".join(map(str, ids))))
        print("batch%d first_id: %d" % (n, ids[0] if ids else -1))
        t3 = [(i, types[i]) for i in ids if i < len(types) and types[i] == 3]
        print("batch%d type3_ids: %s" % (n, t3 if t3 else "none"))
        print("batch%d decoded: %s" % (n, decode(ids)))


if sys.argv[1] == "--ids":
    report([ids_from_file(sys.argv[2])])
elif sys.argv[1] == "--log":
    report(batches_from_log(open(sys.argv[2]).read()))
else:
    sys.exit("bad mode")
