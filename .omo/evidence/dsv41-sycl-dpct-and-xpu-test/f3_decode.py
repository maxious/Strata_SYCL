#!/usr/bin/env python
"""F3 QA decode/analysis helper. Usage:
  f3_decode.py --ids <file>            decode an ids file
  f3_decode.py --log <file>            extract comma-joined id batches from an engine log, decode each
  f3_decode.py --types <file>          extract ids from log, report any id whose token_type is 3
"""
import json, re, sys, pathlib

PACK = pathlib.Path("/home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4")
vocab = json.load(open(PACK / "tokenizer/vocab.json"))
inv = {i: t for t, i in vocab.items()}
types = json.load(open(PACK / "tokenizer/token_type.json"))

BATCH_RE = re.compile(r"(?<![\d.])\d+(?:,\d+)+,?(?![\d.])")


def batches_from_log(text):
    out = []
    for m in BATCH_RE.finditer(text):
        ids = [int(x) for x in m.group(0).rstrip(",").split(",")]
        out.append(ids)
    return out


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
elif sys.argv[1] in ("--log", "--types"):
    report(batches_from_log(open(sys.argv[2]).read()))
else:
    sys.exit("bad mode")
