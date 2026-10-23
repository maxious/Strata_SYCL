#!/usr/bin/env python3
"""Gyro-S against Gyro-M on one or two Intel Arc cards (the SYCL engine).

    python3 tools/bench_gyro_sycl.py --models-dir ~/gyro --runs 2
    python3 tools/bench_gyro_sycl.py --mode dual --runs 1
    python3 tools/bench_gyro_sycl.py --download --models-dir ~/gyro --verify-sha
    python3 tools/bench_gyro_sycl.py --label Gyro-Coder --gguf ~/models/Gyro-Coder.gguf \
        --pack ~/packs/gyro-coder --mode single --contexts 32768,65536,131072,262144

For each size and mode it starts a fresh server (sycl/serve/server_intel.py with this build's engine), waits for
/health, sends the same greedy prompts through the real OpenAI API, and writes
bench/results/<date>-gyro-sycl-<cards>/ (results.json, matrix.md, config-*.json, engine and server logs).

Modes:
  single  one card, ONEAPI_DEVICE_SELECTOR=level_zero:<card>, no layer split
  dual    both cards, ONEAPI_DEVICE_SELECTOR=level_zero:* and --layer-split auto --split-device 1
  both    every size in single and then dual (the default)

A configuration that cannot answer is recorded with the engine's own error and the run continues; the table then
shows the failure, never an invented number.

Requirements: the SYCL engine built on this PC (build-sycl-aot/strata or build-sycl/strata), oneAPI installed (the
script re-runs itself through /opt/intel/oneapi/setvars.sh when needed), an Intel Arc in /sys/class/drm, Python 3.10+
and the model files (--download fetches the pinned ones).  No Docker.

Speeds are the engine's own timings (prompt_per_second, predicted_per_second); time to first token is measured
here.  Decode includes the MTP draft layer (--spec 4), so the acceptance rate is reported beside it.  Every prompt
is new per request (no cache help).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

HF_REPO = "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"
HF_REVISION = "1ada08422e404aca51ab9df665f089f34c419c6b"
MODELS: dict[str, dict[str, Any]] = {
    "gyro-s": {"title": "Gyro-S", "file": "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf",
               "bytes": 58487015808, "sha256": "c6ea1b08ca312f99095b27588ba35976b53f948268e7f91e7761926aa01d9c87"},
    "gyro-m": {"title": "Gyro-M", "file": "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf",
               "bytes": 91965774272, "sha256": "7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f"},
}
INTEL_ARC = {"e223": ("Arc Pro B70", 32.0), "e221": ("Arc Pro B60", 24.0), "e211": ("Arc Pro B60", 24.0),
             "e20b": ("Arc B580", 12.0), "e20c": ("Arc B570", 10.0), "e212": ("Arc B50", 16.0),
             "56a0": ("Arc A770", 16.0), "56a1": ("Arc A750", 8.0), "56a2": ("Arc A580", 8.0),
             "56a5": ("Arc A380", 6.0), "56a6": ("Arc A310", 4.0), "5690": ("Arc A770M", 16.0)}
SUITE = [("short", 512, 128), ("2k", 2048, 128), ("8k", 8192, 64)]
FILLER = ("The archive keeps one copy of every report, and the index lists each one by the day it arrived. "
          "A reader who knows the date can find the folder without asking anyone, which is the whole point of it. ")


def ensure_oneapi(argv: list[str]) -> None:
    if os.environ.get("_STRATA_BENCH_REEXEC") or os.environ.get("ONEAPI_ROOT"):
        return
    setvars = Path("/opt/intel/oneapi/setvars.sh")
    if not setvars.is_file():
        return
    print(f"oneAPI is not in this shell: re-running through {setvars}", file=sys.stderr)
    cmd = (f"set +u; source {setvars} >/dev/null 2>&1; exec {sys.executable} {Path(__file__).resolve()} "
           + " ".join(shlex_quote(a) for a in argv))
    os.execvpe("bash", ["bash", "-lc", cmd], dict(os.environ, _STRATA_BENCH_REEXEC="1"))


def shlex_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def intel_gpus() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if "-" in card.name:
            continue
        dev = card / "device"
        try:
            vendor = (dev / "vendor").read_text().strip().lower()
            devid = (dev / "device").read_text().strip().lower().replace("0x", "")
            driver = os.path.basename(os.path.realpath(dev / "driver")) if (dev / "driver").exists() else ""
        except OSError:
            continue
        if vendor != "0x8086" or driver not in ("xe", "i915"):
            continue
        name, vram = INTEL_ARC.get(devid, (f"Intel GPU {devid}", 0.0))
        if not vram and driver == "i915":
            continue
        found.append({"index": len(found), "name": name, "devid": devid, "driver": driver, "vram_gb": vram})
    return found


def find_engine(explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise SystemExit(f"engine not found: {p}")
        return p
    for p in (ROOT / "build-sycl-aot" / "strata", ROOT / "build-sycl" / "strata"):
        if p.is_file():
            return p
    raise SystemExit("the SYCL engine is not built here: build it as docs/INTEL.md says "
                     "(sycl/tools/build.sh in the strata-sycl-dev image), or pass --engine")


def download(models_dir: Path, which: list[str]) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for key in which:
        d = MODELS[key]
        target = models_dir / d["file"]
        if target.is_file() and target.stat().st_size == d["bytes"]:
            print(f"{d['title']}: already there ({target})")
            continue
        print(f"{d['title']}: downloading {d['bytes'] / 1e9:.1f} GB to {target} (revision {HF_REVISION[:12]})")
        subprocess.run(["hf", "download", HF_REPO, d["file"], "--revision", HF_REVISION,
                        "--local-dir", str(models_dir)], check=True)
        if not target.is_file() or target.stat().st_size != d["bytes"]:
            raise SystemExit(f"{d['title']}: {target} is missing or has the wrong size")


def verify_model(path: Path, d: dict[str, Any], sha: bool) -> None:
    size = path.stat().st_size if path.is_file() else -1
    if size != d["bytes"]:
        raise SystemExit(f"{path}: {size} bytes, expected {d['bytes']} - download it with --download")
    if not sha:
        return
    print(f"  sha256 {path.name} ...", end="", flush=True)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != d["sha256"]:
        raise SystemExit(f"\n{path}: sha256 {h.hexdigest()} does not match {d['sha256']}")
    print(" ok")


def gguf_py() -> Path | None:
    if v := os.environ.get("STRATA_GGUF_PY"):
        p = Path(v).expanduser()
        if (p / "gguf").is_dir():
            return p
    candidates = [ROOT / "third_party" / "llama.cpp" / "gguf-py"]
    candidates += sorted(ROOT.glob("build*/_deps/*/gguf-py")) + sorted(ROOT.glob("sycl/build*/_deps/*/gguf-py"))
    return next((p for p in candidates if (p / "gguf").is_dir()), None)


def prepare_pack(gguf: Path, pack: Path, python: str) -> None:
    if (pack / "native_experts.txt").is_file() and (pack / "tokenizer" / "vocab.json").is_file():
        return
    gp = gguf_py()
    if gp is None:
        raise SystemExit("tools/iq_pack.py needs llama.cpp's gguf-py: set STRATA_GGUF_PY to a checkout's gguf-py folder")
    print(f"  packing {gguf.name} -> {pack} (reads the GGUF; a few minutes)")
    pack.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, STRATA_GGUF_PY=str(gp))
    subprocess.run([python, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(gguf), "--out", str(pack),
                    "--compat-bf16"], check=True, env=env)


def free_port(start: int = 8090) -> int:
    for port in range(start, start + 100):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("no free port in 8090-8189")


def server_config(binary: Path, model_key: str, mode: str, card: int, gguf: Path, shard2: Path | None, pack: Path,
                  mtp_rt: Path, profile: Path | None, ctx: int, prefill: int, reserve_mib: int, port: int,
                  log: Path) -> dict[str, Any]:
    args = [
        "--serve",
        "--pack", str(pack),
        "--native", str(gguf),
        "--expert-cache", "auto",
        "--prefill", str(prefill),
        "--spec", "4", "--spec-min-p", "0.5",
        "--mtp", str(mtp_rt),
        "--max-context", str(ctx),
        "--kv", "int8",
        "--stream-experts", "--vram-reserve-mib", str(reserve_mib),
    ]
    if profile is not None:
        args[3:3] = ["--expert-profile", str(profile)]
    if shard2 is not None:
        args += ["--ple-gguf", str(shard2)]
    env = {
        "SYCL_CACHE_PERSISTENT": "0",
        "ZES_ENABLE_SYSMAN": "1",
        "STRATA_VERIFY_DEVICE_PLAN": "1",
        "STRATA_VERIFY_NO_HOST": "1",
    }
    if mode == "dual":
        args += ["--layer-split", "auto", "--split-device", "1"]
        env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:*"
        env["STRATA_STAGE_TRIM"] = "1"
    else:
        env["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{card}"
    return {
        "backend": "sycl",
        "exe": str(binary),
        "args": args,
        "cwd": str(ROOT),
        "tokenizer": str(pack / "tokenizer"),
        "model_name": f"{model_key}-{mode}",
        "log": str(log),
        "port": port,
        "env": env,
    }


def start_server(cfg: dict[str, Any], timeout: float) -> tuple[subprocess.Popen[Any], float]:
    cmd = [sys.executable, str(ROOT / "sycl" / "serve" / "server_intel.py"),
           "--engine", "strata", "--config", str(cfg["_path"]), "--port", str(cfg["port"])]
    log = Path(cfg["log"] + ".server")
    fh = log.open("w")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, **cfg.get("env", {})})
    t0 = time.monotonic()
    url = f"http://127.0.0.1:{cfg['port']}/health"
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"the server exited ({proc.returncode}); see {log} and {cfg['log']}")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return proc, time.monotonic() - t0
        except Exception:                               # noqa: BLE001 - not up yet
            pass
        time.sleep(2)
    stop_server(proc)
    raise TimeoutError(f"the server did not answer /health in {timeout:.0f} s; see {log} and {cfg['log']}")


def start_server_with_fallback(cfg: dict[str, Any], timeout: float) -> tuple[subprocess.Popen[Any], float]:
    try:
        return start_server(cfg, timeout)
    except RuntimeError:
        text = Path(cfg["log"]).read_text(encoding="utf-8", errors="replace") if Path(cfg["log"]).is_file() else ""
        if ("neither in VRAM nor mirrored" not in text
                or cfg["env"].get("STRATA_VERIFY_NO_HOST") != "1"):
            raise
        cfg["env"]["STRATA_VERIFY_NO_HOST"] = "0"
        cfg["_fallback"] = True
        Path(cfg["_path"]).write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=1))
        print("    a later stage cannot hold every expert: retrying with STRATA_VERIFY_NO_HOST=0", flush=True)
        return start_server(cfg, timeout)


def stop_server(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def chat(port: int, text: str, max_tokens: int, timeout: float = 3600) -> dict[str, Any]:
    body = {"model": "strata", "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, pieces, timings, usage, err = None, 0, None, None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except ValueError:
                    continue
                if j.get("error"):
                    err = str(j["error"])[:300]
                for ch in j.get("choices") or []:
                    dl = ch.get("delta") or {}
                    piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                    if piece:
                        if first is None:
                            first = time.perf_counter() - t0
                        pieces += 1
                timings = j.get("timings") or timings
                usage = j.get("usage") or usage
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read()[:300]!r}"
    except Exception as e:                              # noqa: BLE001
        err = repr(e)[:300]
    total = time.perf_counter() - t0
    u, t = usage or {}, timings or {}
    gen = u.get("completion_tokens") or pieces
    decode = t.get("predicted_per_second")
    if decode is None and first is not None and gen > 1 and total > first:
        decode = round((gen - 1) / (total - first), 2)
    return {"ok": err is None and first is not None, "err": err,
            "ttft_s": None if first is None else round(first, 3),
            "total_s": round(total, 3),
            "prompt_tokens": u.get("prompt_tokens"),
            "generated_tokens": gen,
            "prompt_tps": t.get("prompt_per_second"),
            "decode_tps": decode,
            "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
            "timings": t}


def prompt_text(approx_tokens: int, nonce: str) -> str:
    words = max(32, int(approx_tokens / 1.2))
    reps = words // len(FILLER.split()) + 1
    return (f"Note {nonce}.\n\nRead the notes below and answer with three short sentences about what the archive is for.\n\n"
            + (FILLER * reps))[: approx_tokens * 5]


def median(values: list[Any]):
    v = [x for x in values if isinstance(x, (int, float))]
    return round(statistics.median(v), 2) if v else None


def cell(value: Any) -> Any:
    return "-" if value is None else value


def pack_geometry(pack: Path) -> tuple[int, int]:
    table = pack / "native_experts.txt"
    if not table.is_file():
        return 0, 0
    text = table.read_text(encoding="utf-8", errors="replace").splitlines()
    header = next((line for line in text if line.startswith("#")), "")
    m = re.search(r"n_expert (\d+)", header)
    layers = sum(1 for line in text if line and not line.startswith("#"))
    return layers, int(m.group(1)) if m else 0


def engine_facts(log: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return facts
    for line in text.splitlines():
        if m := re.search(r"expert cache auto: .*-> (\d+) slots", line):
            facts["cache_slots"] = int(m.group(1))
        if m := re.search(r"expert cache (\d+) slots, ([\d.]+) GiB", line):
            facts["cache_slots"], facts["cache_gib"] = int(m.group(1)), float(m.group(2))
        if m := re.search(r"layer split: CUDA1 runs layers (\d+)-(\d+)", line):
            facts["split_stage1_layers"] = f"{m.group(1)}-{m.group(2)}"
        if m := re.search(r"layer split: CUDA0 runs layers (\d+)-(\d+)", line):
            facts["split_stage0_layers"] = f"{m.group(1)}-{m.group(2)}"
        if m := re.search(r"layer split: (\d+)% of the experts resident", line):
            facts["resident_percent"] = int(m.group(1))
        if m := re.search(r"(\d+) MiB of VRAM free with everything loaded", line):
            facts["vram_free_mib"] = int(m.group(1))
        if m := re.search(r"decode expert cache hit rate: ([\d.]+)%", line):
            facts["hit_rate"] = float(m.group(1))
        if m := re.search(r"caches hold (\d+) of (\d+) profiled pairs", line):
            facts["held_pairs"], facts["total_pairs"] = int(m.group(1)), int(m.group(2))
        if "UR_RESULT_ERROR_DEVICE_LOST" in line:
            facts["error"] = line.strip()
        elif "error" in line.lower() and "no error" not in line.lower() and line.startswith("strata "):
            facts["engine_last_line"] = line.strip()
    return facts


def run_config(binary: Path, model_key: str, title: str, mode: str, card: int, args: argparse.Namespace,
               gguf: Path, shard2: Path | None, pack: Path, profile: Path | None, mtp_rt: Path, out_dir: Path,
               results: list[dict[str, Any]]) -> None:
    port = free_port()
    stem = f"{model_key}-{mode}"
    log = out_dir / f"{stem}.log"
    cfg = server_config(binary, model_key, mode, card, gguf, shard2, pack, mtp_rt, profile, args.context,
                        args.prefill or (4096 if args.context <= 32768 else 8192),
                        args.reserve_mib or (1024 if args.context <= 32768 else 2048), port, log)
    cfg["_path"] = str(out_dir / f"config-{stem}.json")
    Path(cfg["_path"]).write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=1))
    log.unlink(missing_ok=True)
    print(f"\n=== {title}, {mode} (card {card if mode == 'single' else '0+1'}, port {port}) ===", flush=True)
    proc = None
    try:
        started = time.monotonic()
        proc, _ = start_server_with_fallback(cfg, timeout=args.load_timeout)
        load_s = time.monotonic() - started
        print(f"    up in {load_s:.0f} s", flush=True)
        facts = engine_facts(log)
        if cfg.get("_fallback"):
            facts["host_fallback"] = True
        failed: str | None = None
        for label, approx, max_new in SUITE:
            for run in range(args.runs):
                r = chat(port, prompt_text(approx, f"{stem}-{label}-{run}-{time.time_ns()}"), max_new)
                r.update({"model": model_key, "mode": mode, "prompt": label, "run": run + 1,
                          "prompt_approx": approx})
                results.append(r)
                if r["ok"]:
                    print(f"    {label:5s} run {run + 1}: prompt {r['prompt_tokens']} tok, "
                          f"{r['prompt_tps']} tok/s prompt, {r['decode_tps']} tok/s decode, "
                          f"ttft {r['ttft_s']} s, drafts {r['draft_accepted']}/{r['draft_n']}", flush=True)
                else:
                    print(f"    {label:5s} run {run + 1}: FAILED: {r['err']}", flush=True)
                    failed = r["err"]
                    break
            if failed:
                break
        facts.update({"model": model_key, "mode": mode, "load_s": round(load_s, 1)})
        if failed:
            facts["failed"] = failed
        (out_dir / f"facts-{stem}.json").write_text(json.dumps(facts, indent=1))
    except Exception as e:                              # noqa: BLE001 - one configuration must not end the run
        print(f"    FAILED to start: {e}", flush=True)
        (out_dir / f"facts-{stem}.json").write_text(json.dumps({"model": model_key, "mode": mode,
                                                               "failed": str(e)}, indent=1))
        results.append({"model": model_key, "mode": mode, "prompt": None, "ok": False, "err": str(e)})
    finally:
        if proc is not None:
            stop_server(proc)


def write_report(out_dir: Path, results: list[dict[str, Any]], cards: list[dict[str, Any]], args: argparse.Namespace,
                 binary: Path, order: list[str], titles: dict[str, str]) -> None:
    (out_dir / "results.json").write_text(json.dumps({
        "date": datetime.date.today().isoformat(),
        "engine": str(binary),
        "cards": cards,
        "context": args.context,
        "runs_per_prompt": args.runs,
        "suite": [{"label": l, "approx_prompt_tokens": a, "max_new_tokens": n} for l, a, n in SUITE],
        "results": results,
    }, indent=1))

    facts: dict[tuple[Any, Any], dict[str, Any]] = {}
    for p in sorted(out_dir.glob("facts-*.json")):
        f = json.loads(p.read_text())
        facts[(f.get("model"), f.get("mode"))] = f
    lines = [f"# {', '.join(titles[k] for k in order)} on Intel Arc (SYCL), {datetime.date.today().isoformat()}", "",
             "Cards: " + ", ".join(f"{c['name']} ({c['vram_gb']:.0f} GB, {c['driver']})" for c in cards) + ". "
             f"Context {args.context}, {args.runs} run(s) per prompt, greedy, thinking off, MTP draft 4.",
             "Speeds are the engine's own timings; prompt tokens and draft acceptance come from the same answers.", "",
             "| Model | Mode | Prompt | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT s | Drafts accepted |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    ok = [r for r in results if r.get("prompt") and r["ok"]]
    for key in [(m, mode) for m in order for mode in ("single", "dual")]:
        rows = [r for r in ok if (r["model"], r["mode"]) == key]
        if not rows:
            f = facts.get(key) or {}
            if f.get("failed"):
                lines.append(f"| {titles[key[0]]} | {key[1]} | - | - | - | - | - | failed: {f['failed']} |")
            continue
        for label, _, _ in SUITE:
            rs = [r for r in rows if r["prompt"] == label]
            if not rs:
                continue
            lines.append(f"| {titles[key[0]]} | {key[1]} | {label} | "
                         f"{cell(median([r['prompt_tokens'] for r in rs]))} | "
                         f"{cell(median([r['prompt_tps'] for r in rs]))} | "
                         f"{cell(median([r['decode_tps'] for r in rs]))} | "
                         f"{cell(median([r['ttft_s'] for r in rs]))} | "
                         f"{cell(median([r['draft_accepted'] for r in rs]))}/{cell(median([r['draft_n'] for r in rs]))} |")
    lines += ["", "## Start-up", "",
              "| Model | Mode | Load s | Cache slots | Split | Experts resident | VRAM free MiB | Host computes misses |",
              "|---|---|---:|---:|---|---:|---:|---|"]
    for key in [(m, mode) for m in order for mode in ("single", "dual")]:
        f = facts.get(key)
        if not f:
            continue
        split = (f"{f['split_stage0_layers']} / {f['split_stage1_layers']}"
                 if f.get("split_stage1_layers") else "one card")
        lines.append(f"| {titles[key[0]]} | {key[1]} | {f.get('load_s', '-')} | {f.get('cache_slots', '-')} | "
                     f"{split} | {f.get('resident_percent', '-')} | {f.get('vram_free_mib', '-')} | "
                     f"{'yes' if f.get('host_fallback') else 'no'} |")
    failed = [f for f in facts.values() if f.get("failed")]
    if failed:
        lines += ["", "## Failures", ""]
        for f in failed:
            lines.append(f"- {titles[f['model']]} {f['mode']}: {f['failed']}")
    lines += ["", "The engine and server logs in this folder hold the start-up lines and every request's timings."]
    (out_dir / "matrix.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def sweep_contexts(binary: Path, model_key: str, title: str, mode: str, card: int, args: argparse.Namespace,
                   gguf: Path, shard2: Path | None, pack: Path, profile: Path | None, mtp_rt: Path,
                   contexts: list[int], out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    n_layers, n_expert = pack_geometry(pack)
    needed = n_layers * n_expert
    print(f"\n=== {title}, {mode}: context sweep {contexts} ({needed or '?'} expert slots needed) ===", flush=True)
    for ctx in contexts:
        stem = f"{model_key}-{mode}-ctx{ctx}"
        log = out_dir / f"sweep-{stem}.log"
        cfg = server_config(binary, model_key, mode, card, gguf, shard2, pack, mtp_rt, profile, ctx,
                            args.prefill or 4096, args.reserve_mib or 1024, free_port(), log)
        cfg["_path"] = str(out_dir / f"config-sweep-{stem}.json")
        Path(cfg["_path"]).write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=1))
        log.unlink(missing_ok=True)
        row: dict[str, Any] = {"model": model_key, "mode": mode, "context": ctx}
        proc = None
        try:
            started = time.monotonic()
            proc, _ = start_server_with_fallback(cfg, timeout=args.load_timeout)
            row["load_s"] = round(time.monotonic() - started, 1)
            r = chat(cfg["port"], prompt_text(256, f"sweep-{ctx}-{time.time_ns()}"), 16)
            row.update({"ok": r["ok"], "err": r["err"], "decode_tps": r["decode_tps"],
                        "prompt_tps": r["prompt_tps"], "prompt_tokens": r["prompt_tokens"]})
            row.update(engine_facts(log))
            if cfg.get("_fallback"):
                row["host_fallback"] = True
            row["gpu_only"] = (bool(r["ok"]) and not row.get("host_fallback")
                               and row.get("resident_percent", 100) == 100
                               and row.get("hit_rate", 100.0) == 100.0
                               and ((row.get("held_pairs", 0) >= row.get("total_pairs", 1 << 60))
                                    or (needed > 0 and row.get("cache_slots", 0) >= needed)))
        except Exception as e:                          # noqa: BLE001 - one context failing is the answer
            row.update({"ok": False, "err": str(e), "gpu_only": False})
            row.update(engine_facts(log))
        finally:
            if proc is not None:
                stop_server(proc)
        rows.append(row)
        print(f"    {ctx:>7} tokens: ok={row['ok']} gpu_only={row.get('gpu_only')} "
              f"resident={row.get('resident_percent', '-')}% hit={row.get('hit_rate', '-')}% "
              f"slots={row.get('cache_slots', '-')} load={row.get('load_s', '-')}s err={row.get('err') or ''}",
              flush=True)

    good = [r["context"] for r in rows if r.get("gpu_only")]
    summary = {"model": model_key, "mode": mode, "contexts": rows,
               "max_gpu_only_context": max(good) if good else None}
    (out_dir / f"contexts-{model_key}-{mode}.json").write_text(json.dumps(summary, indent=1))
    lines = [f"# {title}, {mode}: context against expert spill", "",
             "| Context | Load s | Cache slots | Resident % | Decode hit rate | GPU only | Error |",
             "|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['context']} | {r.get('load_s', '-')} | {r.get('cache_slots', '-')} | "
                     f"{r.get('resident_percent', '-')} | {r.get('hit_rate', '-')} | "
                     f"{'yes' if r.get('gpu_only') else 'no'} | {r.get('err') or ''} |")
    lines += ["", f"Largest context with every expert on the GPU: {summary['max_gpu_only_context'] or 'none'}"]
    (out_dir / f"contexts-{model_key}-{mode}.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return summary


def main(argv: list[str]) -> int:
    ensure_oneapi(argv)
    ap = argparse.ArgumentParser(description="Qwen3.8-Flash-Next sizes (Gyro or explicit GGUFs) on one or two Intel Arc cards (SYCL).")
    ap.add_argument("--models-dir", default=os.environ.get("STRATA_GYRO_MODELS", str(ROOT / "models")))
    ap.add_argument("--packs-dir", default=os.environ.get("STRATA_GYRO_PACKS", str(ROOT / "packs")))
    ap.add_argument("--mtp", default=str(ROOT / "mtp" / "rt"))
    ap.add_argument("--profile", default=str(ROOT / "data" / "expert-profile-sycl.bin"))
    ap.add_argument("--engine", default="")
    ap.add_argument("--mode", choices=("single", "dual", "both"), default="both")
    ap.add_argument("--sizes", default="gyro-s,gyro-m", help="comma list: gyro-s,gyro-m")
    ap.add_argument("--label", default="", help="name for an explicit model (with --gguf)")
    ap.add_argument("--gguf", default="", help="shard 1 of an explicit model")
    ap.add_argument("--shard2", default="", help="the PLE shard of an explicit model (omitted for Gyro)")
    ap.add_argument("--pack", default="", help="the pack directory of an explicit model")
    ap.add_argument("--card", type=int, default=0, help="Level Zero index for single mode (default 0)")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--contexts", default="", help="comma list: load each context and report whether every expert stays on the GPU")
    ap.add_argument("--prefill", type=int, default=0, help="prompt chunk (default 4096, or 8192 above 32K context)")
    ap.add_argument("--reserve-mib", type=int, default=0, help="VRAM reserve (default 1024, or 2048 above 32K)")
    ap.add_argument("--runs", type=int, default=2, help="runs per prompt (default 2)")
    ap.add_argument("--load-timeout", type=float, default=1200, help="seconds to wait for /health (default 1200)")
    ap.add_argument("--download", action="store_true", help="download missing pinned GGUFs")
    ap.add_argument("--verify-sha", action="store_true", help="check each GGUF's sha256 (reads all of it)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    binary = find_engine(args.engine)
    cards = intel_gpus()
    if not cards:
        raise SystemExit("no Intel Arc found in /sys/class/drm (vendor 8086 under xe or i915)")
    if args.mode in ("dual", "both") and len(cards) < 2:
        raise SystemExit(f"--mode {args.mode} needs two Intel GPUs; found {len(cards)}")
    if args.card >= len(cards):
        raise SystemExit(f"--card {args.card}: only {len(cards)} Intel GPU(s)")

    models_dir, packs_dir = Path(args.models_dir).expanduser(), Path(args.packs_dir).expanduser()
    shard2s: dict[str, Path | None] = {}
    if args.gguf:
        if not args.label or not args.pack:
            raise SystemExit("--gguf needs --label and --pack")
        key = re.sub(r"[^a-z0-9]+", "-", args.label.lower()).strip("-") or "model"
        gguf = Path(args.gguf).expanduser()
        if not gguf.is_file():
            raise SystemExit(f"model not found: {gguf}")
        sizes = [key]
        titles = {key: args.label}
        ggufs = {key: gguf}
        shard2s[key] = Path(args.shard2).expanduser() if args.shard2 else None
        packs = {key: Path(args.pack).expanduser()}
        prepare_pack(gguf, packs[key], sys.executable)
    else:
        sizes = [s.strip().lower() for s in args.sizes.split(",") if s.strip()]
        for s in sizes:
            if s not in MODELS:
                raise SystemExit(f"unknown size {s!r}; pick from {', '.join(MODELS)}")
        titles = {s: MODELS[s]["title"] for s in sizes}
        if args.download:
            download(models_dir, sizes)
        ggufs = {}
        for s in sizes:
            p = models_dir / MODELS[s]["file"]
            verify_model(p, MODELS[s], args.verify_sha)
            ggufs[s] = p
            shard2s[s] = None
        packs = {}
        for s in sizes:
            pack = packs_dir / s
            prepare_pack(ggufs[s], pack, sys.executable)
            packs[s] = pack

    mtp_rt = Path(args.mtp).expanduser()
    if not (mtp_rt / "dense.bin").is_file():
        raise SystemExit(f"MTP draft runtime not found at {mtp_rt} (tools/mtp_fetch.py, tools/mtp_pack.py, tools/mtp_rt.py)")
    profile = None if args.profile.lower() == "none" else Path(args.profile).expanduser()
    if profile is not None and not profile.is_file():
        raise SystemExit(f"expert profile not found at {profile}")

    modes = ("single", "dual") if args.mode == "both" else (args.mode,)
    out_dir = Path(args.out) if args.out else (
        ROOT / "bench" / "results" / f"{datetime.date.today().isoformat()}-gyro-sycl-"
        f"{len(cards)}x{re.sub(r'[^A-Za-z0-9]+', '', cards[0]['name'])}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("cards: " + ", ".join(f"[{c['index']}] {c['name']} {c['vram_gb']:.0f} GB ({c['driver']})" for c in cards))
    print(f"engine: {binary}\nresults: {out_dir}")

    if args.contexts:
        contexts = [int(c.strip()) for c in args.contexts.split(",") if c.strip()]
        for mode in modes:
            for s in sizes:
                sweep_contexts(binary, s, titles[s], mode, args.card, args, ggufs[s], shard2s[s], packs[s],
                               profile, mtp_rt, contexts, out_dir)
        return 0

    results: list[dict[str, Any]] = []
    for mode in modes:
        for s in sizes:
            run_config(binary, s, titles[s], mode, args.card, args, ggufs[s], shard2s[s], packs[s], profile, mtp_rt,
                       out_dir, results)
    write_report(out_dir, results, cards, args, binary, sizes, titles)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
