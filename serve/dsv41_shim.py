"""An OpenAI-compatible server for the DeepSeek-V4.1 engine (strata-dsv41 --serve).

The engine has no HTTP surface: it reads 'GEN <max_new> [keys] <ids>' on stdin and streams 'T <id>'
lines until a 'DONE' line. This tool turns POST /v1/chat/completions into that protocol, encoding the
prompt with the pack's vocab.json and decoding the streamed ids back to text.

Run it against a live engine with --attach, or let it start one with --engine and --pack:

    python serve/dsv41_shim.py --engine build-sycl/sycl/strata-dsv41 --pack maya-ds4 --port 8100
    python serve/dsv41_shim.py --pack maya-ds4 --attach-fifo /tmp/ds4-in --attach-log /tmp/engine.log
"""
import argparse
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any, IO

BEGIN, USER, ASSISTANT, END = 0, 128803, 128804, 1
NL = 201
PAD = 0
MAX_PIECE = 40


def load_vocab(pack: str) -> dict[str, int]:
    with open(os.path.join(pack, "tokenizer", "vocab.json"), encoding="utf-8") as f:
        return json.load(f)


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    ids: list[int] = []
    i = 0
    while i < len(text):
        if text[i] == "\n":
            ids.append(NL)
            i += 1
            continue
        j = i
        while j < len(text) and text[j] == " ":
            j += 1
        spaces = j - i
        hit = None
        for length in range(min(len(text) - i + spaces, MAX_PIECE), 0, -1):
            found = vocab.get("Ġ" * spaces + text[i:i + length])
            if found is not None:
                hit = (found, length)
                break
        if hit is not None:
            ids.append(hit[0])
            i += hit[1]
        elif spaces:
            ids.append(vocab["Ġ"])
            i += spaces
        else:
            ids.append(PAD)
            i += 1
    return ids


class Engine:
    def __init__(self, args: argparse.Namespace, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.lock = Lock()
        self.proc: subprocess.Popen[bytes] | None = None
        self.fifo = args.attach_fifo
        self.log = args.attach_log
        self.offset = os.path.getsize(args.attach_log) if args.attach_log else 0
        self.buffer = ""
        self.stdin_f: IO[Any] | None = None
        self.stdout_f: IO[Any] | None = None
        if args.attach_fifo:
            return
        if not args.engine or not args.pack:
            raise SystemExit("--engine and --pack are needed unless --attach-fifo is given")
        cmd = [args.engine, "--pack", args.pack, "--gpus", args.gpus, "--max-context", args.max_context, "--serve"]
        if args.oneapi:
            cmd = ["bash", "-c", f"source {args.oneapi} >/dev/null 2>&1; " + " ".join(cmd)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        stdin_f, stdout_f = proc.stdin, proc.stdout
        if stdin_f is None or stdout_f is None:
            raise SystemExit("the engine must be started with stdin and stdout pipes")
        os.set_blocking(stdout_f.fileno(), False)
        self.proc = proc
        self.stdin_f = stdin_f
        self.stdout_f = stdout_f

    def _lines(self) -> list[str]:
        if self.stdout_f is not None:
            raw = os.read(self.stdout_f.fileno(), 4096)
            if not raw:
                return []
            self.buffer += raw.decode("utf-8", "replace")
        elif self.log is not None:
            with open(self.log, "rb") as f:
                f.seek(self.offset)
                chunk = f.read()
            self.offset += len(chunk)
            self.buffer += chunk.decode("utf-8", "replace")
        else:
            return []
        self.buffer, _, tail = self.buffer.partition("\n")
        out = self.buffer.split("\n")
        self.buffer = tail
        return out

    def ask(self, max_new: int, ids: list[int], keys: str) -> tuple[list[int], str]:
        line = f"GEN {max_new}{keys} {','.join(str(i) for i in ids)}\n"
        if self.stdin_f is not None:
            self.stdin_f.write(line.encode())
            self.stdin_f.flush()
        elif self.fifo is not None:
            with open(self.fifo, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        gen: list[int] = []
        done = ""
        deadline = time.monotonic() + 600.0
        pending = ""
        while not done and time.monotonic() < deadline:
            pending += "\n".join(self._lines()) + "\n"
            lines = pending.split("\n")
            pending = lines.pop()
            for raw in lines:
                if raw.startswith("T "):
                    gen.append(int(raw[2:]))
                elif raw.startswith("DONE") or raw.startswith("ERR"):
                    done = raw
            time.sleep(0.05)
        return gen, done


def chat_ids(messages: list[dict[str, Any]], vocab: dict[str, int]) -> list[int]:
    ids = [BEGIN]
    for m in messages:
        ids.append(USER if m.get("role", "user") == "user" else ASSISTANT)
        ids.extend(encode(m.get("content", ""), vocab))
        ids.append(END)
    ids.append(ASSISTANT)
    return ids


class Handler(BaseHTTPRequestHandler):
    engine: Engine | None = None
    decode_map: dict[int, str] = {}

    def _send(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/v1/models"):
            self._send({"object": "list", "data": [{"id": "dsv41", "object": "model", "owned_by": "strata"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if not self.path.startswith("/v1/chat/completions") or self.engine is None:
            self._send({"error": "not found"}, 404)
            return
        req: Any = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        ids = chat_ids(req.get("messages", []), self.engine.vocab)
        keys = "".join(f" {k}={req[k]}" for k in ("temperature", "top_p", "top_k", "seed") if req.get(k))
        with self.engine.lock:
            gen, done = self.engine.ask(int(req.get("max_tokens") or 64), ids, keys)
        content = "".join(self.decode_map.get(i, "").replace("Ġ", " ").replace("Ċ", "\n") for i in gen)
        content = content.replace("<｜end▁of▁sentence｜>", "")
        self._send({
            "id": "cmpl-dsv41",
            "object": "chat.completion",
            "model": "dsv41",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop" if done.startswith("DONE") else "error"}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen), "total_tokens": len(ids) + len(gen)},
            "engine_done": done,
        })

    def log_message(self, format: str, *args: object) -> None:  # type: ignore[override]
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI-compatible server for strata-dsv41 --serve")
    ap.add_argument("--engine")
    ap.add_argument("--pack")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--max-context", default="8192")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--oneapi", default="/opt/intel/oneapi/setvars.sh")
    ap.add_argument("--attach-fifo")
    ap.add_argument("--attach-log")
    args = ap.parse_args()
    if not args.pack:
        raise SystemExit("--pack is needed to read the pack's vocab.json")
    vocab = load_vocab(args.pack)
    Handler.engine = Engine(args, vocab)
    Handler.decode_map = {i: t for t, i in vocab.items()}
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
