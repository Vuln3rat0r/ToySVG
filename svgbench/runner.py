"""The model runner.

Sends each item's PNG to a vision model with the frozen prompt and stores the
raw answer verbatim. Two invariants this module is built around:

* **It never reads ground truth.** It imports :func:`load_public`, not
  :func:`load_hidden`, and the two live in different directories. A runner that
  could see the answer would eventually leak it into a prompt, an error message,
  or a retry heuristic.
* **Verbatim storage.** Raw model outputs are stored directly without text
  normalization, attribute extraction, or syntax repair, leaving failure
  gating entirely to the downstream scorer. Everything downstream sees
  exactly what the model returned, so every gate in the failure taxonomy
  is applied by the scorer and is auditable after the fact.

Providers are declared by a small base URL and an auth style rather than by
importing an SDK, so adding a provider is a table entry and not a code path.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .prompts import PROMPT_VERSION, build_prompt, prompt_hash
from .tasks import load_manifest, load_public

# ---------------------------------------------------------------- providers


@dataclass(frozen=True)
class Provider:
    name: str
    style: str  # "gemini" | "openai"
    base_url: str
    key_env: tuple


PROVIDERS: Dict[str, Provider] = {
    "gemini": Provider("gemini", "gemini", "https://generativelanguage.googleapis.com/v1beta", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "zenmux": Provider("zenmux", "openai", "https://zenmux.ai/api/v1", ("ZENMUX_API_KEY",)),
    "nvidia": Provider("nvidia", "openai", "https://integrate.api.nvidia.com/v1", ("NVIDIA_API_KEY", "NGC_API_KEY")),
    "openai": Provider("openai", "openai", "https://api.openai.com/v1", ("OPENAI_API_KEY",)),
    "openrouter": Provider("openrouter", "openai", "https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",)),
}


def resolve_key(provider: Provider, key_file: Optional[str] = None, key_index: int = 0, pattern: str = r"AIza[0-9A-Za-z_\-]{30,}") -> str:
    """Find a credential: explicit env first, then a key file.

    The value is never logged, never written to the run directory, and never
    echoed in an error message. Only its length and a short digest are recorded,
    which is enough to tell two attempted keys apart in a report.
    """
    for env_name in provider.key_env:
        val = os.environ.get(env_name)
        if val:
            return val.strip()
    if key_file:
        import re

        text = Path(key_file).expanduser().read_text(encoding="utf-8", errors="replace")
        found = re.findall(pattern, text)
        if not found:
            raise SystemExit(f"no key matching the expected pattern found in {key_file}")
        if key_index >= len(found):
            raise SystemExit(f"--key-index {key_index} out of range ({len(found)} keys found)")
        return found[key_index]
    raise SystemExit(
        f"no credential for provider {provider.name!r}: set one of {', '.join(provider.key_env)} "
        f"or pass --key-file"
    )


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:8]


# ---------------------------------------------------------------- http


class ApiError(RuntimeError):
    """An HTTP failure that keeps its status code *and* its response body.

    The body is the part that matters operationally: a free-tier rejection says
    exactly how long to wait, and discarding it turns a legible ``quota exceeded,
    retry in 34s`` into an opaque 429 that invites blind, wasteful retries.
    """

    def __init__(self, code: int, body: str) -> None:
        self.code = code
        self.body = body
        super().__init__(f"HTTP {code}: {body[:400]}")


def _post_json(url: str, payload: dict, headers: Dict[str, str], timeout: float = 180.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - the status code alone is still useful
            detail = ""
        raise ApiError(exc.code, detail) from None


def call_model(
    provider: Provider,
    key: str,
    model: str,
    prompt: str,
    png_path: Path,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
    timeout: float = 180.0,
    thinking_budget: Optional[int] = None,
) -> Dict[str, object]:
    """One generation. Returns the raw text plus everything needed to audit it.

    ``thinking_budget``: reasoning models spend their *output* token allowance on
    hidden reasoning before emitting a visible answer. Measured here on
    ``gemini-2.5-flash``: with the default budget, 11 of 12 answers were
    truncated mid-path at $\approx 120$ visible tokens, so the run measured the
    harness's token cap rather than the model's perception. Setting the budget to
    zero makes the visible answer the whole answer, which is what a drawing task
    needs. The setting is recorded in the manifest either way.
    """
    b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")

    if provider.style == "gemini":
        url = f"{provider.base_url}/models/{model}:generateContent?key={key}"
        gen_cfg: Dict[str, object] = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
        if thinking_budget is not None:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": b64}}]}],
            "generationConfig": gen_cfg,
        }
        data = _post_json(url, payload, {}, timeout)
        cand = (data.get("candidates") or [{}])[0]
        parts = ((cand.get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata") or {}
        return {
            "text": text,
            "finish_reason": cand.get("finishReason") or data.get("promptFeedback", {}).get("blockReason"),
            "prompt_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "model_reported": data.get("modelVersion") or model,
        }

    headers = {"Authorization": f"Bearer {key}"}
    url = f"{provider.base_url}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    }
    data = _post_json(url, payload, headers, timeout)
    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    return {
        "text": ((choice.get("message") or {}).get("content")) or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "model_reported": data.get("model") or model,
    }


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ApiError):
        return exc.code in (408, 409, 425, 429, 500, 502, 503, 504)
    return isinstance(exc, (TimeoutError, urllib.error.URLError))


class _Throttle:
    """Enforce a minimum spacing between requests, across threads.

    Free API tiers are limited per minute, not per second, so a burst of parallel
    workers converts a healthy quota into a wall of 429s: measured here, 4 workers
    produced 5 failures in 12 calls, while the same 12 calls spaced 7\,s apart all
    succeed. Retrying a per-minute quota with a 2\,s backoff cannot work, so
    spacing is enforced deliberately rather than hoped for.
    """

    def __init__(self, min_interval: float) -> None:
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0.0:
            return
        with self._lock:
            delta = self._last + self.min_interval - time.time()
            if delta > 0:
                time.sleep(delta)
            self._last = time.time()


# ---------------------------------------------------------------- run


@dataclass
class RunSpec:
    pack_dir: str
    out_dir: str
    provider: str
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 8192
    thinking_budget: Optional[int] = None
    retries: int = 4
    jobs: int = 1
    min_interval: float = 0.0
    deadline_s: Optional[float] = None
    force: bool = False
    only: Optional[List[str]] = None
    limit: Optional[int] = None
    key_file: Optional[str] = None
    key_index: int = 0
    key_pattern: str = r"AIza[0-9A-Za-z_\-]{30,}"
    note: str = ""
    tags: List[str] = field(default_factory=list)


def run(spec: RunSpec, progress=None) -> Dict[str, object]:
    """Execute a run over every public task in the pack. Returns the manifest."""
    pack_dir = Path(spec.pack_dir)
    out_dir = Path(spec.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    if spec.provider not in PROVIDERS:
        raise SystemExit(f"unknown provider {spec.provider!r}; known: {', '.join(PROVIDERS)}")
    provider = PROVIDERS[spec.provider]
    key = resolve_key(provider, spec.key_file, spec.key_index, spec.key_pattern)

    pack_manifest = load_manifest(pack_dir)
    tasks = load_public(pack_dir)
    if spec.only:
        wanted = {str(x).strip() for x in spec.only if str(x).strip()}
        tasks = [t for t in tasks if str(t["item_id"]) in wanted]
        missing = wanted - {str(t["item_id"]) for t in tasks}
        if missing:
            raise SystemExit(f"item ids not present in the pack: {sorted(missing)}")
    elif spec.limit:
        tasks = tasks[: spec.limit]

    manifest = {
        "run_id": out_dir.name,
        "provider": provider.name,
        "model": spec.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256_16": prompt_hash(),
        "temperature": spec.temperature,
        "max_output_tokens": spec.max_output_tokens,
        "thinking_budget": spec.thinking_budget,
        "retries": spec.retries,
        "jobs": spec.jobs,
        "min_interval_s": spec.min_interval,
        "deadline_s": spec.deadline_s,
        "key_fingerprint": key_fingerprint(key),
        "pack": {k: pack_manifest.get(k) for k in ("pack_version", "salt", "seed", "canvas", "ss", "source", "budgets", "stratum_cuts_commands")},
        "n_tasks": len(tasks),
        "note": spec.note,
        "tags": list(spec.tags),
        "only": list(spec.only) if spec.only else None,
    }

    throttle = _Throttle(spec.min_interval)

    # ---- resume: keep answers already on disk from an earlier, interrupted run ----
    results_path = out_dir / "results.jsonl"
    prior: Dict[str, Dict[str, object]] = {}
    if results_path.exists() and not spec.force:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = out_dir / str(rec.get("raw_file", ""))
            if rec.get("ok") and raw.exists() and raw.stat().st_size > 0:
                prior[str(rec["item_id"])] = rec
    if prior:
        tasks = [t for t in tasks if str(t["item_id"]) not in prior]
        print(f"resuming: {len(prior)} item(s) already answered, {len(tasks)} to go", flush=True)

    write_lock = threading.Lock()
    started = time.time()

    def append(rec: Dict[str, object]) -> None:
        with write_lock:
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

    def one(task: Dict[str, object]) -> Dict[str, object]:
        item_id = str(task["item_id"])
        png_path = pack_dir / str(task["input_png"])
        prompt = build_prompt(task.get("k"), canvas=int(task.get("canvas", 256)))
        last_err = ""
        t0 = time.time()
        for attempt in range(1, spec.retries + 1):
            try:
                throttle.wait()
                res = call_model(
                    provider, key, spec.model, prompt, png_path,
                    temperature=spec.temperature, max_output_tokens=spec.max_output_tokens,
                    thinking_budget=spec.thinking_budget,
                )
                raw_path = out_dir / "raw" / f"{item_id}.txt"
                raw_path.write_text(str(res.get("text", "")), encoding="utf-8")
                rec = {
                    "item_id": item_id,
                    "budget": task.get("budget"),
                    "k": task.get("k"),
                    "attempts": attempt,
                    "ok": True,
                    "elapsed_s": round(time.time() - t0, 2),
                    "finish_reason": res.get("finish_reason"),
                    "prompt_tokens": res.get("prompt_tokens"),
                    "output_tokens": res.get("output_tokens"),
                    "model_reported": res.get("model_reported"),
                    "answer_chars": len(str(res.get("text", ""))),
                    "raw_file": f"raw/{item_id}.txt",
                }
                append(rec)
                return rec
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < spec.retries and _is_retryable(exc):
                    # Escalating, because the failure that actually happens here is
                    # a per-minute quota, which a short backoff cannot clear.
                    time.sleep(min(6.0 * 2.0 ** (attempt - 1), 75.0))
                    continue
                break
        (out_dir / "raw" / f"{item_id}.txt").write_text("", encoding="utf-8")
        rec = {
            "item_id": item_id, "budget": task.get("budget"), "k": task.get("k"),
            "attempts": spec.retries, "ok": False, "error": last_err,
            "elapsed_s": round(time.time() - t0, 2), "raw_file": f"raw/{item_id}.txt",
        }
        append(rec)
        return rec

    records: List[Dict[str, object]] = list(prior.values())
    pending = list(tasks)
    if spec.jobs and spec.jobs > 1:
        def guarded(task: Dict[str, object]) -> Dict[str, object]:
            if spec.deadline_s and (time.time() - started) > spec.deadline_s:
                return {"item_id": task["item_id"], "ok": False, "skipped": "deadline", "error": "deadline reached"}
            return one(task)

        with ThreadPoolExecutor(max_workers=spec.jobs) as pool:
            for rec in pool.map(guarded, pending):
                records.append(rec)
                if progress:
                    progress(rec)
    else:
        for task in pending:
            if spec.deadline_s and (time.time() - started) > spec.deadline_s:
                rec = {"item_id": task["item_id"], "ok": False, "skipped": "deadline", "error": "deadline reached"}
            else:
                rec = one(task)
            records.append(rec)
            if progress:
                progress(rec)

    manifest["results"] = records
    ok = sum(1 for r in records if r.get("ok"))
    manifest["summary"] = {
        "n": len(records),
        "ok": ok,
        "failed": len(records) - ok,
        "mean_latency_s": round(sum(float(r.get("elapsed_s", 0.0)) for r in records) / max(1, len(records)), 2),
        "total_output_tokens": sum(int(r.get("output_tokens") or 0) for r in records),
    }
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- cli


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run a vision model over a ToySVG pack.")
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--out", required=True, help="run directory, e.g. data/runs/gemini-2.5-flash")
    ap.add_argument("--provider", default="gemini", choices=sorted(PROVIDERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="0 disables hidden reasoning; unset leaves the provider default")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="minimum seconds between requests; use ~15s on free tiers")
    ap.add_argument("--deadline", type=float, default=None, help="wall-clock seconds before stopping; run is resumable")
    ap.add_argument("--force", action="store_true", help="ignore and overwrite previous answers in this run directory")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated item ids to run")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--key-index", type=int, default=0)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    spec = RunSpec(
        pack_dir=args.pack, out_dir=args.out, provider=args.provider, model=args.model,
        temperature=args.temperature, max_output_tokens=args.max_output_tokens,
        thinking_budget=args.thinking_budget,
        retries=args.retries, jobs=args.jobs, min_interval=args.min_interval,
        deadline_s=args.deadline, force=args.force, limit=args.limit,
        only=[s for s in (args.only or "").split(",") if s.strip()] or None,
        key_file=args.key_file, key_index=args.key_index, note=args.note,
    )

    def progress(rec: Dict[str, object]) -> None:
        flag = "ok " if rec.get("ok") else "ERR"
        print(f"  [{flag}] {rec['item_id']}  k={rec.get('k')}  {rec.get('elapsed_s')}s  "
              f"{rec.get('answer_chars', 0)} chars  {rec.get('finish_reason')}", flush=True)

    manifest = run(spec, progress=progress)
    print(json.dumps(manifest["summary"], indent=2))
    if manifest["summary"]["failed"]:
        print("failures:")
        for r in manifest["results"]:
            if not r.get("ok"):
                print(f"  {r['item_id']}: {r.get('error')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
