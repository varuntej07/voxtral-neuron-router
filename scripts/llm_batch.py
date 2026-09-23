"""Disk-cached runner for the OpenAI Batch API.

Every request is keyed by a hash of its exact body. A finished response lands in
cache/responses/<key>.json, so re-running any script only pays for requests it
has never seen, and a crash mid-batch keeps everything already downloaded.
A submitted-but-uncollected batch is remembered in cache/pending/, so a rerun
resumes waiting on it instead of paying to submit the same work twice.
"""

import hashlib
import json
import time
from pathlib import Path

from openai import OpenAI

from tooling import ROOT

CACHE_DIR = ROOT / "cache" / "responses"
PENDING_DIR = ROOT / "cache" / "pending"
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def request_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def cached_count(bodies: list[dict]) -> int:
    return sum(_cache_path(request_key(b)).exists() for b in bodies)


def run_batch(bodies: list[dict], label: str, poll_seconds: int = 30) -> dict[str, dict]:
    """Return {key: chat completion body}. Keys whose request failed are absent."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)

    keyed = {request_key(b): b for b in bodies}
    todo = {k: b for k, b in keyed.items() if not _cache_path(k).exists()}
    print(f"[{label}] {len(keyed)} requests, {len(keyed) - len(todo)} cached, {len(todo)} to submit")

    if todo:
        client = OpenAI()
        batch_id = _resume_or_submit(client, todo, label)
        _wait_and_collect(client, batch_id, label, poll_seconds)

    results = {k: json.loads(_cache_path(k).read_text(encoding="utf-8")) for k in keyed if _cache_path(k).exists()}
    if len(results) < len(keyed):
        print(f"[{label}] {len(keyed) - len(results)} requests failed; rerun to retry only those")
    return results


def _resume_or_submit(client: OpenAI, todo: dict[str, dict], label: str) -> str:
    set_key = hashlib.sha256("".join(sorted(todo)).encode()).hexdigest()[:16]
    marker = PENDING_DIR / f"{label}-{set_key}.txt"
    if marker.exists():
        batch_id = marker.read_text().strip()
        if client.batches.retrieve(batch_id).status not in {"failed", "cancelled"}:
            print(f"[{label}] resuming batch {batch_id}")
            return batch_id

    jsonl = "\n".join(
        json.dumps({"custom_id": k, "method": "POST", "url": "/v1/chat/completions", "body": b}) for k, b in todo.items()
    )
    upload = client.files.create(file=(f"{label}.jsonl", jsonl.encode()), purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"label": label},
    )
    marker.write_text(batch.id)
    print(f"[{label}] submitted batch {batch.id}")
    return batch.id


def _wait_and_collect(client: OpenAI, batch_id: str, label: str, poll_seconds: int) -> None:
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        progress = f"{counts.completed}/{counts.total} done, {counts.failed} failed" if counts else "no counts yet"
        print(f"[{label}] {batch.status}: {progress}")
        if batch.status in TERMINAL:
            break
        time.sleep(poll_seconds)

    if batch.output_file_id:
        for line in client.files.content(batch.output_file_id).text.splitlines():
            item = json.loads(line)
            response = item.get("response") or {}
            if response.get("status_code") == 200:
                _cache_path(item["custom_id"]).write_text(json.dumps(response["body"]), encoding="utf-8")

    if batch.error_file_id:
        errors = client.files.content(batch.error_file_id).text.splitlines()
        print(f"[{label}] {len(errors)} errors, first: {errors[0][:300] if errors else ''}")

    for marker in PENDING_DIR.glob(f"{label}-*.txt"):
        if marker.read_text().strip() == batch_id:
            marker.unlink()


def usage_totals(responses: list[dict]) -> dict[str, int]:
    """Prompt, cached-prompt and completion tokens across responses."""
    totals = {"prompt": 0, "cached": 0, "completion": 0}
    for body in responses:
        usage = body.get("usage") or {}
        totals["prompt"] += usage.get("prompt_tokens", 0)
        totals["completion"] += usage.get("completion_tokens", 0)
        totals["cached"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return totals
