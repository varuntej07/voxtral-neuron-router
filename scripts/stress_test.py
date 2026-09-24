"""Do we need every full tool schema in every request to route correctly?

Routes the val split under four prompt variants and compares accuracy against
prompt size. The answer decides what goes into Voxtral's prompt: prompt tokens
are prefill, and prefill is the latency floor on Inferentia2.

  full      all 34 full schemas as native tools          (~6.9k tokens, production-like)
  bm25_k    top-k full schemas by BM25 over the utterance (production's discovery step)
  compact   one line per tool: name, arg names, 1 sentence (~370 tokens)
  names     tool names only                               (~100 tokens)

Uses a text LLM as a stand-in for the router. The same variants get rerun on
base and fine-tuned Voxtral later; this run sizes the prompt before any Neuron
compile, since a context bucket is fixed at compile time.

    python scripts/stress_test.py --dry-run
    python scripts/stress_test.py --variants full compact names bm25_k --k 7
"""

import argparse
import json
import math
import re
from collections import Counter

from llm_batch import cached_count, request_key, run_batch, usage_totals
from tooling import (DEFAULT_MODEL, NO_TOOL, RESULTS_DIR, compact_catalog, label_space, score,
                     load_tools, names_catalog, openai_tools, read_rows)

USD_PER_M_INPUT = 0.20  # gpt-4.1-mini batch price; check current pricing
ROUTER_RULES = (
    "You route a voice assistant's user turn. If one tool fits, pick it and fill its arguments from the utterance. "
    "If the assistant should just talk, pick no_tool."
)


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.tokens = [self._tok(d) for d in docs]
        self.avg = sum(map(len, self.tokens)) / len(self.tokens)
        df = Counter(t for doc in self.tokens for t in set(doc))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))

    def top(self, query: str, k: int) -> list[int]:
        q = self._tok(query)
        scores = []
        for i, doc in enumerate(self.tokens):
            tf = Counter(doc)
            s = sum(
                self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(doc) / self.avg))
                for t in q
            )
            scores.append((s, i))
        return [i for _, i in sorted(scores, reverse=True)[:k]]


def json_route_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "route",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"tool": {"type": "string", "enum": label_space()}, "arguments_json": {"type": "string"}},
                "required": ["tool", "arguments_json"],
                "additionalProperties": False,
            },
        },
    }


def build_body(variant: str, utterance: str, model: str, bm25: BM25, k: int) -> dict:
    # Static parts first, utterance last: that prefix is what the provider can cache.
    body = {"model": model, "temperature": 0, "max_tokens": 400}
    if variant in {"full", "bm25_k"}:
        names = None
        if variant == "bm25_k":
            names = [load_tools()[i]["name"] for i in bm25.top(utterance, k)]
        body["tools"] = openai_tools(names)
        body["messages"] = [{"role": "system", "content": ROUTER_RULES + " No tool call means no_tool."},
                            {"role": "user", "content": utterance}]
    else:
        catalog = compact_catalog() if variant == "compact" else names_catalog()
        body["response_format"] = json_route_format()
        body["messages"] = [{"role": "system", "content": f"{ROUTER_RULES}\nTools:\n{catalog}\nArguments go in arguments_json as a JSON object string, {{}} for no_tool."},
                            {"role": "user", "content": utterance}]
    return body


def parse(variant: str, response: dict) -> tuple[str, dict]:
    message = response["choices"][0]["message"]
    try:
        if variant in {"full", "bm25_k"}:
            calls = message.get("tool_calls") or []
            if not calls:
                return NO_TOOL, {}
            return calls[0]["function"]["name"], json.loads(calls[0]["function"]["arguments"] or "{}")
        out = json.loads(message["content"])
        return out["tool"], json.loads(out["arguments_json"] or "{}")
    except (json.JSONDecodeError, KeyError, TypeError):
        return "<unparseable>", {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["full", "bm25_k", "compact", "names"])
    parser.add_argument("--k", type=int, default=7, help="production keeps about 7 tools per turn")
    parser.add_argument("--limit", type=int, default=0, help="cap val rows, 0 = all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_rows("val")
    if args.limit:
        rows = rows[: args.limit]
    bm25 = BM25([f"{t['name']} {t['description']}" for t in load_tools()])

    plan = {v: [(r, build_body(v, r["utterance"], args.model, bm25, args.k)) for r in rows] for v in args.variants}
    for variant, pairs in plan.items():
        bodies = [b for _, b in pairs]
        approx = sum(len(json.dumps(b.get("tools", ""))) + len(json.dumps(b["messages"])) for b in bodies) // 4
        print(f"{variant:8s} {len(bodies)} requests, {cached_count(bodies)} cached, ~{approx // max(len(bodies), 1):,} tok/request, ~${approx / 1e6 * USD_PER_M_INPUT:.2f}")
    if args.dry_run:
        return

    # One batch for all variants: fewer jobs to wait on, same per-request cache.
    all_bodies = [b for pairs in plan.values() for _, b in pairs]
    responses = run_batch(all_bodies, label="stress")

    report = {"model": args.model, "k": args.k, "val_rows": len(rows), "variants": {}}
    if "bm25_k" in plan:
        tool_rows = [r for r in rows if r["tool"] != NO_TOOL]
        hits = sum(r["tool"] in {load_tools()[i]["name"] for i in bm25.top(r["utterance"], args.k)} for r in tool_rows)
        report["bm25_recall_at_k"] = hits / max(len(tool_rows), 1)

    for variant, pairs in plan.items():
        preds, used = {}, []
        for row, body in pairs:
            response = responses.get(request_key(body))
            if response is not None:
                preds[row["id"]] = parse(variant, response)
                used.append(response)
        usage = usage_totals(used)
        result = score(rows, preds)
        result["avg_prompt_tokens"] = usage["prompt"] / max(len(used), 1)
        result["cached_share"] = usage["cached"] / max(usage["prompt"], 1)
        report["variants"][variant] = result

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "stress_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        f"# Prompt stress test ({args.model}, {len(rows)} val rows)\n",
        "| variant | tool acc | false trigger | missed call | arg keys | avg prompt tok | cached |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant, r in report["variants"].items():
        lines.append(
            f"| {variant} | {r['tool_accuracy']:.1%} | {r['false_trigger_rate']:.1%} | {r['missed_call_rate']:.1%} "
            f"| {r['arg_keys_match']:.1%} | {r['avg_prompt_tokens']:,.0f} | {r['cached_share']:.0%} |"
        )
    if "bm25_recall_at_k" in report:
        lines.append(f"\nBM25 recall@{args.k} (gold tool survives retrieval): {report['bm25_recall_at_k']:.1%}")
    for variant, r in report["variants"].items():
        lines.append(f"\n**{variant} top confusions:** " + "; ".join(f"{c} ({n})" for c, n in r["top_confusions"]))
    (RESULTS_DIR / "stress_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
