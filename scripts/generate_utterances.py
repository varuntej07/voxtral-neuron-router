"""Generate (spoken utterance, gold tool call) rows with one Batch API job.

Each request carries ONE target tool's full schema plus the compact catalog of
the other 33, so the generator can steer clear of utterances that would really
belong to a neighbouring tool. Sending all 34 full schemas per request would
cost ~18x the input tokens for no better labels.

    python scripts/generate_utterances.py --dry-run     # count and price only
    python scripts/generate_utterances.py               # submit, wait, write rows
"""

import argparse
import hashlib
import json
from datetime import date

from llm_batch import cached_count, request_key, run_batch, usage_totals
from tooling import DEFAULT_MODEL, NO_TOOL, ROWS_PATH, args_valid, compact_catalog, load_tools

PER_REQUEST = 25
VAL_FRACTION = 0.12

# Batch price for gpt-4.1-mini (half of list). Check current pricing before a big run.
USD_PER_M_INPUT = 0.20
USD_PER_M_OUTPUT = 0.80

TOOL_STYLES = {
    "direct": "Plain, direct requests said out loud.",
    "casual": "Casual speech with filler words, false starts or self-corrections (uh, wait no, actually).",
    "indirect": "Indirect requests where the intent is clear but the action is never named outright.",
    "terse": "Very short commands of 2 to 6 words.",
    "detailed": "Longer requests that state most of the arguments explicitly.",
    "contextual": "A request wrapped in a sentence of context about the speaker's day.",
}

NO_TOOL_STYLES = {
    "smalltalk": "Greetings, small talk, thanks, and conversational backchannels (okay, got it, cool).",
    "knowledge": "Timeless general-knowledge or explanation questions the assistant can answer from its own knowledge, with no need for live data.",
    "venting": "The speaker sharing feelings or thinking out loud, with no request for an action.",
    "unsupported": "Requests for actions no listed tool can do (order food, book a flight, call a phone number, change device brightness).",
    "about_reply": "Reactions to what the assistant just said (say that again, why, that's wrong, shorter please).",
    "brainstorm": "Asking for ideas, advice or feedback that the assistant answers directly in conversation.",
}
NO_TOOL_REQUESTS_PER_STYLE = 4

PERSONAS = ["college student", "self-taught developer", "job seeker preparing for interviews", "graduate student", "freelancer"]

SYSTEM_PROMPT = f"""You write training data for a voice assistant's tool router.
Every utterance is a transcription of real speech: no emoji, no markdown, no symbols a person would not say, numbers as a person would say them.
The assistant has exactly these tools:
{compact_catalog()}

Rules:
- Make every utterance unambiguous for its target label. If a phrasing could fairly belong to another tool, do not use it.
- Vary vocabulary, sentence shape and argument values across the set. No two utterances may share a template.
- arguments_json is a JSON object string. For a tool, include only arguments the utterance states or clearly implies, always include required arguments, and match the schema's types and enums exactly. For no_tool it is "{{}}".
Today's date is {date.today().isoformat()}."""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "examples",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"utterance": {"type": "string"}, "arguments_json": {"type": "string"}},
                        "required": ["utterance", "arguments_json"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["examples"],
            "additionalProperties": False,
        },
    },
}


def build_requests(model: str) -> list[tuple[dict, dict]]:
    """(body, meta) pairs. The system prompt is identical across all of them and comes first."""
    requests = []
    for i, tool in enumerate(load_tools()):
        for j, (style, style_text) in enumerate(TOOL_STYLES.items()):
            persona = PERSONAS[(i + j) % len(PERSONAS)]
            user = (
                f"Target label: {tool['name']}\nFull schema:\n{json.dumps(tool, indent=1)}\n\n"
                f"Style: {style_text}\nSpeaker: a {persona}.\nWrite {PER_REQUEST} examples."
            )
            requests.append((_body(model, user), {"tool": tool["name"], "style": style}))
    for style, style_text in NO_TOOL_STYLES.items():
        for n in range(NO_TOOL_REQUESTS_PER_STYLE):
            persona = PERSONAS[n % len(PERSONAS)]
            user = (
                f"Target label: {NO_TOOL} (nothing to call; the assistant just talks)\n"
                f"Style: {style_text}\nSpeaker: a {persona}. Variation {n + 1} of {NO_TOOL_REQUESTS_PER_STYLE}: pick topics other variations would not.\n"
                f"Write {PER_REQUEST} examples."
            )
            requests.append((_body(model, user), {"tool": NO_TOOL, "style": style}))
    return requests


def _body(model: str, user: str) -> dict:
    return {
        "model": model,
        "temperature": 1.0,
        "max_tokens": 4000,
        "response_format": RESPONSE_FORMAT,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
    }


def split_for(utterance: str) -> str:
    bucket = int(hashlib.sha256(utterance.lower().encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < VAL_FRACTION else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requests = build_requests(args.model)
    bodies = [b for b, _ in requests]
    approx_in = sum(len(json.dumps(b["messages"])) for b in bodies) // 4
    approx_out = len(bodies) * PER_REQUEST * 40
    cost = approx_in / 1e6 * USD_PER_M_INPUT + approx_out / 1e6 * USD_PER_M_OUTPUT
    print(f"{len(bodies)} requests, {cached_count(bodies)} already cached, ~{approx_in:,} input tokens, ~${cost:.2f} worst case")
    if args.dry_run:
        return

    responses = run_batch(bodies, label="generate")
    rows, seen, rejected, truncated = [], set(), {}, []
    for body, meta in requests:
        response = responses.get(request_key(body))
        if response is None:
            continue
        choice = response["choices"][0]
        # A run to max_tokens is an HTTP 200 whose JSON is cut mid-string; the same
        # prompt would degenerate again, so drop the request and name it.
        if choice["finish_reason"] == "length":
            truncated.append(f"{meta['tool']}/{meta['style']}")
            continue
        for example in json.loads(choice["message"]["content"])["examples"]:
            utterance = " ".join(example["utterance"].split())
            norm = utterance.lower().strip(" .?!")
            try:
                arguments = json.loads(example["arguments_json"])
            except json.JSONDecodeError:
                arguments = None
            if not utterance or norm in seen or not isinstance(arguments, dict) or not args_valid(meta["tool"], arguments):
                rejected[meta["tool"]] = rejected.get(meta["tool"], 0) + 1
                continue
            seen.add(norm)
            rows.append({
                "id": hashlib.sha256(norm.encode()).hexdigest()[:12],
                "tool": meta["tool"],
                "arguments": arguments,
                "utterance": utterance,
                "style": meta["style"],
                "split": split_for(utterance),
            })

    ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROWS_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    per_tool = {}
    for r in rows:
        per_tool[r["tool"]] = per_tool.get(r["tool"], 0) + 1
    usage = usage_totals(list(responses.values()))
    print(f"wrote {len(rows)} rows ({sum(r['split'] == 'val' for r in rows)} val) to {ROWS_PATH}")
    print(f"tokens: {usage['prompt']:,} prompt ({usage['cached']:,} cached), {usage['completion']:,} completion")
    print(f"thinnest labels: {sorted(per_tool.items(), key=lambda kv: kv[1])[:5]}")
    if rejected:
        print(f"rejected (dupe or schema-invalid): {dict(sorted(rejected.items(), key=lambda kv: -kv[1])[:8])}")
    if truncated:
        print(f"dropped {len(truncated)} truncated responses (hit max_tokens): {truncated}")


if __name__ == "__main__":
    main()
