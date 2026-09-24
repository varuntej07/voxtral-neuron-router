"""Turn results/*.json into blocks you can paste straight into a post.

Everything this project measures lands as JSON, which is right for machines and wrong for
writing. This renders it as fixed width blocks, aligned with spaces, 80 columns, no
markdown pipes anywhere. Paste a block into a Medium <pre> and the columns stay put.

    python scripts/report.py                      # everything that exists
    python scripts/report.py --only accuracy cost

Blocks are skipped when their input file is missing, so this is safe to run at any point
and the output grows as the work does. Nothing is invented: a number appears here only if
some file in results/ contains it.

The last block is DEAD ENDS, built from lessons.txt entries still marked open or worked
around. That is deliberate. The failures are the part worth reading, and they are already
being recorded in the right shape, with the evidence next to the claim.
"""

import argparse
import json
from pathlib import Path
from textwrap import wrap

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WIDTH = 78

# On demand pricing, us-east-1. Check these before publishing a cost; they move.
HOURLY_USD = {"trn1.2xlarge": 1.34, "trn1.32xlarge": 21.50, "inf2.xlarge": 0.76,
              "inf2.8xlarge": 1.97}


def load(name: str):
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def block(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    return f"{title}\n" + "\n".join(("  " + line).rstrip() for line in lines) + "\n"


def kv(pairs: list[tuple[str, object]], pad: int = 26) -> list[str]:
    """Label left, value left aligned in one column, so the eye runs straight down."""
    return [f"{str(k):<{pad}}{'' if v is None else v}" for k, v in pairs if v is not None or True]


def columns(header: list[str], rows: list[list[str]]) -> list[str]:
    """Space aligned columns. Numbers are right aligned, the first column is not."""
    table = [header] + rows
    widths = [max(len(str(r[i])) for r in table) for i in range(len(header))]
    out = []
    for n, row in enumerate(table):
        cells = []
        for i, cell in enumerate(row):
            cells.append(f"{str(cell):<{widths[i]}}" if i == 0 else f"{str(cell):>{widths[i]}}")
        out.append("  ".join(cells).rstrip())
        if n == 0:
            out.append("  ".join("-" * w for w in widths).rstrip())
    return out


def pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def toolchain() -> str:
    pre = load("neuron_preflight.json")
    train = load("train_lora.json") or load("train_smoke.json")
    versions = (pre or {}).get("versions") or (train or {}).get("versions")
    if not versions:
        return ""
    lines = kv([(k, v) for k, v in versions.items() if v])
    env = (pre or {}).get("env") or {}
    flags = env.get("NEURON_CC_FLAGS") or (train or {}).get("neuron_cc_flags")
    instance = (train or {}).get("instance_type")
    lines += kv([("instance", instance), ("NEURON_CC_FLAGS", flags)])
    devices = (pre or {}).get("neuron_devices")
    if devices:
        lines += [""] + [d for d in str(devices).splitlines() if d.strip()][:6]
    return block("TOOLCHAIN", lines)


def shapes() -> str:
    shape, ref = load("prompt_shape.json"), load("reference_forward.json")
    if not shape and not ref:
        return ""
    lines = []
    if shape:
        by_clip = shape.get("by_clip_seconds") or {}
        if by_clip:
            rows = [[f"{k}s clip", str(v["total"]), str(v["audio_tokens"]), str(v["audio_start"])]
                    for k, v in sorted(by_clip.items(), key=lambda kv: int(kv[0]))]
            lines += columns(["", "prompt", "audio tok", "audio at"], rows)
            lines += ["", "the audio costs 375 positions at every clip length, so the prefill",
                      "is constant and the decoder needs one graph at one shape"]
    if ref:
        span = ref.get("audio_span") or {}
        lines += [""] + kv([
            ("measured prefill", ref.get("prefill_tokens")),
            ("audio span", f"[{span.get('start')}, {span.get('start', 0) + span.get('count', 0)}) "
                           f"contiguous={span.get('contiguous')}" if span else None),
        ])
    return block("SHAPES", lines)


def compile_and_step() -> str:
    lines = []
    for name, label in (("train_smoke.json", "smoke"), ("train_lora.json", "full run")):
        data = load(name)
        if not data:
            continue
        steady = data.get("steady") or {}
        metrics = data.get("xla_metrics") or {}
        compiles = (metrics.get("CompileTime") or {}).get("count")
        fallbacks = metrics.get("aten_fallbacks") or {}
        lines += [f"{label}:"] + kv([
            ("  steps", data.get("steps")),
            ("  first step (compile)", f"{data['first_step_s']:.1f}s" if data.get("first_step_s") else None),
            ("  steady s/step", f"{steady['s_per_step']:.2f}" if steady.get("s_per_step") else None),
            ("  samples/s", f"{steady['samples_per_s']:.2f}" if steady.get("samples_per_s") else None),
            ("  wall clock", f"{data['wall_s'] / 60:.1f} min" if data.get("wall_s") else None),
            ("  graphs compiled", compiles),
            ("  CPU fallbacks", sum(fallbacks.values()) if fallbacks else 0),
        ]) + [""]
    if lines and any("graphs compiled" in l for l in lines):
        lines += ["a compile count that stops rising after warmup is the claim that this",
                  "is one graph; any aten:: fallback is work that left the device"]
    return block("COMPILE AND STEP TIME", lines)


def parity() -> str:
    data = load("neuron_parity.json")
    if not data:
        return ""
    rows = []
    for stage, rec in data.items():
        if not rec.get("compiled"):
            rows.append([stage, "FAILED", "-", "-", "-"])
            continue
        for t in rec.get("parity", []):
            rows.append([
                f"{stage}/{t['tensor']}",
                f"{t.get('cosine_similarity', 0):.4f}",
                f"{t.get('relative_frobenius_error', 0):.4f}",
                f"{t.get('max_abs_diff', 0):.3f}",
                f"{rec.get('graph_latency_ms', {}).get('p50', '-')}",
            ])
    if not rows:
        return ""
    lines = columns(["tensor", "cosine", "rel err", "max diff", "p50 ms"], rows)
    worst = min((float(r[1]) for r in rows if r[1] != "FAILED"), default=1.0)
    if worst < 0.99:
        lines += ["", f"cosine {worst:.3f} is not parity. a latency number from this graph",
                  "measures how fast it computes the wrong answer, so it is not quotable yet"]
    return block("NUMERIC PARITY AGAINST THE CPU REFERENCE", lines)


def accuracy() -> str:
    """The headline: did fine-tuning actually help, on TTS and on real speech."""
    found = {}
    for path in sorted(RESULTS.glob("eval_*.json")):
        if path.name.endswith(".pred.jsonl"):
            continue
        data = load(path.name)
        if not data or "scores" not in data:
            continue
        kind = "real speech" if data.get("audio_is_real_speech") else "TTS"
        found.setdefault(kind, {})["tuned" if data.get("adapter") else "base"] = data
    if not found:
        return ""
    rows = []
    for kind in ("TTS", "real speech"):
        pair = found.get(kind)
        if not pair:
            continue
        base, tuned = pair.get("base"), pair.get("tuned")
        n = (tuned or base)["scores"]["n"]
        b = base["scores"]["tool_accuracy"] if base else None
        t = tuned["scores"]["tool_accuracy"] if tuned else None
        delta = f"{(t - b) * 100:+.1f}" if (b is not None and t is not None) else "-"
        rows.append([f"{kind} ({n})", pct(b), pct(t), delta])
    lines = columns(["set", "base", "tuned", "delta"], rows)

    detail = []
    for kind, pair in found.items():
        for which, data in sorted(pair.items()):
            s = data["scores"]
            detail.append([f"{kind}/{which}", pct(s.get("tool_accuracy")),
                           pct(s.get("false_trigger_rate")), pct(s.get("missed_call_rate")),
                           pct(s.get("json_valid_rate")), pct(s.get("arg_keys_match"))])
    lines += [""] + columns(["", "acc", "false trig", "missed", "json ok", "args"], detail)

    for kind, pair in found.items():
        for which, data in sorted(pair.items()):
            conf = data["scores"].get("top_confusions") or []
            if conf:
                lines += ["", f"{kind}/{which} top confusions:"]
                lines += [f"  {c[0]}  x{c[1]}" for c in conf[:5]]
    if "real speech" not in found:
        lines += ["", "TTS only so far. the criterion is real speech: record it with",
                  "scripts/record_real_speech.py before quoting this as the result"]
    return block("ROUTING ACCURACY", lines)


def cost() -> str:
    lines, train = [], load("train_lora.json")
    if train and train.get("wall_s"):
        inst = train.get("instance_type")
        rate = HOURLY_USD.get(inst)
        hours = train["wall_s"] / 3600
        lines += kv([("instance", inst), ("training hours", f"{hours:.2f}"),
                     ("USD/hr", rate), ("training cost", f"${hours * rate:.2f}" if rate else None)])
    parity_data = load("neuron_parity.json") or {}
    latencies = [r.get("graph_latency_ms", {}).get("p50") for r in parity_data.values()
                 if isinstance(r, dict) and r.get("graph_latency_ms")]
    latencies = [l for l in latencies if l]
    if latencies:
        turn_ms = sum(latencies)
        inst = next((r.get("instance") for r in parity_data.values()
                     if isinstance(r, dict) and r.get("instance")), None)
        rate = HOURLY_USD.get(inst) if inst else None
        lines += [""] + kv([
            ("serving instance", inst),
            ("graph time per turn", f"{turn_ms:.0f} ms ({' + '.join(f'{l:.0f}' for l in latencies)})"),
            ("turns per hour, 1 core", f"{3600 / (turn_ms / 1000):,.0f}"),
            ("USD per 1000 turns", f"${rate * (1000 * turn_ms / 1000) / 3600:.3f}" if rate else None),
        ])
        lines += ["", "graph time only. it is the floor for a turn, not the turn: no",
                  "detokenisation, no network, no queueing"]
    return block("COST", lines)


def dead_ends() -> str:
    path = ROOT / "lessons.txt"
    if not path.exists():
        return ""
    try:
        lessons = json.loads(path.read_text(encoding="utf-8"))["lessons"]
    except (json.JSONDecodeError, KeyError):
        return ""
    open_items = [l for l in lessons
                  if any(w in str(l.get("status", "")).lower()
                         for w in ("open", "workaround", "not been", "has not", "not yet",
                                   "not diagnosed", "not tested"))]
    if not open_items:
        return ""
    lines = []
    for item in open_items:
        lines.append(f"{item['id']}:")
        for label, key in (("", "lesson"), ("status: ", "status")):
            text = str(item.get(key, "")).strip()
            if text:
                # Wrapped, not truncated: a cut-off sentence is not quotable, and these
                # lines are the ones most likely to be pasted into a post as they are.
                lines += ["  " + l for l in wrap(label + text, WIDTH - 4)]
        lines.append("")
    return block("STILL OPEN, OR FIXED ONLY BY WORKING AROUND IT", lines)


BLOCKS = {
    "toolchain": toolchain, "shapes": shapes, "compile": compile_and_step,
    "parity": parity, "accuracy": accuracy, "cost": cost, "dead-ends": dead_ends,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=sorted(BLOCKS), default=sorted(BLOCKS))
    parser.add_argument("--out", default="results/report.txt")
    args = parser.parse_args()

    order = [n for n in ("toolchain", "shapes", "compile", "parity", "accuracy", "cost",
                         "dead-ends") if n in args.only]
    parts = [BLOCKS[name]() for name in order]
    text = "\n".join(p for p in parts if p)
    if not text:
        text = "nothing measured yet: results/ holds no file this reporter understands\n"
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[written to {args.out}, {len(text.splitlines())} lines]")


if __name__ == "__main__":
    main()
