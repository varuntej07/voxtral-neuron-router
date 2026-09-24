"""Prove the trn1 instance can do what the training run needs, before the training run.

Every check here is something that fails slowly and confusingly if you find it during
training instead of now. It takes a couple of minutes and compiles one small graph.

    source scripts/neuron_env.sh
    python scripts/neuron_preflight.py

Writes results/neuron_preflight.json.
"""

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WANT_ENV = ["NEURON_CC_FLAGS", "NEURON_COMPILE_CACHE_URL"]
# torch-neuronx 2.9.0.2.15 declares no compiler pin, and 2.27 fails on every graph it emits.
# See pin-neuronx-cc-to-match-torch-neuronx in lessons.txt.
WANT_NEURONX_CC = "2.26.6360.0"


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (out.stdout or out.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def versions() -> dict:
    import importlib

    found = {}
    for name in ("torch", "torch_xla", "torch_neuronx", "neuronx_distributed",
                 "transformers", "peft", "mistral_common"):
        try:
            found[name] = getattr(importlib.import_module(name), "__version__", "present")
        except ImportError:
            found[name] = None
    from importlib.metadata import PackageNotFoundError, version

    try:
        found["neuronx-cc"] = version("neuronx-cc")
    except PackageNotFoundError:
        found["neuronx-cc"] = None
    return found


def voxtral_api_check() -> dict:
    """train_lora.py calls get_audio_embeds; the name has moved between transformers releases."""
    try:
        from transformers import VoxtralForConditionalGeneration
    except ImportError as exc:
        return {"ok": False, "reason": str(exc)}
    return {"ok": hasattr(VoxtralForConditionalGeneration, "get_audio_embeds")}


def compile_probe() -> dict:
    """Compile and run a 64x64 Linear plus gelu, twice.

    This is the toolchain canary from lessons.txt: under a mismatched compiler this exact
    graph fails with the same error as the full model, in about 30 seconds instead of 20
    minutes.

    The first call pays compilation, the second should hit the cache. If the second is
    as slow as the first, NEURON_COMPILE_CACHE_URL is not being honoured and every
    training restart will pay full compile time.
    """
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    timings, value = [], float("nan")
    for _ in range(2):
        started = time.time()
        torch.manual_seed(0)
        layer = torch.nn.Linear(64, 64).to(device=device, dtype=torch.bfloat16)
        x = torch.randn(8, 64, device=device, dtype=torch.bfloat16)
        c = torch.nn.functional.gelu(layer(x)).sum()
        xm.mark_step()
        value = c.item()
        timings.append(round(time.time() - started, 2))
    return {"seconds": timings, "finite": bool(value == value), "cache_hit": timings[1] < timings[0] / 2}


def shape_check() -> dict:
    """The cache has to exist and agree with what the training step assumes."""
    meta_path = ROOT / "data" / "cache" / "train.meta.json"
    if not meta_path.exists():
        return {"ok": False, "reason": "run scripts/prepare_sft_cache.py first"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cache = ROOT / "data" / "cache"
    # Either audio source is legitimate. --audio-embeds keeps the encoder out of the graph
    # and does not need the mels at all, so demanding them would fail a valid setup.
    has_mel = (cache / "train.features.npy").exists()
    embeds_side = cache / "train.audio_embeds.meta.json"
    has_embeds = (cache / "train.audio_embeds.npy").exists() and embeds_side.exists()
    if not (has_mel or has_embeds):
        return {"ok": False, "reason": "neither train.features.npy nor train.audio_embeds.npy; "
                                       "run prepare_sft_cache.py, then precompute_audio_embeds.py"}

    keep = meta["seq"] - meta["prompt_len"] + 1
    report = {
        "ok": meta["longest_row"] <= meta["seq"] and keep > 1,
        "seq": meta["seq"],
        "prompt_len": meta["prompt_len"],
        "logits_kept": keep,
        "audio_span": [meta["audio_span_start"], meta["audio_span_start"] + meta["audio_span_width"]],
        "built_from_silence": meta.get("silence", False),
        "audio_source": "both" if has_mel and has_embeds else ("embeds" if has_embeds else "mel"),
    }
    if has_embeds:
        side = json.loads(embeds_side.read_text(encoding="utf-8"))
        report["embeds"] = {k: side.get(k) for k in
                            ("rows", "rows_done", "complete", "store_dtype", "device", "shape")}
        # A partial cache is the failure that looks like a working one: training would refuse
        # to start, but only after the model load, which is minutes in.
        if not side.get("complete"):
            report["ok"] = False
            report["reason"] = (f"audio embeds are {side.get('rows_done')} of {side.get('rows')} "
                                f"rows; finish precompute_audio_embeds.py")
    return report


def main() -> None:
    report = {
        "env": {k: os.environ.get(k) for k in WANT_ENV},
        "missing_env": [k for k in WANT_ENV if not os.environ.get(k)],
        "neuron_devices": run(["neuron-ls"]),
        "versions": versions(),
        "shapes": shape_check(),
        "voxtral_api": voxtral_api_check(),
    }

    if report["versions"].get("torch_xla"):
        try:
            report["compile_probe"] = compile_probe()
        except Exception as exc:  # a failure here is the point of the check
            report["compile_probe"] = {"failed": str(exc)}
    else:
        report["compile_probe"] = {"skipped": "torch_xla not installed"}

    blocking = []
    if report["missing_env"]:
        blocking.append(f"unset env: {', '.join(report['missing_env'])}, source scripts/neuron_env.sh")
    for name in ("torch_xla", "torch_neuronx", "peft", "transformers"):
        if not report["versions"].get(name):
            blocking.append(f"{name} is not installed")
    if report["versions"].get("neuronx-cc") != WANT_NEURONX_CC:
        blocking.append(f"neuronx-cc is {report['versions'].get('neuronx-cc')}, want {WANT_NEURONX_CC}")
    if not report["voxtral_api"].get("ok"):
        blocking.append("VoxtralForConditionalGeneration has no get_audio_embeds in this transformers")
    if not report["shapes"].get("ok"):
        blocking.append(f"cache not usable: {report['shapes'].get('reason', 'shapes disagree')}")
    if report["compile_probe"].get("failed"):
        blocking.append("a one-line graph did not compile, nothing else will")
    report["blocking"] = blocking

    out = ROOT / "results" / "neuron_preflight.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nREADY" if not blocking else "\nNOT READY:\n  " + "\n  ".join(blocking))


if __name__ == "__main__":
    main()
