"""Thirty seconds that tell you whether the Neuron toolchain works at all.

Compile a 64x64 Linear plus gelu through torch_neuronx.trace. No model weights, no
reference npz, no SFT cache, nothing else that can fail first. Run this before any real
graph on any fresh box, venv, or compiler version.

Why this specific graph. Under neuronx-cc 2.27.5334 the full 3B Voxtral encoder failed with
[INTERNAL_ERROR] [NCC_ISMP902] Simplifier error: is_subset(), and so did a 64x64 Linear plus
gelu, byte for byte the same error. One compiler version apart, the tiny graph compiled on
the first attempt. Scale told you nothing about the cause. Four hypotheses were tested and
discarded, at roughly twenty minutes per compile, before anyone tried the thirty second
graph. lessons.txt records this as blog-identical-failure-at-every-scale.

There is already a canary like this in neuron_preflight.py, but it goes through
torch_xla and xm.xla_device(), which is the trn1 training path, and that script cannot run
without an SFT cache it has no reason to need. This one exercises the torch_neuronx.trace
path that inf2 serving actually uses, and depends on nothing.

    source scripts/neuron_env_inf2.sh
    python -u scripts/trace_canary.py

It also answers a separate question: whether a host with no Neuron device can still compile.
Tracing and compiling are host CPU work, and only executing the NEFF should need a chip. If
that holds, compile-error debugging can move off an inf2 onto any cheap x86 Linux box,
including WSL. The script compiles first, then tries to execute, and reports the two results
separately, so a device-less host gives compiled true and executed false rather than one
ambiguous failure. Do not source neuron_env_inf2.sh on a device-less host: its NEURON_RT_*
runtime variables are the easiest way to manufacture a misleading failure.

Writes results/trace_canary.json.
"""

import argparse
import glob
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def versions() -> dict:
    """Record what produced this result. A canary is only useful next to its toolchain."""
    found = {}
    for name in ("torch", "torch_neuronx", "torch_xla", "libneuronxla", "transformers"):
        try:
            found[name] = __import__(name).__version__
        except Exception as exc:
            found[name] = f"unavailable: {type(exc).__name__}"
    try:
        from importlib.metadata import version

        found["neuronx-cc"] = version("neuronx-cc")
    except Exception as exc:
        found["neuronx-cc"] = f"unavailable: {type(exc).__name__}"
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="~/neuron_artifacts")
    parser.add_argument("--compiler-args",
                        default="--target=inf2 --model-type=transformer --auto-cast=none",
                        help="what torch_neuronx.trace is given, which is what decides "
                             "rather than NEURON_CC_FLAGS")
    parser.add_argument("--size", type=int, default=64, help="Linear width")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "trace_canary.json"
    devices = sorted(glob.glob("/dev/neuron*"))
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph": f"Linear({args.size},{args.size}) + gelu, bf16",
        "compiler_args": args.compiler_args,
        "neuron_devices": devices,
        "has_neuron_device": bool(devices),
        "instance": os.environ.get("NEURON_INSTANCE_TYPE", "unrecorded"),
        "platform": f"{platform.system()} {platform.machine()}",
        "phase": "started",
    }

    def bank() -> None:
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    bank()
    print(f"neuron devices: {devices or 'none'}", flush=True)

    try:
        import torch
        import torch_neuronx
    except Exception as exc:
        record["phase"] = "import failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        bank()
        raise SystemExit(f"cannot import the toolchain: {exc}")

    record["versions"] = versions()
    record["phase"] = "imported"
    bank()
    print(json.dumps(record["versions"], indent=2), flush=True)

    torch.manual_seed(0)
    layer = torch.nn.Linear(args.size, args.size).to(torch.bfloat16).eval()

    class TinyGraph(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            return torch.nn.functional.gelu(self.layer(x))

    graph = TinyGraph(layer).eval()
    example = (torch.randn(8, args.size, dtype=torch.bfloat16),)
    with torch.no_grad():
        want = graph(*example).float().numpy()

    artifacts = Path(args.artifacts).expanduser()
    artifacts.mkdir(parents=True, exist_ok=True)
    workdir = artifacts / "workdir_canary"

    record["phase"] = "compiling"
    record["compiler_workdir"] = str(workdir)
    bank()
    print(f"compiling, workdir {workdir}", flush=True)

    started = time.perf_counter()
    try:
        traced = torch_neuronx.trace(
            graph, example,
            compiler_workdir=str(workdir),
            compiler_args=args.compiler_args.split(),
        )
    except Exception:
        record["compiled"] = False
        record["compile_seconds"] = round(time.perf_counter() - started, 1)
        record["error"] = traceback.format_exc()[-4000:]
        record["phase"] = "compile failed"
        record["verdict"] = (
            "The toolchain cannot compile a one line graph. Nothing larger will compile "
            "either, so stop here and fix the toolchain. The usual cause is a neuronx-cc "
            "that does not match torch-neuronx: pin neuronx-cc==2.26.6360.0."
        )
        bank()
        print(record["verdict"], flush=True)
        raise SystemExit("canary failed to compile, see results/trace_canary.json")

    record["compiled"] = True
    record["compile_seconds"] = round(time.perf_counter() - started, 1)
    record["phase"] = "compiled"
    bank()
    print(f"compiled in {record['compile_seconds']}s", flush=True)

    # Execution is a separate question from compilation, and on a host with no Neuron
    # device it is the only part expected to fail.
    try:
        got = traced(*example).float().numpy()
        record["executed"] = True
        diff = abs(got - want)
        record["max_abs_diff"] = float(diff.max())
        record["finite"] = bool(got.size and (got == got).all())
        record["phase"] = "done"
        record["verdict"] = "Toolchain compiles and executes. Proceed to the real graphs."
    except Exception as exc:
        record["executed"] = False
        record["execute_error"] = f"{type(exc).__name__}: {exc}"
        record["phase"] = "compiled but did not execute"
        record["verdict"] = (
            "Compiled without executing. On a host with no Neuron device this is the "
            "expected and useful result: it means compile-error debugging can happen here "
            "and only the parity and latency runs need a real chip."
            if not devices else
            "Compiled but failed to execute on a host that does have a Neuron device. That "
            "is a runtime or driver problem, not a compiler one."
        )

    bank()
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
