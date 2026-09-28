"""GELU and the permute-then-add pattern on Neuron, traced in isolation.

conv_stem_probe.py showed that the HF Voxtral encoder at depth 0 scores cosine 0.0975 on
Neuron while a hand copy of the same body passes, and that their HLO differs only in GELU:
after `import torch_neuronx`, torch.nn.functional.gelu lowers to a custom call
(AwsNeuronGelu) instead of erf arithmetic. This script asks whether that custom call is the
bug. It is not: GELU alone is correct every way. The failure is the add of a weight after a
permute (see repro_permute_add.py), which the GELU lowering only switches on or off in the
full graph.

Variants, each checked against exact GELU in fp64 on CPU:

    custom              torch.nn.functional.gelu looked up at call time (AwsNeuronGelu)
    erf                 0.5 * x * (1 + erf(x / sqrt 2)) written out
    tanh                torch.nn.functional.gelu(approximate="tanh")
    custom_permute      custom, then permute(0, 2, 1)
    erf_permute         erf, then permute(0, 2, 1)
    *_permute_add       custom, erf or plain (no GELU), then permute, then + a [T, C] buffer

Each case records whether AwsNeuronGelu reached the compiler, and the add variants also
record the cosine against "the buffer read as [C, T]", the signature of the bug.

    source ~/venv-trace/bin/activate && source scripts/neuron_env_inf2.sh
    python -u scripts/gelu_probe.py
    python -u scripts/gelu_probe.py --shapes 1x8x16 --dtypes float32 --variants plain_permute_add

Writes results/gelu_probe.json after every case; compiler workdirs go to ~/gelu_wd.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gelu_probe.json"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den else 0.0


class Custom(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)  # resolved at call time, like HF does


class Erf(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


class CustomPermute(torch.nn.Module):
    # The smallest shape of the encoder failure: custom-call GELU, then swap the last two axes.
    def forward(self, x):
        return torch.nn.functional.gelu(x).permute(0, 2, 1)


class ErfPermute(Erf):
    def forward(self, x):
        return super().forward(x).permute(0, 2, 1)


class CustomPermuteAdd(torch.nn.Module):
    # CustomPermute plus the fp32 position add the encoder does next.
    def __init__(self, shape):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.register_buffer("pos", torch.randn(shape[2], shape[1], generator=g))  # [T, C], fp32

    def gelu(self, x):
        return torch.nn.functional.gelu(x)

    def forward(self, x):
        h = self.gelu(x).permute(0, 2, 1)
        return (h + self.pos).to(h.dtype)


class PlainPermuteAdd(CustomPermuteAdd):
    # No GELU at all: is the permute-then-add rewrite wrong on its own?
    def gelu(self, x):
        return x


class ErfPermuteAdd(CustomPermuteAdd):
    def gelu(self, x):
        return 0.5 * x * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


class Tanh(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate="tanh")


def write(record: dict) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(tmp, OUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="1x1280x3000,1x1280x1500,1x8x16")
    parser.add_argument("--dtypes", default="bfloat16,float32")
    parser.add_argument("--variants", default="custom,erf,tanh")
    parser.add_argument("--no-inline-weights", action="store_true",
                        help="trace with inline_weights_to_neff=False: weights are passed at run time, not baked into the NEFF")
    parser.add_argument("--compiler-args", default="--target=inf2 --model-type=transformer --auto-cast=none")
    args = parser.parse_args()

    import torch_neuronx

    record = {
        "torch": torch.__version__,
        "torch_neuronx": getattr(torch_neuronx, "__version__", None),
        "compiler_args": args.compiler_args,
        "inline_weights_to_neff": not args.no_inline_weights,
        "cases": [],
    }
    try:
        import neuronxcc
        record["neuronx_cc"] = neuronxcc.__version__
    except Exception:
        record["neuronx_cc"] = None

    variants = {"custom": Custom, "erf": Erf, "tanh": Tanh, "custom_permute": CustomPermute,
                "erf_permute": ErfPermute, "custom_permute_add": CustomPermuteAdd,
                "erf_permute_add": ErfPermuteAdd, "plain_permute_add": PlainPermuteAdd}
    torch.manual_seed(0)
    for shape_s in args.shapes.split(","):
        shape = [int(d) for d in shape_s.split("x")]
        for dtype_s in args.dtypes.split(","):
            dtype = getattr(torch, dtype_s)
            x = (torch.randn(*shape) * 2).to(dtype)
            # Exact GELU in fp64 on CPU is the truth for every variant.
            for v in args.variants.split(","):
                module = variants[v](shape) if v.endswith("_add") else variants[v]()
                with torch.no_grad():
                    # Exact GELU in fp64 on CPU, then whatever the variant does after it.
                    truth = x.double() if v.startswith("plain") else torch.nn.functional.gelu(x.double())
                    if "permute" in v:
                        truth = truth.permute(0, 2, 1)
                    if v.endswith("_add"):
                        truth = truth + module.pos.double()
                    truth = truth.numpy()
                case = {"variant": v, "shape": shape, "dtype": dtype_s}
                record["cases"].append(case)
                started = time.perf_counter()
                try:
                    workdir = Path(os.path.expanduser("~/gelu_wd")) / f"{v}_{shape_s}_{dtype_s}"
                    traced = torch_neuronx.trace(module.eval(), (x,), compiler_args=args.compiler_args.split(),
                                                 compiler_workdir=str(workdir),
                                                 inline_weights_to_neff=not args.no_inline_weights)
                except Exception as exc:
                    case["error"] = str(exc)[-1500:]
                    print(f"{v:6s} {shape_s:13s} {dtype_s:8s} COMPILE FAILED", flush=True)
                    write(record)
                    continue
                case["compile_seconds"] = round(time.perf_counter() - started, 1)
                # Which GELU actually reached the compiler: the custom call, or erf arithmetic.
                hlo = (workdir / "model" / "graph.hlo").read_bytes()
                case["hlo_has_AwsNeuronGelu"] = b"AwsNeuronGelu" in hlo
                case["hlo_has_erf"] = b"erf" in hlo
                got = traced(x).double().numpy()
                case["cosine"] = round(cosine(got, truth), 6)
                case["max_abs_diff"] = float(np.abs(got - truth).max())
                case["got_abs_max"] = float(np.abs(got).max())
                case["truth_abs_max"] = float(np.abs(truth).max())
                # Layout check: is it GELU of the input with its last two axes swapped, read
                # back in the original shape? That is what an elementwise kernel that assumed
                # the other memory order would produce.
                if len(shape) == 3 and "permute" not in v:
                    swapped = torch.nn.functional.gelu(x.double().transpose(1, 2).contiguous()).reshape(shape).numpy()
                    case["cosine_vs_transposed_input"] = round(cosine(got, swapped), 6)
                if v.endswith("_permute_add"):
                    # Hypothesis: the permute is dropped and GELU's [B, C, T] output is read
                    # back as [B, T, C] by a plain reshape before the add.
                    g = x.double() if v.startswith("plain") else torch.nn.functional.gelu(x.double())
                    # What neuronx-cc was seen computing: pos's [T, C] memory read as [C, T].
                    bug = g.permute(0, 2, 1) + module.pos.double().reshape(shape[1], shape[2]).T
                    case["cosine_vs_pos_reinterpreted"] = round(cosine(got, bug.numpy()), 6)
                    case["cosine_vs_reshape_not_permute"] = round(cosine(got, (g.reshape(shape[0], shape[2], shape[1]) + module.pos.double()).numpy()), 6)
                    np.save(OUT.parent / f"gelu_probe_{v}_{shape_s}_{dtype_s}.npy", got)
                # Right values, wrong places: sorted outputs match even if positions do not.
                case["cosine_sorted"] = round(cosine(np.sort(got.ravel()), np.sort(truth.ravel())), 6)
                print(
                    f"{v:6s} {shape_s:13s} {dtype_s:8s} cosine {case['cosine']:.6f}  "
                    f"sorted {case['cosine_sorted']:.6f}  "
                    f"vs-transposed {case.get('cosine_vs_transposed_input', float('nan')):.6f}  "
                    f"max_abs_diff {case['max_abs_diff']:.3g}  "
                    f"AwsNeuronGelu {case['hlo_has_AwsNeuronGelu']}  "
                    f"vs-pos-reinterpreted {case.get('cosine_vs_pos_reinterpreted', float('nan')):.6f}",
                    flush=True,
                )
                write(record)
                del traced
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
