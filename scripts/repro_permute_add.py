"""Minimal repro: torch_neuronx.trace computes x.permute(0, 2, 1) + pos wrong.

Found while porting the Voxtral (Whisper-style) audio encoder to Inferentia2: the encoder does
gelu(conv(x)).permute(0, 2, 1) + embed_positions.weight. On Neuron the result equals

    x.permute(0, 2, 1) + pos.reshape(C, T).T

that is, the [T, C] tensor added after the permute is read as if its memory were [C, T].
It looks like the compiler sinks the transpose below the add and reshapes the other operand
instead of transposing it.

Three ways of computing the same thing, each checked against CPU:

    buffer            pos is a registered buffer, added after the permute (fails)
    input             pos is passed in as a second input instead of a buffer
    add_then_permute  pos kept as [C, T] and added before the permute (the model-side fix)

    python repro_permute_add.py                 # default shape 1x8x16, fp32
    python repro_permute_add.py --c 1280 --t 1500
"""

import argparse
import json

import numpy as np
import torch
import torch_neuronx


class Buffer(torch.nn.Module):
    def __init__(self, pos):
        super().__init__()
        self.register_buffer("pos", pos)  # [T, C]

    def forward(self, x):  # x: [1, C, T]
        return x.permute(0, 2, 1) + self.pos


class Input(torch.nn.Module):
    def forward(self, x, pos):
        return x.permute(0, 2, 1) + pos


class AddThenPermute(torch.nn.Module):
    def __init__(self, pos):
        super().__init__()
        self.register_buffer("pos_ct", pos.T.contiguous())  # [C, T]

    def forward(self, x):
        return (x + self.pos_ct).permute(0, 2, 1)


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=8)
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--compiler-args", default="--target=inf2 --auto-cast=none")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    x = torch.randn(1, args.c, args.t)
    pos = torch.randn(args.t, args.c)
    want = (x.permute(0, 2, 1) + pos).numpy()
    bug = (x.permute(0, 2, 1) + pos.reshape(args.c, args.t).T).numpy()

    cases = {
        "buffer": (Buffer(pos), (x,)),
        "input": (Input(), (x, pos)),
        "add_then_permute": (AddThenPermute(pos), (x,)),
    }
    results = {"shape": [1, args.c, args.t], "compiler_args": args.compiler_args, "cases": {}}
    for name, (module, inputs) in cases.items():
        traced = torch_neuronx.trace(module.eval(), inputs, compiler_args=args.compiler_args.split())
        got = traced(*inputs).numpy()
        r = {
            "cosine_vs_cpu": round(cosine(got, want), 6),
            "max_abs_diff": float(np.abs(got - want).max()),
            "cosine_vs_pos_read_as_CxT": round(cosine(got, bug), 6),
        }
        results["cases"][name] = r
        print(f"{name:17s} cosine vs CPU {r['cosine_vs_cpu']:.6f}  max_abs_diff {r['max_abs_diff']:.3g}  "
              f"cosine vs 'pos read as [C,T]' {r['cosine_vs_pos_read_as_CxT']:.6f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
