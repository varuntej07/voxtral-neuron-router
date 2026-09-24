"""Does Voxtral's forward pass compile and run on Inferentia2, and does it match the reference.

Two graphs, traced and compared separately:

    encoder   input_features [1, 128, 3000] -> (encoder_hidden, audio_embeds)
    decoder   inputs_embeds  [1, S, 3072]   -> last position logits [1, vocab]

They are separate on purpose. The splice between them, writing 375 audio embeddings into
the text embedding sequence, is a boolean masked assignment in modeling_voxtral, which is
a data dependent shape. reference_forward.py proves the audio span is one contiguous run
at a constant offset, so on Neuron it is a static slice and it belongs on the host, not in
either graph. Splitting also means a failure tells you which half failed, and it keeps two
smaller compilations instead of one large one, which matters because inf2.xlarge has 4
vCPU and 16 GB of host memory and the compiler runs on the host.

Shape decisions, since Neuron compiles one graph per shape:

  * mel is [1, 128, 3000] always. VoxtralEncoder raises unless the mel length is exactly
    max_source_positions * 2, so the 30 s window is not a choice, it is the contract. A
    4 s clip and a 29 s clip are the same graph, which is why prompt_shape.json shows 375
    audio tokens regardless of clip length.
  * the decoder sequence is whatever reference_forward.py measured, not a padded bucket.
    The prompt is the frozen catalog plus a fixed 375 audio tokens plus fixed turn
    markers, so the prefill length is a constant and there is nothing to pad. The causal
    mask is therefore a plain lower triangle, built on the host and carried as a graph
    constant rather than an input. It has to be handed in rather than left to the model:
    transformers builds masks with torch.vmap and torch.jit.trace cannot capture that.
  * --auto-cast=none, so the compiler does not silently change precision. The weights are
    loaded in bf16 and the reference is computed in bf16, so any difference that shows up
    is the hardware and the compiler, not a dtype the compiler chose on its own.

    source scripts/neuron_env_inf2.sh
    python scripts/neuron_forward_smoke.py --stage encoder
    python scripts/neuron_forward_smoke.py --stage decoder

Writes results/neuron_parity.json, merging one stage at a time so a later failure cannot
erase an earlier success. A failed compile is a result too and is recorded, with the
unsupported operator report from torch_neuronx.analyze when one can be produced.
"""

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np

# Both scripts have to load the model and ask for logits the same way, so the loader lives
# in one place and the reference is the place it lives.
from reference_forward import load_model, logits_keep_kwarg

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COMPARE = {
    "encoder": ("encoder_hidden", "audio_embeds"),
    "decoder": ("prefill_logits",),
}


def compare(name: str, got: np.ndarray, want: np.ndarray) -> dict:
    """Absolute and relative error, plus the thing that actually decides the routing
    output: whether the argmax moved.
    """
    got, want = got.astype(np.float64), want.astype(np.float64)
    if got.shape != want.shape:
        return {"tensor": name, "shape_mismatch": [list(got.shape), list(want.shape)]}
    diff = np.abs(got - want)
    denom = np.linalg.norm(want) or 1.0
    flat_got, flat_want = got.reshape(-1), want.reshape(-1)
    cosine = float(flat_got @ flat_want / ((np.linalg.norm(flat_got) or 1.0) * (np.linalg.norm(flat_want) or 1.0)))
    stats = {
        "tensor": name,
        "shape": list(got.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "relative_frobenius_error": float(np.linalg.norm(got - want) / denom),
        "cosine_similarity": round(cosine, 8),
        "reference_abs_max": float(np.abs(want).max()),
    }
    if name == "prefill_logits":
        top_got = np.argsort(-flat_got)[:5]
        top_want = np.argsort(-flat_want)[:5]
        stats["top1_agrees"] = bool(top_got[0] == top_want[0])
        stats["top5_overlap"] = int(len(set(top_got.tolist()) & set(top_want.tolist())))
        stats["neuron_top5_ids"] = [int(i) for i in top_got]
        stats["reference_top5_ids"] = [int(i) for i in top_want]
    return stats


def causal_mask(seq_len: int, dtype):
    """An additive lower triangular mask, built on the host and baked into the graph.

    transformers builds attention masks with torch.vmap, and torch.jit.trace, which
    torch_neuronx.trace runs first, cannot capture vmap: it fails with an unhelpful
    "invalid unordered_map key" from functorch. masking_utils returns any already 4D mask
    untouched, so handing one in skips the mask builder entirely. The sequence is exactly
    full, so the correct mask is a plain lower triangle with no padding to account for.
    """
    import torch

    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(mask, diagonal=1)[None, None]


def build_graph(stage: str, model_dir: Path, dtype, attn: str, seq_len: int):
    """Return (module, example_inputs, output_names). The wrapper keeps only the
    submodules the stage needs so the rest of the 3B can be freed before compiling.
    """
    import torch

    model = load_model(model_dir, dtype, attn)

    if stage == "encoder":
        intermediate = model.config.audio_config.intermediate_size

        class EncoderGraph(torch.nn.Module):
            def __init__(self, tower, projector):
                super().__init__()
                self.audio_tower = tower
                self.projector = projector

            def forward(self, input_features):
                hidden = self.audio_tower(input_features).last_hidden_state
                return hidden, self.projector(hidden.reshape(-1, intermediate))

        graph = EncoderGraph(model.audio_tower, model.multi_modal_projector).eval()
        names = COMPARE["encoder"]
    else:
        language_model = model.language_model
        keep = logits_keep_kwarg(language_model)

        class DecoderPrefill(torch.nn.Module):
            def __init__(self, language_model, mask):
                super().__init__()
                self.language_model = language_model
                # A buffer, not an input: the mask is the same tensor on every call, so it
                # belongs in the graph as a constant and the graph keeps one input.
                self.register_buffer("causal_mask", mask, persistent=False)

            def forward(self, inputs_embeds):
                extra = {keep: 1} if keep else {}
                return self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=self.causal_mask,
                    use_cache=False,
                    **extra,
                ).logits[:, -1, :]

        graph = DecoderPrefill(language_model, causal_mask(seq_len, dtype)).eval()
        names = COMPARE["decoder"]

    del model
    gc.collect()
    return graph, names


def example_input(stage: str, reference: dict, dtype):
    import torch

    key = "input_features" if stage == "encoder" else "inputs_embeds"
    return (torch.from_numpy(reference[key]).to(dtype),)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["encoder", "decoder"])
    parser.add_argument("--model", default="~/models/voxtral-mini-3b")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--artifacts", default="~/neuron_artifacts")
    parser.add_argument("--reuse", action="store_true", help="load a saved .pt instead of compiling")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compiler-args",
                        default="--target=inf2 --model-type=transformer --auto-cast=none",
                        help="passed to torch_neuronx.trace, which is what decides rather "
                             "than NEURON_CC_FLAGS, so the inf2 target is named here too")
    parser.add_argument("--skip-analyze", action="store_true",
                        help="do not run torch_neuronx.analyze after a failed trace")
    args = parser.parse_args()

    import torch
    import torch_neuronx

    dtype = getattr(torch, args.dtype)
    npz = RESULTS / "reference_forward.npz"
    if not npz.exists():
        raise SystemExit("run scripts/reference_forward.py first; there is nothing to compare against")
    reference = dict(np.load(npz))
    ref_facts = json.loads((RESULTS / "reference_forward.json").read_text(encoding="utf-8"))

    artifacts = Path(args.artifacts).expanduser()
    artifacts.mkdir(parents=True, exist_ok=True)
    saved = artifacts / f"voxtral_{args.stage}_{args.dtype}.pt"

    record = {
        "stage": args.stage,
        "dtype": args.dtype,
        "attn_implementation": args.attn,
        "compiler_args": args.compiler_args,
        "reference_dtype": ref_facts.get("dtype"),
        "reference_prefill_tokens": ref_facts.get("prefill_tokens"),
        "wav_is_synthetic": ref_facts.get("wav_is_synthetic"),
        "instance": __import__("os").environ.get("NEURON_INSTANCE_TYPE", "unrecorded"),
    }
    if ref_facts.get("dtype") != args.dtype:
        record["warning"] = (
            f"reference is {ref_facts.get('dtype')} but this run is {args.dtype}; the "
            "difference below includes the dtype change, not just Neuron"
        )

    inputs = example_input(args.stage, reference, dtype)
    record["input_shape"] = list(inputs[0].shape)

    if args.reuse and saved.exists():
        traced = torch.jit.load(str(saved))
        record["compiled"] = True
        record["reused_artifact"] = str(saved)
    else:
        graph, _ = build_graph(
            args.stage, Path(args.model).expanduser(), dtype, args.attn, int(inputs[0].shape[1])
        )
        started = time.perf_counter()
        try:
            traced = torch_neuronx.trace(
                graph,
                inputs,
                compiler_workdir=str(artifacts / f"workdir_{args.stage}"),
                compiler_args=args.compiler_args.split(),
            )
            record["compiled"] = True
            record["compile_seconds"] = round(time.perf_counter() - started, 1)
            torch.jit.save(traced, str(saved))
            record["artifact"] = str(saved)
            record["artifact_mb"] = round(saved.stat().st_size / 1e6, 1)
        except Exception:
            record["compiled"] = False
            record["compile_seconds"] = round(time.perf_counter() - started, 1)
            record["error"] = traceback.format_exc()[-4000:]
            if not args.skip_analyze:
                # The useful question after a failed trace is which operator is not
                # supported, and analyze answers it per operator instead of per graph.
                try:
                    record["analysis"] = str(torch_neuronx.analyze(graph, inputs))[:4000]
                except Exception as exc:
                    record["analysis_error"] = f"{type(exc).__name__}: {exc}"
            write(record)
            raise SystemExit(f"{args.stage}: trace failed, see results/neuron_parity.json")

    names = COMPARE[args.stage]
    outputs = traced(*inputs)
    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    record["parity"] = [
        compare(name, outputs[i].float().numpy(), reference[name])
        for i, name in enumerate(names)
    ]

    # Graph latency on one NeuronCore, which is not the end to end turn latency. It is a
    # floor for it, and it is the number that says whether the compile is worth keeping.
    for _ in range(3):
        traced(*inputs)
    samples = []
    for _ in range(args.iters):
        started = time.perf_counter()
        traced(*inputs)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    record["graph_latency_ms"] = {
        "p50": round(samples[len(samples) // 2], 2),
        "p95": round(samples[min(len(samples) - 1, int(len(samples) * 0.95))], 2),
        "min": round(samples[0], 2),
        "iters": args.iters,
        "note": "single graph, one NeuronCore, batch 1, host to device copy included",
    }
    write(record)
    print(json.dumps(record, indent=2))


def write(record: dict) -> None:
    """One file, one entry per stage, so running the stages on different days still leaves
    a single readable record.
    """
    path = RESULTS / "neuron_parity.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing[record["stage"]] = record
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
