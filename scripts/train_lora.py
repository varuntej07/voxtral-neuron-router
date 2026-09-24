"""LoRA fine-tune of Voxtral Mini 3B on Trainium, with a CPU path for the reference run.

Same file both ways so the CPU run is the reference the Neuron run is checked against.
--device cpu is plain eager PyTorch; --device xla is torch-neuronx on trn1.

    # trn1.2xlarge, Neuron DLAMI. Two NeuronCores, so two workers.
    source scripts/neuron_env.sh

    # Ahead-of-time compile. Extracts the graphs and fills the cache without
    # spending a real run on compilation. Rerun whenever a shape or the step changes.
    neuron_parallel_compile torchrun --nproc_per_node=2 scripts/train_lora.py --steps 12

    torchrun --nproc_per_node=2 scripts/train_lora.py --epochs 3
    python scripts/train_lora.py --device cpu --steps 2   # reference, needs no Neuron

    # Parity: same batches, same order, no dropout, once on CPU fp32 and once on one core.
    python scripts/train_lora.py --device cpu --steps 5 --seed 0 --no-shuffle --dropout 0 \
        --results results/train_parity_cpu.json
    python scripts/train_lora.py --steps 5 --seed 0 --no-shuffle --dropout 0 \
        --results results/train_parity_neuron.json

Four things here exist because of Neuron, not because of LoRA.

1. The audio splice. Stock Voxtral writes audio embeddings into the text embeddings
   with a boolean mask (`inputs_embeds[input_ids == audio_token_id] = audio_embeds`).
   The number of selected positions is data dependent, which XLA cannot shape at trace
   time, so it recompiles per distinct count or falls back to CPU. The cache guarantees
   the span is [225, 600) in every row, so `splice_audio` rebuilds the sequence with a
   cat over constant offsets. Same math, one graph.

2. One graph, executed on a schedule. Every shape that reaches the device is fixed in
   prepare_sft_cache.py. MpDeviceLoader issues the mark_step that actually runs the
   accumulated graph each iteration; without it XLA keeps tracing and never executes.
   Nothing in the loop reads a device tensor, because reading one blocks the pipeline.
   Losses are printed through add_step_closure, which runs after the step lands.

3. A narrow loss. The vocabulary is 131072 wide, so logits over all 768 positions
   would be the largest tensor in the step by a wide margin, and almost all of it
   scores padding. The prompt is byte-identical in every row, so the target begins at
   a constant index and the forward keeps only the logits from there on.

4. bf16 base, fp32 adapter. The frozen 3B stays bf16 to fit HBM, but AdamW moments on
   bf16 parameters lose the small updates LoRA depends on, so the adapter tensors are
   promoted to fp32. PEFT casts between them inside the LoRA branch.

The audio tower and the projector stay frozen: the encoder is the expensive half and
the routing decision lives in the language model. Because the encoder is frozen its
output never changes, so it runs under no_grad here, and caching it outright is the
next speedup worth taking.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"

# Attention and MLP projections of the language model. The audio tower is left alone.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class CachedPairs(torch.utils.data.Dataset):
    def __init__(self, split: str):
        data = np.load(CACHE_DIR / f"{split}.npz")
        self.input_ids = torch.from_numpy(data["input_ids"].astype(np.int64))
        self.labels = torch.from_numpy(data["labels"].astype(np.int64))
        # 7.5 GB for train. Memory mapped, so the torchrun workers share one copy in the
        # page cache instead of each holding their own on a 32 GB host.
        self.input_features = np.load(CACHE_DIR / f"{split}.features.npy", mmap_mode="r")
        self.meta = json.loads((CACHE_DIR / f"{split}.meta.json").read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "input_features": torch.from_numpy(np.array(self.input_features[i])),
        }


def splice_audio(model, input_ids: torch.Tensor, input_features: torch.Tensor,
                 start: int, width: int) -> torch.Tensor:
    """Text embeddings with the audio span replaced, using constant offsets only."""
    base = model.get_input_embeddings()(input_ids)
    with torch.no_grad():  # the encoder is frozen, so nothing here needs an activation kept
        audio = model.get_audio_embeds(input_features)
    audio = audio.reshape(input_ids.shape[0], width, -1).to(base.dtype)
    return torch.cat([base[:, :start], audio, base[:, start + width :]], dim=1)


def build_model(model_dir: Path, dtype: torch.dtype, rank: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model
    from transformers import VoxtralForConditionalGeneration

    model = VoxtralForConditionalGeneration.from_pretrained(model_dir, torch_dtype=dtype)
    model.config.use_cache = False  # a KV cache in training is a second set of shapes
    model.audio_tower.requires_grad_(False)
    model.multi_modal_projector.requires_grad_(False)

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
        # Those projection names exist in the encoder too; keep the adapter off it.
        exclude_modules=r"^audio_tower\..*",
    )
    model = get_peft_model(model, config)

    # bf16 moments would swallow the updates, so the trainable tensors go back to fp32.
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model


def peak_host_memory_gb() -> float | None:
    """Host RAM is the binding constraint on trn1.2xlarge: 32 GB shared by both workers.

    Worth recording because the failure it predicts, the OOM killer taking a worker
    mid-epoch, looks like an unexplained hang rather than a memory error.
    """
    # Broad except on purpose: this runs at the end of a multi-hour job, and a
    # bookkeeping number must never be the thing that loses the results file.
    try:
        import resource  # Linux only; absent on the Windows dev box

        # ru_maxrss is kilobytes on Linux, and it is the peak rather than the current.
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    except Exception:
        pass
    try:
        import psutil  # ships as a torch-neuronx dependency

        return round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:
        return None


def package_versions() -> dict:
    import importlib
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in ("torch", "torch_xla", "torch_neuronx", "transformers", "peft"):
        try:
            found[name] = getattr(importlib.import_module(name), "__version__", "present")
        except ImportError:
            found[name] = None
    try:
        found["neuronx-cc"] = version("neuronx-cc")
    except PackageNotFoundError:
        found["neuronx-cc"] = None
    return found


def xla_metrics() -> dict:
    """Compile count and CPU fallbacks. A flat CompileTime count after warmup means one graph."""
    import torch_xla.debug.metrics as met

    out = {}
    for name in ("CompileTime", "ExecuteTime"):
        data = met.metric_data(name)
        if data:
            out[name] = {"count": int(data[0]), "total_s": data[1] / 1e9}
    # aten:: counters are ops XLA could not lower, so they ran on the host instead.
    out["aten_fallbacks"] = {n: met.counter_value(n) for n in met.counter_names() if n.startswith("aten::")}
    return out


def save_adapter(model, out: Path) -> None:
    from peft import get_peft_model_state_dict

    # Adapter tensors live on the device; pull them to host before writing.
    state = {k: v.to("cpu") for k, v in get_peft_model_state_dict(model).items()}
    model.save_pretrained(out, state_dict=state)
    print(f"adapter written to {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/models/voxtral-mini-3b")
    parser.add_argument("--device", choices=["xla", "cpu"], default="xla")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=1, help="per NeuronCore")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0, help="stop early, for compile and smoke runs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", default="checkpoints/lora")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true", help="fixed row order, for parity runs")
    parser.add_argument("--save-every", type=int, default=0, help="adapter to <out>/step-N every N steps")
    parser.add_argument("--warmup-steps", type=int, default=3,
                        help="steps left out of the steady s/step; the first ones pay compilation")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="recompute decoder activations in backward; the fallback if HBM overflows")
    parser.add_argument("--results", default="", help="write measured numbers to this JSON")
    parser.add_argument("--notes", default="", help="what this run was for, recorded in --results")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    on_xla = args.device == "xla"
    if on_xla:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_backend  # noqa: F401  registers the xla backend

        # torchrun sets these. Running bare python is a single worker, which is valid
        # but leaves the second NeuronCore idle.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            torch.distributed.init_process_group("xla")
        device = xm.xla_device()
        dtype = torch.bfloat16
        rank, is_master = xm.get_ordinal(), xm.is_master_ordinal()
    else:
        xm = pl = None
        world_size, rank, is_master = 1, 0, True
        device = torch.device("cpu")
        dtype = torch.float32

    data = CachedPairs(args.split)
    start, width = data.meta["audio_span_start"], data.meta["audio_span_width"]
    prompt_len, seq = data.meta["prompt_len"], data.meta["seq"]
    keep = seq - prompt_len + 1  # logits[:, k] predicts the label at prompt_len + k

    if is_master:
        print(f"{len(data)} rows, seq {seq}, audio span [{start}, {start + width}), "
              f"loss over {seq - prompt_len} of {seq} positions, {world_size} worker(s)", flush=True)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            data, num_replicas=world_size, rank=rank, shuffle=not args.no_shuffle,
            seed=args.seed, drop_last=True,
        )
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None and not args.no_shuffle,
        drop_last=True,  # a short last batch is a second graph
        num_workers=0,
    )

    model = build_model(Path(args.model).expanduser(), dtype, args.rank, args.alpha, args.dropout)
    if args.gradient_checkpointing:
        # Non-reentrant, because the embeddings are frozen and nothing upstream needs grad.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    model.train()
    if is_master:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable {trainable / 1e6:.1f}M of {total / 1e9:.2f}B", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def sync() -> None:
        if on_xla:
            xm.wait_device_ops()  # only at measurement edges; per step it would stall the queue

    losses: list[tuple[int, float]] = []
    first_step_s = steady_start = None
    step, t0 = 0, time.time()
    stop = False
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        # MpDeviceLoader moves each batch to the device and issues the mark_step that
        # executes the traced graph. On CPU the plain loader is already what we want.
        epoch_loader = pl.MpDeviceLoader(loader, device) if on_xla else loader

        for batch in epoch_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            features = batch["input_features"].to(device, dtype)

            inputs_embeds = splice_audio(model, input_ids, features, start, width)
            logits = model(inputs_embeds=inputs_embeds, logits_to_keep=keep).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, prompt_len:].reshape(-1),
                ignore_index=-100,
            )
            loss.backward()

            if on_xla:
                # Reduces gradients across NeuronCores, then steps.
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step == 1:
                sync()
                first_step_s = time.time() - t0
            if step == args.warmup_steps:
                sync()
                steady_start = (step, time.time())

            if is_master and (step % args.log_every == 0 or step <= 5):
                rate = (time.time() - t0) / step
                if on_xla:
                    # Reading loss here would block until the step lands and stall the
                    # pipeline, so the read is queued behind it instead.
                    def report(value, s=step, r=rate):
                        losses.append((s, value.item()))
                        print(f"step {s} loss {value.item():.4f} {r:.2f}s/step (host clock)", flush=True)
                    xm.add_step_closure(report, args=(loss,))
                else:
                    losses.append((step, loss.item()))
                    print(f"step {step} loss {loss.item():.4f} {rate:.2f}s/step", flush=True)

            if is_master and args.save_every and step % args.save_every == 0:
                save_adapter(model, Path(args.out).expanduser() / f"step-{step}")

            if args.steps and step >= args.steps:
                stop = True
                break
        if stop:
            break

    sync()
    end = time.time()
    if on_xla:
        xm.rendezvous("training done")

    if is_master:
        save_adapter(model, Path(args.out).expanduser())

    if is_master and args.results:
        steady = None
        if steady_start and step > steady_start[0]:
            s_per_step = (end - steady_start[1]) / (step - steady_start[0])
            samples_per_s = world_size * args.batch_size / s_per_step
            steady = {
                "steps": step - steady_start[0],
                "s_per_step": s_per_step,
                "samples_per_s": samples_per_s,
                # Every row is the full padded sequence, so this is positions pushed
                # through the decoder per second, not useful tokens.
                "tokens_per_s": samples_per_s * seq,
            }
        record = {
            "device": args.device,
            "notes": args.notes,
            "instance_type": os.environ.get("NEURON_INSTANCE_TYPE"),
            "peak_host_memory_gb": peak_host_memory_gb(),
            "world_size": world_size,
            "args": vars(args),
            "versions": package_versions(),
            "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS"),
            "rows": len(data),
            "seq": seq,
            "steps": step,
            "wall_s": end - t0,
            "first_step_s": first_step_s,
            "steady": steady,
            "losses": [{"step": s, "loss": v} for s, v in sorted(losses)],
        }
        if on_xla:
            try:
                record["xla_metrics"] = xla_metrics()
            except Exception as exc:  # never lose the run's numbers over the metrics report
                record["xla_metrics_error"] = f"{type(exc).__name__}: {exc}"
        path = Path(args.results)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"results written to {path}", flush=True)


if __name__ == "__main__":
    main()
