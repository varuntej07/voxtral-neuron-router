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
        self.input_features = torch.from_numpy(data["input_features"])
        self.meta = json.loads((CACHE_DIR / f"{split}.meta.json").read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "input_features": self.input_features[i],
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
    args = parser.parse_args()

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
            data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=True,  # a short last batch is a second graph
        num_workers=0,
    )

    model = build_model(Path(args.model).expanduser(), dtype, args.rank, args.alpha, args.dropout)
    model.to(device)
    model.train()
    if is_master:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable {trainable / 1e6:.1f}M of {total / 1e9:.2f}B", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

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

            if is_master and step % args.log_every == 0:
                rate = (time.time() - t0) / step
                if on_xla:
                    # Reading loss here would block until the step lands and stall the
                    # pipeline, so the print is queued behind it instead.
                    xm.add_step_closure(
                        lambda value, s=step, r=rate: print(
                            f"step {s} loss {value.item():.4f} {r:.2f}s/step", flush=True),
                        args=(loss,),
                    )
                else:
                    print(f"step {step} loss {loss.item():.4f} {rate:.2f}s/step", flush=True)

            if args.steps and step >= args.steps:
                stop = True
                break
        if stop:
            break

    if on_xla:
        xm.rendezvous("training done")

    if is_master:
        from peft import get_peft_model_state_dict

        # Adapter tensors live on the device; pull them to host before writing.
        state = {k: v.to("cpu") for k, v in get_peft_model_state_dict(model).items()}
        out = Path(args.out).expanduser()
        model.save_pretrained(out, state_dict=state)
        print(f"adapter written to {out}", flush=True)


if __name__ == "__main__":
    main()
