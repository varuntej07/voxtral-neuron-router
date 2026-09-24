# voxtral-neuron-router

Fine-tune Voxtral Mini 3B on Trainium, serve it on Inferentia2, and use it as the
speech-and-routing step of a voice turn: audio in, tool call out.

34 tools from Aura, plus `no_tool`, frozen into `data/tools.json`.

## Where things stand

| Step | State |
|---|---|
| Text dataset | done, 4903 train / 665 val in `data/sft/` |
| TTS clips | running on Colab T4 (`scripts/colab_synthesize_audio.py`) |
| Shapes measured | done, `results/prompt_shape.json` |
| Trainium training path | written, not yet run on hardware |
| Inferentia2 serving | not started |

The training path does not wait on the clips. `prepare_sft_cache.py --silence`
builds the same fixed shapes with silent audio, which is enough to compile the
graph and get a real step time.

## The numbers that drive everything

Measured from the real tokenizer, not assumed.

- **375 audio tokens, always.** The encoder takes one fixed 30 second window and
  the projector packs 4 encoder frames per token, so 1500/4 = 375. A 2 second
  utterance costs the same prefill as a 30 second one. This is the biggest
  latency lever on inf2.
- **Audio span is [225, 600), constant.** The prompt is byte-identical in every
  row, so the span never moves. That constant is what makes the graph static.
- **Sequence length 768.** 618 tokens for a typical turn, rounded up.
- **Vocabulary 131072.** Wide enough that logits over the full sequence would be
  the largest tensor in a training step, so the loss reads only the target slice.
- **Tekken v7 rejects system messages alongside audio.** The tool catalog rides
  in the user turn as a text chunk ahead of the clip.

## Scripts

Order matters.

| Script | Does |
|---|---|
| `export_tools.py` | Refresh the tool snapshot from Aura. Only run on purpose. |
| `generate_utterances.py` | Build the text dataset via the OpenAI Batch API. |
| `stress_test.py` | Prompt size vs accuracy, which picked the catalog format. |
| `build_sft.py` | Rows to chat pairs in `data/sft/`. |
| `colab_synthesize_audio.py` | Speak the user turns on a T4. Whisper checks each clip. |
| `synthesize_audio.py` | Same thing on an EC2 GPU box. |
| `probe_prompt_shape.py` | Measure the token shape. Needs only `tekken.json`. |
| `fetch_voxtral.py` | Download the model, write `results/voxtral_shapes.json`. |
| `prepare_sft_cache.py` | Pairs to fixed-shape arrays in `data/cache/`. |
| `neuron_preflight.py` | Prove the trn1 box works before spending time on it. |
| `train_lora.py` | LoRA on Trainium, or on CPU as the reference. |

## Running it on trn1

```bash
source scripts/neuron_env.sh
python scripts/neuron_preflight.py                        # prints READY or blockers
python scripts/fetch_voxtral.py
python scripts/prepare_sft_cache.py --split train --silence
neuron_parallel_compile torchrun --nproc_per_node=2 scripts/train_lora.py --steps 12
torchrun --nproc_per_node=2 scripts/train_lora.py --epochs 3
```

trn1.2xlarge is one Trainium, two NeuronCores, hence `--nproc_per_node=2`.

## Why the training script looks the way it does

Neuron compiles one graph per input shape, so anything data dependent either
recompiles or falls back to CPU silently.

- Stock Voxtral splices audio in with a boolean mask, whose selected count is
  data dependent. Replaced with a `cat` over the constant span.
- The loss slices constant positions instead of scoring 768 of them.
- `MpDeviceLoader` issues the `mark_step` that actually executes the graph.
  Without it XLA traces forever and looks like a hang.
- Nothing reads a device tensor in the loop. Logging goes through
  `add_step_closure`.
- Base weights bf16, adapter fp32, because AdamW moments in bf16 lose the small
  updates LoRA depends on.
- Audio tower and projector are frozen. The routing decision lives in the
  language model.

## Rules

- Never import Aura at runtime. `data/tools.json` is the only contract.
- Every LLM call goes through `llm_batch.run_batch` so nothing is paid twice.
- `--dry-run` and a cost estimate before any paid batch.
- Quote numbers only from `results/`.
- Stop EC2 instances when a step finishes.
