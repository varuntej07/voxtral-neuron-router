# voxtral-neuron-router

Fine-tune Voxtral Mini 3B on Trainium, serve it on Inferentia2, and use it as the
speech-and-routing step of a voice turn: audio in, tool call out.

34 tools from Aura, plus `no_tool`, frozen into `data/tools.json`.

## Where things stand

| Step | State |
|---|---|
| Text dataset | done, 4903 train / 665 val in `data/sft/` |
| TTS clips | done, all 5568 at 16 kHz, 6.6 hours |
| Real-speech test set | not recorded (`scripts/record_real_speech.py`) |
| Shapes measured | done, `results/prompt_shape.json` |
| Trainium training path | written, not yet run on hardware |
| Routing accuracy | written, not yet run (`scripts/eval_routing.py`) |
| Inferentia2 serving | encoder compiles, output does not match the reference |

Two things are deliberately not claimed yet. The inf2 encoder compiles and runs
at 82 ms, but its output sits at cosine 0.37 against the CPU reference, so that
latency measures an unverified graph and is not quotable. And every clip so far
is TTS, which is not the real speech the success criterion names.

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
| `precompute_audio_embeds.py` | Run the frozen audio half once, off the training loop. |
| `neuron_preflight.py` | Prove the trn1 box works before spending time on it. |
| `train_lora.py` | LoRA on Trainium, or on CPU as the reference. |
| `eval_routing.py` | Routing accuracy from audio, base vs fine-tuned. |
| `record_real_speech.py` | Record the val utterances yourself, 16 kHz mono. |
| `trace_canary.py` | 30 s toolchain check. Run it first on any fresh box or compiler. |
| `reference_forward.py` | The golden CPU forward the Neuron run is checked against. |
| `neuron_forward_smoke.py` | Compile and score the two inference graphs on inf2. |
| `encoder_bisect.py` | Find where the Neuron encoder stops matching. |
| `report.py` | `results/*.json` to paste-ready fixed-width blocks. |

## Running it on trn1

Two steps do not belong on the trn1 clock at all, and both are cheaper elsewhere.

```bash
# On a GPU box, ahead of time. Encoder plus projector is about 2.3 TFLOP per row:
# under a second on a T4, tens of seconds on the 8 vCPU of a trn1.2xlarge.
python -u scripts/prepare_sft_cache.py --split train
python -u scripts/precompute_audio_embeds.py --split train --device cuda --self-check
python -u scripts/precompute_audio_embeds.py --split val --device cuda

# Base accuracy. eval_routing.py touches no Neuron code, so pay CPU rates for it.
python -u scripts/eval_routing.py --results results/eval_base_tts.json
```

Then on the trn1:

```bash
source scripts/neuron_env.sh
python -u scripts/trace_canary.py                         # 30 s toolchain check, first
python -u scripts/fetch_voxtral.py
python -u scripts/neuron_preflight.py                     # prints READY or blockers
neuron_parallel_compile torchrun --nproc_per_node=2 scripts/train_lora.py \
    --audio-embeds --steps 12 --out /tmp/throwaway
torchrun --nproc_per_node=2 scripts/train_lora.py --audio-embeds --max-minutes 75 \
    --epochs 3 --results results/train_lora.json
python -u scripts/eval_routing.py --adapter checkpoints/lora --results results/eval_lora_tts.json
python scripts/report.py
```

Measure base accuracy before training. Once the adapter exists, recovering the
baseline costs a second model load, and without it there is no delta to report.

`--out /tmp/throwaway` on the parallel compile is not cosmetic: under graph
extraction the outputs are invalid, and the script would otherwise write that
garbage adapter over a real one.

`--max-minutes` ends the run on the clock rather than on an epoch count guessed
before the step time was known, and it still exits through the adapter save.

trn1.2xlarge is one Trainium, two NeuronCores, hence `--nproc_per_node=2`.

## Why the encoder is cached rather than trained through

The audio tower and projector are frozen, so their output is a function of the clip
alone and computing it inside every step buys nothing. It costs two things.

The first is correctness, and it is why this is the default rather than an
optimisation. The Voxtral conv frontend is the one component with an open numeric
bug on Neuron: with **zero** transformer layers it still disagrees with the CPU
reference at cosine 0.0975, and the compiler reports PASS. An encoder inside the
training graph would feed that into every gradient, the loss would fall anyway, and
nothing would say so.

The cache is exact, not an approximation: Voxtral hardcodes encoder dropout,
layerdrop and activation dropout to 0.0, so the frozen half is deterministic in
`model.train()` too.

One counterintuitive consequence: **the embeddings are bigger than the mels they
come from.** 375 x 3072 in bf16 is 2.20 MiB a row against 1.46 MiB for a
128 x 3000 fp32 mel, because the projector trades four times fewer time positions
for twenty-four times wider channels. The full train split is 10.5 GiB.

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
- Audio tower and projector are frozen, and with `--audio-embeds` they are not in
  the graph at all. See the section above for why that is a correctness decision.
- The causal mask is built on the host and handed in. Left to itself transformers
  builds masks with `torch.vmap`, which is a pile of ops for a constant and a
  plausible source of silent CPU fallbacks.

## Rules

- Never import Aura at runtime. `data/tools.json` is the only contract.
- Every LLM call goes through `llm_batch.run_batch` so nothing is paid twice.
- `--dry-run` and a cost estimate before any paid batch.
- Quote numbers only from `results/`.
- Stop EC2 instances when a step finishes.
