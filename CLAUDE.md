# voxtral-neuron-router

Source of the tool set: Aura, a voice companion app (https://auravoiceapp.com) whose code lives at `C:\Users\varun\mobileapps\Aura` (backend) and `C:\Users\varun\mobileapps\aura-desktop` (desktop client).

## Goal

A portfolio demo for AWS Annapurna Labs. It fine-tunes Voxtral Mini 3B on Trainium (trn1) and serves it on Inferentia2 (inf2) as the speech-and-routing step of a LiveKit voice turn. Routing is the easy part; the Neuron work is what is being judged: compile, training, porting, numeric parity with a GPU/CPU reference, and latency.

Success looks like:
- the fine-tuned model beats base Voxtral on a real-speech test set
- Neuron output matches the reference within 1 point of accuracy
- end-of-speech to tool decision is reported at p50/p95 on inf2.xlarge, with cost per 1,000 turns

## Layout

- `data/tools.json`: frozen snapshot of 34 Aura voice tools (`speak_only` excluded). Labels are these 34 plus `no_tool`.
- `scripts/export_tools.py`: the only code that touches Aura. Rerun it only to refresh the snapshot on purpose.
- `scripts/llm_batch.py`: OpenAI Batch API runner with a disk cache (`cache/`).
- `scripts/generate_utterances.py`: generates the text dataset.
- `scripts/stress_test.py`: prompt-size vs accuracy check.
- `scripts/build_sft.py`: rows to LoRA chat pairs in `data/sft/`; after TTS each user turn is replaced by the audio clip with the same id.
- `data/text/rows.jsonl` is the text dataset; `results/` holds measured numbers.

## Rules

- Never import Aura at runtime. The demo works from `data/tools.json` alone and never talks to Aura's backend, Firebase, or real user data.
- Every LLM call goes through `llm_batch.run_batch` so nothing is paid for twice. Keep static prompt parts first and the per-row part last so prefixes stay cacheable.
- Run `--dry-run` and state the estimated cost before any paid batch. Ask before starting EC2 instances, and stop them when a step finishes.
- API keys live only in the gitignored `.env` (loaded by `tooling.py`) or the shell environment. Never commit them, print them, or put them on a command line.
- Report numbers only from files in `results/`; never quote a latency or accuracy that was not measured.
- Neuron compiles one graph per input shape: fix shapes and buckets deliberately and write down why.
- No em dashes anywhere. Commit messages read human-written, with no AI attribution. Never push without being asked.
