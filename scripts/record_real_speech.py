"""Record yourself saying the val utterances, so the headline claim is about real speech.

Every clip in data/audio is TTS. The success criterion says the fine-tuned model has to
beat base Voxtral on real speech, and TTS is not that: it has no room noise, no breath, no
disfluency, and a Whisper check already confirmed it is transcribable almost perfectly.
A model can look good on synthetic audio and fall over on a person.

    python scripts/record_real_speech.py --count 150        # stratified across all tools
    python scripts/record_real_speech.py --count 150        # rerun resumes where it stopped
    python scripts/record_real_speech.py --tool web_surf    # top up one label

Per row: the text is printed, Enter starts recording, Enter stops it, then it plays back
and you keep it, redo it, or skip it. Clips land in data/audio_real/<id>.wav at 16 kHz
mono, named by the same row id as the TTS set, so eval_routing.py scores the two sets
against identical gold labels:

    python scripts/eval_routing.py --audio-dir data/audio_real --results results/eval_base_real.json

Rows are chosen round robin across tools rather than at random. 34 tools over 665 val rows
means a uniform sample would leave several labels with one clip or none, and a per-tool
accuracy computed from one clip is noise.

Read the line as written. The gold arguments are extracted from that exact wording, so
paraphrasing quietly breaks the label.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tooling import read_rows

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
MAX_SECONDS = 30  # the encoder window; anything longer cannot be fed to Voxtral at all
QUIET_PEAK = 0.02  # below this the mic was muted or the wrong device was selected


def stratified(rows: list[dict], count: int, done: set) -> list[dict]:
    """Round robin across tools, so every label gets clips before any label gets seconds."""
    by_tool = defaultdict(list)
    for row in rows:
        if row["id"] not in done:
            by_tool[row["tool"]].append(row)
    picked, tools = [], sorted(by_tool)
    while len(picked) < count and any(by_tool[t] for t in tools):
        for tool in tools:
            if by_tool[tool] and len(picked) < count:
                picked.append(by_tool[tool].pop(0))
    return picked


def record_clip() -> np.ndarray:
    """Open the mic, collect until Enter. Returns mono float32 at SAMPLE_RATE."""
    import sounddevice as sd

    frames = []

    def callback(indata, _frames, _time, status):
        if status:
            print(f"  audio status: {status}", flush=True)
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input("  recording, Enter to stop ")
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames)[:, 0]


def describe(audio: np.ndarray) -> str:
    seconds = len(audio) / SAMPLE_RATE
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    notes = []
    if seconds > MAX_SECONDS:
        notes.append(f"TOO LONG, over the {MAX_SECONDS}s window")
    if peak < QUIET_PEAK:
        notes.append("TOO QUIET, check the input device")
    if peak >= 0.999:
        notes.append("CLIPPING, move back from the mic")
    return f"  {seconds:.1f}s, peak {peak:.2f}" + ("  <- " + "; ".join(notes) if notes else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--tool", default="", help="only rows for this tool")
    parser.add_argument("--out", default="data/audio_real")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    import sounddevice as sd
    import soundfile as sf

    if args.list_devices:
        print(sd.query_devices())
        return

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "recorded.jsonl"

    done = {json.loads(l)["id"] for l in manifest.read_text(encoding="utf-8").splitlines()
            if l.strip()} if manifest.exists() else set()
    # A wav with no manifest line, or the reverse, would silently desync the two.
    done = {i for i in done if (out_dir / f"{i}.wav").exists()}

    rows = read_rows(args.split)
    if args.tool:
        rows = [r for r in rows if r["tool"] == args.tool]
    queue = stratified(rows, args.count, done)

    print(f"{len(done)} already recorded, {len(queue)} queued, writing to {out_dir}")
    print(f"input device: {sd.query_devices(kind='input')['name']}")
    print("Enter starts, Enter stops. Then k keep, r redo, s skip, q quit.\n")

    kept = 0
    with manifest.open("a", encoding="utf-8") as sink:
        for i, row in enumerate(queue, 1):
            print(f"[{i}/{len(queue)}] {row['tool']}")
            print(f'  "{row["utterance"]}"')
            while True:
                choice = input("  Enter to record, s skip, q quit ").strip().lower()
                if choice == "q":
                    print(f"\nstopped. {kept} recorded this session, {len(done) + kept} total.")
                    return
                if choice == "s":
                    break
                audio = record_clip()
                print(describe(audio))
                if len(audio) == 0:
                    print("  nothing captured, try again")
                    continue
                sd.play(audio, SAMPLE_RATE)
                sd.wait()
                verdict = input("  k keep, r redo, s skip, q quit ").strip().lower()
                if verdict == "q":
                    print(f"\nstopped. {kept} recorded this session, {len(done) + kept} total.")
                    return
                if verdict == "s":
                    break
                if verdict != "k":
                    continue
                if len(audio) / SAMPLE_RATE > MAX_SECONDS:
                    print("  refusing: longer than the encoder window, record it again")
                    continue
                # PCM_16 to match the TTS clips exactly, so the two sets differ only in
                # who spoke, not in how they were stored.
                sf.write(out_dir / f"{row['id']}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
                sink.write(json.dumps({
                    "id": row["id"], "tool": row["tool"], "text": row["utterance"],
                    "seconds": round(len(audio) / SAMPLE_RATE, 2),
                    "peak": round(float(np.abs(audio).max()), 3),
                }) + "\n")
                sink.flush()
                kept += 1
                print("  kept\n")
                break

    print(f"done. {kept} recorded this session, {len(done) + kept} total in {out_dir}")


if __name__ == "__main__":
    main()
