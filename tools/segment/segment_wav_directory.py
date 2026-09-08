#!/usr/bin/env python3
"""Silence-based WAV segmentation (Applio-identical).

Vendored from /media/tom/SATAM/Applio41/segment_wav_directory.py so
PhonePaint's --bfa / --mfa / --falcon pipelines share the same cut points as
~/a3t/run_inpaint_pipeline.py. Only ASR and alignment backends differ.

Outputs 32 kHz mono PCM via ffmpeg, matching Applio.
"""
import argparse
import subprocess
from pathlib import Path
from pydub import AudioSegment, silence


def longest_silence(audio: AudioSegment,
                    start_ms: int,
                    end_ms: int,
                    thresh_db: float):
    region = audio[start_ms:end_ms]
    sils = silence.detect_silence(region,
                                  min_silence_len=75,
                                  silence_thresh=thresh_db)
    if not sils:
        return None
    sils = [(s + start_ms, e + start_ms) for s, e in sils]
    return max(sils, key=lambda x: x[1] - x[0])


def cut_with_ffmpeg(src: Path,
                    dst: Path,
                    start_ms: int,
                    end_ms: int):
    start_sec = f"{start_ms/1000:.3f}"
    end_sec = f"{end_ms/1000:.3f}"
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-accurate_seek",
        "-i", str(src),
        "-ss", start_sec,
        "-to", end_sec,
        "-c:a", "pcm_s16le",
        "-ar", "32000",
        "-ac", "1",
        str(dst)
    ]
    subprocess.run(cmd, check=True)


def segment_file(path: Path,
                 out_dir: Path,
                 thresh_db: float):
    audio = AudioSegment.from_file(path)
    total_len = len(audio)
    cut_points = []
    prev = 0

    while total_len - prev >= 30000:          # need ≥22.5 s left after 7.5 s buffer
        rng_start = prev + 7500              # 10 s after last cut
        rng_end = rng_start + 7500           # window 10 s long
        sil = longest_silence(audio, rng_start, rng_end, thresh_db)
        if sil:
            cut = sil[0] + (sil[1] - sil[0]) // 3
        else:
            cut = rng_start + 3750            # midpoint fallback
        cut_points.append(cut)
        prev = cut

    tail_start = max(prev, total_len - 22500)
    sil = longest_silence(audio, tail_start, total_len, thresh_db)
    if sil:
        cut = sil[0] + (sil[1] - sil[0]) // 3
        if prev < cut < total_len:
            cut_points.append(cut)
        else:
            cut = (total_len + tail_start) // 2
            cut_points.append(cut)
    else:
        cut = (total_len + tail_start) // 2
        cut_points.append(cut)

    segs = [0] + cut_points + [total_len]
    stem = path.stem
    for i in range(len(segs) - 1):
        out_path = out_dir / f"{stem}_segment_{i}.wav"
        cut_with_ffmpeg(path, out_path, segs[i], segs[i + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--silence_level", type=float, default=-40.0,
                    help="Silence threshold in dBFS (negative)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Drop stale chunks so a prior N-way split cannot leave orphan segment_K.wav
    # after a re-run that produces fewer pieces (Applio itself did not clear).
    for old in args.output_dir.glob("*.wav"):
        old.unlink()
    for wav in sorted(args.input_dir.glob("**/*.wav")):
        segment_file(wav, args.output_dir, args.silence_level)


if __name__ == "__main__":
    main()
