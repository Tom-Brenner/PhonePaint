"""Torch-native MAPS command-line interface."""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import statistics
import tempfile
import warnings
from pathlib import Path

import natsort
import numpy as np
import soxr
import torch
from scipy.io import wavfile
from textgrid import textgrid
from tqdm import tqdm

from args import build_arg_parser
from maps_torch.alignment import force_align, load_dictionary
from maps_torch.cli_types import WordString
from maps_torch.features import extract_features_batch, read_mono_wav
from maps_torch.model import load_checkpoint
from maps_torch.textgrid_io import make_textgrid

EPS = 1e-8
FRAME_INTERVAL = 0.01


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        device = torch.device("cuda")
    elif name == "cpu":
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"Unknown device {name!r}; choose auto, cpu, or cuda.")
    return device


def discover_models(model_path: Path) -> list[Path]:
    if model_path.suffix == ".tf":
        raise RuntimeError(
            f"TensorFlow models ({model_path}) are not supported at runtime. "
            "Run the one-time conversion:\n"
            "  conda run -n TF_maps python tools/export_tf_models.py\n"
            "  conda run -n torch-maps python tools/build_torch_checkpoints.py\n"
            "Then pass a .pt file or a directory of .pt ensemble checkpoints."
        )
    if model_path.suffix == ".pt" and model_path.is_file():
        return [model_path]
    if model_path.is_dir():
        models = natsort.natsorted(p for p in model_path.iterdir() if p.suffix == ".pt")
        if not models:
            raise RuntimeError(
                f"Could not find a model named {model_path}, nor any .pt models within that path."
            )
        return models
    raise RuntimeError(
        f"Could not find a model named {model_path}. Expected a .pt file or directory of .pt checkpoints."
    )


def textgrid_output_path(audio_input: Path, wav_path: Path) -> Path:
    """Place TextGrids beside the original input WAVs, not temp resampled copies."""
    if audio_input.is_file():
        return audio_input.with_suffix(".TextGrid")
    return audio_input / f"{wav_path.stem}.TextGrid"


def run_cli(argv: list[str] | None = None) -> None:
    p = build_arg_parser()
    args = vars(p.parse_args(argv))

    wavname_path = Path(args["audio"])
    if not wavname_path.is_file() and not wavname_path.is_dir():
        raise RuntimeError(f"Could not find {wavname_path}. Please check the spelling and try again.")

    original_wavnames: list[Path]
    if wavname_path.is_dir():
        original_wavnames = sorted(wavname_path / x for x in os.listdir(wavname_path) if x.lower().endswith(".wav"))
    else:
        original_wavnames = [wavname_path]

    wavnames = list(original_wavnames)
    temp_dir: tempfile.TemporaryDirectory | None = None

    if args["resample"]:
        print("RESAMPLING TO 16,000 HZ...")
        temp_dir = tempfile.TemporaryDirectory()
        temp_d = Path(temp_dir.name)
        resampled: list[Path] = []
        for wname in tqdm(original_wavnames):
            sr, samples = wavfile.read(wname)
            if samples.ndim == 2:
                samples = samples[:, 0]
            samples = soxr.resample(samples, sr, 16_000)
            out_path = temp_d / wname.name
            wavfile.write(out_path, 16_000, samples)
            resampled.append(out_path)
        wavnames = resampled

    model_path = Path(args["model"])
    model_names = discover_models(model_path)
    use_ensemble = len(model_names) > 1
    rm_ensemble = args["rm_ensemble"]
    ensemble_table = args["ensemble_table"]
    ensemble_json = args["ensemble_json"]

    transcription_path = Path(args["text"])
    if not transcription_path.is_file() and not transcription_path.is_dir():
        raise RuntimeError(f"Could not find {transcription_path}. Please check the spelling and try again.")
    if transcription_path.is_dir():
        transcriptions = [transcription_path / Path(x.name).with_suffix(".txt") for x in original_wavnames]
    else:
        transcriptions = [transcription_path]

    w_set = {x.stem for x in original_wavnames}
    t_set = {x.stem for x in transcriptions}
    mismatched = [w for w in original_wavnames if w.stem not in t_set]
    mismatched += [t for t in transcriptions if t.stem not in w_set]
    if mismatched:
        raise RuntimeError(
            "The following files did not have a corresponding WAV or txt match. "
            f"Please add matches or remove the files. Note that name matching is case-sensitive.\n"
            f"{','.join(str(x) for x in mismatched)}"
        )

    d_path = Path(args["dict"])
    if not d_path.is_file():
        raise RuntimeError(f"Could not find {d_path}. Please check the spelling and try again.")

    word2phone = load_dictionary(d_path)
    use_interp = args["interp"] == "true"
    add_sil = args["sil"] == "true"
    quiet = args["quiet"]
    overwrite = args["overwrite"]

    tgnames = [textgrid_output_path(wavname_path, w) for w in original_wavnames]
    filenames = list(zip(tgnames, wavnames, transcriptions, original_wavnames))

    word_list: list[str] = []
    for t in transcriptions:
        with open(t, "r") as f:
            word_list += f.read().upper().split()
    ood_words = {w for w in word_list if w not in word2phone}
    if ood_words:
        raise RuntimeError(
            "The following words were not found in the dictionary. "
            f"Please add them to the dictionary and run the aligner again.\n{', '.join(sorted(ood_words))}"
        )

    device = resolve_device(args["device"])
    print(f"Using device: {device}", flush=True)

    if not quiet:
        print("BEGINNING ALIGNMENT")

    loaded_models: list[tuple[Path, torch.nn.Module]] = []
    for m_name in model_names:
        loaded_models.append((m_name, load_checkpoint(m_name, device=device)))

    for m_I, (m_name, model) in enumerate(loaded_models, start=1):
        print(f"USING MODEL {m_name.name} ({m_I}/{len(loaded_models)})", flush=True)
        file_iter = tqdm(filenames) if not quiet else filenames
        have_stereo_warned = False

        for tgname_base, wavname, transcription, _original_wav in file_iter:
            if use_ensemble:
                tgname = tgname_base.parent / tgname_base.name.replace(".TextGrid", f"_{m_name.stem}.TextGrid")
            else:
                tgname = tgname_base

            if tgname.is_file() and not overwrite:
                continue

            sr, samples = read_mono_wav(wavname)
            if samples.ndim == 2:
                samples = samples[:, 0]
                if not have_stereo_warned:
                    warnings.warn(
                        "Stereo files were found. Automatically extracting the left (first) channel to make mono files."
                    )
                    have_stereo_warned = True

            duration = samples.size / sr
            x = extract_features_batch(samples, sr)

            with torch.inference_mode():
                tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
                yhat = model(tensor).cpu().numpy()

            with open(transcription, "r") as f:
                word_labels = f.read().upper().split()

            if add_sil and duration >= 0.045:
                word_labels = ["sil"] + word_labels + ["sil"]
            elif add_sil:
                warnings.warn(
                    f"Silence segments not added to ends of transcription for {wavname} because duration "
                    f"of {duration} s is too short to have silence padding."
                )
            word_chain = [word2phone[w] for w in word_labels]

            best_score = np.inf
            best_w_string: WordString | None = None
            best_seq = None
            best_M = None

            check_variants = args["check_variants"]
            variant_shuffle = args["variant_shuffle"]
            if variant_shuffle:
                seed = args["variant_seed"]
                r = random.Random() if seed is None else random.Random(seed)
                chains = list(itertools.product(*word_chain))
                r.shuffle(chains)
            else:
                chains = itertools.product(*word_chain)

            variant_limit = args["variant_limit"]
            if variant_limit is None:
                variant_limit = float("inf")

            tried_variants: set[str] = set()
            variant_counter = 0

            for c in chains:
                if add_sil:
                    this_word_labels = [x for cI, x in zip(c, word_labels) if cI]
                    c = [x for x in c if x]
                else:
                    this_word_labels = word_labels

                w_string = WordString(this_word_labels, c)
                if add_sil and len(w_string.collapsed_string) > (duration - 0.015) / 0.01:
                    if this_word_labels and this_word_labels[0] == "sil":
                        this_word_labels = this_word_labels[1:]
                        c = c[1:]
                        warnings.warn(
                            f"File {wavname} with duration {duration} too short for adding silence to "
                            f"transcription {w_string.collapsed_string}. Removing first silence label."
                        )
                    if this_word_labels and this_word_labels[-1] == "sil":
                        this_word_labels = this_word_labels[:-1]
                        c = c[:-1]
                        warnings.warn(
                            f"File {wavname} with duration {duration} too short for adding silence to "
                            f"transcription {w_string.collapsed_string}. Removing final silence label."
                        )
                    w_string = WordString(this_word_labels, c)

                if best_w_string is None:
                    best_w_string = w_string

                checking = " ".join(w_string.collapsed_string)
                if checking in tried_variants:
                    continue
                tried_variants.add(checking)

                seq, m = force_align(w_string.collapsed_string, yhat)
                if m[-1, -1] < best_score:
                    best_seq = seq
                    best_M = m
                    best_score = m[-1, -1]
                    best_w_string = w_string

                variant_counter += 1
                if (not check_variants) or (variant_counter == variant_limit):
                    break

            assert best_w_string is not None and best_seq is not None and best_M is not None

            n_segs = len(best_w_string.collapsed_string)
            if n_segs > 1 and duration < 0.015 + (0.01 * n_segs):
                warnings.warn(
                    f"File {wavname} with duration {duration} too short for collapsed "
                    f"{len(best_w_string.collapsed_string)}-segment best transcription "
                    f"{best_w_string.collapsed_string}. Assigning equal durations for each segment."
                )
                intervals = []
                for i, mark in enumerate(best_w_string.collapsed_string):
                    d_min = i / n_segs * duration
                    d_max = (i + 1) / n_segs * duration
                    intervals.append(textgrid.Interval(minTime=d_min, maxTime=d_max, mark=mark))
                tier = textgrid.IntervalTier("segments")
                tier.intervals = intervals
                from maps_torch.textgrid_io import make_word_tier

                word_tier = make_word_tier(tier, best_w_string)
                tg = textgrid.TextGrid()
                tg.tiers.append(word_tier)
                tg.tiers.append(tier)
                tg.write(tgname)
                continue

            make_textgrid(
                best_seq,
                tgname,
                duration,
                best_w_string,
                interpolate=use_interp,
                probs=best_M.T,
            )

    if use_ensemble:
        print("ENSEMBLING", flush=True)
        f_path = j_path = None
        if ensemble_table:
            f_path = Path(f"{'_'.join(wavname_path.parts)}_{model_path.name}_alignment_results.tsv")
            col_names = [
                "file",
                "word",
                "word_mintime",
                "word_maxtime",
                "segment",
                "segment_mintime",
                "segment_maxtime",
                "segment_lo_ci",
                "segment_hi_ci",
            ]
            with open(f_path, "a") as w:
                w.write("\t".join(col_names) + "\n")
        if ensemble_json:
            j_path = Path(f"{'_'.join(wavname_path.parts)}_{model_path.name}_alignment_results.json")

        all_tg_names: list[Path] = []
        for tgname_base, _, _, _ in tqdm(filenames):
            ensemble_tg_path = tgname_base.parent / f"{tgname_base.stem}_ensemble.TextGrid"
            ens_intervals = textgrid.IntervalTier(name="segments")
            intervals: list[textgrid.Interval] = []
            cis: list[textgrid.Point] = []
            tg_names = [
                tgname_base.parent / tgname_base.name.replace(".TextGrid", f"_{m_name.stem}.TextGrid")
                for m_name, _ in loaded_models
            ]
            all_tg_names.extend(tg_names)
            if ensemble_tg_path.is_file() and not overwrite:
                continue

            tgs = [textgrid.TextGrid() for _ in tg_names]
            for tg, tg_name in zip(tgs, tg_names):
                tg.read(tg_name, round_digits=1000)

            n_tgs = len(tgs)
            n_intervals = len(tgs[0].tiers[1].intervals)

            for i in range(n_intervals):
                lab = tgs[0].tiers[1].intervals[i].mark
                mintimes = [tgs[tier_i].tiers[1].intervals[i].minTime for tier_i in range(n_tgs)]
                maxtimes = [tgs[tier_i].tiers[1].intervals[i].maxTime for tier_i in range(n_tgs)]
                mintime = statistics.median(mintimes)
                maxtime = statistics.median(maxtimes)
                times_sorted = sorted(maxtimes)
                ci_lo = times_sorted[1]
                ci_hi = times_sorted[8]
                if ci_lo == ci_hi:
                    ci_lo -= EPS
                    ci_hi += EPS
                intervals.append(textgrid.Interval(minTime=mintime, maxTime=maxtime, mark=lab))
                if i < n_intervals - 1:
                    cis += [
                        textgrid.Point(mark=f"{lab}_cilo", time=ci_lo),
                        textgrid.Point(mark=f"{lab}_cihi", time=ci_hi),
                    ]

            n_word_intervals = len(tgs[0].tiers[0].intervals)
            word_intervals = []
            for i in range(n_word_intervals):
                lab = tgs[0].tiers[0].intervals[i].mark
                mintimes = [tgs[tier_i].tiers[0].intervals[i].minTime for tier_i in range(n_tgs)]
                maxtimes = [tgs[tier_i].tiers[0].intervals[i].maxTime for tier_i in range(n_tgs)]
                word_intervals.append(
                    textgrid.Interval(
                        minTime=statistics.median(mintimes),
                        maxTime=statistics.median(maxtimes),
                        mark=lab,
                    )
                )

            ens_tg = textgrid.TextGrid(maxTime=tgs[0].maxTime)
            word_tier = textgrid.IntervalTier(name="words")
            word_tier.intervals = word_intervals
            ens_tg.tiers.append(word_tier)
            int_tier = textgrid.IntervalTier(name="segments")
            int_tier.intervals = intervals
            ens_tg.tiers.append(int_tier)
            ci_tier = textgrid.PointTier(name="95-CIs")
            ci_tier.points = cis
            ens_tg.tiers.append(ci_tier)
            ens_tg.write(ensemble_tg_path)

            if ensemble_table or ensemble_json:
                fname = ensemble_tg_path.name
                j_out = ensemble_tg_path.with_suffix(".json")
                word_iter = iter(word_intervals)
                word = next(word_iter)
                for x_I, x in enumerate(intervals):
                    if x_I == len(intervals) - 1:
                        segment_lo_ci = x.maxTime
                        segment_hi_ci = x.maxTime
                    else:
                        segment_lo_ci = cis[x_I * 2].time
                        segment_hi_ci = cis[x_I * 2 + 1].time
                    row = [
                        fname,
                        word.mark,
                        word.minTime,
                        word.maxTime,
                        x.mark,
                        x.minTime,
                        x.maxTime,
                        segment_lo_ci,
                        segment_hi_ci,
                    ]
                    if ensemble_table and f_path is not None:
                        with open(f_path, "a") as w:
                            w.write("\t".join(str(z) for z in row) + "\n")
                    if ensemble_json:
                        ens_j = {
                            "file": fname,
                            "word": word.mark,
                            "word_mintime": word.minTime,
                            "word_maxtime": word.maxTime,
                            "segment": x.mark,
                            "segment_mintime": x.minTime,
                            "segment_maxtime": x.maxTime,
                            "segment_lo_ci": segment_lo_ci,
                            "segment_hi_ci": segment_hi_ci,
                        }
                        with open(j_out, "a") as w:
                            w.write(json.dumps(ens_j, indent=4))
                    if x.maxTime == word.maxTime and x_I < len(intervals) - 1:
                        word = next(word_iter)

        if rm_ensemble:
            for n in all_tg_names:
                n.unlink(missing_ok=True)

    if temp_dir is not None:
        temp_dir.cleanup()


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
