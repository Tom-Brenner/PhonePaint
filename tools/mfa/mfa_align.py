#!/usr/bin/env python3
"""mfa_align_from_json.py – MFA alignment from pre-provided text or CSV
Skips missing audio and missing .TextGrid outputs; processes intersection only.

CSV rule: row j maps to wav "{j:04d}.wav"; text taken from column 'transcript'.
"""

from __future__ import annotations
import argparse, csv, json, os, shutil, subprocess as sp, tempfile, re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Iterable
from textgrid import TextGrid, IntervalTier  # pip install textgrid

_MFA_PRETRAINED_BASE = Path.home() / "Documents" / "MFA" / "pretrained_models"
_MFA_DICT_DIR = _MFA_PRETRAINED_BASE / "dictionary"
_MFA_ACOUSTIC_DIR = _MFA_PRETRAINED_BASE / "acoustic"
_MFA_G2P_DIR = _MFA_PRETRAINED_BASE / "g2p"
_DEFAULT_MFA_CONDA_ENV = "mfa_env"
_PHONEPAINT_MFA_ENVS = frozenset({"PhonePaintMFA"})


def _active_conda_env_name() -> Optional[str]:
    name = (os.environ.get("CONDA_DEFAULT_ENV") or "").strip()
    if name:
        return name
    prefix = (os.environ.get("CONDA_PREFIX") or "").strip()
    if prefix:
        return Path(prefix).name
    return None


def _is_phonepaint_mfa_env() -> bool:
    return _active_conda_env_name() in _PHONEPAINT_MFA_ENVS


def resolve_mfa_bin() -> str:
    """Locate the MFA CLI binary; never import montreal_forced_aligner.

    Resolution order:
      1. MFA_BIN
      2. Active env: ``$CONDA_PREFIX/bin/mfa`` (same-env; preferred)
      3. ``mfa`` on PATH when it lives under ``$CONDA_PREFIX``
      4. Fallback conda env ``MFA_CONDA_ENV`` / ``mfa_env`` (legacy sidecar) —
         skipped while PhonePaintMFA is active (MFA+MAPS share that env)
      5. Any ``mfa`` on PATH (also skipped while PhonePaintMFA is active)
    """
    explicit = os.environ.get("MFA_BIN")
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"MFA_BIN is set but not a file: {explicit}")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    candidates: List[Path] = []
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "mfa")
        which = shutil.which("mfa")
        if which:
            which_path = Path(which).resolve()
            prefix_path = Path(conda_prefix).resolve()
            try:
                which_path.relative_to(prefix_path)
                candidates.append(which_path)
            except ValueError:
                pass

    allow_sidecar = not _is_phonepaint_mfa_env()
    if allow_sidecar:
        env_name = os.environ.get("MFA_CONDA_ENV", _DEFAULT_MFA_CONDA_ENV)
        if conda_prefix:
            candidates.append(Path(conda_prefix).parent / env_name / "bin" / "mfa")
        home = Path.home()
        for base in (home / "miniconda3", home / "anaconda3", home / "mambaforge", home / "miniforge3"):
            candidates.append(base / "envs" / env_name / "bin" / "mfa")
        which = shutil.which("mfa")
        if which:
            candidates.append(Path(which))

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return str(cand)
    if _is_phonepaint_mfa_env():
        env = _active_conda_env_name() or "PhonePaint"
        raise FileNotFoundError(
            f"{env} is active but MFA CLI was not found in this env "
            "($CONDA_PREFIX/bin/mfa). Install MFA in-env or set MFA_BIN; "
            f"the mfa_env sidecar is not used while {env} is active."
        )
    env_name = os.environ.get("MFA_CONDA_ENV", _DEFAULT_MFA_CONDA_ENV)
    raise FileNotFoundError(
        "MFA CLI not found. Install MFA in the active env (e.g. PhonePaint), "
        "create mfa_env (./create_mfa_envs.sh), or set MFA_BIN "
        f"(looked for active env and fallback {env_name!r})."
    )


def mfa_cli_env(**extra: str) -> dict:
    """Env for subprocesses that run the MFA CLI.

    Calling ``mfa`` by absolute path is not enough: MFA's third-party check and
    Kaldi helpers need ``sox`` / Kaldi binaries from the MFA conda env on PATH
    (and their shared libs).
    """
    env = os.environ.copy()
    mfa_bin = Path(resolve_mfa_bin())
    env_bin = mfa_bin.parent
    env_root = env_bin.parent
    env["PATH"] = f"{env_bin}{os.pathsep}{env.get('PATH', '')}"
    lib_dir = env_root / "lib"
    if lib_dir.is_dir():
        prev_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{lib_dir}{os.pathsep}{prev_ld}" if prev_ld else str(lib_dir)
        )
    # MFA looks for conda-installed third-party tools relative to this prefix.
    env["CONDA_PREFIX"] = str(env_root)
    env["MFA_BIN"] = str(mfa_bin)
    if extra:
        env.update(extra)
    return env


def _resolve_model_file(kind: str, model: str) -> Path:
    p = Path(os.path.expanduser(os.path.expandvars(model)))
    if p.exists():
        return p
    if kind == "dictionary":
        cand = _MFA_DICT_DIR / f"{model}.dict"
    elif kind == "acoustic":
        cand = _MFA_ACOUSTIC_DIR / f"{model}.zip"
    elif kind == "g2p":
        cand = _MFA_G2P_DIR / f"{model}.zip"
    else:
        raise ValueError(kind)
    if not cand.exists():
        raise FileNotFoundError(f"Missing MFA {kind} model file: {cand}")
    return cand


def _load_dictionary_vocab(dict_path: Path) -> set[str]:
    vocab: set[str] = set()
    for line in dict_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        w = line.split()[0]
        w = re.sub(r"\(\d+\)$", "", w)
        vocab.add(w.lower())
    return vocab


# ARPABET to IPA conversion mapping for MFA compatibility
ARPABET_TO_IPA = {
    # Vowels
    'AA': 'ɑ', 'AA0': 'ɑ', 'AA1': 'ɑ', 'AA2': 'ɑ',
    'AE': 'æ', 'AE0': 'æ', 'AE1': 'æ', 'AE2': 'æ',
    'AH': 'ʌ', 'AH0': 'ə', 'AH1': 'ʌ', 'AH2': 'ʌ',  # AH0 -> schwa
    'AO': 'ɔ', 'AO0': 'ɔ', 'AO1': 'ɔ', 'AO2': 'ɔ',
    'AW': 'aʊ', 'AW0': 'aʊ', 'AW1': 'aʊ', 'AW2': 'aʊ',
    'AY': 'aɪ', 'AY0': 'aɪ', 'AY1': 'aɪ', 'AY2': 'aɪ',
    'EH': 'ɛ', 'EH0': 'ɛ', 'EH1': 'ɛ', 'EH2': 'ɛ',
    'ER': 'ɜr', 'ER0': 'ər', 'ER1': 'ɜr', 'ER2': 'ɜr',  # ER0 -> schwar
    'EY': 'eɪ', 'EY0': 'eɪ', 'EY1': 'eɪ', 'EY2': 'eɪ',
    'IH': 'ɪ', 'IH0': 'ɪ', 'IH1': 'ɪ', 'IH2': 'ɪ',
    'IY': 'i', 'IY0': 'i', 'IY1': 'i', 'IY2': 'i',
    'OW': 'oʊ', 'OW0': 'oʊ', 'OW1': 'oʊ', 'OW2': 'oʊ',
    'OY': 'ɔɪ', 'OY0': 'ɔɪ', 'OY1': 'ɔɪ', 'OY2': 'ɔɪ',
    'UH': 'ʊ', 'UH0': 'ʊ', 'UH1': 'ʊ', 'UH2': 'ʊ',
    'UW': 'u', 'UW0': 'u', 'UW1': 'u', 'UW2': 'u',
    
    # Consonants
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f', 'G': 'ɡ',
    'HH': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n',
    'NG': 'ŋ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ', 'T': 't',
    'TH': 'θ', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}


def convert_arpabet_to_ipa(phone: str) -> str:
    """Convert ARPABET phone to IPA phone used by english_mfa.
    
    Args:
        phone: ARPABET phone (e.g., 'AY1', 'B', 'SH')
        
    Returns:
        IPA phone (e.g., 'aɪ', 'b', 'ʃ')
    """
    phone = phone.strip().upper()
    
    # Direct lookup
    if phone in ARPABET_TO_IPA:
        return ARPABET_TO_IPA[phone]
    
    # Try without stress marker for vowels
    base_phone = re.sub(r'[012]$', '', phone)
    if base_phone in ARPABET_TO_IPA:
        return ARPABET_TO_IPA[base_phone]
    
    # If no conversion found, return original (might already be IPA)
    return phone.lower()


def _ffmpeg_convert(src: Path, dst: Path, gain_db: float | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
    ]
    # Optional volume gain for the WAV that MFA aligns against.
    # Applied only when explicitly requested to preserve default behavior.
    if gain_db is not None:
        cmd += ["-af", f"volume={gain_db}dB"]
    cmd.append(str(dst))
    sp.run(cmd, check=True)


def run_mfa(
    corpus: Path,
    acoustic: str,
    dictionary: str,
    threads: int,
    *,
    out_dir: Optional[Path] = None,
    tag: Optional[str] = None,
) -> Path:
    """Run MFA alignment into a *dedicated* output directory.

    Important: the old implementation always wrote to corpus/out and thus overwrote
    previous passes. Incremental/fallback logic requires distinct output dirs.
    """
    if out_dir is None:
        out_dir = corpus / "out"
    out_dir = Path(out_dir)

    # Keep per-pass history/log to avoid clobbering across multiple runs.
    tag = tag or out_dir.name
    hist = corpus / f"history_{tag}"
    hist.mkdir(exist_ok=True)
    log = corpus / f"mfa_{tag}.log"

    mfa_cmd = resolve_mfa_bin()
    dict_arg = str(_resolve_model_file("dictionary", dictionary))
    acoustic_arg = str(_resolve_model_file("acoustic", acoustic))
    n_jobs = max(1, int(threads))
    cmd = [
        mfa_cmd,
        "align",
        str(corpus),
        dict_arg,
        acoustic_arg,
        str(out_dir),
        "--clean",
        "--single_speaker",
        "--num_jobs",
        str(n_jobs),
        "--output_format",
        "long_textgrid",
        "--debug",
        "--verbose",
        "--overwrite",
    ]
    # --use_mp with num_jobs=1 still pays multiprocessing overhead for no gain.
    if n_jobs > 1:
        cmd.append("--use_mp")
    env = mfa_cli_env(MFA_HISTORY_DIR=str(hist))

    try:
        result = sp.run(cmd, capture_output=True, text=True, check=False, env=env)
        log.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8", errors="ignore")
        if result.returncode != 0:
            print(f"MFA command failed with return code {result.returncode}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Log file: {log}")
            if log.exists():
                print("Last 10 lines of log:")
                with log.open("r") as rf:
                    _lines = rf.readlines()
                    for line in _lines[-10:]:
                        print(f"  {line.rstrip()}")
            raise RuntimeError(f"MFA failed (see {log})")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"MFA not found at {mfa_cmd}. Install MFA in the active env, set "
            "MFA_BIN, or create mfa_env (./create_mfa_envs.sh)."
        ) from exc
    return out_dir


def count_spn_instances(output_dir: Path) -> int:
    """Count total 'spn' phone instances in all TextGrid files."""
    spn_count = 0
    for tg_file in output_dir.glob("*.TextGrid"):
        try:
            words, phones = parse_textgrid(tg_file)
            spn_count += sum(1 for _, _, phone in phones if phone.strip().lower() == 'spn')
        except Exception as e:
            print(f"Warning: Could not parse {tg_file}: {e}")
    return spn_count


def extract_spn_segments(output_dir: Path) -> dict:
    """Extract information about spn segments for targeted re-alignment."""
    spn_segments = {}
    for tg_file in output_dir.glob("*.TextGrid"):
        try:
            words, phones = parse_textgrid(tg_file)
            file_spns = []
            for start, end, phone in phones:
                if phone.strip().lower() == 'spn':
                    file_spns.append((start, end, phone))
            if file_spns:
                spn_segments[tg_file.stem] = file_spns
        except Exception as e:
            print(f"Warning: Could not parse {tg_file}: {e}")
    return spn_segments


def create_spn_targeted_corpus(original_corpus: Path, spn_segments: dict, text_data: dict) -> Path:
    """Create a new corpus containing only audio/text segments that had spn instances."""
    spn_corpus = original_corpus.parent / f"{original_corpus.name}_spn_targeted"
    spn_corpus.mkdir(exist_ok=True)
    
    # Copy files that had spn instances
    for filename in spn_segments.keys():
        wav_file = f"{filename}.wav"
        lab_file = f"{filename}.lab"
        
        src_wav = original_corpus / wav_file
        src_lab = original_corpus / lab_file
        
        if src_wav.exists() and src_lab.exists():
            shutil.copy2(src_wav, spn_corpus / wav_file)
            shutil.copy2(src_lab, spn_corpus / lab_file)
    
    return spn_corpus


def find_time_overlapping_segments(segments_list: list, target_start: float, target_end: float, tolerance: float = 0.01) -> list:
    """Find segments that overlap with a target time range."""
    overlapping = []
    for i, (start, end, text) in enumerate(segments_list):
        # Check for any overlap with tolerance
        if not (end <= target_start - tolerance or start >= target_end + tolerance):
            overlapping.append((i, start, end, text))
    return overlapping

def _build_aligned_dict(out_dir: Path, included: list[str]) -> dict[str, dict]:
    """Create the aligned JSON dict from TextGrids in out_dir for the given files."""
    words_all, phones_all = defaultdict(list), defaultdict(list)
    for fn in included:
        tg_path = out_dir / Path(fn).with_suffix(".TextGrid")
        if not tg_path.exists():
            continue
        w, p = parse_textgrid(tg_path)
        words_all[fn], phones_all[fn] = w, p

    aligned: dict[str, dict] = {}
    for fn in included:
        if fn not in words_all and fn not in phones_all:
            continue
        aligned[fn] = {
            "words": {
                str(i): {"xmin": round(x0, 3), "xmax": round(x1, 3), "text": t}
                for i, (x0, x1, t) in enumerate(words_all.get(fn, []))
            },
            "phones": {
                str(i): {"xmin": round(x0, 3), "xmax": round(x1, 3), "text": t}
                for i, (x0, x1, t) in enumerate(phones_all.get(fn, []))
            },
        }
    return aligned


def _alignment_needs_fallback(out_dir: Path, included: list[str]) -> bool:
    """True when alignment still has OOV/spn/sil gaps or a missing TextGrid."""
    targets = {"<unk>", "unk", "</unk>", "spn", "sil"}
    for fn in included:
        tg_path = out_dir / Path(fn).with_suffix(".TextGrid")
        if not tg_path.exists():
            return True
        try:
            words, phones = parse_textgrid(tg_path)
        except Exception:
            return True
        for _, _, w in words:
            if (w or "").strip().lower() in targets:
                return True
        for _, _, p in phones:
            if (p or "").strip().lower() in targets:
                return True
    return False


def _needs_g2p(out_dir: Path, included: list[str]) -> bool:
    """Return True if any included file still has unk/spn/sil tokens in words or phones."""
    return _alignment_needs_fallback(out_dir, included)

def run_incremental_alignment_with_model_pairs(
    corpus: Path,
    model_pairs: list[tuple[str, str]],
    threads: int,
    text_data: dict[str, dict],
    use_g2p: bool = False,
    g2p_model: str = "english_us_mfa",
    *,
    pre_g2p_json_path: Optional[Path] = None,
    included_fns: Optional[list[str]] = None,
) -> tuple[Path, dict]:
    """
    IMPROVED Pipeline (fixes 'homie' -> '<unk>' issue):
      1) Align with primary pair → primary out_dir.
      2) For each subsequent pair (e.g., ARPA): align only when primary still has
         <unk>/spn/sil gaps; surgically replace spn in primary using that pass.
      3) Optional G2P pass LAST: build enhanced dict from input texts; align; surgically replace remaining spn.
      4) Return the modified primary out_dir and stats.

    Replacement semantics:
      - Replace ONLY primary 'spn' phones using fallback phones clipped to the primary word span.
      - Words tier: replace '<unk>' with fallback word(s) if available; keep recognized primary words as-is.
      - No borrowing from neighbors; original primary word boundaries preserved.
      - G2P is used LAST, only for remaining OOV issues after dictionary fallbacks.
    """
    included = included_fns or list(text_data.keys())

    # 1) Primary alignment
    primary_acoustic, primary_dict = model_pairs[0]
    primary_out = run_mfa(corpus, primary_acoustic, primary_dict, threads, out_dir=(corpus / "out_primary"), tag="primary")

    # Stats init
    stats = {
        "models_tried": [f"{primary_acoustic}+{primary_dict}"],
        "spn_counts": [],
        "improvements": [],  # delta vs previous
    }
    spn0 = count_spn_instances(primary_out)
    stats["spn_counts"].append(spn0)

    # 2) Subsequent model pairs as additional fallbacks (e.g., english_us_arpa)
    for k, (acoustic, dictionary) in enumerate(model_pairs[1:], start=1):
        if not _alignment_needs_fallback(primary_out, included):
            print(
                f"  Skipping fallback pair {acoustic}+{dictionary}: "
                "primary alignment complete (no <unk>/spn/sil)."
            )
            break
        print(f"\n  Running fallback pair {acoustic}+{dictionary} …")
        fb_out = run_mfa(
            corpus,
            acoustic,
            dictionary,
            threads,
            out_dir = (corpus / f"out_fallback_{k:02d}"),
            tag = f"fallback_{k:02d}")
        is_arpa_model = 'arpa' in acoustic.lower() or 'arpa' in dictionary.lower()
        surgical_spn_replacement(primary_out, fb_out, convert_arpabet=is_arpa_model)
        spn_now = count_spn_instances(primary_out)
        stats["models_tried"].append(f"{acoustic}+{dictionary}")
        stats["improvements"].append(stats["spn_counts"][-1] - spn_now)
        stats["spn_counts"].append(spn_now)
        if not _alignment_needs_fallback(primary_out, included):
            print("  Alignment complete after fallback; skipping remaining dictionary pairs.")
            break

    # 3) Optional G2P pass LAST (only after dictionary fallbacks have been tried)
    # Pre-G2P JSON snapshot, if requested
    if use_g2p and pre_g2p_json_path is not None:
        if not included:
            raise RuntimeError("included_fns must be provided when pre_g2p_json_path is set.")
        pre_aligned = _build_aligned_dict(primary_out, included)
        pre_g2p_json_path.parent.mkdir(parents=True, exist_ok=True)
        pre_g2p_json_path.write_text(json.dumps(pre_aligned, indent=2, ensure_ascii=False))
        print(f"[pre-G2P] JSON snapshot written to: {pre_g2p_json_path}")

    if use_g2p:
        # Only run G2P if targeted tokens remain
        if not _needs_g2p(primary_out, included):
            print("  Skipping G2P fallback: no <unk>/unk/</unk>/spn/sil tokens detected.")
            return primary_out, stats

        print(f"\n  Applying G2P fallback (last resort for remaining OOV words)...")
        tmp_g2p_dir = corpus / "_g2p"
        tmp_g2p_dir.mkdir(exist_ok=True)
        all_words = extract_words_from_texts(text_data)  # naive tokenization
        if generate_g2p_pronunciations(all_words, g2p_model, tmp_g2p_dir):
            enhanced_dict = create_enhanced_dictionary(primary_dict, tmp_g2p_dir / "g2p_pronunciations.dict", tmp_g2p_dir)
            if enhanced_dict and enhanced_dict.exists():
                g2p_out = run_mfa(corpus, primary_acoustic, str(enhanced_dict), threads, out_dir=(corpus / "out_g2p"),tag="g2p")
                surgical_spn_replacement(primary_out, g2p_out, convert_arpabet=False)
                spn_g2p = count_spn_instances(primary_out)
                stats["models_tried"].append(f"{primary_acoustic}+G2P({g2p_model})")
                stats["improvements"].append(stats["spn_counts"][-1] - spn_g2p)
                stats["spn_counts"].append(spn_g2p)
            else:
                print("  G2P fallback skipped: could not create enhanced dictionary")
        else:
            print("  G2P fallback skipped: could not generate pronunciations")

    # 4) Return the modified primary out_dir and stats
    return primary_out, stats



def _stretch_segments_to_primary_boundaries(primary_start: float, primary_end: float, fallback_segments: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """Stretch fallback segments (phones or words) to fit primary boundaries using proportional scaling.
    
    NEVER use fallback timestamps directly. Always scale relative to primary boundaries.
    
    Example: 'homie' primary (0.85-1.12), fallback phones [(0.85, 0.94, 'h'), (0.94, 0.97, 'oʊ'), (0.97, 1.04, 'm'), (1.04, 1.10, 'i')]
    
    Algorithm:
    1. Convert fallback to relative times (0-based)
    2. Calculate scaling factor to fit primary duration 
    3. Scale each segment proportionally
    4. Add primary_start to get final timestamps
    """
    if not fallback_segments:
        return []
    
    # Get fallback boundaries
    fallback_start = min(s for s, e, lab in fallback_segments)
    fallback_end = max(e for s, e, lab in fallback_segments)
    fallback_duration = fallback_end - fallback_start
    
    if fallback_duration <= 1e-9:
        # Degenerate case: all segments at same time, distribute equally
        primary_duration = primary_end - primary_start
        segment_duration = primary_duration / len(fallback_segments)
        result = []
        for i, (_, _, lab) in enumerate(fallback_segments):
            start = primary_start + i * segment_duration
            end = primary_start + (i + 1) * segment_duration
            result.append((start, end, lab))
        return result
    
    # Calculate scaling factor
    primary_duration = primary_end - primary_start
    scale_factor = primary_duration / fallback_duration
    
    # Scale each segment
    result = []
    for fb_start, fb_end, lab in fallback_segments:
        # Convert to relative times (0-based from fallback_start)
        rel_start = fb_start - fallback_start
        rel_end = fb_end - fallback_start
        
        # Scale to primary duration
        scaled_start = rel_start * scale_factor
        scaled_end = rel_end * scale_factor
        
        # Add primary_start to get final absolute times
        final_start = primary_start + scaled_start
        final_end = primary_start + scaled_end
        
        result.append((final_start, final_end, lab))
    
    # Ensure last phone ends exactly at primary_end (handle floating point precision)
    if result:
        last_start, _, last_lab = result[-1]
        result[-1] = (last_start, primary_end, last_lab)
    
    return result


def word_bounds_for(primary_words: Optional[IntervalTier], span_start: float, span_end: float) -> Tuple[float, float]:
    """Find the primary word boundaries that contain the given span.
    
    CRITICAL: Always return primary model boundaries, never fallback boundaries.
    This ensures timestamps like 'homie' (0.85-1.12 primary vs 0.85-1.10 fallback) 
    use the primary boundaries (0.85-1.12).
    """
    if primary_words is None:
        return (span_start, span_end)
    
    # Find the primary word that overlaps with this span
    for w in primary_words:
        # Check if primary word overlaps with the span (not strict containment)
        if (w.maxTime > span_start + 1e-9 and w.minTime < span_end - 1e-9):
            if os.environ.get("MFA_DEBUG_BOUNDARIES"):
                word_label = (w.mark or "").strip()
                print(f"    word_bounds_for({span_start:.3f}-{span_end:.3f}) -> primary word '{word_label}' ({w.minTime:.3f}-{w.maxTime:.3f})")
            return (w.minTime, w.maxTime)
    
    # If no overlapping word found, return the span as-is
    if os.environ.get("MFA_DEBUG_BOUNDARIES"):
        print(f"    word_bounds_for({span_start:.3f}-{span_end:.3f}) -> no overlapping word found, using span")
    return (span_start, span_end)


def _iter_textgrid_paths(root: Path) -> Iterable[Path]:
    patterns = ("*.TextGrid", "*.Textgrid", "*.textgrid")
    for pat in patterns:
        yield from root.rglob(pat)



def surgical_spn_replacement(primary_path: Path, fallback_path: Path, convert_arpabet: bool = True) -> None:
    """
    Recalculate per pseudocode. Copy primary word boundaries; only replace primary 'spn' with fallback phones
    remapped to primary span by equal per-phone adjustment; never copy fallback timestamps.
    """
        # Handle directory vs file inputs up front
    if Path(primary_path).is_dir() and Path(fallback_path).is_dir():
        fb_index = {q.stem: q for q in Path(fallback_path).rglob("*.TextGrid")}
        for prim in Path(primary_path).rglob("*.TextGrid"):
            mate = fb_index.get(prim.stem)
            if mate is None:
                continue
            surgical_spn_replacement(prim, mate, convert_arpabet)
        return
    if Path(primary_path).is_file() and Path(fallback_path).is_file():
        pass
    else:
        raise ValueError("surgical_spn_replacement expects both paths to be files or both to be directories")
    tol = 1e-3

    def pick_tiers(tg: TextGrid):
        phones_tier = None
        words_tier = None
        for t in tg.tiers:
            if not isinstance(t, IntervalTier):
                continue
            lname = (t.name or "").lower()
            if ("phone" in lname) or ("segment" in lname):
                phones_tier = t
            if "word" in lname:
                words_tier = t
        return phones_tier, words_tier

    def find_span_indices(tier: IntervalTier, start: float, end: float):
        s = None; e = None
        for i, it in enumerate(tier):
            if it.maxTime < start - tol:
                continue
            if it.minTime > end + tol:
                break
            if s is None and abs(it.minTime - start) <= 5e-3:
                s = i
            if abs(it.maxTime - end) <= 5e-3:
                e = i
        if s is None:
            for i, it in enumerate(tier):
                if it.minTime >= start - tol:
                    s = i; break
        if e is None:
            for i in range(len(tier)-1, -1, -1):
                it = tier[i]
                if it.maxTime <= end + tol:
                    e = i; break
        return s, e

    def get_word_by_index(tier: IntervalTier, k: int):
        return tier[k] if (tier is not None and 0 <= k < len(tier)) else None

    primary_tg = TextGrid(); primary_tg.read(primary_path)
    fallback_tg = TextGrid(); fallback_tg.read(fallback_path)

    p_phones, p_words = pick_tiers(primary_tg)
    f_phones, f_words = pick_tiers(fallback_tg)

    if p_words is None or p_phones is None:
        raise ValueError("Primary TextGrid missing phones/words")

    new_tiers = []
    rewritten_phones = IntervalTier(name=p_phones.name or "phones", minTime=p_phones.minTime, maxTime=p_phones.maxTime)
    rewritten_words = IntervalTier(name=p_words.name or "words", minTime=p_words.minTime, maxTime=p_words.maxTime)

    for wi, pw in enumerate(p_words):
        p_xmin, p_xmax = pw.minTime, pw.maxTime
        p_text = (pw.mark or "").strip()

        pi_s, pi_e = find_span_indices(p_phones, p_xmin, p_xmax)

        fw = get_word_by_index(f_words, wi)
        f_xmin = fw.minTime if fw is not None else p_xmin
        f_xmax = fw.maxTime if fw is not None else p_xmax
        f_text = (fw.mark or "").strip() if fw is not None else ""

        fi_s = fi_e = None
        if f_phones is not None and fw is not None:
            fi_s, fi_e = find_span_indices(f_phones, f_xmin, f_xmax)

        if pi_s is None or pi_e is None or pi_s > pi_e:
            rewritten_words.add(p_xmin, p_xmax, p_text)
            continue

        p_first = (p_phones[pi_s].mark or "").strip().lower()
        f_first = ""
        if fi_s is not None and fi_e is not None and fi_s <= fi_e and f_phones is not None:
            f_first = (f_phones[fi_s].mark or "").strip().lower()

        use_fallback = (p_first == "spn" and f_first != "spn" and (fi_s is not None and fi_e is not None))

        if not use_fallback:
            for idx in range(pi_s, pi_e + 1):
                ph = p_phones[idx]
                rewritten_phones.add(ph.minTime, ph.maxTime, ph.mark)
            rewritten_words.add(p_xmin, p_xmax, p_text)
        else:
            n = fi_e - fi_s + 1
            if n <= 0:
                for idx in range(pi_s, pi_e + 1):
                    ph = p_phones[idx]
                    rewritten_phones.add(ph.minTime, ph.maxTime, ph.mark)
                rewritten_words.add(p_xmin, p_xmax, p_text)
            else:
                fb = []
                for idx in range(fi_s, fi_e + 1):
                    it = f_phones[idx]
                    lab = (it.mark or "").strip()
                    if convert_arpabet:
                        lab = convert_arpabet_to_ipa(lab)
                    fb.append((it.minTime, it.maxTime, lab))
                f_dur = f_xmax - f_xmin
                p_dur = p_xmax - p_xmin
                delta = f_dur - p_dur
                cur = p_xmin
                for j, (s, e, lab) in enumerate(fb):
                    d = max(0.0, e - s)
                    d_adj = d - (delta / n)
                    if j == n - 1:
                        start = cur
                        end = p_xmax
                    else:
                        d_adj = max(1e-4, d_adj)
                        start = cur
                        end = cur + d_adj
                    if start < p_xmin: start = p_xmin
                    if end > p_xmax: end = p_xmax
                    rewritten_phones.add(start, end, lab)
                    cur = end
                final_word_text = f_text if f_text else p_text
                rewritten_words.add(p_xmin, p_xmax, final_word_text)

    for t in primary_tg.tiers:
        lname = (t.name or "").lower()
        if "phone" in lname or "segment" in lname:
            continue
        if "word" in lname:
            continue
        new_tiers.append(t)

    merged = TextGrid(minTime=primary_tg.minTime, maxTime=primary_tg.maxTime)
    merged.append(rewritten_phones)
    merged.append(rewritten_words)
    for t in new_tiers:
        merged.append(t)
    merged.write(primary_path)
    return

def parse_textgrid(tg_path: Path):
    tg = TextGrid()
    tg.read(tg_path)
    words, phones = [], []
    for tier in tg:
        name = (tier.name or "").lower()
        tgt = words if "word" in name else phones if ("phone" in name or "segment" in name) else None
        if tgt is None:
            continue
        tgt += [(i.minTime, i.maxTime, i.mark) for i in tier if (i.mark or "").strip()]
    return words, phones

def detect_fail_words_from_textgrid(tg_path: Path, eps_s: float = 0.010) -> list[int]:
    """
    fail_word definition (your condition 1):
      Word k is a fail_word iff within the word interval [ws,we]:
        - there exists exactly one phone interval P with label in {'spn','sil'}
          such that |P.start-ws|<=eps and |P.end-we|<=eps
        - and there are no other phones overlapping the word by more than eps
    Returns word indices (0-based) into the words tier order.
    """
    tg = TextGrid(); tg.read(tg_path)
    phones_tier, words_tier = None, None
    for t in tg.tiers:
        if not isinstance(t, IntervalTier):
            continue
        lname = (t.name or "").lower()
        if ("phone" in lname) or ("segment" in lname):
            phones_tier = t
        if "word" in lname:
            words_tier = t
    if phones_tier is None or words_tier is None:
        return []

    def overlap(a0, a1, b0, b1) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    out: list[int] = []
    for k, w in enumerate(words_tier):
        ws, we = float(w.minTime), float(w.maxTime)
        # Candidate "single phone equals word interval"
        candidates = []
        for p in phones_tier:
            ps, pe = float(p.minTime), float(p.maxTime)
            lab = (p.mark or "").strip().lower()
            if lab in {"spn", "sil"} and abs(ps - ws) <= eps_s and abs(pe - we) <= eps_s:
                candidates.append(p)
        if len(candidates) != 1:
            continue
        # Ensure no other phone overlaps the word substantially
        ok = True
        for p in phones_tier:
            ps, pe = float(p.minTime), float(p.maxTime)
            if overlap(ws, we, ps, pe) > eps_s:
                if p is not candidates[0]:
                    ok = False
                    break
        if ok:
            out.append(k)
    return out

def patch_fail_words_from_fallback(
    primary_out_dir: Path,
    fallback_out_dir: Path,
    fail_words_by_wav: dict[str, list[int]],
    *,
    eps_s: float = 0.010,
    convert_arpabet: bool = False,
) -> None:
    """
    Implements your condition 3:
      - keep primary words tier (timestamps + labels)
      - replace only fail_words (by word index) where primary has a single spn/sil phone == word interval
      - use only fallback phones inside fallback word interval, discard fallback spn/sil/empty
      - retimestamp by preserving fallback phone duration ratios and projecting onto primary word interval
    """
    def pick_tiers(tg: TextGrid):
        phones_tier, words_tier = None, None
        for t in tg.tiers:
            if not isinstance(t, IntervalTier):
                continue
            lname = (t.name or "").lower()
            if ("phone" in lname) or ("segment" in lname):
                phones_tier = t
            if "word" in lname:
                words_tier = t
        return phones_tier, words_tier

    def merge_adjacent(segs: list[tuple[float,float,str]]) -> list[tuple[float,float,str]]:
        out: list[tuple[float,float,str]] = []
        for s,e,lab in segs:
            if e <= s + 1e-9:
                continue
            if out and out[-1][2] == lab and abs(out[-1][1] - s) <= 1e-6:
                out[-1] = (out[-1][0], e, lab)
            else:
                out.append((s,e,lab))
        return out

    def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

    fb_index = {q.stem: q for q in Path(fallback_out_dir).rglob("*.TextGrid")}
    for prim in Path(primary_out_dir).rglob("*.TextGrid"):
        mate = fb_index.get(prim.stem)
        if mate is None:
            continue
        wav_name = prim.with_suffix(".wav").name  # matches included_wavs naming convention
        fail_idxs = fail_words_by_wav.get(wav_name, [])
        if not fail_idxs:
            continue

        ptg = TextGrid(); ptg.read(prim)
        ftg = TextGrid(); ftg.read(mate)
        p_phones, p_words = pick_tiers(ptg)
        f_phones, f_words = pick_tiers(ftg)
        if p_phones is None or p_words is None or f_phones is None or f_words is None:
            continue

        # Build phone replacement plan: list of (word_start, word_end, replacement_segments)
        plan: list[tuple[float,float,list[tuple[float,float,str]]]] = []
        for k in fail_idxs:
            if not (0 <= k < len(p_words)) or not (0 <= k < len(f_words)):
                continue

            pw = p_words[k]; fw = f_words[k]
            pws, pwe = float(pw.minTime), float(pw.maxTime)
            fws, fwe = float(fw.minTime), float(fw.maxTime)

            # Verify primary really has the single spn/sil phone == word interval (within eps)
            carrier = None
            for p in p_phones:
                lab = (p.mark or "").strip().lower()
                if lab in {"spn","sil"} and abs(float(p.minTime)-pws) <= eps_s and abs(float(p.maxTime)-pwe) <= eps_s:
                    carrier = p
                    break
            if carrier is None:
                continue

            # Collect fallback phones within fallback word interval; discard spn/sil/empty
            fb = []
            for p in f_phones:
                ps, pe = float(p.minTime), float(p.maxTime)
                if pe <= fws or ps >= fwe:
                    continue
                lab = (p.mark or "").strip()
                if convert_arpabet:
                    lab = convert_arpabet_to_ipa(lab)
                lab_l = lab.strip().lower()
                if not lab_l or lab_l in {"spn","sil"}:
                    continue
                cs = clamp(ps, fws, fwe)
                ce = clamp(pe, fws, fwe)
                if ce > cs + 1e-9:
                    fb.append((cs, ce, lab))
            if not fb:
                continue

            durs = [ce - cs for cs, ce, _ in fb]
            total = sum(durs)
            if total <= 1e-9:
                continue

            # Project ratios onto primary word interval
            span = pwe - pws
            cur = pws
            repl: list[tuple[float,float,str]] = []
            for i, (_, _, lab) in enumerate(fb):
                if i == len(fb) - 1:
                    ns, ne = cur, pwe
                else:
                    frac = durs[i] / total
                    ne = cur + frac * span
                    ns = cur
                repl.append((ns, ne, lab))
                cur = ne
            repl = merge_adjacent(repl)
            plan.append((pws, pwe, repl))

        if not plan:
            continue
        plan.sort(key=lambda x: x[0])

        # Rebuild phones tier with replacements; keep all other tiers untouched
        new_phones = IntervalTier(name=p_phones.name or "phones", minTime=p_phones.minTime, maxTime=p_phones.maxTime)
        i_plan = 0
        for p in p_phones:
            ps, pe = float(p.minTime), float(p.maxTime)
            lab = (p.mark or "")
            if i_plan < len(plan):
                ws, we, repl = plan[i_plan]
                if abs(ps - ws) <= eps_s and abs(pe - we) <= eps_s and (lab.strip().lower() in {"spn","sil"}):
                    for ns, ne, nl in repl:
                        new_phones.add(ns, ne, nl)
                    i_plan += 1
                    continue
            new_phones.add(ps, pe, lab)

        # Write merged TextGrid: replace phones tier only
        merged = TextGrid(minTime=ptg.minTime, maxTime=ptg.maxTime)
        for t in ptg.tiers:
            lname = (t.name or "").lower()
            if ("phone" in lname) or ("segment" in lname):
                merged.append(new_phones)
            else:
                merged.append(t)
        merged.write(prim)

def extract_words_from_texts(text_data: dict) -> set:
    """Extract unique words from all text entries."""
    all_words = set()
    for entry in text_data.values():
        text = entry.get('text', '')
        if isinstance(text, str):
            # Basic word extraction - split on whitespace and punctuation
            words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
            all_words.update(words)
    return all_words


def detect_oov_words(words: set, dictionary: str) -> set:
    dict_path = _resolve_model_file("dictionary", dictionary)
    vocab = _load_dictionary_vocab(dict_path)
    return {w for w in words if w.lower() not in vocab}


def generate_g2p_pronunciations(oov_words: set, g2p_model: str, output_path: Path) -> bool:
    """Generate pronunciations for OOV words using G2P model."""
    if not oov_words:
        print("  [g2p] OOV count: 0 → skipping G2P generation")
        return False
        
    try:
        print(f"  [g2p] OOV count: {len(oov_words)}")
        # Create temporary file with OOV words
        oov_file = output_path / "oov_words.txt"
        with open(oov_file, 'w') as f:
            for word in sorted(oov_words):
                f.write(f"{word}\n")
        
        # Generate pronunciations using MFA G2P (sidecar CLI)
        g2p_output = output_path / "g2p_pronunciations.dict"
        mfa_cmd = resolve_mfa_bin()
        g2p_model_path = str(_resolve_model_file("g2p", g2p_model))
        result = sp.run([
            mfa_cmd, "g2p", str(oov_file), g2p_model_path, str(g2p_output)
        ], capture_output=True, text=True, check=True, env=mfa_cli_env())
        if g2p_output.exists():
            lines = g2p_output.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            print(f"  [g2p] output lines: {len(lines)}")
            return len(lines) > 0
        else:
            print("  [g2p] output file missing after g2p")
            return False
        
    except sp.CalledProcessError as e:
        print(f"Warning: G2P generation failed: {e}")
        return False

def create_enhanced_dictionary(base_dict: str, g2p_dict_path: Path, output_path: Path) -> Optional[Path]:
    """Create an enhanced dictionary: base dictionary + G2P pronunciations.

    This is *not* MFA's built-in "enhanced dictionary" logic; it's a simple merge to
    ensure only OOV words gain pronunciations while preserving the original dictionary.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    enhanced_dict = output_path / f"{base_dict.replace('/', '_')}_enhanced.dict"

    # Read G2P pronunciations (may be empty if G2P failed)
    g2p_content = ""
    if g2p_dict_path.exists():
        g2p_content = g2p_dict_path.read_text(encoding="utf-8", errors="ignore").strip()

    base_path = _resolve_model_file("dictionary", base_dict)
    base_dict_content = base_path.read_text(encoding="utf-8", errors="ignore").strip()

    # If neither base nor G2P content exists, skip
    if not (base_dict_content or g2p_content):
        print("Warning: Could not create enhanced dictionary, skipping G2P pass")
        return None

    # Write and then verify *after closing* (avoids zero-size stat on some filesystems)
    with enhanced_dict.open("w", encoding="utf-8") as out_f:
        if base_dict_content:
            out_f.write(base_dict_content)
            out_f.write("\n")
        if g2p_content:
            out_f.write(g2p_content)
            out_f.write("\n")
            print("Added G2P pronunciations to enhanced dictionary")

    if enhanced_dict.exists() and enhanced_dict.stat().st_size > 0:
        return enhanced_dict

    print("Warning: Could not create enhanced dictionary, skipping G2P pass")
    return None


def _load_texts_from_json(p: Path) -> dict[str, dict]:
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--input_json must map filename → {'text': str}")
    return data


def _load_texts_from_csv(p: Path,prefix:str) -> dict[str, dict]:
    # Row position j (0-based) → filename "{j:04d}.wav"; text from 'transcript'
    out: dict[str, dict] = {}
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "transcript" not in (reader.fieldnames or []):
            raise SystemExit("CSV must contain a 'transcript' column")
        for j, row in enumerate(reader):
            txt = (row.get("transcript") or "").strip()
            if not txt:
                continue  # skip missing/empty rows
            fn = f"{prefix}_{j:04d}.wav"
            out[fn] = {"text": txt}
    return out


def parse_model_pairs(args) -> list[tuple[str, str]]:
    """Parse model pairs from arguments, with smart defaults."""
    model_pairs = []
    
    # If explicit model pairs provided, use those
    if args.mfa_model_pairs and any(args.mfa_model_pairs):
        for pair_list in args.mfa_model_pairs:
            for pair_str in pair_list:
                if ':' in pair_str:
                    acoustic, dictionary = pair_str.split(':', 1)
                    model_pairs.append((acoustic.strip(), dictionary.strip()))
                else:
                    # If no colon, assume it's a model name to use for both
                    model_pairs.append((pair_str.strip(), pair_str.strip()))
    else:
        # Use smart defaults based on backward compatibility
        # Create matched pairs from the fallback lists
        if hasattr(args, 'mfa_acoustic_fallbacks') and hasattr(args, 'mfa_dict_fallbacks'):
            # Primary pair: first of each list
            primary_acoustic = args.mfa_acoustic_fallbacks[0]
            primary_dict = args.mfa_dict_fallbacks[0]
            model_pairs.append((primary_acoustic, primary_dict))
            
            # ARPA pair only if explicitly listed in both fallback lists
            if 'english_us_arpa' in args.mfa_acoustic_fallbacks and 'english_us_arpa' in args.mfa_dict_fallbacks:
                model_pairs.append(('english_us_arpa', 'english_us_arpa'))
            
        # Fallback to primary english_mfa only (G2P handles OOVs when requested)
        if not model_pairs:
            model_pairs = [
                ('english_mfa', 'english_mfa'),
            ]
    
    return model_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", type=Path, required=True)
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--input_json", type=Path, help="filename → {'text': str}")
    mx.add_argument("--input_csv", type=Path, help="CSV with a 'transcript' column")
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--prefix", type=str, help="audio filename prefix when csv is used")
    ap.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Optional volume gain (dB) applied when converting audio to 16k mono WAV for MFA alignment.",
    )
    ap.add_argument("-j", "--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--mfa_acoustic", default="english_mfa", 
                    help="Primary acoustic model (deprecated, use --mfa_model_pairs)")
    ap.add_argument("--mfa_dict", default="english_mfa",
                    help="Primary dictionary (deprecated, use --mfa_model_pairs)")
    ap.add_argument("--mfa_model_pairs", nargs='*', action='append',
                    help="Model pairs in format 'acoustic:dictionary' (e.g. english_mfa:english_mfa english_us_arpa:english_us_arpa)")
    ap.add_argument("--mfa_dict_fallbacks", nargs='+', 
                    default=["english_mfa"], 
                    help="Dictionary models to try in order (deprecated, use --mfa_model_pairs)")
    ap.add_argument("--mfa_acoustic_fallbacks", nargs='+',
                    default=["english_mfa"], 
                    help="Acoustic models to try in order (deprecated, use --mfa_model_pairs)")  
    ap.add_argument("--use_g2p_fallback", action='store_true',
                    help="After dictionary pass(es), run G2P only for remaining OOV/fail words")
    ap.add_argument("--g2p_model", default="english_us_mfa",
                    help="G2P model for OOV pronunciation generation (default: english_us_mfa)")
    ap.add_argument("--debug_keep_tmp", action="store_true")
    ap.add_argument("--fallback_tolerance_ms", type=float, default=50.0,
                    help = "Tolerance window around primary spn intervals for capturing fallback segments (ms)")
    ap.add_argument("--pre_g2p_output_json", type=Path,
                    help = "When --use_g2p_fallback is set, write an additional JSON just before G2P fallback to this path")
    args = ap.parse_args()
    
    # Parse model pairs
    model_pairs = parse_model_pairs(args)
    print(f"Using model pairs: {' → '.join([f'{a}+{d}' for a, d in model_pairs])}")
    if args.use_g2p_fallback and not args.pre_g2p_output_json:
        raise SystemExit("--pre_g2p_output_json is required when --use_g2p_fallback is used")


    if args.input_json:
        data = _load_texts_from_json(args.input_json)
    else:
        data = _load_texts_from_csv(args.input_csv,args.prefix)  # type: ignore[arg-type]

    if args.debug_keep_tmp:
        tmp = Path(tempfile.mkdtemp(prefix="mfa_corpus_"))
        cleanup = False
    else:
        td = tempfile.TemporaryDirectory()
        tmp = Path(td.name)
        cleanup = True

    included: list[str] = []
    for fn, val in data.items():
        src = args.audio_dir / fn
        if not src.exists():
            continue  # skip missing audio
        text = (val or {}).get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue  # skip malformed/empty entry
        dst_wav = tmp / fn
        _ffmpeg_convert(src, dst_wav, gain_db=args.gain)
        (tmp / Path(fn).with_suffix(".lab")).write_text(text.strip(), encoding="utf-8")
        included.append(fn)

    # Use incremental fallback system with model pairs for alignment
    if included:
        try:
            out_dir, alignment_stats = run_incremental_alignment_with_model_pairs(
                tmp, 
                model_pairs,
                args.threads,
                text_data=data,
                use_g2p=args.use_g2p_fallback,
                g2p_model=args.g2p_model,
                pre_g2p_json_path=args.pre_g2p_output_json,
                included_fns = included
            )
            
            # Store alignment stats for summary
            globals()['alignment_stats'] = alignment_stats
            
        except RuntimeError as e:
            print(f"Error: {e}")
            print("Alignment failed completely.")
            out_dir = tmp / "out"  # Create empty output dir
            out_dir.mkdir(exist_ok=True)
            globals()['alignment_stats'] = {'models_tried': [], 'spn_counts': [], 'improvements': []}
    else:
        out_dir = tmp / "out"
        globals()['alignment_stats'] = {'models_tried': [], 'spn_counts': [], 'improvements': []}

    # Expose tolerance to surgical replacer
    os.environ["MFA_FALLBACK_TOL"] = f"{max(0.0, (args.fallback_tolerance_ms / 1000.0)):.3f}"
    aligned = _build_aligned_dict(out_dir, included)

    args.output_json.write_text(json.dumps(aligned, indent=2, ensure_ascii=False))
    
    # Print detailed summary with alignment statistics
    # Print detailed summary with alignment statistics
    print(f"\n=== Alignment Completed Successfully! ===")
    print(f"Output written to: {args.output_json}")
    print(f"Total files processed: {len(aligned)}")
    
    # Count final spn instances in output
    total_spn_instances = 0
    for file_data in aligned.values():
        phones = file_data.get('phones', {})
        total_spn_instances += sum(1 for phone_data in phones.values() 
                                 if phone_data.get('text', '').strip().lower() == 'spn')
    
    print(f"Final 'spn' instances in output: {total_spn_instances}")
    
    # Print alignment statistics if available
    if 'alignment_stats' in globals() and globals()['alignment_stats']['models_tried']:
        stats = globals()['alignment_stats']
        print(f"\n=== Fallback Performance ===")
        for i, model in enumerate(stats['models_tried']):
            spn_count = stats['spn_counts'][i]
            if i > 0 and i-1 < len(stats['improvements']):
                improvement = stats['improvements'][i-1]
                print(f"{model}: {spn_count} spn instances (improved by {improvement})")
            else:
                print(f"{model}: {spn_count} spn instances")
                
        if len(stats['spn_counts']) > 1:
            total_improvement = stats['spn_counts'][0] - stats['spn_counts'][-1]
            reduction_pct = (total_improvement / stats['spn_counts'][0] * 100) if stats['spn_counts'][0] > 0 else 0
            print(f"Total improvement: -{total_improvement} spn instances ({reduction_pct:.1f}% reduction)")
    
    print(f"\nConfiguration used:")
    print(f"  Model pairs: {' → '.join([f'{a}+{d}' for a, d in model_pairs])}")
    if args.use_g2p_fallback:
        print(f"  G2P model: {args.g2p_model}")
    else:
        print(f"  G2P fallback: Disabled")

    if cleanup:
        shutil.rmtree(tmp, ignore_errors=True)
    elif not args.debug_keep_tmp:
        print(f"[debug] corpus kept at {tmp}")


if __name__ == "__main__":
    main()
