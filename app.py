#!/usr/bin/env python3
"""Gradio front end for PhonePaint: drop a wav, pick phones, get a wav back.

Two steps, because you cannot sensibly choose phones before seeing which ones
are in the audio:

  1. Analyze  — segment, transcribe (Whisper), and align. Reports every phone
                found, and which of them this checkpoint can actually edit.
  2. Inpaint  — resume from that alignment and replace the phones you picked.

Analysis is the slow part; inpainting resumes from the same work dir, so trying
another substitution costs a few seconds rather than a full re-run.

This process does not import torch. It shells out to the pipeline using the
PhonePaint conda env, so the UI can live in a small env of its own and the
pinned inference env stays untouched. Point PHONEPAINT_PYTHON elsewhere to
override which interpreter runs the pipeline.

On a Hugging Face Space the same file runs with SPACE_ID set, which changes
three things: there is one interpreter rather than two envs, MAPS is the only
aligner (MFA needs a Kaldi conda sidecar), and the handlers are wrapped in
@spaces.GPU so ZeroGPU attaches a card for the duration of a call. The
allocation reaches the pipeline subprocess and the inference subprocess below
it — measured, not assumed — so the shell-out design carries over unchanged.

Run:
    python app.py                 # http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

IS_SPACE = bool(os.environ.get("SPACE_ID"))

if IS_SPACE:
    # Imported before anything that could touch CUDA. Off ZeroGPU this package
    # is a no-op, but it is only installed on the Space, hence the gate.
    import spaces

    def gpu(duration):
        return spaces.GPU(duration=duration)
else:
    def gpu(duration):
        return lambda fn: fn

# Reserved GPU seconds per call, as a function of clip length.
#
# Both stages are a fixed model-load cost plus a per-second-of-audio cost: on
# the live Space a 3.6 s clip takes ~18 s to analyze and ~8 s to inpaint, nearly
# all of it loading Whisper, the MAPS ensemble, and the 1.1 GB EnCodec. The
# subprocess design means that load happens on every call — there is no warm
# model to reuse — so the fixed term dominates for demo-length audio.
#
# These are declared rather than left at the default because ZeroGPU compares
# the *requested* duration against a visitor's remaining daily quota: an
# inflated number turns people away from a run that would have fit, and a small
# one also ranks higher in the queue.
# Sized from live measurements on a 3.6 s clip: analyze ~20 s, inpaint ~9 s,
# times the customary 1.4 safety factor. The earlier, more generous numbers
# were actively harmful — a 92 s reservation was refused against 64 s of
# remaining quota for a job that takes 20 s, so the visitor lost a run that
# would have fitted comfortably.
ANALYZE_FIXED = int(os.environ.get("PHONEPAINT_ANALYZE_FIXED", 28))
INPAINT_FIXED = int(os.environ.get("PHONEPAINT_INPAINT_FIXED", 16))
PER_AUDIO_SECOND = 4
MAX_RESERVATION = 240


def _clip_seconds(path) -> float:
    """Length of a wav, or 0 when it cannot be read cheaply.

    stdlib wave only, because this runs in the main web process where importing
    an audio stack (and anything that might pull in CUDA) is not welcome.
    """
    import contextlib
    import wave

    try:
        with contextlib.closing(wave.open(str(path), "rb")) as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:  # noqa: BLE001 - an unreadable file just gets the default
        return 0.0


def _reserve(fixed: int, seconds: float) -> int:
    return min(MAX_RESERVATION, fixed + int(PER_AUDIO_SECOND * seconds))


# Gradio passes progress= positionally, so estimators must swallow extras.
def _estimate_analyze(audio_path=None, *_args, **_kwargs) -> int:
    return _reserve(ANALYZE_FIXED, _clip_seconds(audio_path) if audio_path else 0.0)


def _estimate_inpaint(state=None, *_args, **_kwargs) -> int:
    source = (state or {}).get("input") if isinstance(state, dict) else None
    return _reserve(INPAINT_FIXED, _clip_seconds(source) if source else 0.0)

BOOT_PROBLEMS: list[str] = []
if IS_SPACE:
    from bootstrap import DEFAULT_ASR_MODEL, bootstrap  # noqa: E402

    ASR_MODEL = os.environ.get("PHONEPAINT_ASR_MODEL", DEFAULT_ASR_MODEL)
    BOOT_PROBLEMS = bootstrap(ASR_MODEL)
else:
    ASR_MODEL = os.environ.get("PHONEPAINT_ASR_MODEL", "")

# Pure stdlib (re/unicodedata), so it imports fine without torch.
from phone_labels import DEFAULT_TARGET_PHONES, build_label_to_phone  # noqa: E402

PIPELINE = REPO_ROOT / "inpaint_pipeline.py"
CHECKPOINT = Path(os.environ.get("PHONEPAINT_CHECKPOINT", REPO_ROOT / "weights" / "best.pt"))
MAX_PAIRS = 5  # one per colour in PAIR_COLOURS; the sixth click sends you local
EXAMPLE_WAV = REPO_ROOT / "tests" / "p225_001_mic2.wav"


def _pipeline_python() -> str:
    """Interpreter that can import torch and run the pipeline."""
    explicit = os.environ.get("PHONEPAINT_PYTHON")
    if explicit:
        return explicit
    if IS_SPACE:
        return sys.executable  # one env on a Space; nothing to hop to
    conda = Path.home() / "miniconda3" / "envs" / "PhonePaint" / "bin" / "python"
    return str(conda) if conda.is_file() else sys.executable


PYTHON = _pipeline_python()


def _supported_phones() -> list[str]:
    """Phones this checkpoint was trained on, straight from the checkpoint.

    Falls back to the training default if the checkpoint cannot be read, so the
    UI still starts and the failure surfaces on the first run instead.
    """
    if CHECKPOINT.is_file():
        code = (
            "import torch,json;"
            f"print(json.dumps(torch.load(r'{CHECKPOINT}',map_location='cpu',"
            "weights_only=False).get('phones')))"
        )
        try:
            out = subprocess.run(
                [PYTHON, "-c", code], capture_output=True, text=True, timeout=180
            )
            phones = json.loads(out.stdout.strip())
            if isinstance(phones, list) and phones:
                return phones
        except Exception:
            pass
    return list(DEFAULT_TARGET_PHONES)


SUPPORTED = _supported_phones()
LABEL_TO_PHONE = build_label_to_phone(SUPPORTED)


def _run(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    log = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, log


def _backend_flags(backend: str, ensemble: bool) -> list[str]:
    if IS_SPACE:
        # MFA needs Kaldi binaries from a conda sidecar, which a pip/apt Space
        # image cannot provide. MAPS is pure PyTorch and runs from this tree.
        backend = "MAPS"
    flags = ["--maps"] if backend == "MAPS" else ["--mfa"]
    if backend == "MAPS" and ensemble:
        flags.append("--maps-ensemble")
    return flags


def _asr_flags() -> list[str]:
    return ["--asr-model", ASR_MODEL] if ASR_MODEL else []


def _whole_file_entry(aligned: dict, stem: str) -> dict:
    """The stitched-together entry, not a per-segment one."""
    exact = aligned.get(f"{stem}.wav")
    if isinstance(exact, dict) and exact.get("phones"):
        return exact
    best: dict = {}
    for value in aligned.values():
        if isinstance(value, dict) and len(value.get("phones", {})) > len(
            best.get("phones", {})
        ):
            best = value
    return best


def _user_phone(entry: dict) -> str | None:
    return LABEL_TO_PHONE.get(entry.get("text")) or LABEL_TO_PHONE.get(
        entry.get("canonical")
    )


# MFA shells out to a conda env carrying Kaldi, OpenFst and pynini, none of
# which exist as pip wheels, so a Space image cannot offer it at all.
ALIGNERS = ["MAPS"] if IS_SPACE else ["MAPS", "MFA"]
ALIGNER_INFO = (
    "MAPS runs in-process. MFA is usually more accurate but needs a conda "
    "Kaldi sidecar, which a Space image cannot install."
    if IS_SPACE else
    "MAPS runs in-process. MFA is usually more accurate but needs the "
    "mfa_env sidecar (./create_mfa_envs.sh)."
)

REPO_URL = "https://github.com/Tom-Brenner/PhonePaint"
LIMIT_NOTICE = (
    f"That is {MAX_PAIRS} substitutions, the most this demo will take in one "
    "pass — each one costs GPU time that visitors share. To edit more phones "
    f"at once, clone [the public repo]({REPO_URL}) and run PhonePaint locally, "
    "where nothing is capped."
)

# A pair is only readable as a pair because both of its keys carry the same
# colour, so the two keyboards index into one list. Colour alone would be a
# poor carrier of meaning, hence the written list of pairs under the keys.
KEY_COLOURS = (
    ("red", "#dc2626", "#ffffff"),
    ("blue", "#2563eb", "#ffffff"),
    ("green", "#16a34a", "#ffffff"),
    ("yellow", "#eab308", "#1f2937"),
    ("purple", "#7c3aed", "#ffffff"),
)
PAIR_COLOURS = tuple(name for name, _, _ in KEY_COLOURS)
# elem_classes land on the button in current Gradio, but the descendant
# selector costs nothing and survives the class moving to the wrapper.
# opacity is pinned because an assigned key is also a disabled key, and the
# default disabled styling would wash the colour out.
KEY_CSS = "\n".join(
    f".pp-c{i}, .pp-c{i} button {{ background: {bg} !important; "
    f"color: {fg} !important; border-color: {bg} !important; "
    "opacity: 1 !important; }"
    for i, (_, bg, fg) in enumerate(KEY_COLOURS)
)
KEYS_PER_ROW = 6


def _key_rows(phones: list[str]) -> list[list[str]]:
    return [phones[i:i + KEYS_PER_ROW] for i in range(0, len(phones), KEYS_PER_ROW)]


def _blank_selection(editable=()) -> dict:
    """Selection state: completed pairs, plus the half-made one, if any.

    `editable` is what the current clip actually contains, which decides whether
    the substitution block appears at all; it does not gate individual keys.
    """
    return {"pairs": [], "pending": None, "editable": list(editable), "at_limit": False}


def _copy_selection(sel) -> dict:
    sel = dict(sel or _blank_selection())
    sel["pairs"] = [list(pair) for pair in sel.get("pairs", [])]
    return sel


def pick_source(phone, sel):
    """Open a pair. The source keyboard stays locked until a target lands."""
    sel = _copy_selection(sel)
    if len(sel["pairs"]) >= MAX_PAIRS:
        sel["at_limit"] = True
        return sel
    sel["pending"] = phone
    return sel


def pick_target(phone, sel):
    """Close the pair opened by pick_source and hand the keyboard back."""
    sel = _copy_selection(sel)
    pending = sel.get("pending")
    if not pending:
        return sel
    sel["pairs"].append([pending, phone])
    sel["pending"] = None
    return sel


def clear_selection(sel):
    """Start the substitution over without re-running alignment."""
    return _blank_selection((sel or {}).get("editable", ()))


def inpaint_ready(sel):
    """Runnable only with at least one pair and no half-made one."""
    sel = sel or {}
    return gr.update(interactive=bool(sel.get("pairs")) and not sel.get("pending"))


def _keyboard(handler, live: bool, colours: dict, allowed, sel_state):
    """One 22-key board. Keys are always drawn; only pressability changes."""
    for row in _key_rows(SUPPORTED):
        with gr.Row():
            for phone in row:
                index = colours.get(phone)
                key = gr.Button(
                    phone, size="sm", min_width=64,
                    interactive=live and (allowed is None or phone in allowed),
                    elem_classes=[f"pp-c{index}"] if index is not None else [],
                )

                def clicked(selection, _phone=phone):
                    return handler(_phone, selection)

                key.click(clicked, inputs=[sel_state], outputs=[sel_state])


def ensemble_for_backend(backend):
    """The ensemble is a MAPS-only feature, so lock the box under MFA.

    `--maps-ensemble` without `--maps` makes the CLI exit, so leaving the box
    live under MFA offers a click whose only outcome is a failed run. Selecting
    MAPS restores it, checked: the ensemble is the better default.
    """
    maps = backend == "MAPS"
    return gr.update(value=maps, interactive=maps)


def _analysis_result(summary, rows=None, state=None, editable=()):
    """Every return path of analyze(), assembled in one place.

    The handler has five exits and eight outputs; building the tuple by hand at
    each one is how an off-by-one lands in the keyboards. Clearing the player and
    the status line is part of the contract: a result from the previous clip left
    sitting there is indistinguishable from a result for this one.
    """
    editable = list(editable)
    return (
        summary,
        rows,
        state,
        _blank_selection(editable),
        gr.update(interactive=False),
        gr.update(visible=bool(editable)),
        None,
        "",
    )


def invalidate(_audio_path):
    """A new clip makes the previous alignment meaningless — drop all of it.

    Without this, Inpaint stays armed and still points at the old work dir, so
    it happily returns an edit of the file you just replaced.
    """
    return _analysis_result("")


@gpu(_estimate_analyze)
def analyze(audio_path, backend, ensemble):
    """Step 1: run the pipeline up to and including alignment."""
    if not audio_path:
        return _analysis_result("Upload or record a clip first.")
    if not CHECKPOINT.is_file():
        return _analysis_result(
            f"Checkpoint missing: {CHECKPOINT}\nRun: python tools/fetch_assets.py"
        )

    work_root = Path(tempfile.mkdtemp(prefix="phonepaint_ui_"))
    # The Audio component is set to format="wav", but keep whatever suffix we
    # are actually handed rather than mislabelling another container as wav.
    src = Path(audio_path)
    stem = src.stem
    local_in = work_root / f"{stem}{src.suffix or '.wav'}"
    shutil.copy2(src, local_in)
    work_dir = work_root / "work"
    output = work_root / f"{stem}_analyze.wav"  # never written; --to-stage align

    # phones_in/out are required by the CLI but unused before the inpaint
    # stage, so any valid pair works as a placeholder here.
    cmd = [
        PYTHON, str(PIPELINE), *_backend_flags(backend, ensemble),
        "--input", str(local_in), "--output", str(output),
        "--work-dir", str(work_dir), "--checkpoint", str(CHECKPOINT),
        "--phones_in", SUPPORTED[0], "--phones_out", SUPPORTED[0],
        "--to-stage", "align",
        *_asr_flags(),
    ]
    ok, log = _run(cmd)
    if not ok:
        return _analysis_result(f"Alignment failed.\n\n{log[-3000:]}")

    aligned_path = work_dir / "03_aligned.json"
    if not aligned_path.is_file():
        return _analysis_result(f"No alignment written.\n\n{log[-2000:]}")

    aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
    entry = _whole_file_entry(aligned, stem)

    transcript = ""
    transcripts_path = work_dir / "02_transcripts.json"
    if transcripts_path.is_file():
        data = json.loads(transcripts_path.read_text(encoding="utf-8"))
        transcript = " ".join(
            v.get("text", "") for v in data.values() if isinstance(v, dict)
        ).strip()

    rows, counts = [], {}
    for item in entry.get("phones", {}).values():
        if not isinstance(item, dict):
            continue
        user = _user_phone(item)
        rows.append([
            item.get("text", ""),
            f'{float(item.get("xmin", 0)):.2f}',
            f'{float(item.get("xmax", 0)):.2f}',
            user or "—",
        ])
        if user:
            counts[user] = counts.get(user, 0) + 1

    editable = sorted(counts)
    if editable:
        summary = (
            f'Heard: "{transcript}"\n\n'
            f"{len(rows)} phones aligned. This checkpoint can edit "
            + ", ".join(f"{p} (×{counts[p]})" for p in editable)
            + "\n\nChoose substitutions below, then Inpaint."
        )
    else:
        summary = (
            f'Heard: "{transcript}"\n\n'
            f"{len(rows)} phones aligned, but none are editable by this "
            "checkpoint, which only handles consonants:\n"
            + ", ".join(SUPPORTED)
        )

    state = {
        "work_dir": str(work_dir),
        "work_root": str(work_root),
        "input": str(local_in),
        "stem": stem,
    }
    return _analysis_result(summary, rows, state, editable)


@gpu(_estimate_inpaint)
def inpaint(state, backend, ensemble, selection):
    """Step 2: resume from the existing alignment with the chosen phone map."""
    if not state:
        return None, "Run Analyze first."

    phones_in, phones_out = [], []
    for pair in (selection or {}).get("pairs", []):
        src, dst = (list(pair) + [None, None])[:2]
        if src and dst:
            phones_in.append(src)
            phones_out.append(dst)
    if not phones_in:
        return None, (
            "Pick at least one substitution: a phone on the left keyboard, "
            "then its replacement on the right."
        )

    # A fresh path per run: a leftover file from a previous substitution must
    # never be mistaken for this one's result (and it stops the browser
    # caching the old audio in the player).
    out = Path(state["work_root"]) / f"{state['stem']}_out_{uuid.uuid4().hex[:8]}.wav"

    # --force is essential. stage_should_run() skips any stage already marked
    # done in the work dir manifest when --resume is set, so without it the
    # second substitution would skip inpaint and stitch entirely and quietly
    # return the first result. --from-stage inpaint still excludes the earlier
    # stages by ordering, so this re-runs inpaint and stitch only.
    cmd = [
        PYTHON, str(PIPELINE), *_backend_flags(backend, ensemble),
        "--input", state["input"], "--output", str(out),
        "--work-dir", state["work_dir"], "--checkpoint", str(CHECKPOINT),
        "--phones_in", *phones_in,
        "--phones_out", *phones_out,
        "--resume", "--from-stage", "inpaint", "--force",
        *_asr_flags(),
    ]
    ok, log = _run(cmd)
    if not ok:
        return None, f"Inpainting failed.\n\n{log[-3000:]}"
    if not out.is_file():
        return None, f"No output written.\n\n{log[-2000:]}"

    mapping = ", ".join(f"{a} to {b}" for a, b in zip(phones_in, phones_out))
    return str(out), f"Replaced {mapping}."


with gr.Blocks(title="PhonePaint") as demo:
    # Injected as markup rather than passed to Blocks(css=...): that argument
    # moved to launch() in Gradio 6 and is ignored there, which would silently
    # cost the keyboards their colours off the Space.
    gr.HTML(f"<style>{KEY_CSS}</style>")
    gr.Markdown(
        "# PhonePaint\n"
        "Correct mispronounced phones in a recording, or play around editing "
        "speech phone by phone. This tool is especially useful for removing "
        "lateralized pronunciation and pharyngeal compensatory effects, "
        "replacing each affected phone with itself (e.g. s --> s), see "
        "examples on "
        "https://storage.googleapis.com/phone-paint-demo/demo.html."
    )
    if BOOT_PROBLEMS:
        gr.Markdown(
            "**Startup incomplete — runs will fail until this is fixed:**\n\n"
            + "\n".join(f"- `{problem}`" for problem in BOOT_PROBLEMS)
        )
    # A hidden JSON rather than gr.State: the payload is just work-dir paths,
    # and gr.State is invisible to the HTTP API, which makes the Inpaint step
    # impossible to exercise from gradio_client — the one check that catches a
    # silently-skipped resume.
    state = gr.JSON(visible=False)
    # Same reasoning, and it is what the keyboards are drawn from.
    sel_state = gr.JSON(value=_blank_selection(), visible=False)

    gr.Markdown("## 1. Analyze")
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(
                label="Input audio", type="filepath", format="wav",
                sources=["upload", "microphone"],
            )
            # MFA is dropped from the choices on a Space rather than the whole
            # control being hidden: naming the aligner in use matters more than
            # hiding an option, and offering MFA there would only ever error.
            backend = gr.Radio(
                ALIGNERS, value="MAPS", label="Aligner", info=ALIGNER_INFO,
            )
            ensemble = gr.Checkbox(
                value=True,
                label="10-model ensemble" if IS_SPACE else
                      "10-model ensemble (MAPS only)",
                info="Median boundaries across 10 checkpoints: steadier, slower.",
            )
            analyze_btn = gr.Button("Analyze", variant="secondary")
            if EXAMPLE_WAV.is_file():
                gr.Examples(
                    examples=[[str(EXAMPLE_WAV)]],
                    inputs=[audio_in],
                    label="Example clip (VCTK p225)",
                )
        with gr.Column():
            summary = gr.Textbox(
                label="What was found", lines=8, interactive=False,
                placeholder="Analyze a clip to see which phones it contains.",
            )
            with gr.Accordion("Alignment detail", open=False):
                table = gr.Dataframe(
                    headers=["phone", "start (s)", "end (s)", "editable as"],
                    label="Aligned phones",
                    wrap=True,
                )

    # Hidden until there is an alignment to choose from: a keyboard above an
    # inert button invites clicks that can only fail.
    with gr.Column(visible=False) as subs_block:
        gr.Markdown(
            "## 2. Substitute\n"
            "Press a phone on the left, then its replacement on the right. "
            "The pair is marked in one colour, and the left keyboard is held "
            "until you have chosen the replacement."
        )

        # Redrawn from the selection rather than updated key by key: with two
        # 22-key boards, a click would otherwise have to return 44 updates in a
        # fixed order, and the colour and locking rules are simply a reading of
        # the state.
        @gr.render(inputs=[sel_state])
        def keyboards(sel):
            sel = sel or _blank_selection()
            pairs = [tuple(pair) for pair in sel.get("pairs", [])]
            pending = sel.get("pending")
            editable = set(sel.get("editable") or [])

            # First assignment wins, so a phone reused as a target keeps the
            # colour of the pair that claimed it.
            source_colours: dict = {}
            target_colours: dict = {}
            for index, (src, dst) in enumerate(pairs):
                source_colours.setdefault(src, index)
                target_colours.setdefault(dst, index)
            if pending is not None:
                source_colours.setdefault(pending, len(pairs))

            with gr.Row():
                # The whole inventory is drawn, but only phones this clip's
                # transcript actually contains can be pressed: a substitution
                # for a phone that was never spoken has nothing to act on.
                # Phones already claimed by an earlier pair lock too, since two
                # pairs cannot share a --phones_in.
                with gr.Column():
                    gr.Markdown(
                        "**Choose source phones to replace**  \n"
                        "Only phones found in this clip can be selected; the "
                        "rest are greyed out."
                    )
                    _keyboard(
                        pick_source,
                        live=pending is None,
                        colours=source_colours,
                        allowed=editable - {src for src, _ in pairs},
                        sel_state=sel_state,
                    )
                with gr.Column():
                    gr.Markdown(
                        "**Choose new (target) phones; disfluency correction "
                        "requires phones in = phones out**"
                    )
                    _keyboard(
                        pick_target,
                        live=pending is not None,
                        colours=target_colours,
                        allowed=None,
                        sel_state=sel_state,
                    )

            if pairs or pending is not None:
                lines = [
                    f"- **{src} → {dst}** ({PAIR_COLOURS[i]})"
                    for i, (src, dst) in enumerate(pairs)
                ]
                if pending is not None:
                    lines.append(
                        f"- **{pending} → ?** ({PAIR_COLOURS[len(pairs)]}) — "
                        "waiting for a target phone on the right"
                    )
                gr.Markdown("\n".join(lines))

            if sel.get("at_limit"):
                gr.Markdown(LIMIT_NOTICE)

        with gr.Row():
            inpaint_btn = gr.Button("Inpaint", variant="primary", interactive=False)
            clear_btn = gr.Button("Clear selection", variant="secondary")
        audio_out = gr.Audio(label="Result", type="filepath")
        status = gr.Textbox(label="Inpaint status", lines=4, interactive=False)

    gr.Markdown(
        "Forced alignment by [MAPS](https://github.com/deepneurallearning/MAPS). "
        "Audio codec weights are VoiceCraft's 16 kHz EnCodec, licensed for "
        "**non-commercial use only**."
    )

    analysis_outputs = [
        summary, table, state, sel_state, inpaint_btn, subs_block,
        audio_out, status,
    ]
    analyze_btn.click(
        analyze,
        inputs=[audio_in, backend, ensemble],
        outputs=analysis_outputs,
    )
    audio_in.change(invalidate, inputs=[audio_in], outputs=analysis_outputs)
    backend.change(ensemble_for_backend, inputs=[backend], outputs=[ensemble])
    # Derived here rather than returned by every key handler: the keys are
    # created inside the render block, so they cannot name a component the
    # layout defines after them.
    sel_state.change(inpaint_ready, inputs=[sel_state], outputs=[inpaint_btn])
    clear_btn.click(clear_selection, inputs=[sel_state], outputs=[sel_state])
    inpaint_btn.click(
        inpaint,
        inputs=[state, backend, ensemble, sel_state],
        outputs=[audio_out, status],
    )


if __name__ == "__main__":
    # Work dirs live under the system temp dir; without this Gradio refuses to
    # serve the finished wav back to the browser.
    default_host = "0.0.0.0" if IS_SPACE else "127.0.0.1"
    demo.launch(
        server_name=os.environ.get("PHONEPAINT_HOST", default_host),
        allowed_paths=[tempfile.gettempdir()],
    )
