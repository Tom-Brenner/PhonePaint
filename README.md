# PhonePaint

**PhonePaint** is a speech-processing pipeline for **targeted phoneme
correction** in unconstrained recordings — a disfluency / compensatory-error
corrector, not a phone-swapping or voice-conversion tool.

It uses off-the-shelf transcription (Whisper) and forced alignment — either the
[Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
(`--mfa`) or the [Mason-Alberta Phonetic Segmenter](https://github.com/MasonPhonLab/MAPS)
(`--maps`). A small (<9M)
masked model then inpaints only the selected phone frames in
[VoiceCraft EnCodec](https://github.com/jasonppy/VoiceCraft) latent space.
Surrounding frames are left unchanged (waveform copied outside the corrected
phones; frame-synchronous with the original).

**Current use case:** correction of lateralized alveolar fricatives (`s`, `z`)
and compensatory errors — pharyngeal production of postalveolar
fricatives/affricates (`sh`, `jh`, `ch`) and velar stops (`k`, `g`).

**[Hear before/after examples](https://storage.googleapis.com/phone-paint-demo/demo.html)**

Built on [A³T](https://arxiv.org/abs/2203.09690), [ESPnet](https://github.com/espnet/espnet),
[Tacotron 2](https://arxiv.org/abs/1712.05884), and
[VoiceCraft](https://arxiv.org/abs/2403.16973).

---

## Setup — run this whole block

Before you start you need: **conda**, a **CUDA 12.x driver**, and your own
PhonePaint **checkpoint** (`.pt`). Everything else is installed below.

```bash
# 1. get the code
git clone https://github.com/Tom-Brenner/PhonePaint.git
cd PhonePaint

# 2. create the PhonePaint env (Whisper + Torch MAPS + PhonePaint inpaint)
./create_phonepaint_env.sh

# 3. create the MFA sidecar env (needed for --mfa)
./create_mfa_envs.sh

# 4. download the VoiceCraft 16 kHz EnCodec weights
./tools/setup_voicecraft.sh

# 5. use it
conda activate PhonePaint
python inpaint_pipeline.py --mfa \
  --input in.wav --output out.wav \
  --checkpoint /path/to/your/checkpoint.pt \
  --phones_in s --phones_out s
```

That is the whole install. Step 5 writes the corrected audio to `out.wav` and
keeps intermediate files under `work/<input_stem>/`.

### What each step gives you

| Step | Creates | Needed for |
|------|---------|------------|
| 2 | conda env `PhonePaint` + MAPS `.pt` weights | everything |
| 3 | conda env `mfa_env` + English MFA models | `--mfa` only |
| 4 | `third_party/VoiceCraft/pretrained_models/encodec_4cb2048_giga.th` | everything |

MFA never shares an env with CUDA Torch: `--mfa` runs the `mfa_env` CLI as a
CPU subprocess while Whisper and inpainting stay on the GPU.

### Alternatives

**Use MAPS instead of MFA** — skip step 3 and swap the flag:

```bash
python inpaint_pipeline.py --maps \
  --input in.wav --output out.wav \
  --checkpoint /path/to/your/checkpoint.pt \
  --phones_in s --phones_out s
```

Upstream [MAPS](https://github.com/MasonPhonLab/MAPS) ships TensorFlow
SavedModels and wants Python 3.11. PhonePaint instead loads PyTorch ports of
those weights from [TORCH-MAPS](https://github.com/Tom-Brenner/TORCH-MAPS),
which is what lets `--maps` run in-process in the `PhonePaint` env with no
TensorFlow and no conda hop. `--maps-ensemble` uses the 10-model ensemble for
median boundaries.

**Do steps 2–4 in one command:**

```bash
./setup_from_clone.sh --mfa
```

**Build the env by hand** (only if `./create_phonepaint_env.sh` fails):

```bash
# flexible priority: the pytorch-channel CUDA build is excluded under strict
CONDA_CHANNEL_PRIORITY=flexible conda env create -n PhonePaint -f environment.yml
conda activate PhonePaint
pip install -e . --no-deps   # ESPnet from this repo, not PyPI
conda install -c pytorch -c nvidia -c conda-forge \
  'pytorch=2.5.1=py3.12_cuda12.4*' pytorch-cuda=12.4 torchaudio=2.5.1 -y
python tools/fetch_assets.py        # checkpoints; --check reports what is missing
```

**Point at an existing VoiceCraft install** — set `VOICECRAFT_ROOT` instead of
running step 4. **Point at an existing MFA install** — set `MFA_BIN`, or pass
`--env-mfa <env name>`, instead of running step 3.

---

## Run inference

Entrypoint: [`inpaint_pipeline.py`](inpaint_pipeline.py). Run
`python inpaint_pipeline.py --help` for every flag.

```bash
conda activate PhonePaint

python inpaint_pipeline.py --maps \
  --input recording.wav \
  --output corrected.wav \
  --checkpoint weights/best.pt \
  --phones_in s --phones_out s
```

`--phones_in` and `--phones_out` are matched pairs of equal length, so
`--phones_in s z --phones_out s z` corrects both `s` and `z` in one pass. Swap
`--maps` for `--mfa` to align with the `mfa_env` sidecar instead.

Defaults include `--ext 0.02` and `--align-threads segments` (MFA `-j` =
segment count). Pipeline stages: segment → Whisper → MFA|MAPS → inpaint →
stitch. Workspace: `work/<stem>/`.

---

## How it works

```
recording ──► VoiceCraft EnCodec ──────────────► z_e  [T × 128]
                                                   │
phone alignment ──► frame mask ──► zero masked frames
phone token     ──► text_embed ──► ESPnetMLMEncAsDecoderModel
                                                   │
                                             Postnet (Tacotron 2)
                                                   │
                              reconstructed z_e ──► EnCodec decoder ──► corrected speech
```

No separate vocoder; no voice cloning or full-utterance generation/conversion.
Only the targeted phoneme frames are filled in.

---

## Target phones (current use case)

| Phone | Role in correction |
|-------|-------------------|
| `s`, `z` | Lateralized alveolar fricatives |
| `sh`, `jh`, `ch` | Compensatory pharyngeal postalveolars / affricates |
| `k`, `g` | Compensatory velar stops |

A wider inventory lives in `phone_labels.py` for training and aliases; the
shipped use case is the disfluency / compensatory set above.

---

## Project layout (inference)

```
PhonePaint/
├── inpaint_pipeline.py       # production CLI
├── pipeline_scripts/         # segment / ASR / align / inpaint / stitch
├── infer_phone_ec_cond.py    # EnCodec inpaint worker
├── voicecraft_encodec.py     # VoiceCraft EnCodec loader
├── unified_speech.py         # Whisper + shared speech helpers
├── phone_labels.py
├── environment.yml           # PhonePaint conda env (Whisper + MAPS + PP)
├── create_phonepaint_env.sh  # create/update PhonePaint (+ MFA models)
├── create_mfa_envs.sh        # mfa_env sidecar (Kaldi MFA CLI) for --mfa
├── tools/setup_voicecraft.sh
├── tools/mfa/                # MFA JSON aligner
└── tools/maps/               # MAPS aligner (in-process)
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Third-party checkpoints and datasets keep their own terms (VoiceCraft 16 kHz
EnCodec is non-commercial).

## Citation

**A³T**

```bibtex
@inproceedings{bai2022a3t,
  title     = {{A$^3$T}: Alignment-Aware Acoustic and Text Pretraining for Speech Synthesis and Editing},
  author    = {Bai, He and Shi, Renjie and Lin, Junyi and Gao, Ye and Tao, Jiquan and
               Hao, Yongqiang and Zhou, Peng},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {162},
  year      = {2022},
}
```

**ESPnet**

```bibtex
@inproceedings{watanabe2018espnet,
  title     = {{ESPnet}: End-to-End Speech Processing Toolkit},
  author    = {Watanabe, Shinji and Hori, Takaaki and Karita, Shigeki and others},
  booktitle = {Interspeech},
  year      = {2018},
}
```

**Tacotron 2**

```bibtex
@inproceedings{shen2018tacotron2,
  title     = {Natural {TTS} Synthesis by Conditioning {WaveNet} on Mel Spectrogram Predictions},
  author    = {Shen, Jonathan and Pang, Ruoming and Weiss, Ron J. and Schuster, Mike and
               Jaitly, Navdeep and Yang, Zongheng and Chen, Zhifeng and Zhang, Yu and
               Wang, Yuxuan and Skerry-Ryan, R. J. and Saurous, Rif A. and
               Agiomyrgiannakis, Yannis and Wu, Yonghui},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2018},
}
```

**MAPS** (used by `--maps`)

```bibtex
@article{kelley2024maps,
  title   = {The {Mason-Alberta} Phonetic Segmenter: a forced alignment system
             based on deep neural networks and interpolation},
  author  = {Kelley, M. and Perry, S. and Tucker, B.},
  journal = {Phonetica},
  volume  = {81},
  number  = {5},
  pages   = {451--508},
  year    = {2024},
  doi     = {10.1515/phon-2024-0015},
}
```

**Montreal Forced Aligner** (used by `--mfa`)

```bibtex
@inproceedings{mcauliffe2017mfa,
  title     = {{Montreal Forced Aligner}: Trainable Text-Speech Alignment Using {Kaldi}},
  author    = {McAuliffe, Michael and Socolof, Michaela and Mihuc, Sarah and
               Wagner, Michael and Sonderegger, Morgan},
  booktitle = {Interspeech},
  pages     = {498--502},
  year      = {2017},
}
```

**VoiceCraft**

```bibtex
@article{peng2024voicecraft,
  title   = {{VoiceCraft}: Zero-Shot Speech Editing and Text-to-Speech in the Wild},
  author  = {Peng, Puyuan and Huang, Po-Yao and Li, Daniel and Mohamed, Abdelrahman and Harwath, David},
  journal = {arXiv preprint arXiv:2403.16973},
  year    = {2024},
}
```
