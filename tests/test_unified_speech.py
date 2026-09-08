from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from phone_labels import canonical_phone_label, canonical_user_phone
from pipeline_utils import merge_segment_alignments
from pipeline_scripts.spans import collect_labeled_spans
from unified_speech import _bfa_result_to_alignment, segment_wav_directory


class PhoneInventoryTest(unittest.TestCase):
    def test_target_consonants_have_canonical_labels(self) -> None:
        expected = {
            "ch": "CH", "sh": "SH", "g": "G", "soft_g": "JH",
            "s": "S", "r": "R", "ʒ": "ZH", "z": "Z", "k": "K",
            "t": "T", "d": "D", "p": "P", "b": "B", "h": "HH",
            "θ": "TH", "ð": "DH", "f": "F", "v": "V", "n": "N",
            "l": "L", "m": "M", "w": "W",
        }
        self.assertEqual(
            {phone: canonical_user_phone(phone) for phone in expected},
            expected,
        )

    def test_bfa_ipa_normalization_preserves_contrasts(self) -> None:
        self.assertEqual(canonical_phone_label("ʃ"), "SH")
        self.assertEqual(canonical_phone_label("ʒ"), "ZH")
        self.assertEqual(canonical_phone_label("tʃ"), "CH")
        self.assertEqual(canonical_phone_label("dʒ"), "JH")
        self.assertEqual(canonical_phone_label("ɹ"), "R")
        self.assertEqual(canonical_phone_label("ɾ"), "T")

    def test_mfa_c_and_k_normalize_to_k(self) -> None:
        self.assertEqual(canonical_phone_label("c"), "K")
        self.assertEqual(canonical_phone_label("k"), "K")
        self.assertEqual(canonical_user_phone("c"), "K")
        self.assertEqual(canonical_user_phone("k"), "K")

    def test_mfa_c_and_k_match_either_cli_flag(self) -> None:
        phones = {
            "0": {"xmin": 0.1, "xmax": 0.2, "text": "c"},
            "1": {"xmin": 0.3, "xmax": 0.4, "text": "k"},
        }
        for wanted in (["k"], ["c"]):
            spans = collect_labeled_spans(phones, wanted=wanted)
            self.assertEqual(
                [(start, end, label) for start, end, label, _ in spans],
                [(0.1, 0.2, "K"), (0.3, 0.4, "K")],
            )


class BfaContractTest(unittest.TestCase):
    def test_bfa_output_converts_to_pipeline_contract(self) -> None:
        result = {
            "segments": [{
                "coverage_analysis": {"coverage_ratio": 1.0},
                "phoneme_ts": [{
                    "ipa_label": "ʒ",
                    "start_ms": 100.0,
                    "end_ms": 180.0,
                    "confidence": 0.9,
                    "is_estimated": False,
                    "target_seq_idx": 0,
                }, {
                    "ipa_label": "ɹ",
                    "start_ms": 220.0,
                    "end_ms": 260.0,
                    "confidence": 0.8,
                    "is_estimated": False,
                    "target_seq_idx": 1,
                }],
                "word_num": [0, 0],
                "words_ts": [{
                    "word": "genre",
                    "start_ms": 90.0,
                    "end_ms": 400.0,
                    "confidence": 0.8,
                }],
            }],
        }
        converted = _bfa_result_to_alignment(result, duration=0.5)
        phone = converted["phones"]["0"]
        self.assertEqual(phone["canonical"], "ZH")
        self.assertEqual((phone["xmin"], phone["xmax"]), (0.1, 0.2))
        self.assertEqual(converted["phones"]["1"]["xmin"], 0.2)
        self.assertEqual(phone["core_xmax"], 0.18)
        self.assertEqual(converted["words"]["0"]["text"], "genre")

    def test_bfa_contract_survives_merge_and_span_selection(self) -> None:
        name = "sample_segment_0.wav"
        per_segment = {
            name: {
                "words": {},
                "phones": {
                    "0": {
                        "xmin": 0.1,
                        "xmax": 0.2,
                        "core_xmin": 0.12,
                        "core_xmax": 0.18,
                        "text": "ʒ",
                        "canonical": "ZH",
                        "confidence": 0.9,
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sf.write(str(root / name), np.zeros(16_000, dtype=np.float32), 16_000)
            merged = merge_segment_alignments(per_segment, [name], root)
        phone = merged["sample.wav"]["phones"]["0"]
        self.assertEqual(phone["canonical"], "ZH")
        self.assertEqual((phone["core_xmin"], phone["core_xmax"]), (0.12, 0.18))
        spans = collect_labeled_spans({"0": phone}, wanted=["ʒ"])
        self.assertEqual(spans, [(0.1, 0.2, "ZH", False)])


class SegmentationTest(unittest.TestCase):
    def test_segmentation_preserves_full_timeline(self) -> None:
        sr = 16_000
        speech = np.full(sr * 3, 0.1, dtype=np.float32)
        silence = np.zeros(sr, dtype=np.float32)
        wav = np.concatenate([speech, silence, speech])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            output = root / "segments"
            source.mkdir()
            sf.write(str(source / "sample.wav"), wav, sr)
            names = segment_wav_directory(
                source,
                output,
                min_split_sec=0.0,
            )
            total_dur = sum(sf.info(str(output / name)).duration for name in names)
            self.assertAlmostEqual(total_dur, len(wav) / sr, places=2)
            self.assertGreaterEqual(len(names), 2)

    def test_file_between_min_split_and_max_still_splits(self) -> None:
        """Applio always adds a tail cut for files ≥ min_split_sec (a3t behavior)."""
        sr = 16_000
        speech = np.full(sr * 3, 0.1, dtype=np.float32)
        silence = np.zeros(int(sr * 0.8), dtype=np.float32)
        wav = np.concatenate([speech, silence, speech])  # 6.8 s
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            output = root / "segments"
            source.mkdir()
            sf.write(str(source / "sample.wav"), wav, sr)
            names = segment_wav_directory(
                source,
                output,
                min_split_sec=6.0,
            )
            self.assertGreaterEqual(len(names), 2)
            total_dur = sum(sf.info(str(output / name)).duration for name in names)
            self.assertAlmostEqual(total_dur, len(wav) / sr, places=2)

    def test_short_file_stays_one_segment(self) -> None:
        sr = 16_000
        wav = np.full(sr * 4, 0.1, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            output = root / "segments"
            source.mkdir()
            sf.write(str(source / "sample.wav"), wav, sr)
            names = segment_wav_directory(source, output, min_split_sec=6.0)
            self.assertEqual(names, ["sample_segment_0.wav"])

    def test_nikki12_matches_a3t_applio_cuts(self) -> None:
        """Regression: Nikki12 must split like ~/a3t (0.112s + 7.088s), not one chunk."""
        src = Path("/home/tom/Dropbox/datasets/Nikki12_segment_1_org.wav")
        if not src.is_file():
            self.skipTest(f"missing {src}")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            output = root / "segments"
            source.mkdir()
            import shutil
            shutil.copy2(src, source / src.name)
            names = segment_wav_directory(source, output, min_split_sec=6.0)
            durs = [sf.info(str(output / n)).duration for n in names]
            self.assertEqual(len(names), 2)
            self.assertAlmostEqual(durs[0], 0.112, places=3)
            self.assertAlmostEqual(durs[1], 7.088, places=3)


if __name__ == "__main__":
    unittest.main()
