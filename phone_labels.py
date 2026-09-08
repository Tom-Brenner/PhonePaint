"""Map user-facing phone names to alignment JSON labels."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Sequence

# User flag / CLI name → primary IPA label in aligned_json phones[].text
# (MFA english_mfa / english_us_mfa inventory).
ALIGN_LABEL_ALIASES: dict[str, str] = {
    "r": "ɹ",
    "ch": "tʃ",
    "sh": "ʃ",
    "g": "ɡ",          # hard /g/ — MFA uses U+0261 ɡ, not ASCII g
    "soft_g": "dʒ",    # soft /g/ as in "gem" (voiced palato-alveolar affricate)
}

# ARPAbet labels used by LibriSpeech TextGrids / english_us_arpa alignments.
ARPABET_LABEL_ALIASES: dict[str, str] = {
    "s": "S",
    "r": "R",
    "ch": "CH",
    "sh": "SH",
    "g": "G",
    "soft_g": "JH",
    "ʒ": "ZH",
    "z": "Z",
    "k": "K",
    "c": "K",  # MFA english_mfa /k/ allophone (e.g. "clears")
    "t": "T",
    "d": "D",
    "p": "P",
    "b": "B",
    "h": "HH",
    "θ": "TH",
    "ð": "DH",
    "f": "F",
    "v": "V",
    "n": "N",
    "l": "L",
    "m": "M",
    "w": "W",
}

# Canonical inventory shared by MFA, BFA/espeak, and PhonePaint.  Alignment
# adapters may preserve their native ``text`` while attaching this value as
# ``canonical``.  Stress digits are intentionally omitted: PhonePaint edits
# phone identities, not lexical stress.
_ARPABET_BASES = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH",
    "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N",
    "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH",
})

_IPA_TO_ARPABET: dict[str, str] = {
    # Vowels and diphthongs used by espeak's en/en-us voices.
    "ɑ": "AA", "ɑː": "AA", "ɒ": "AA", "æ": "AE", "ʌ": "AH",
    "ə": "AH", "ɔ": "AO", "ɔː": "AO", "aʊ": "AW", "aɪ": "AY",
    "ɛ": "EH", "ɝ": "ER", "ɚ": "ER", "ɜ": "ER", "ɜː": "ER",
    "eɪ": "EY", "ɪ": "IH", "i": "IY", "iː": "IY", "oʊ": "OW",
    "əʊ": "OW", "ɔɪ": "OY", "ʊ": "UH", "u": "UW", "uː": "UW",
    # Consonants.  American-English /ɾ/ is the alveolar flap produced from
    # /t/ or /d/ (for example, espeak phonemizes "butterfly" with /ɾ/), not /r/.
    "b": "B", "tʃ": "CH", "d": "D", "ð": "DH", "f": "F", "g": "G",
    "ɡ": "G", "h": "HH", "ç": "HH", "dʒ": "JH", "k": "K", "c": "K", "l": "L",
    "ɫ": "L", "ʎ": "L", "m": "M", "n": "N", "ŋ": "NG", "p": "P",
    "r": "R", "ɹ": "R", "ɻ": "R", "s": "S", "ʃ": "SH", "t": "T",
    "ɾ": "T", "ʔ": "T", "θ": "TH", "v": "V", "w": "W", "j": "Y",
    "z": "Z", "ʒ": "ZH",
    # Common syllabic allophones.
    "l̩": "L", "n̩": "N", "m̩": "M",
}

_IPA_DECORATIONS_RE = re.compile(r"[ˈˌ.'‿|]")


def canonical_phone_label(label: str) -> str:
    """Translate MFA/BFA/espeak labels to stressless canonical ARPAbet.

    Unknown labels are returned unchanged so diagnostics and conditioning do
    not silently lose information.
    """
    raw = unicodedata.normalize("NFC", str(label).strip())
    if not raw:
        return raw
    upper = re.sub(r"\d+$", "", raw.upper())
    if upper in _ARPABET_BASES:
        return upper

    # Normalize espeak decorations while retaining length and syllabic marks,
    # which disambiguate several vowels/allophones.
    ipa = _IPA_DECORATIONS_RE.sub("", raw).replace("͡", "").replace("͜", "")
    mapped = _IPA_TO_ARPABET.get(ipa)
    if mapped is not None:
        return mapped
    if ipa.endswith("ː"):
        mapped = _IPA_TO_ARPABET.get(ipa[:-1])
        if mapped is not None:
            return mapped
    return raw


def canonical_user_phone(phone: str) -> str:
    """Canonical label for a PhonePaint CLI phone name."""
    arpa = ARPABET_LABEL_ALIASES.get(phone)
    if arpa is not None:
        return arpa
    return canonical_phone_label(align_label(phone))


# MFA allophonic variants that should map to the same user phone.
ALIGN_LABEL_EXTRAS: dict[str, tuple[str, ...]] = {
    "g": ("g",),       # ASCII fallback if an aligner emits it
    "t": ("t̪", "tʰ"),
    "d": ("d̪",),
    "p": ("pʰ", "pʲ"),
    "b": ("bʲ",),
    "k": ("kʰ", "c"),
    "c": ("c", "k", "kʰ"),
    "l": ("ɫ", "ʎ"),
    "h": ("ç",),
}


def align_label(phone: str) -> str:
    return ALIGN_LABEL_ALIASES.get(phone, phone)


def alignment_labels_for_phone(phone: str) -> frozenset[str]:
    """All alignment label strings that should map to this user phone."""
    labels = {phone, align_label(phone)}
    labels.update(ALIGN_LABEL_EXTRAS.get(phone, ()))
    arpa = ARPABET_LABEL_ALIASES.get(phone)
    if arpa is not None:
        labels.add(arpa)
    labels.add(canonical_user_phone(phone))
    return frozenset(labels)


def build_label_to_phone(phones: Sequence[str]) -> Dict[str, str]:
    label_to_phone: Dict[str, str] = {}
    for ph in phones:
        for label in alignment_labels_for_phone(ph):
            label_to_phone[label] = ph
    return label_to_phone


def mask_key(phone: str) -> str:
    return f"{phone}_mask"


# Character inventory for A3T-style full-phone-string conditioning (prep + train).
_ALIGNMENT_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ɡʃɹʒθðəɚʔçɫʎt̪d̪tʰpʰkʰbʲpʲ"
    " #-'"
)
ANY_TOKEN = "<any>"


def build_alignment_char_vocab() -> tuple[list[str], dict[str, int]]:
    """Build char-level vocab for tokenized phone strings (token_list[0] = <any>)."""
    chars = sorted(set(_ALIGNMENT_CHARS), key=lambda c: (not c.isascii(), c))
    token_list = [ANY_TOKEN] + chars
    return token_list, {t: i for i, t in enumerate(token_list)}


def encode_label_to_ids(label: str, char2id: dict[str, int]) -> list[int]:
    """Tokenize one alignment phone label into char ids (unknown chars dropped)."""
    out: list[int] = []
    for ch in label.strip():
        idx = char2id.get(ch)
        if idx is not None:
            out.append(idx)
    return out


# Default phone set for the conditioned pipeline (prep_encodec_cond / train_phone_ec_cond).
DEFAULT_TARGET_PHONES: tuple[str, ...] = (
    "ch", "sh", "g", "soft_g", "s", "r", "ʒ", "z", "k",
    "t", "d", "p", "b", "h", "θ", "ð", "f", "v", "n", "l", "m", "w",
)
