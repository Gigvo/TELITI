"""Locale layer — makes the product work in English today and Indonesian later.

## The problem this solves

The rule layer was built Indonesian-first: the risk-phrase lexicon, the UMK wage
table and the salary parser all assume Indonesian text. On an English scam ad only
2 of 9 rules fire, and the three highest-weighted ones are silent.

Meanwhile the Indonesian *evaluation* data may not materialise. The product must
therefore be fully useful with English resources alone, and gain Indonesian
capability by dropping files in — no code change, no redeploy logic.

## How it works

A locale is a bundle of data resources: a risk-phrase lexicon, a wage reference, a
currency parser and a set of language markers. Each is OPTIONAL and discovered at
load time. `LocaleRegistry.available()` reports which locales have enough resources
to be usable, so the API can advertise its real capability rather than a hoped-for one.

Rules ask the registry for resources instead of importing Indonesian files directly.
A rule whose resource is missing for the active locale reports `available=False` —
the same "not assessed" state used for redacted corpora (`api/rules/base.py`), never
"clean". That distinction is what stops a missing lexicon from silently looking like
a clean ad.

## Adding Indonesian later

Drop `data/reference/risk_phrases_id.yaml` and `umk_2025.json` into place. They are
detected on next start, `available()` begins listing `id`, and the language detector
starts routing Indonesian text to them. Nothing else changes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REFERENCE_DIR = Path("data/reference")

LOCALE_EN = "en"
LOCALE_ID = "id"

#: Used when detection is inconclusive or the detected locale has no resources.
FALLBACK_LOCALE = LOCALE_EN


@dataclass(frozen=True)
class WageReference:
    """Minimum-wage table for judging salary plausibility."""

    currency: str
    period: str  # "month" | "year"
    regions: dict[str, int]
    aliases: dict[str, str]
    fallback: int
    label: str

    def lookup(self, text: str) -> tuple[int, str]:
        """Best-guess minimum wage for the location named in `text`."""
        lowered = text.lower()
        for alias, region in sorted(self.aliases.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                value = self.regions.get(region)
                if value:
                    return value, f"{alias.title()} ({region.title()})"
        for region, value in sorted(self.regions.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(region)}\b", lowered):
                return value, region.title()
        return self.fallback, self.label


@dataclass(frozen=True)
class Locale:
    """Everything the rule layer needs to work in one language."""

    code: str
    name: str
    lexicon: dict[str, Any] | None = None
    wages: WageReference | None = None
    #: Words that indicate text is in this language, for detection.
    markers: tuple[str, ...] = ()
    #: Phrases marking a role as entry-level.
    entry_level_markers: tuple[str, ...] = ()
    #: Phrases marking a role as senior.
    senior_markers: tuple[str, ...] = ()
    #: Roles that are entry-level by nature.
    entry_level_roles: tuple[str, ...] = ()
    #: Currency symbols/codes appearing before or after an amount.
    currency_tokens: tuple[str, ...] = ()
    #: True when "." is a thousands separator and "," a decimal point (id-style).
    dot_is_thousands: bool = True
    missing: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_lexicon(self) -> bool:
        return bool(self.lexicon and self.lexicon.get("groups"))

    @property
    def has_wages(self) -> bool:
        return self.wages is not None

    @property
    def is_usable(self) -> bool:
        """A locale needs at least a lexicon to contribute anything meaningful.

        Wage data is a bonus: without it the salary rule reports itself unassessed
        rather than guessing against a foreign wage table.
        """
        return self.has_lexicon


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8")) or None


def _load_indonesian_wages(path: Path) -> WageReference | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return WageReference(
        currency="IDR",
        period="month",
        regions=raw.get("provinces", {}),
        aliases=raw.get("city_aliases", {}),
        fallback=raw.get("_meta", {}).get("national_median_fallback", 3_100_000),
        label="median nasional",
    )


def _load_english_wages(path: Path) -> WageReference | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return WageReference(
        currency=raw.get("_meta", {}).get("currency", "USD"),
        period=raw.get("_meta", {}).get("period", "month"),
        regions=raw.get("regions", {}),
        aliases=raw.get("aliases", {}),
        fallback=raw.get("_meta", {}).get("fallback", 1_300_000),
        label=raw.get("_meta", {}).get("fallback_label", "national median"),
    )


ENGLISH_MARKERS = (
    " the ", " and ", " with ", " for ", " you ", " your ", " we ", " our ",
    " will ", " must ", " have ", " this ", " that ", " are ", " is ",
    "experience", "requirements", "responsibilities", "salary", "apply",
)

INDONESIAN_MARKERS = (
    " yang ", " dan ", " untuk ", " dengan ", " dari ", " akan ", " tidak ",
    " kami ", " anda ", " di ", " ke ", " atau ", " pada ",
    "lowongan", "pengalaman", "kualifikasi", "gaji", "lamaran", "perusahaan",
    "dibutuhkan", "pelamar", "kerja",
)


@lru_cache(maxsize=1)
def load_registry(reference_dir: str = str(REFERENCE_DIR)) -> "LocaleRegistry":
    directory = Path(reference_dir)

    en_lexicon = _load_yaml(directory / "risk_phrases_en.yaml")
    id_lexicon = _load_yaml(directory / "risk_phrases_id.yaml")

    english = Locale(
        code=LOCALE_EN,
        name="English",
        lexicon=en_lexicon,
        wages=_load_english_wages(directory / "wages_en.json"),
        markers=ENGLISH_MARKERS,
        entry_level_markers=(
            "no experience", "no prior experience", "entry level", "entry-level",
            "fresh graduate", "no experience necessary", "no experience required",
            "beginners welcome", "will train", "training provided", "any age",
            "high school", "students welcome", "no skills",
        ),
        senior_markers=(
            "manager", "supervisor", "director", "head of", "senior", "lead ",
            "principal", "specialist", "engineer", "developer", "architect",
            "analyst", "consultant", "years of experience", "phd", "master's",
        ),
        entry_level_roles=(
            "data entry", "clerk", "administrative assistant", "packer", "courier",
            "typist", "receptionist", "cashier", "customer service", "warehouse",
            "sales assistant", "helper", "operator", "assistant",
        ),
        currency_tokens=("$", "usd", "us$", "£", "gbp", "€", "eur"),
        dot_is_thousands=False,  # "$5,000.50": comma thousands, dot decimal
        missing=tuple(
            n for n, present in (
                ("risk_phrases_en.yaml", en_lexicon is not None),
                ("wages_en.json", (directory / "wages_en.json").exists()),
            ) if not present
        ),
    )

    indonesian = Locale(
        code=LOCALE_ID,
        name="Bahasa Indonesia",
        lexicon=id_lexicon,
        wages=_load_indonesian_wages(directory / "umk_2025.json"),
        markers=INDONESIAN_MARKERS,
        entry_level_markers=(
            "tanpa pengalaman", "tidak perlu pengalaman", "fresh graduate",
            "fresh grad", "lulusan baru", "pemula", "semua umur", "minimal sma",
            "min sma", "min smp", "lulusan sma", "tidak ada syarat", "tanpa skill",
            "pelajar", "mahasiswa", "tanpa pengalaman kerja",
        ),
        senior_markers=(
            "manager", "manajer", "supervisor", "kepala", "direktur", "senior",
            "spesialis", "engineer", "developer", "programmer", "analis",
            "konsultan", "dokter", "apoteker", "arsitek", "akuntan", "auditor",
            "pengalaman minimal", "pengalaman min", "tahun pengalaman", "s2", "magister",
        ),
        entry_level_roles=(
            "admin online", "admin olshop", "data entry", "packing", "kurir",
            "pengetik", "penjaga", "helper", "operator", "kasir",
            "customer service", "cs online", "staff gudang", "sales promotion",
            "spg", "buruh",
        ),
        currency_tokens=("rp", "idr"),
        dot_is_thousands=True,  # "Rp9.000.000": dot thousands, comma decimal
        missing=tuple(
            n for n, present in (
                ("risk_phrases_id.yaml", id_lexicon is not None),
                ("umk_2025.json", (directory / "umk_2025.json").exists()),
            ) if not present
        ),
    )

    return LocaleRegistry({LOCALE_EN: english, LOCALE_ID: indonesian})


@dataclass(frozen=True)
class LocaleRegistry:
    locales: dict[str, Locale]

    def get(self, code: str) -> Locale:
        locale = self.locales.get(code)
        if locale is None:
            raise KeyError(f"Unknown locale {code!r}. Known: {sorted(self.locales)}")
        return locale

    def available(self) -> tuple[str, ...]:
        """Locales with enough resources to actually contribute.

        The API reports this so it advertises real capability. A locale whose
        lexicon has not been written yet must not be presented as supported.
        """
        return tuple(code for code, locale in sorted(self.locales.items()) if locale.is_usable)

    def resolve(self, text: str, requested: str | None = None) -> Locale:
        """Pick the locale to score with.

        An explicit request wins if that locale is usable. Otherwise detect, and
        fall back to English when the detected locale has no resources — scoring
        Indonesian text with English rules is degraded but honest, whereas scoring
        it with an empty rule set is silently useless.
        """
        if requested and requested in self.locales:
            locale = self.get(requested)
            if locale.is_usable:
                return locale

        detected = detect_language(text)
        locale = self.locales.get(detected)
        if locale is not None and locale.is_usable:
            return locale

        fallback = self.locales[FALLBACK_LOCALE]
        if fallback.is_usable:
            return fallback

        # Nothing is usable; return the fallback anyway so rules can report
        # themselves unavailable rather than the request failing outright.
        return fallback


def detect_language(text: str) -> str:
    """Cheap marker-count language detection.

    Deliberately not a dependency. Job ads are short and the two languages share
    almost no function words, so counting markers is accurate enough and costs
    microseconds. Ties go to English, the guaranteed-resourced locale.
    """
    padded = f" {text.lower()} "
    indonesian = sum(1 for marker in INDONESIAN_MARKERS if marker in padded)
    english = sum(1 for marker in ENGLISH_MARKERS if marker in padded)
    return LOCALE_ID if indonesian > english else LOCALE_EN
