"""Three Witnesses: fusing the stone, the language and the records.

The stone has a veto. A reading is only eligible if the stone physically
supports it about as well as the best reading (within `stone_tolerance`).
Language and records may only choose BETWEEN physically plausible readings;
they can never overrule the carving. This is what stops the system from
'inventing' a nice-sounding name.
"""
from dataclasses import dataclass, field
import re

MONTHS = {"JAN", "JANUARY", "FEB", "FEBRUARY", "MAR", "MARCH", "APR", "APRIL", "MAY", "JUNE", "JUN",
          "JULY", "JUL", "AUG", "AUGUST", "SEPT", "SEP", "SEPTEMBER", "OCT", "OCTOBER", "NOV",
          "NOVEMBER", "DEC", "DECEMBER"}
EPITAPH_WORDS = {"IN", "LOVING", "MEMORY", "OF", "AFFECTIONATE", "REMEMBRANCE", "SACRED", "TO", "THE",
                 "DIED", "AGED", "YEARS", "YEAR", "MONTHS", "DAYS", "WHO", "DEPARTED", "THIS", "LIFE",
                 "ALSO", "ABOVE", "HIS", "HER", "WIFE", "SON", "DAUGHTER", "HUSBAND", "BELOVED", "REST",
                 "PEACE", "AT", "ON", "AND", "WHO", "FELL", "ASLEEP", "JESUS", "THY", "WILL", "BE", "DONE",
                 "R.I.P.", "RIP", "ERECTED", "BY", "FATHER", "MOTHER", "INFANT", "LATE"}


@dataclass
class Judgement:
    text: str
    stone: float
    language: float | None = None
    records: float | None = None
    eligible: bool = True
    final: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self):
        return {"text": self.text, "stone": round(self.stone, 4), "language": self.language,
                "records": self.records, "eligible": self.eligible, "final": round(self.final, 4),
                "notes": self.notes}


def offline_language_score(text: str, known_names: set[str] | None = None) -> float:
    """A small, transparent stand-in for the Nemotron language witness, used for
    offline tests and the benchmark. The live system asks Nemotron instead."""
    words = re.findall(r"[A-Z.']+|\d+", text.upper())
    if not words:
        return 0.0
    good = 0.0
    for w in words:
        if w.isdigit():
            n = int(w)
            good += 1.0 if (len(w) == 4 and 1500 <= n <= 2030) or (len(w) <= 3 and n <= 110) else 0.3
        elif w in EPITAPH_WORDS or w in MONTHS:
            good += 1.0
        elif known_names and w in known_names:
            good += 1.0
        else:
            good += 0.35  # unknown word: possible but unproven
    return good / len(words)


def fuse(stone_scores: dict[str, float], language: dict[str, float] | None = None,
         records: dict[str, float] | None = None, stone_tolerance=0.012,
         w_lang=0.04, w_rec=0.06) -> list[Judgement]:
    best_stone = max(stone_scores.values())
    out = []
    for text, st in stone_scores.items():
        j = Judgement(text, st, (language or {}).get(text), (records or {}).get(text))
        j.eligible = best_stone - st <= stone_tolerance
        if not j.eligible:
            j.notes.append("rejected: strokes not supported by the stone")
        j.final = st + w_lang * (j.language or 0) + w_rec * (j.records or 0)
        if not j.eligible:
            j.final -= 1.0
        out.append(j)
    return sorted(out, key=lambda j: j.final, reverse=True)
