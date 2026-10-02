"""WornBench: does the verifier pick the true reading over near-miss distractors
as erosion increases? Every stone is synthetic, so the ground truth is known."""
import random

import numpy as np

from .synth import make_stone, capture
from .stereo import reconstruct, find_lines
from .verify import compare, crop_line
from .witnesses import fuse, offline_language_score

FIRST = ["WILLIAM", "MARY", "JOHN", "ELIZABETH", "THOMAS", "SARAH", "GEORGE", "ANN", "JAMES", "MARGARET",
         "ROBERT", "JANE", "HENRY", "ALICE", "JOSEPH", "ELLEN", "EDWARD", "HANNAH"]
LAST = ["HARRIS", "SMITH", "ROBSON", "TAYLOR", "BROWN", "WILSON", "THOMPSON", "WALKER", "WHITE", "HALL",
        "WOOD", "BELL", "CHARLTON", "DIXON", "MILBURN", "ELLIOTT", "ARMSTRONG", "HEDLEY"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUNE", "JULY", "AUG", "SEPT", "OCT", "NOV", "DEC"]
CONFUSE = {"1": "47", "4": "17", "7": "1", "3": "85", "8": "360", "6": "80", "0": "86", "5": "36", "9": "08", "2": "7",
           "I": "LT", "L": "IE", "E": "FB", "F": "EP", "H": "NM", "N": "HM", "M": "NW", "O": "CQ", "C": "OG",
           "G": "CO", "R": "BP", "B": "RE", "P": "RF", "T": "I", "A": "R", "S": "B", "W": "M", "D": "O",
           "J": "I", "Y": "V", "U": "O"}


def distractors(text, k, rng):
    """Near misses: one plausible character confused, as a human reader might."""
    idx = [i for i, c in enumerate(text) if c in CONFUSE]
    out = set()
    tries = 0
    while len(out) < k and tries < 200:
        tries += 1
        i = rng.choice(idx)
        c = rng.choice(CONFUSE[text[i]])
        alt = text[:i] + c + text[i + 1:]
        if alt != text:
            out.add(alt)
    return list(out)


def random_stone_text(rng):
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    year = rng.randint(1790, 1935)
    date = f"DIED {rng.randint(1, 28)} {rng.choice(MONTHS)} {year}"
    age = f"AGED {rng.randint(1, 95)} YEARS"
    return ["IN LOVING MEMORY OF", name, date, age]


def run(wears=(0.2, 0.4, 0.6, 0.8), n_stones=12, n_distract=4, seed=0, verbose=False):
    rng = random.Random(seed)
    results = {}
    for wear in wears:
        correct = total = fused_ok = 0
        for k in range(n_stones):
            lines = random_stone_text(rng)
            stone = make_stone(lines, wear=wear, lichen=rng.uniform(0.3, 0.8), seed=rng.randint(0, 10_000))
            imgs, lights = capture(stone, seed=k)
            surf = reconstruct(imgs, lights)
            boxes = find_lines(surf.groove)
            if len(boxes) != len(lines):
                # fall back to known geometry so the benchmark isolates the verifier
                boxes = stone.line_boxes
            for li in (1, 2, 3):
                truth = lines[li]
                cands = [truth] + distractors(truth, n_distract, rng)
                ranked = compare(crop_line(surf.groove, boxes[li]), cands)
                ok = ranked[0].text == truth
                correct += ok
                total += 1
                names = set(FIRST) | set(LAST)
                fused = fuse({v.text: v.score for v in ranked},
                             language={c: offline_language_score(c, names) for c in cands})
                fused_ok += fused[0].text == truth
                if verbose and not ok:
                    tv = next(v for v in ranked if v.text == truth)
                    w = ranked[0]
                    diff = [i for i, (a, b) in enumerate(zip(w.text, truth)) if a != b]
                    print(f"  wear {wear}: {truth!r} lost to {w.text!r} "
                          f"({w.score:.3f} vs {tv.score:.3f}; letters {[round(tv.char_scores[i],3) for i in diff]} vs "
                          f"{[round(w.char_scores[i],3) for i in diff]}; box_found={len(find_lines(surf.groove)) == len(lines)})")
        results[wear] = {"stone_only": round(correct / total, 3), "three_witness": round(fused_ok / total, 3), "n": total}
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run(verbose=True), indent=2))
