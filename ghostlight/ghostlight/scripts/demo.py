"""Run Ghostlight end to end on a synthetic worn stone.

    python scripts/demo.py                 # offline: reconstruction + verification
    python scripts/demo.py --agent         # also run the Nemotron agent (needs API keys)

Images are written to ./out so you can see the before-and-after.
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghostlight.synth import make_stone, capture, plain_photo
from ghostlight.stereo import reconstruct, relight, enhanced, find_lines, to_uint8
from ghostlight.verify import compare, crop_line

LINES = ["IN LOVING MEMORY OF", "WILLIAM HARRIS", "DIED 14 MAY 1874", "AGED 37 YEARS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wear", type=float, default=0.6)
    ap.add_argument("--lichen", type=float, default=0.6)
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    stone = make_stone(LINES, wear=a.wear, lichen=a.lichen, seed=3)
    imgs, lights = capture(stone)
    surf = reconstruct(imgs, lights)
    cv2.imwrite(f"{a.out}/1_plain_photo.png", to_uint8(plain_photo(stone)))
    cv2.imwrite(f"{a.out}/2_relit.png", to_uint8(relight(surf, 200, 18)))
    cv2.imwrite(f"{a.out}/3_enhanced.png", to_uint8(enhanced(surf)))
    print(f"Images written to {a.out}/")

    boxes = find_lines(surf.groove) or stone.line_boxes
    ranked = compare(crop_line(surf.groove, boxes[1]),
                     ["WILLIAM HARRIS", "WILLIAM MORRIS", "WILLIAM HARRIES", "WALLACE HARRIS"])
    print("\nLine 2, candidates ranked by the stone:")
    for v in ranked:
        print(f"  {v.score:.3f}  {v.text}")

    if a.agent:
        from ghostlight.agent import GhostlightAgent
        agent = GhostlightAgent(surf, boxes, on_event=lambda e: print(f"[{e.kind}]",
                                json.dumps({k: v for k, v in e.detail.items() if k != 'image'})[:300]))
        print("\nAgent result:", json.dumps(agent.run(), indent=2))


if __name__ == "__main__":
    main()
