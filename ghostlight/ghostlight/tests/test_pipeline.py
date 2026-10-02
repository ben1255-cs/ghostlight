"""Fast end-to-end checks on a synthetic stone (no API keys needed)."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ghostlight.synth import make_stone, capture, normals_from_height, light_dir
from ghostlight.stereo import reconstruct, find_lines
from ghostlight.verify import compare, crop_line
from ghostlight.witnesses import fuse
from ghostlight.capture import Sphere, light_from_highlight
from ghostlight.agent import GhostlightAgent

LINES = ["IN LOVING MEMORY OF", "WILLIAM HARRIS", "DIED 14 MAY 1874", "AGED 37 YEARS"]


@pytest.fixture(scope="module")
def stone_and_surface():
    s = make_stone(LINES, wear=0.4, lichen=0.6, seed=3)
    imgs, lights = capture(s)
    return s, reconstruct(imgs, lights)


def test_reconstruction_recovers_normals(stone_and_surface):
    s, surf = stone_and_surface
    err = np.degrees(np.arccos(np.clip((surf.normals * normals_from_height(s.height)).sum(2), -1, 1)))
    assert err.mean() < 2.0


def test_lines_found(stone_and_surface):
    _, surf = stone_and_surface
    assert len(find_lines(surf.groove)) == len(LINES)


def test_verifier_prefers_truth(stone_and_surface):
    s, surf = stone_and_surface
    ranked = compare(crop_line(surf.groove, s.line_boxes[1]),
                     ["WILLIAM MORRIS", "WILLIAM HARRIS", "WALLACE HARRIS"])
    assert ranked[0].text == "WILLIAM HARRIS"


def test_stone_veto():
    j = fuse({"REAL NAME": 0.9, "NICE NAME": 0.7}, language={"NICE NAME": 1.0}, records={"NICE NAME": 1.0})
    assert j[0].text == "REAL NAME" and not j[1].eligible


def test_light_calibration():
    h = w = 160; cx = cy = 80; r = 50
    yy, xx = np.mgrid[:h, :w]
    nx, ny = (xx - cx) / r, (yy - cy) / r
    inside = nx ** 2 + ny ** 2 < 1
    N = np.dstack([nx, ny, np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))])
    L = light_dir(120, 25)
    Hh = (L + [0, 0, 1]) / np.linalg.norm(L + [0, 0, 1])
    img = np.where(inside, np.clip(N @ Hh, 0, 1) ** 400, 0).astype(np.float32)
    est = light_from_highlight(img, Sphere(cx, cy, r))
    assert np.degrees(np.arccos(np.clip(est @ L, -1, 1))) < 2.0


class FakeClient:
    """Scripted stand-in for Token Factory so the agent loop can be tested offline."""
    def __init__(self, script):
        self.script = list(script)
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model, messages, **kw):
        if kw.get("tools") is None:   # vision call
            return NS(choices=[NS(message=NS(content='{"text": "WILL?AM HARRIS", "unclear": "5th letter"}', tool_calls=None))])
        name, args = self.script.pop(0)
        call = NS(id=f"c{len(self.script)}", function=NS(name=name, arguments=json.dumps(args)))
        return NS(choices=[NS(message=NS(content="", tool_calls=[call]))])


def test_agent_loop(stone_and_surface):
    s, surf = stone_and_surface
    script = [
        ("look", {"line": 1, "azimuth": 200, "elevation": 18}),
        ("verify", {"line": 1, "candidates": ["WILLIAM HARRIS", "WILLAAM HARRIS"],
                    "language_scores": {"WILLIAM HARRIS": 0.95, "WILLAAM HARRIS": 0.05}}),
        ("search_records", {"query": "William Harris died 1874"}),
        ("finish", {"lines": [{"text": "WILLIAM HARRIS", "status": "confirmed", "basis": "read+language"}]}),
    ]
    tav = NS(search=lambda q, **k: {"results": [{"title": "Burial index", "url": "https://example.org", "content": "William Harris 1874"}]})
    agent = GhostlightAgent(surf, s.line_boxes, client=FakeClient(script), tavily=tav,
                            reasoning_model="r", vision_model="v")
    result = agent.run()
    kinds = [e.kind for e in agent.trace]
    assert result["lines"][0]["text"] == "WILLIAM HARRIS"
    assert {"look", "verify", "search", "finish"} <= set(kinds)
    verify_ev = next(e for e in agent.trace if e.kind == "verify")
    assert verify_ev.detail["result"]["ranking"][0]["text"] == "WILLIAM HARRIS"
