"""The Ghostlight agent: an epigrapher with a torch, a dictionary and a library card.

A NVIDIA Nemotron reasoning model (served by Nebius Token Factory) runs the
investigation through four tools:

  look            relight the reconstructed stone from any angle and have a
                  vision model read one line (Witness 1: the stone)
  verify          render candidate readings as virtual carving and score every
                  letter against the real grooves; fuse with language and
                  record evidence under the stone's veto (Three Witnesses)
  search_records  search public records and archives via Tavily (Witness 3)
  finish          submit the final reading with evidence

The language witness (Witness 2) is the reasoning model itself: it proposes
candidate readings from epitaph conventions and rates their plausibility.
Every step is recorded in `trace`, which the app shows as the live reasoning trail.
"""
import base64
import json
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .stereo import Surface, relight, to_uint8
from .verify import compare, crop_line, ambiguous
from .witnesses import fuse

TOKEN_FACTORY_URL = os.environ.get("TOKEN_FACTORY_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")

SYSTEM_PROMPT = """You are Ghostlight, an expert epigrapher reading a weathered gravestone.
The stone has been reconstructed in 3D from a light sweep. You cannot see it directly;
you investigate it with tools.

Protocol (the Three Witnesses):
1. THE STONE. For each line, call `look` from several low raking angles (elevation 10-30 deg).
   Strokes perpendicular to the light show best, so vary the azimuth. Where letters are
   unclear, look again from new angles before guessing.
2. THE LANGUAGE. Propose 2-6 candidate readings per uncertain line, using epitaph
   conventions (set phrases, date and age formats, period names). Rate each candidate's
   linguistic plausibility from 0 to 1.
3. THE STONE'S VETO. Call `verify` with your candidates and plausibility scores. A reading
   whose strokes are not physically present is rejected, however plausible it sounds.
   Never override the stone.
4. THE RECORDS. When a name and date are partly read, call `search_records` to find real
   people who match (burial indexes, war memorials, newspaper notices). Pass any record
   support back into `verify` as record scores.
5. Call `finish` once each line is either confirmed or honestly marked uncertain. Use '?'
   for letters you could not establish. Distinguish what was READ from the stone from what
   was INFERRED from records. Cite sources. Never invent a person.
Be efficient: aim for at most 3 looks per line."""

TOOLS = [
    {"type": "function", "function": {
        "name": "look",
        "description": "Relight the reconstructed stone and read one text line with the vision model.",
        "parameters": {"type": "object", "properties": {
            "line": {"type": "integer", "description": "0-based line index"},
            "azimuth": {"type": "number", "description": "light direction around the stone, degrees 0-360"},
            "elevation": {"type": "number", "description": "light height above the surface, degrees 5-60"}},
            "required": ["line", "azimuth", "elevation"]}}},
    {"type": "function", "function": {
        "name": "verify",
        "description": ("Score candidate readings of one line against the physical carving, letter by letter, "
                        "and fuse with language/record evidence. The stone can veto."),
        "parameters": {"type": "object", "properties": {
            "line": {"type": "integer"},
            "candidates": {"type": "array", "items": {"type": "string"}},
            "language_scores": {"type": "object", "additionalProperties": {"type": "number"},
                                "description": "candidate -> plausibility 0..1"},
            "record_scores": {"type": "object", "additionalProperties": {"type": "number"},
                              "description": "candidate -> support from records 0..1"}},
            "required": ["line", "candidates"]}}},
    {"type": "function", "function": {
        "name": "search_records",
        "description": "Search public records, memorial databases and archives for people matching a partial reading.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Submit the final transcription.",
        "parameters": {"type": "object", "properties": {
            "lines": {"type": "array", "items": {"type": "object", "properties": {
                "text": {"type": "string"},
                "status": {"type": "string", "enum": ["confirmed", "probable", "uncertain"]},
                "basis": {"type": "string", "description": "read | read+language | read+records"}}}},
            "person": {"type": "string"},
            "summary": {"type": "string", "description": "2-4 sentence sourced summary of who this was, or why unknown"},
            "sources": {"type": "array", "items": {"type": "string"}}},
            "required": ["lines"]}}},
]


@dataclass
class Event:
    kind: str
    detail: dict
    t: float = field(default_factory=time.time)


class GhostlightAgent:
    def __init__(self, surface: Surface, line_boxes, client=None, tavily=None,
                 reasoning_model=None, vision_model=None, location_hint="", max_steps=30,
                 on_event=None):
        self.surface = surface
        self.boxes = line_boxes
        self.client = client or _default_client()
        self.tavily = tavily if tavily is not None else _default_tavily()
        self.reasoning_model = reasoning_model or os.environ.get("GHOSTLIGHT_REASONING_MODEL")
        self.vision_model = vision_model or os.environ.get("GHOSTLIGHT_VISION_MODEL")
        self.location_hint = location_hint
        self.max_steps = max_steps
        self.trace: list[Event] = []
        self.on_event = on_event
        self.result = None

    # ---- events -------------------------------------------------------------
    def _emit(self, kind, **detail):
        ev = Event(kind, detail)
        self.trace.append(ev)
        if self.on_event:
            self.on_event(ev)

    # ---- tools --------------------------------------------------------------
    def _line_image(self, line, azimuth, elevation, margin=14):
        img = relight(self.surface, azimuth, elevation)
        x0, y0, x1, y1 = self.boxes[line]
        H, W = img.shape
        crop = img[max(0, y0 - margin):min(H, y1 + margin), max(0, x0 - margin):min(W, x1 + margin)]
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".png", to_uint8(crop))
        return base64.b64encode(buf.tobytes()).decode()

    def tool_look(self, line, azimuth, elevation):
        if not 0 <= line < len(self.boxes):
            return {"error": f"line must be 0..{len(self.boxes) - 1}"}
        b64 = self._line_image(line, azimuth, elevation)
        prompt = ("This is one line of a carved inscription on a weathered gravestone, lit by a raking "
                  "light. Transcribe it exactly in capitals. Use '?' for any letter you cannot make out. "
                  'Reply with JSON only: {"text": "...", "unclear": "short note on which letters are doubtful"}')
        resp = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
            temperature=0.0, max_tokens=200)
        raw = resp.choices[0].message.content or ""
        parsed = _parse_json(raw) or {"text": raw.strip()}
        self._emit("look", line=line, azimuth=azimuth, elevation=elevation, reading=parsed, image=b64)
        return parsed

    def tool_verify(self, line, candidates, language_scores=None, record_scores=None):
        if not 0 <= line < len(self.boxes):
            return {"error": f"line must be 0..{len(self.boxes) - 1}"}
        cands = [c.upper().strip() for c in candidates if c and c.strip()][:8]
        if not cands:
            return {"error": "no candidates"}
        norm = lambda d: {k.upper().strip(): float(v) for k, v in (d or {}).items()}
        ranked = compare(crop_line(self.surface.groove, self.boxes[line]), cands)
        judged = fuse({v.text: v.score for v in ranked}, norm(language_scores), norm(record_scores))
        ties = [v.text for v in ambiguous(ranked)]
        out = {
            "ranking": [j.as_dict() for j in judged],
            "stone_cannot_separate": ties if len(ties) > 1 else [],
            "weakest_letters_of_best": [
                {"pos": i, "char": c, "support": round(s, 3)} for i, c, s in ranked[0].weakest(3)],
        }
        self._emit("verify", line=line, result=out)
        return out

    def tool_search_records(self, query):
        if self.tavily is None:
            return {"error": "records search unavailable (no TAVILY_API_KEY)"}
        q = f"{query} {self.location_hint}".strip()
        res = self.tavily.search(q, max_results=5, search_depth="advanced")
        items = [{"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:400]}
                 for r in res.get("results", [])]
        self._emit("search", query=q, results=items)
        return {"results": items}

    # ---- loop ---------------------------------------------------------------
    def run(self, user_note=""):
        n = len(self.boxes)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (f"The stone has {n} detected text lines (0..{n - 1}). "
                                         f"Location: {self.location_hint or 'unknown'}. {user_note}".strip())},
        ]
        self._emit("start", lines=n)
        for step in range(self.max_steps):
            resp = self.client.chat.completions.create(
                model=self.reasoning_model, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.2)
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None) or []
            assistant = {"role": "assistant", "content": msg.content or ""}
            if calls:
                assistant["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in calls]
            messages.append(assistant)
            if msg.content:
                self._emit("thought", text=msg.content)
            if not calls:
                messages.append({"role": "user", "content": "Continue with the tools, and call `finish` when done."})
                continue
            for c in calls:
                args = _parse_json(c.function.arguments) or {}
                name = c.function.name
                try:
                    if name == "finish":
                        self.result = args
                        self._emit("finish", result=args)
                        return args
                    fn = {"look": self.tool_look, "verify": self.tool_verify,
                          "search_records": self.tool_search_records}.get(name)
                    out = fn(**args) if fn else {"error": f"unknown tool {name}"}
                except Exception as e:  # tools must never crash the investigation
                    out = {"error": f"{type(e).__name__}: {e}"}
                messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(out)[:6000]})
        self._emit("stopped", reason="step limit")
        return self.result


def _parse_json(s):
    if not s:
        return None
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    try:
        return json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(s[a:b + 1])
            except Exception:
                return None
    return None


def _default_client():
    from openai import OpenAI
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError("Set NEBIUS_API_KEY (Nebius Token Factory) to run the agent.")
    return OpenAI(base_url=TOKEN_FACTORY_URL, api_key=key)


def _default_tavily():
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return None
    from tavily import TavilyClient
    return TavilyClient(api_key=key)
