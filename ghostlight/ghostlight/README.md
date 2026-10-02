# Ghostlight: the AI that reads gravestones time has erased

Weather is wiping names off gravestones. Ghostlight rebuilds worn stone in 3D, and a **NVIDIA Nemotron** agent on **Nebius Token Factory** reads it, checking every guess against the carving and real records (via **Tavily**). Nothing invented.

## How NVIDIA and Nebius are used

| Component | Technology | Role |
|---|---|---|
| Reasoning agent | NVIDIA Nemotron on Nebius Token Factory | Runs the investigation, proposes readings, weighs evidence |
| Vision reader | NVIDIA Nemotron vision model on Nebius Token Factory | Reads each relit line of the stone |
| Reconstruction | PyTorch + CUDA on Nebius AI Cloud GPUs | Batched photometric stereo over every pixel |
| Records witness | Tavily | Searches archives and memorial databases |

### Where Token Factory accelerated our workflow

- **No model hosting to manage.** Token Factory's OpenAI-compatible endpoint (`https://api.tokenfactory.nebius.com/v1/`) let us call Nemotron with the standard `openai` Python client, so the agent's tool-calling loop needed no custom serving code.
- **Swapping models in one line.** Model IDs live in environment variables, so we can compare Nemotron reasoning and vision models (`scripts/list_models.py` lists what's available) without changing code.
- **Tool calling out of the box.** The agent's four tools (`look`, `verify`, `search_records`, `finish`) use standard function calling served by Token Factory.

### Other Nebius services

- **Nebius AI Cloud GPUs** run the photometric stereo reconstruction: one batched least-squares solve over every pixel in PyTorch on CUDA (`ghostlight/stereo.py`), with an automatic NumPy fallback on machines without a GPU.

## The Three Witnesses

1. **The stone**: the agent relights the 3D stone from any angle and reads it, looking again where letters are unclear.
2. **The language**: Nemotron proposes candidate readings from epitaph conventions.
3. **The records**: Tavily finds real people matching partial readings.

Every candidate is rendered as virtual carving and checked **letter by letter** against the real grooves (analysis by synthesis). The stone has a veto: language and records can choose between physically plausible readings, but never overrule the carving.

## Live demo

https://ghostlight-stones.netlify.app

## Quick start

```bash
pip install -r requirements.txt
python scripts/demo.py            # offline: synthetic stone, before/after images in ./out
python -m pytest -q tests         # 6 tests, no API keys needed
python scripts/run_bench.py       # WornBench
```

To run the live agent:

```bash
cp .env.example .env              # add your keys, then export them
python scripts/list_models.py     # find the exact Nemotron model IDs
python scripts/demo.py --agent
```

## WornBench (early results)

Each true reading competes against four near-miss alternatives (e.g. "DIED" vs "DIFD"). Chance is 20%.

| Wear | Stone alone | Three Witnesses |
|---|---|---|
| Light | 93% | 100% |
| Medium | 73% | 87% |
| Heavy | 47% | 87% |

15 lines per wear level on synthetic stones; the language witness in this offline benchmark is a simple word list standing in for Nemotron.

## Repository layout

| File | Purpose |
|---|---|
| `ghostlight/capture.py` | Phone video to calibrated frames (mirror-ball light estimation, stabilisation) |
| `ghostlight/stereo.py` | Photometric stereo, height integration, virtual relighting, line detection |
| `ghostlight/verify.py` | Analysis-by-synthesis letter verification |
| `ghostlight/witnesses.py` | Three Witnesses fusion with the stone's veto |
| `ghostlight/agent.py` | Nemotron tool-calling agent with live reasoning trace |
| `ghostlight/synth.py` | Synthetic worn stones with ground truth |
| `ghostlight/bench.py` | WornBench |

## Capturing a real stone

Prop your phone on a small tripod facing the stone. Put a shiny ball (ball bearing or glossy marble) in the corner of the frame. Record about 15 seconds while sweeping a torch slowly around the stone at low angles. Shade or overcast light works best.

## Licence

MIT
