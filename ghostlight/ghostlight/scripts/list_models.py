"""List models available on your Nebius Token Factory account, NVIDIA ones first."""
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ.get("TOKEN_FACTORY_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
                api_key=os.environ["NEBIUS_API_KEY"])
ids = sorted(m.id for m in client.models.list().data)
nv = [i for i in ids if "nvidia" in i.lower() or "nemotron" in i.lower()]
print("NVIDIA / Nemotron models:")
for i in nv:
    print("  ", i)
print(f"\n{len(ids) - len(nv)} other models available.")
