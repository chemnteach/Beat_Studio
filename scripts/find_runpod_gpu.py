"""Find available RunPod GPUs for our serverless endpoint.

Filters for >=80GB VRAM, excludes Blackwell, requires uninterruptablePrice,
and cross-references per-datacenter availability with network volume support.
"""
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.environ["RUNPOD_API_KEY"]
GRAPHQL_URL = f"https://api.runpod.io/graphql?api_key={API_KEY}"

DATACENTERS = [
    "US-KS-2", "US-CA-2", "US-GA-2", "US-MD-1", "US-MO-2",
    "US-NC-1", "US-NC-2", "US-TX-2", "US-TX-3",
    "EUR-IS-1", "EUR-NO-1", "EU-RO-1", "EU-CZ-1",
]

# Network volume NOT supported in these DCs
VOLUME_UNSUPPORTED = {"US-NC-2", "US-TX-2"}

BLACKWELL_KEYWORDS = {"blackwell", "rtx pro 6000", "rtx 5090", "b200", "b100", "b300"}

MIN_VRAM_GB = 80


def gql(query: str, variables: dict = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(GRAPHQL_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data and not data.get("data"):
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data.get("data", {})


# ── Step 1: fetch all GPU types (no lowestPrice here — needs gpuCount) ────────

GPU_LIST_QUERY = "{gpuTypes{id displayName memoryInGb}}"

print("Fetching GPU types...", flush=True)
data = gql(GPU_LIST_QUERY)
all_gpus = data["gpuTypes"]

# ── Step 2: filter by VRAM and exclude Blackwell ──────────────────────────────

candidates = []
for gpu in all_gpus:
    name_lower = (gpu.get("displayName") or gpu.get("id") or "").lower()
    vram = gpu.get("memoryInGb") or 0

    if vram < MIN_VRAM_GB:
        continue

    if any(kw in name_lower for kw in BLACKWELL_KEYWORDS):
        print(f"  skip (Blackwell/next-gen): {gpu.get('displayName')}", flush=True)
        continue

    candidates.append(gpu)

print(f"\n{len(candidates)} candidate GPU(s) >=  {MIN_VRAM_GB}GB VRAM (non-Blackwell):")
for g in candidates:
    print(f"  {g['displayName']} — {g['memoryInGb']}GB")

# ── Step 3: per-GPU per-DC query ──────────────────────────────────────────────

DC_QUERY = """
query GpuDC($gpuId: String!, $dc: String!) {
  gpuTypes(input: { id: $gpuId }) {
    id displayName memoryInGb
    lowestPrice(input: { gpuCount: 1, dataCenterId: $dc }) {
      uninterruptablePrice
      stockStatus
      minimumBidPrice
    }
  }
}
"""

total = len(candidates) * len(DATACENTERS)
print(f"\nChecking {len(candidates)} GPUs × {len(DATACENTERS)} DCs = {total} queries...", flush=True)

rows = []
n = 0
for gpu in candidates:
    gpu_id = gpu["id"]
    gpu_name = gpu["displayName"]
    vram = gpu["memoryInGb"]

    for dc in DATACENTERS:
        n += 1
        try:
            dc_data = gql(DC_QUERY, {"gpuId": gpu_id, "dc": dc})
            types = dc_data.get("gpuTypes") or []
            if not types:
                continue
            lp = types[0].get("lowestPrice") or {}
            price = lp.get("uninterruptablePrice")
            stock = lp.get("stockStatus") or "None"
            if price is None:
                continue
            volume_ok = dc not in VOLUME_UNSUPPORTED
            rows.append({
                "gpu": gpu_name,
                "vram": vram,
                "dc": dc,
                "price": price,
                "stock": stock,
                "volume": volume_ok,
            })
            print(f"  [{n}/{total}] {gpu_name} @ {dc}: ${price:.2f}/hr stock={stock}", flush=True)
        except Exception as e:
            print(f"  [{n}/{total}] warn: {gpu_name} @ {dc}: {e}", file=sys.stderr, flush=True)
        time.sleep(0.05)  # gentle rate limiting

# ── Step 4: print sorted table ────────────────────────────────────────────────

if not rows:
    print("\nNo results found with uninterruptable pricing.")
    sys.exit(1)

rows.sort(key=lambda r: (r["price"], r["gpu"], r["dc"]))

best = [r for r in rows if r["stock"] in ("High", "Medium") and r["volume"]]

print(f"\n{'GPU':<28} {'VRAM':>6} {'DC':<12} {'$/hr':>7} {'Stock':<10} {'Volume'}")
print("-" * 75)
for r in rows:
    vol_mark = "YES" if r["volume"] else "no"
    flag = " <--" if r in best[:5] else ""
    print(
        f"{r['gpu']:<28} {r['vram']:>5}G {r['dc']:<12} {r['price']:>7.2f} {r['stock']:<10} {vol_mark}{flag}"
    )

print(f"\nTotal: {len(rows)} options | In-stock + volume support: {len(best)}")
if best:
    b = best[0]
    print(f"\nTop pick: {b['gpu']} in {b['dc']} @ ${b['price']:.2f}/hr")
