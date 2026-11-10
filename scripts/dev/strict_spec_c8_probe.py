#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Strict-spec concurrency probe (PR #336 merge-regime replication).

12 requests: 6 response_format=json_object + 6 response_format=json_schema
(strict), interleaved, temperature 0, thinking off, max_tokens=400. Phase C1
runs them sequentially; phase C8 runs all 12 through a pool of 8. Streams each
request to split TTFT from decode; records per-request decode tok/s,
finish_reason, raw content for cross-arm byte-identity checks, and
valid-JSON/schema-key checks.

Usage: strict_spec_c8_probe.py --url http://127.0.0.1:8893 --arm on --out on.json
"""
import argparse, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "city": {"type": "string"},
        "temperature_c": {"type": "integer"},
        "conditions": {"type": "string", "enum": ["sunny", "cloudy", "rain", "snow"]},
        "wind": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "speed_kmh": {"type": "number"},
                "direction": {"type": "string"},
            },
            "required": ["speed_kmh", "direction"],
        },
    },
    "required": ["city", "temperature_c", "conditions", "wind"],
}
RF_SCHEMA = {"type": "json_schema",
             "json_schema": {"name": "weather_report", "strict": True, "schema": SCHEMA}}
RF_OBJECT = {"type": "json_object"}

CITIES = ["Paris", "Osaka", "Lima", "Oslo", "Nairobi", "Havana"]


def post_stream(url, body, headers=None):
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps({**body, "stream": True,
                         "stream_options": {"include_usage": True}}).encode(),
        headers=hdrs)
    t0 = time.time()
    ttft = None
    content_parts = []
    finish = None
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                delta = ch.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    content_parts.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    wall = time.time() - t0
    return {
        "content": "".join(content_parts),
        "finish_reason": finish,
        "wall_s": wall,
        "ttft_s": ttft if ttft is not None else wall,
        "completion_tokens": usage.get("completion_tokens", 0),
        "server_ttft_ms": usage.get("time_to_first_token_ms"),
        "server_tok_s": usage.get("response_token/s"),
    }


def one_request(url, model, idx, fmt, headers=None):
    kind, rf = fmt
    city = CITIES[idx % len(CITIES)]
    n = idx // len(CITIES) + 1
    body = {
        "model": model,
        "messages": [{"role": "user", "content":
                      f"What is the current weather in {city} (report variant {n})? "
                      "Answer with JSON only."}],
        "response_format": rf,
        "temperature": 0,
        "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = post_stream(url, body, headers)
    except Exception as e:  # noqa: BLE001
        return {"idx": idx, "kind": kind, "error": f"{type(e).__name__}: {e}"}
    decode_s = max(r["wall_s"] - r["ttft_s"], 1e-6)
    toks = r["completion_tokens"]
    try:
        obj = json.loads(r["content"])
        valid = True
        keys = (set(SCHEMA["required"]) <= set(obj.keys())) if kind == "schema" else None
    except Exception:
        valid, keys = False, False
    # Prefer the server-attested decode rate; fall back to client wall split.
    srv = r.get("server_tok_s")
    return {"idx": idx, "kind": kind, "finish_reason": r["finish_reason"],
            "wall_s": r["wall_s"], "ttft_s": r["ttft_s"], "decode_s": decode_s,
            "completion_tokens": toks,
            "decode_tok_s": srv if srv else (toks / decode_s if toks else 0.0),
            "server_ttft_ms": r.get("server_ttft_ms"),
            "valid_json": valid, "schema_keys": keys, "content": r["content"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--commit", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--token-file", default=None,
                    help="read a bearer token and send it directly (bypasses proxies)")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    headers = None
    if args.token_file:
        token = open(args.token_file).read().strip()
        headers = {"Authorization": f"Bearer {token}"}
    # interleave object/schema so each C8 batch mixes both kinds
    order = []
    for _ in range(6):
        order.append(("object", RF_OBJECT))
        order.append(("schema", RF_SCHEMA))
    rec = {"arm": args.arm, "model": args.model, "commit": args.commit,
           "note": args.note, "phases": {}}
    rec["phases"]["c1"] = [one_request(url, args.model, i, f, headers)
                           for i, f in enumerate(order)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rec["phases"]["c8"] = list(pool.map(
            lambda t: one_request(url, args.model, t[0], t[1], headers), enumerate(order)))
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    for ph, rows in rec["phases"].items():
        rates = sorted(r["decode_tok_s"] for r in rows if r.get("decode_tok_s"))
        med = rates[len(rates) // 2] if rates else 0.0
        valid = sum(1 for r in rows if r.get("valid_json"))
        keys = sum(1 for r in rows if r.get("schema_keys"))
        errs = sum(1 for r in rows if "error" in r)
        print(f"{ph}: n={len(rows)} median decode {med:.1f} tok/s "
              f"(min {rates[0]:.1f} max {rates[-1]:.1f}) valid {valid}/12 "
              f"schema_keys {keys}/6 errors {errs}")
        for r in sorted(rows, key=lambda x: x["idx"]):
            print(f"  [{r['idx']:2d}] {r['kind']:6s} "
                  f"{r.get('decode_tok_s', 0):6.1f} tok/s "
                  f"{r.get('finish_reason', 'ERR'):>6s} "
                  f"valid={r.get('valid_json')} keys={r.get('schema_keys')}")


if __name__ == "__main__":
    sys.exit(main())
