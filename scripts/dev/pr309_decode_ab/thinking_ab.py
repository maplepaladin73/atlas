#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""GLM code-decode A/B that splits reasoning from answer.

One invocation measures one booted server (one arm). It streams each request,
timestamps every chunk, and splits the output into the reasoning phase
(`delta.reasoning_content`) and the answer phase (`delta.content`), using the
server's usage (`completion_tokens_details.reasoning_tokens`,
`accepted_prediction_tokens`) for token counts.

Variants (same prompt, same sampling, only the thinking fields differ):
  default    no thinking fields, max_tokens 512 (the original code cell)
  matched    thinking on, reasoning_effort low, thinking_token_budget 256,
             max_tokens 2048 (same settings on every image)
  max_nobud  thinking on, no effort, no budget, max_tokens 2048
             (PR 309's default behaviour, without the 512 cap)
  sq_shape   thinking on, reasoning_effort low, max_tokens 384
             (the shape of upstream's sq_bench decode cell on #338/#346)

Usage:
  thinking_ab.py --url http://127.0.0.1:8888 --tag pr-tip --reps 3 \
      --variants default,matched,max_nobud,sq_shape --out results.jsonl
  thinking_ab.py --summarize results.jsonl
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

CODE_PROMPT = (
    "Write a complete Python implementation of a MinHeap class with push, pop, "
    "peek, heapify and __len__, plus unit tests using unittest. Return only code."
)

VARIANTS = {
    "default": {"max_tokens": 512},
    "matched": {
        "max_tokens": 2048,
        "reasoning_effort": "low",
        "thinking_token_budget": 256,
        "chat_template_kwargs": {"enable_thinking": True},
    },
    "max_nobud": {
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": True},
    },
    "sq_shape": {
        "max_tokens": 384,
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": True},
    },
}


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def rate(tokens, t0, t1):
    if tokens is None or tokens < 2 or t0 is None or t1 is None or t1 <= t0:
        return None
    return (tokens - 1) / (t1 - t0)


def run_once(url, model, prompt, variant, rep, temperature):
    body = {
        "model": model,
        # Unique prefix per rep so no prefix-cache hit can shorten prefill.
        "messages": [{"role": "user", "content": f"[rep {rep}] {prompt}"}],
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(VARIANTS[variant])
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t_send = time.perf_counter()
    first = {"reasoning": None, "content": None}
    last = {"reasoning": None, "content": None}
    chunks = {"reasoning": 0, "content": 0}
    chars = {"reasoning": 0, "content": 0}
    usage, finish = None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            now = time.perf_counter()
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                delta = ch.get("delta") or {}
                for phase, key in (("reasoning", "reasoning_content"), ("content", "content")):
                    text = delta.get(key)
                    if text:
                        if first[phase] is None:
                            first[phase] = now
                        last[phase] = now
                        chunks[phase] += 1
                        chars[phase] += len(text)
    t_end = time.perf_counter()
    usage = usage or {}
    details = usage.get("completion_tokens_details") or {}
    completion = usage.get("completion_tokens")
    reasoning = details.get("reasoning_tokens")
    answer = completion - reasoning if completion is not None and reasoning is not None else None
    starts = [t for t in first.values() if t is not None]
    ends = [t for t in last.values() if t is not None]
    t_first = min(starts) if starts else None
    t_last = max(ends) if ends else None
    return {
        "variant": variant,
        "rep": rep,
        "request": {k: v for k, v in body.items() if k != "messages"},
        "finish_reason": finish,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "answer_tokens": answer,
        "accepted_prediction_tokens": details.get("accepted_prediction_tokens"),
        "server_tps": usage.get("response_tokens_per_second"),
        "server_ttft_ms": usage.get("time_to_first_token_ms"),
        "ttft_s": (t_first - t_send) if t_first else None,
        "wall_s": t_end - t_send,
        "reasoning_s": (last["reasoning"] - first["reasoning"]) if first["reasoning"] else None,
        "answer_s": (last["content"] - first["content"]) if first["content"] else None,
        "chunks": chunks,
        "chars": chars,
        "decode_tps": rate(completion, t_first, t_last),
        "reasoning_tps": rate(reasoning, first["reasoning"], last["reasoning"]),
        "answer_tps": rate(answer, first["content"], last["content"]),
    }


def fmt(x, spec=".2f"):
    return "-" if x is None else format(x, spec)


def med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def summarize(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("kind") == "rep"]
    groups = {}
    for r in rows:
        groups.setdefault((r["tag"], r["variant"]), []).append(r)
    cols = ["decode_tps", "answer_tps", "reasoning_tps", "reasoning_tokens",
            "answer_tokens", "accepted_prediction_tokens", "ttft_s"]
    print("| tag | variant | n | " + " | ".join(cols) + " | finish |")
    print("|" + "---|" * (len(cols) + 4))
    for (tag, variant), rs in sorted(groups.items()):
        fins = ",".join(sorted({str(r["finish_reason"]) for r in rs}))
        cells = [fmt(med(rs, c), ".2f" if c.endswith(("tps", "_s")) else ".0f") for c in cols]
        print(f"| {tag} | {variant} | {len(rs)} | " + " | ".join(cells) + f" | {fins} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--tag", help="arm label, e.g. old-3439912 or pr-354da2156")
    ap.add_argument("--model", help="model name; default: first id from /v1/models")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variants", default="default,matched,max_nobud,sq_shape")
    ap.add_argument("--prompt-file", help="use this prompt instead of the built-in code prompt")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=1, help="unrecorded warmup requests")
    ap.add_argument("--note", default="", help="free-text fingerprint (image digest, env file, flags)")
    ap.add_argument("--out", default="thinking_ab.jsonl")
    ap.add_argument("--summarize", metavar="JSONL", help="print the median table and exit")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
        return
    if not a.tag:
        ap.error("--tag is required")
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            ap.error(f"unknown variant {v!r}; choose from {', '.join(VARIANTS)}")
    prompt = open(a.prompt_file).read().strip() if a.prompt_file else CODE_PROMPT
    models = get_json(f"{a.url}/v1/models")
    model = a.model or models["data"][0]["id"]
    with open(a.out, "a") as f:
        f.write(json.dumps({"kind": "fingerprint", "tag": a.tag, "time": time.time(),
                            "url": a.url, "model": model, "models": models, "note": a.note,
                            "prompt": prompt, "variants": {v: VARIANTS[v] for v in variants},
                            "temperature": a.temperature}) + "\n")
        for w in range(a.warmup):
            run_once(a.url, model, prompt, "matched", f"warmup{w}", a.temperature)
        # Interleave variants within each rep so drift hits every variant alike.
        for rep in range(a.reps):
            for v in variants:
                rec = run_once(a.url, model, prompt, v, rep, a.temperature)
                rec.update({"kind": "rep", "tag": a.tag})
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[{a.tag}] {v} rep{rep}: decode {fmt(rec['decode_tps'])} "
                      f"answer {fmt(rec['answer_tps'])} reasoning {fmt(rec['reasoning_tps'])} tok/s; "
                      f"reasoning/answer tokens {rec['reasoning_tokens']}/{rec['answer_tokens']}; "
                      f"accepted {rec['accepted_prediction_tokens']}; finish {rec['finish_reason']}",
                      flush=True)
    summarize(a.out)


if __name__ == "__main__":
    sys.exit(main())
