# RESULTS — PR #336 strict-spec, three-arm hardware A/B (batched verify)

Date: 2026-11-10. Fork-only validation; nothing here is posted upstream.

## Build under test

- Code: `devin/strict-spec-notes` head `f62215013`, which contains PR #336
  (`ATLAS_STRICT_SPEC=1` keeps strict structured requests speculative when the
  verify path writes full logits) plus the batched-verify follow-up
  (`339770813`: strict sequences share the batched MTP verify).
- Checkpoint: `nvidia--Qwen3.8-Flash-Next-FP4-Local` (FP4-Local, TP=2/EP=2).
- Serve profile: production profile — 8 lanes × 262K context,
  `gpu-memory-utilization=0.80`, `--oom-guard-mb=4096`, KV bf16,
  prefix caching, `--max-prefill-tokens=16384`, `--speculative --num-drafts=7`,
  `--max-num-seqs=8 --max-batch-size=8 --max-seq-len=262144`,
  `reasoning_effort=low` chat-template default, `--disable-tool-grammar=true`,
  Bearer auth. `ATLAS_MTP_ACCEPT_DEBUG=1` in every arm.
- Harness: `scripts/dev/strict_spec_ab.py`, 3 reps per case,
  `max_tokens=800`, temperature 0. Raw JSON records and full server logs are
  retained privately; only aggregates appear here.

## Arms (fresh boot of both ranks per arm, pair idle during measurement)

| arm | env delta vs production profile |
|---|---|
| `off` | `ATLAS_STRICT_SPEC` unset |
| `serial` | `ATLAS_STRICT_SPEC=1`, `ATLAS_GRAMMAR_SERIAL_VERIFY=1` (PR #336 only, pre-batch behavior) |
| `on` | `ATLAS_STRICT_SPEC=1` (batched strict verify) |

## §3a baseline

The historical ~86–89 tok/s plain figure does not reproduce on this build or
the previous production image under the short-prompt method: both measure
~50–55 tok/s plain. A separate ~800–1024-token counting-style prompt measures
~90–108 tok/s with much higher `tok_step`, so the earlier number is
prompt/output-acceptance sensitive, not an image regression. The plain control
inside this A/B is identical across arms (see below), which is what matters
for attribution.

## Noise floor: `off` vs `off2` (same build, same profile)

| case | off tok/s | off2 tok/s | ratio |
|---|---|---|---|
| schema@c1 | 36.5 | 36.5 | 1.00 |
| schema@c2 | 57.0 | 55.9 | 0.98 |
| schema@c4 | 90.0 | 89.6 | 1.00 |
| schema@c8 | 134.5 | 133.6 | 0.99 |
| object@c1 | 36.3 | 36.4 | 1.00 |
| plain@c1 | 50.0 | 50.0 | 1.00 |
| sampled@c1 | 35.7 | 35.7 | 1.00 |
| sampled@c4 | 87.1 | 85.4 | 0.98 |
| think@c1 | 36.2 | 36.3 | 1.00 |
| logprobs@c1 | 35.5 | 35.6 | 1.00 |
| mixed@c4 | 78.9 | 78.5 | 0.99 |

Noise floor ≈ ±2%. Temperature-0 outputs are byte-identical between the two
records in every deterministic case.

## Headline: `on` vs `serial` (batched-verify effect)

| case | serial tok/s | on tok/s | on/serial |
|---|---|---|---|
| schema@c1 | 68.5 | 70.8 | 1.03 |
| schema@c2 | 69.2 | 101.3 | 1.46 |
| schema@c4 | 67.6 | 149.0 | 2.20 |
| schema@c8 | 66.7 | 207.4 | 3.11 |
| object@c1 | 64.5 | 68.9 | 1.07 |
| plain@c1 | 49.8 | 50.0 | 1.00 |
| sampled@c1 | 61.3 | 63.0 | 1.03 |
| sampled@c4 | 58.6 | 124.6 | 2.13 |
| think@c1 | 62.8 | 64.4 | 1.03 |
| logprobs@c1 | 62.8 | 64.0 | 1.02 |
| mixed@c4 | 83.1 | 124.0 | 1.49 |

## `serial` vs `off` (PR #336-only effect)

| case | off tok/s | serial tok/s | serial/off |
|---|---|---|---|
| schema@c1 | 36.5 | 68.5 | 1.87 |
| schema@c2 | 57.0 | 69.2 | 1.21 |
| schema@c4 | 90.0 | 67.6 | 0.75 |
| schema@c8 | 134.5 | 66.7 | 0.50 |
| object@c1 | 36.3 | 64.5 | 1.78 |
| plain@c1 | 50.0 | 49.8 | 1.00 |
| sampled@c1 | 35.7 | 61.3 | 1.72 |
| sampled@c4 | 87.1 | 58.6 | 0.67 |
| think@c1 | 36.2 | 62.8 | 1.73 |
| logprobs@c1 | 35.5 | 62.8 | 1.77 |
| mixed@c4 | 78.9 | 83.1 | 1.05 |

PR #336 alone helps single-stream structured decode (~1.8×) but loses badly
at concurrency ≥4 because every strict sequence occupies its own verify
forward. The batched follow-up removes that cliff: `on` beats `off` at every
concurrency level and beats `serial` everywhere except plain@c1 (noise).

## `on` vs `off` (total effect)

| case | off tok/s | on tok/s | on/off |
|---|---|---|---|
| schema@c1 | 36.5 | 70.8 | 1.94 |
| schema@c2 | 57.0 | 101.3 | 1.78 |
| schema@c4 | 90.0 | 149.0 | 1.66 |
| schema@c8 | 134.5 | 207.4 | 1.54 |
| object@c1 | 36.3 | 68.9 | 1.90 |
| plain@c1 | 50.0 | 50.0 | 1.00 |
| sampled@c1 | 35.7 | 63.0 | 1.77 |
| sampled@c4 | 87.1 | 124.6 | 1.43 |
| think@c1 | 36.2 | 64.4 | 1.78 |
| logprobs@c1 | 35.5 | 64.0 | 1.80 |
| mixed@c4 | 78.9 | 124.0 | 1.57 |

## Correctness / validity

- **Outputs identical across arms.** Every deterministic case is byte-identical
  between off, serial, and on (3/3, 6/6, 12/12, 24/24 per concurrency level).
  No arm-only divergence observed; consistent with no grammar-matcher
  desynchronization.
- **Truncated-JSON caveat.** The harness's 800-token budget truncates the
  schema/sampled/think/logprobs/mixed structured replies
  (`finish_reason=length`), so `compare` reports "invalid" counts. The counts
  and the failure positions are *identical in all three arms* — the invalidity
  is workload truncation, not arm-specific. The strict "zero invalid"
  criterion is therefore not literally met by this run; it needs either a
  larger `max_tokens` or a shorter schema workload to clear. The independent
  format probe (below) does produce complete valid JSON.
- **Format probe.** `tools_rfmt_probe.py --runs 5` and `--runs 5 --thinking`:
  5/5 schema-valid final answers in all three arms, exit 0.
  - Probe fix required: with `--disable-tool-grammar=true`, a request carrying
    both `tools` and `response_format` takes the documented escape-hatch path
    and leaves the final answer unconstrained, so the probe's turn-2 answer
    was prose. Sending `tool_choice: "none"` on the final turn pins the
    response_format grammar (OpenAI-conformant agentic pattern) and the probe
    passes. This is a harness fix, committed on this branch; the server
    behavior is pre-existing and unrelated to strict-spec.
- **Request errors:** 0 in all arms.

## Engagement (server-side, `ATLAS_MTP_ACCEPT_DEBUG=1`)

- `off`: structured `Done:` lines show `serial=1.00 mtp=0.00 tok_step=1.000`;
  plain shows `mtp=1.00 tok_step≈2.1`. Lever correctly gates speculation off.
- `serial`: all requests `mtp=1.00`, `tok_step≈2.8`; `MTP accept` batches are
  almost entirely `n=1` (156) with a few `n=3` — strict sequences verify alone.
- `on`: `mtp=1.00`, `tok_step` 3.0–3.6 on structured streams; `MTP accept`
  batch sizes reach `n=8` (45 samples) and `n=4` (69 samples) — strict
  sequences share the batched verify as intended.
- Plain requests speculate in all arms (`mtp=1.00`).

## Thermals / throttling (§3b)

- Cap in force on both nodes: `gpu-clock-cap.service` (systemd oneshot,
  enabled, `RemainAfterExit=yes`) runs `nvidia-smi -lgc 300,2200` — a locked
  clock *range*, max 2200 MHz. Active since 2026-10-14 on both nodes, i.e.
  identical during the historical morning measurement; it survives reboot.
  Not a power limit (`power.limit` reports N/A on GB10).
- Per-arm `nvidia-smi -q -d PERFORMANCE` before/after: zero deltas in
  `SW Thermal Slowdown`, `HW Thermal Slowdown`, `HW Power Braking` on both
  nodes, all arms. `SW Power Capping` counter unchanged (steady-state
  normal on GB10).
- 5 s sampler during each arm, both nodes: SM clock pinned 2184–2190 MHz
  (under the 2200 cap), GPU temp ≤ ~49 °C, throttle-reason bitmask 0x0
  throughout. No arm showed a thermal event; no repeats required.

## Bottom line

Under fingerprint: build `f62215013`, FP4-Local checkpoint, production
8×262K profile, 2026-11-10, idle pair, 2200 MHz clock cap.

- Structured output at concurrency now *exceeds* the plain control:
  schema@c1 70.8 vs plain 50.0 tok/s; schema@c8 207.4 tok/s aggregate.
- The batched follow-up is what removes the PR #336-only concurrency cliff
  (serial arm: 66.7 tok/s flat at c8 vs off's 134.5).
- No correctness divergence: temp-0 outputs identical across arms; both
  format probes 5/5 in every arm.
- Open item: harness `max_tokens=800` truncates structured replies
  symmetrically; a confirming run with a larger budget (or shorter workload)
  is needed to satisfy the literal "zero invalid" gate.
