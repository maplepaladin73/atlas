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

## Validity re-run: `on` arm, `max_tokens=2400`

Same build, checkpoint, profile, and arm env as the `on` arm above; only the
output budget changed (2400 vs 800) so structured replies can complete.
Harness record `on-valid.json`, commit `d3ec45df2` (docs-only delta vs the
tested tree `f62215013`), 3 reps per case, 0 request errors.

Invalid structured replies per case — **all zero**:

| case | tok/s | invalid |
|---|---|---|
| schema@c1 | 70.5 | 0 |
| schema@c2 | 101.4 | 0 |
| schema@c4 | 145.2 | 0 |
| schema@c8 | 201.9 | 0 |
| object@c1 | 69.3 | 0 |
| plain@c1 | 50.0 | 0 |
| sampled@c1 | 64.3 | 0 |
| sampled@c4 | 122.7 | 0 |
| think@c1 | 66.1 | 0 |
| logprobs@c1 | 63.7 | 0 |
| mixed@c4 | 122.3 | 0 |

Every reply in the record has `finish_reason=stop`; no invalid reply exists,
so none reports `(length)`. The earlier "invalid" counts were purely the
800-token truncation and are now cleared for the batched arm. Rates match the
800-token run within noise (schema@c1 70.5 vs 70.8, schema@c8 201.9 vs 207.4).

## Production-profile `ATLAS_*` environment (complete list)

All variables set by the serve profile used in every arm (alphabetical;
`ATLAS_STRICT_SPEC=1` and `ATLAS_MTP_ACCEPT_DEBUG=1` are the only arm/profile
deltas on top of the base production env — `off` unsets `ATLAS_STRICT_SPEC`,
`serial` adds `ATLAS_GRAMMAR_SERIAL_VERIFY=1`):

```
ATLAS_CONTENT_LOOP_MIN_REPEATS=32
ATLAS_CONTENT_LOOP_WATCHDOG=1
ATLAS_EP_PROTOCOL=v2
ATLAS_GLM_CMD_RDMA=1
ATLAS_GLM_ZERO_ROWS=1
ATLAS_LOOP_STEER=1
ATLAS_LT_KCHAIN_PIN=1
ATLAS_MTP_ACCEPT_DEBUG=1
ATLAS_MTP_DEEP_MAX_SEQS=2
ATLAS_MTP_DFLASH_CTX_SCOPED=1
ATLAS_MTP_DYNAMIC_DEPTH=1
ATLAS_MTP_GATE_FORCE=1
ATLAS_MTP_MAX_SEQS=8
ATLAS_MTP_SINGLE_DEPTH_ADAPT=1
ATLAS_NO_MTP_DCUT=1
ATLAS_PLE_FAULT_POOL=1
ATLAS_PLE_NVME_KEEPALIVE_MS=10
ATLAS_PLE_PACKED_DIR=/atlas-ple
ATLAS_PLE_WARM_BG=1
ATLAS_PREFILL_CODISPATCH=1
ATLAS_PREFILL_CODISPATCH_SETTLE_MS=3
ATLAS_PREFILL_CODISPATCH_WINDOW_MS=20
ATLAS_QWEN4EXP_AUX_PINNED=1
ATLAS_QWEN4EXP_BATCH_FAST=1
ATLAS_QWEN4EXP_BATCH_SMALL=1
ATLAS_QWEN4EXP_CKPT_AUX_SHARE=1
ATLAS_QWEN4EXP_DECODE_FUSE=1
ATLAS_QWEN4EXP_DECODE_GRAPH=1
ATLAS_QWEN4EXP_DECODE_GRAPH_COLLECTIVES=1
ATLAS_QWEN4EXP_DECODE_GRAPH_WIDE=1
ATLAS_QWEN4EXP_DRAFT_HEAD_NVFP4=1
ATLAS_QWEN4EXP_EOS_BAN=1
ATLAS_QWEN4EXP_EXACT_DEFER=1
ATLAS_QWEN4EXP_EXACT_VERIFY=1
ATLAS_QWEN4EXP_FINISH_LEAF=1
ATLAS_QWEN4EXP_GDN_COMMIT_FUSE=1
ATLAS_QWEN4EXP_HC_FAST=1
ATLAS_QWEN4EXP_HC_MMA=1
ATLAS_QWEN4EXP_HC_STAGE_FIT=1
ATLAS_QWEN4EXP_HC_WIDE=1
ATLAS_QWEN4EXP_KV_LOOKAHEAD=64
ATLAS_QWEN4EXP_LMHEAD_BATCHM=1
ATLAS_QWEN4EXP_LMHEAD_SPLIT=1
ATLAS_QWEN4EXP_LMHEAD_SPLIT_VERIFY=1
ATLAS_QWEN4EXP_MOE_FAST=1
ATLAS_QWEN4EXP_MOE_NO_CLAMP=1
ATLAS_QWEN4EXP_MOE_TC=1
ATLAS_QWEN4EXP_MTP_DEPTH=7
ATLAS_QWEN4EXP_MTP_DRAFT_TP=1
ATLAS_QWEN4EXP_MTP_GROUPED_NORM=1
ATLAS_QWEN4EXP_MTP_KV_KEEP=1
ATLAS_QWEN4EXP_PREFILL_BA_ROWS=1
ATLAS_QWEN4EXP_PREFILL_FP8_W2=1
ATLAS_QWEN4EXP_PREFILL_GDN=1
ATLAS_QWEN4EXP_PREFILL_GEMM_RASTER=1
ATLAS_QWEN4EXP_PREFILL_HC=1
ATLAS_QWEN4EXP_PREFILL_HOST_IDS=1
ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT=1
ATLAS_QWEN4EXP_PREFILL_MOE=1
ATLAS_QWEN4EXP_PREFILL_MOE_NODUP=1
ATLAS_QWEN4EXP_PREFILL_MOE_W2=1
ATLAS_QWEN4EXP_PREFILL_MULTI=1
ATLAS_QWEN4EXP_PREFILL_MULTI_CACHED=1
ATLAS_QWEN4EXP_PREFILL_QSA_SCORE=1
ATLAS_QWEN4EXP_PREFILL_QSA_SPLIT=1
ATLAS_QWEN4EXP_PREFILL_ROWINV=1
ATLAS_QWEN4EXP_PREFILL_SP=1
ATLAS_QWEN4EXP_PREFILL_SP_PIPE=1
ATLAS_QWEN4EXP_PREFILL_SP_ROUTE=1
ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE=1
ATLAS_QWEN4EXP_PREFILL_SP_SHARED=1
ATLAS_QWEN4EXP_QKV_ROWS_2D=1
ATLAS_QWEN4EXP_QSA_COMMIT_ROWS=1
ATLAS_QWEN4EXP_QSA_COMMIT_TABLE=1
ATLAS_QWEN4EXP_QSA_DECODE_ROWS=1
ATLAS_QWEN4EXP_QSA_KEY_ONLY=1
ATLAS_QWEN4EXP_QSA_REUSE=1
ATLAS_QWEN4EXP_QSA_TOPK_WIDE=1
ATLAS_QWEN4EXP_ROWS32_TILE=1
ATLAS_QWEN4EXP_SNAPSHOT_AUX_MB=2048
ATLAS_QWEN4EXP_SP_PREALLOC=1
ATLAS_QWEN4EXP_VERIFY_BT_PINNED=1
ATLAS_QWEN4EXP_W4_ROWS=1
ATLAS_QWEN4EXP_W4_ROWS_WIDE=1
ATLAS_RDMA_ALLREDUCE=1
ATLAS_RDMA_ONESHOT=1
ATLAS_RDMA_ONESHOT_MAX=8388608
ATLAS_RDMA_PAIR_CHAIN=1
ATLAS_STRICT_SPEC=1
ATLAS_VERIFY_PICK_PAR=1
```

(`ATLAS_GLM_STRICT_SPEC` remains unset — Qwen full-logits path only.)

## Representative `Done:` lines (rank-0 logs, `ATLAS_MTP_ACCEPT_DEBUG=1`)

Case attribution: `schema@c1` is the first harness case (three sequential
800-token reps right after the warmup request); `plain@c1` is the only case
producing 538-token `stop` replies at ~50 tok/s. Lines are verbatim minus the
log prefix.

### `serial` arm — schema@c1

```
Done: 800 tokens (length) 68.0 tok/s, TTFT=166.8ms, serial=0.00 mtp=1.00 p1=0.870 mean_na=2.162 tok_step=3.162 regime_reprobes=0 depth=k4 depth_switches=1 acc=0.87,0.72,0.58,-,-,-,- surv=0.87,0.72,0.58,0.00,0.00,0.00,0.00
Done: 800 tokens (length) 69.6 tok/s, TTFT=168.3ms, serial=0.00 mtp=1.00 p1=0.870 mean_na=2.162 tok_step=3.162 regime_reprobes=0 depth=k4 depth_switches=1 acc=0.87,0.72,0.58,-,-,-,- surv=0.87,0.72,0.58,0.00,0.00,0.00,0.00
Done: 800 tokens (length) 69.5 tok/s, TTFT=167.7ms, serial=0.00 mtp=1.00 p1=0.870 mean_na=2.162 tok_step=3.162 regime_reprobes=0 depth=k4 depth_switches=1 acc=0.87,0.72,0.58,-,-,-,- surv=0.87,0.72,0.58,0.00,0.00,0.00,0.00
```

### `serial` arm — plain@c1

```
Done: 538 tokens (stop) 48.6 tok/s, TTFT=133.9ms, serial=0.00 mtp=1.00 p1=0.627 mean_na=1.157 tok_step=2.157 regime_reprobes=0 depth=k4 depth_switches=0 acc=0.63,0.37,0.17,-,-,-,- surv=0.63,0.37,0.16,0.00,0.00,0.00,0.00
Done: 538 tokens (stop) 50.4 tok/s, TTFT=131.6ms, serial=0.00 mtp=1.00 p1=0.627 mean_na=1.157 tok_step=2.157 regime_reprobes=0 depth=k4 depth_switches=0 acc=0.63,0.37,0.17,-,-,-,- surv=0.63,0.37,0.16,0.00,0.00,0.00,0.00
Done: 538 tokens (stop) 50.5 tok/s, TTFT=134.2ms, serial=0.00 mtp=1.00 p1=0.627 mean_na=1.157 tok_step=2.157 regime_reprobes=0 depth=k4 depth_switches=0 acc=0.63,0.37,0.17,-,-,-,- surv=0.63,0.37,0.16,0.00,0.00,0.00,0.00
```

### `on` arm — schema@c1

```
Done: 800 tokens (length) 70.6 tok/s, TTFT=167.9ms, serial=0.00 mtp=1.00 p1=0.819 mean_na=2.629 tok_step=3.629 regime_reprobes=0 depth=k7 depth_switches=3 acc=0.82,0.66,0.54,0.47,0.39,0.30,- surv=0.82,0.66,0.53,0.36,0.22,0.04,0.00
Done: 800 tokens (length) 72.1 tok/s, TTFT=171.9ms, serial=0.00 mtp=1.00 p1=0.819 mean_na=2.629 tok_step=3.629 regime_reprobes=0 depth=k7 depth_switches=3 acc=0.82,0.66,0.54,0.47,0.39,0.30,- surv=0.82,0.66,0.53,0.36,0.22,0.04,0.00
Done: 800 tokens (length) 72.0 tok/s, TTFT=168.1ms, serial=0.00 mtp=1.00 p1=0.819 mean_na=2.629 tok_step=3.629 regime_reprobes=0 depth=k7 depth_switches=3 acc=0.82,0.66,0.54,0.47,0.39,0.30,- surv=0.82,0.66,0.53,0.36,0.22,0.04,0.00
```

### `on` arm — plain@c1

```
Done: 538 tokens (stop) 48.8 tok/s, TTFT=130.3ms, serial=0.00 mtp=1.00 p1=0.627 mean_na=1.157 tok_step=2.157 regime_reprobes=0 depth=k4 depth_switches=0 acc=0.63,0.37,0.17,-,-,-,- surv=0.63,0.37,0.16,0.00,0.00,0.00,0.00
Done: 538 tokens (stop) 50.7 tok/s, TTFT=132.4ms, serial=0.00 mtp=1.00 p1=0.627 mean_na=1.157 tok_step=2.157 regime_reprobes=0 depth=k4 depth_switches=0 acc=0.63,0.37,0.17,-,-,-,- surv=0.63,0.37,0.16,0.00,0.00,0.00,0.00
Done: 538 tokens (stop) 50.6 tok/s, TTFT=130.7ms, serial=0.00 mtp=1.00 p1=0.627 mean_na=1.157 tok_step=2.157 regime_reprobes=0 depth=k4 depth_switches=0 acc=0.63,0.37,0.17,-,-,-,- surv=0.63,0.37,0.16,0.00,0.00,0.00,0.00
```

Note the `on` schema@c1 lines run deeper drafts (`depth=k7`, `tok_step=3.629`)
than `serial` (`depth=k4`, `tok_step=3.162`) at the same wall rate — the
batched path converts draft depth into throughput at concurrency.

## Merge-regime replication (the #336 merge A/B, run on the Spark pair)

#336 was merged upstream after an A/B on one GB10 (27B target, DFlash2 γ=8
profile): 6 `json_object` + 6 strict `json_schema` requests, greedy, thinking
off, 400 max tokens, at C1 then C8, scoring per-request decode tok/s and
byte-identity. That run flagged the switch as *slower at C8* (per-request
median 9.5 → 6.2 tok/s). This section replays the same regime against the
three arms on the pair build (`f62215013`, production profile, fresh boot per
arm, idle pair). Probe: `scripts/dev/strict_spec_c8_probe.py`; decode rate is
the server-attested `usage.response_token/s` from the streamed reply.

### Per-request decode, median (tok/s)

| phase | kind | off | serial (#336 as merged) | on (#343) |
|---|---|---|---|---|
| C1 | json_object | 36.2 | 64.1 | 64.4 |
| C1 | json_schema | 37.0 | 58.3 | 59.3 |
| C1 | all 12 | 36.7 | 59.6 | 60.5 |
| C8 | json_object | 16.8 | 9.0 | 25.8 |
| C8 | json_schema | 18.0 | 7.4 | 21.8 |
| C8 | all 12 | 17.5 | 8.1 | 24.0 |

### Aggregate (713 completion tokens per phase in every arm)

| phase | off | serial | on |
|---|---|---|---|
| C1 (sequential) | 30.7 | 44.8 | 53.2 |
| C8 (12 via pool of 8) | 155.0 | 77.1 | 197.2 |

- **The C8 regression reproduces for `serial`** (#336 alone): per-request
  median 17.5 → 8.1 tok/s, aggregate 155 → 77. Same shape as the merge A/B's
  9.5 → 6.2 on the 27B profile.
- **`on` removes it**: C8 per-request median 24.0 (1.37× off, 2.96× serial);
  aggregate 197 tok/s (1.27× off, 2.56× serial). C1 is unchanged vs serial
  (60.5 vs 59.6, within noise).
- **Validity**: 12/12 valid JSON and 6/6 schema keys in every arm and phase;
  every reply `finish_reason=stop`; 0 request errors.
- **Byte-identity**: all 12 replies byte-identical across off / serial / on at
  C1 and at C8, and C1 ≡ C8 within each arm. Identical `completion_tokens`
  (713) in every arm/phase.
- **Engagement** (`Done:` lines this boot): serial-arm C8 requests show
  `mtp=1.00 tok_step≈2.2–3.4` at 6.6–10 tok/s — speculation is accepted but
  each strict row pays its own verify forward; `MTP accept` batches are all
  `n=1`. The `on` arm's C8 rows ran at 19–35 tok/s with shared verifies.

## PR #343 follow-up boots (`HANDOFF_PR343_followup.md`)

Same image/build as above (Rust tree unchanged; docs commit `1f2c89b90`),
production profile, fresh boot per arm, idle pair, 2200 MHz cap, no throttle
events. `max_tokens=2400`.

### Boot `T` — step timing (`ATLAS_STRICT_SPEC=1 ATLAS_MTP_TIMING=1`)

Rates with the timing switch armed: schema@c1 **70.5**, object@c1 **69.7**,
plain@c1 **50.3** tok/s. schema@c1 was 70.8 without the switch — timing
overhead is within noise.

`Done:` per case:

```
schema : Done: 872 tokens (stop) 71.6 tok/s, TTFT=185.7ms, serial=0.00 mtp=1.00 p1=0.808 mean_na=2.629 tok_step=3.629 depth=k7 depth_switches=3 acc=0.81,0.65,0.53,0.47,0.37,0.26,-
object : Done: 868 tokens (stop) 70.9 tok/s, TTFT=167.4ms, serial=0.00 mtp=1.00 p1=0.833 mean_na=2.360 tok_step=3.360 depth=k5 depth_switches=1 acc=0.84,0.67,0.53,0.41,-,-,-
plain  : Done: 538 tokens (stop) 51.1 tok/s, TTFT=131.8ms, serial=0.00 mtp=1.00 p1=0.627 mean_na=1.157 tok_step=2.157 depth=k4 depth_switches=0 acc=0.63,0.37,0.17,-,-,-,-
```

`MTP verify timing [25 steps]` — five consecutive from mid-reply, verbatim
minus the log prefix (zero-valued `loop_*`/`GAP` fields elided for width):

schema@c1:
```
[seq_len=369]: fwd=40.69ms fast_greedy=1.20ms pipeline=1.20ms commit=0.01ms save_hidden=0.01ms propose=5.95ms step_mtp=48.05ms TOTAL=47.85ms
[seq_len=457]: fwd=43.38ms fast_greedy=1.31ms d2h=0.01ms(x0.1) dequant=0.24ms(x0.1) pipeline=1.99ms(x1.7) penalties=0.00ms argmax=0.04ms commit=0.01ms save_hidden=0.01ms propose=7.44ms step_mtp=52.35ms TOTAL=52.64ms
[seq_len=557]: fwd=42.82ms fast_greedy=1.55ms d2h=0.01ms(x0.1) dequant=0.13ms(x0.1) pipeline=2.29ms(x1.7) penalties=0.00ms argmax=0.04ms commit=0.01ms save_hidden=0.01ms propose=7.47ms step_mtp=52.38ms TOTAL=52.32ms
[seq_len=661]: fwd=43.22ms fast_greedy=1.53ms pipeline=1.53ms commit=0.01ms save_hidden=0.01ms propose=7.50ms step_mtp=52.23ms TOTAL=52.26ms
[seq_len=774]: fwd=43.89ms fast_greedy=1.48ms d2h=0.01ms(x0.1) dequant=0.13ms(x0.1) pipeline=1.84ms(x1.5) penalties=0.00ms argmax=0.02ms commit=0.01ms save_hidden=0.01ms propose=7.76ms step_mtp=53.30ms TOTAL=53.43ms
```

object@c1:
```
[seq_len=332]: fwd=40.63ms fast_greedy=1.43ms d2h=0.00ms dequant=0.05ms pipeline=1.62ms(x1.2) penalties=0.00ms argmax=0.01ms commit=0.01ms save_hidden=0.01ms propose=5.94ms step_mtp=47.97ms TOTAL=48.14ms
[seq_len=423]: fwd=40.52ms fast_greedy=1.52ms d2h=0.00ms dequant=0.05ms pipeline=1.75ms(x1.2) penalties=0.00ms argmax=0.01ms commit=0.01ms save_hidden=0.01ms propose=5.96ms step_mtp=48.24ms TOTAL=48.17ms
[seq_len=506]: fwd=40.76ms fast_greedy=1.37ms d2h=0.00ms(x0.1) dequant=0.08ms(x0.1) pipeline=1.78ms(x1.4) penalties=0.00ms argmax=0.02ms commit=0.01ms save_hidden=0.01ms propose=5.97ms step_mtp=48.43ms TOTAL=48.37ms
[seq_len=585]: fwd=40.45ms fast_greedy=1.47ms d2h=0.00ms(x0.1) dequant=0.09ms(x0.1) pipeline=1.97ms(x1.4) penalties=0.00ms argmax=0.02ms commit=0.01ms save_hidden=0.01ms propose=6.00ms step_mtp=48.22ms TOTAL=48.24ms
[seq_len=676]: fwd=41.06ms fast_greedy=1.32ms pipeline=1.32ms commit=0.01ms save_hidden=0.00ms propose=6.02ms step_mtp=48.45ms TOTAL=48.40ms
```

plain@c1:
```
[seq_len=123]: fwd=37.67ms fast_greedy=0.00ms propose=4.44ms step_mtp=42.15ms TOTAL=42.12ms
[seq_len=182]: fwd=37.31ms fast_greedy=0.00ms propose=4.44ms step_mtp=41.78ms TOTAL=41.77ms
[seq_len=235]: fwd=37.71ms fast_greedy=0.00ms propose=4.46ms step_mtp=42.12ms TOTAL=42.18ms
[seq_len=289]: fwd=37.37ms fast_greedy=0.00ms propose=4.48ms step_mtp=41.87ms TOTAL=41.86ms
[seq_len=342]: fwd=37.92ms fast_greedy=0.00ms propose=4.48ms step_mtp=42.45ms TOTAL=42.41ms
```

Reading: a strict schema step is ~52 ms vs ~42 ms plain. Of the ~10 ms
difference, ~5–6 ms is `fwd` (deeper draft window → more verify positions:
k7 vs k4) and ~3 ms is `propose` (7.5 vs 4.5 ms, again depth). The grammar
host work itself — `fast_greedy` + `pipeline` (mask, d2h, dequant, argmax) —
is **1.2–2.3 ms per step**, i.e. 3–4% of the step. The remaining cost is
draft depth, which the deeper window converts into `tok_step` 3.6 vs 2.2. The
mask path is not the bottleneck; a lighter mask would buy ≤2 ms/step.

### Boots `U-off` / `U-on` — tool grammar enabled (tool-call-or-JSON union)

Production profile **without** `--disable-tool-grammar`; `U-on` adds
`ATLAS_STRICT_SPEC=1`.

Probe results (`tools_rfmt_probe.py --runs 5`):

| probe | U-off exit / valid | U-on exit / valid | turn 1 called tool? | run-1 final text identical U-off ≡ U-on |
|---|---|---|---|---|
| `--final-tool-choice auto` | 0 / 5/5 | 0 / 5/5 | **no, 0/5** (both boots) | yes |
| `--final-tool-choice auto --thinking` | 0 / 5/5 | 0 / 5/5 | **no, 0/5** (both boots) | yes |
| default (`tool_choice: none`) | 0 / 5/5 | 0 / 5/5 | **no, 0/5** (both boots) | yes |

Run-1 final text in every probe, both boots:
```
{\n  "city": "Paris"\n    , "temperature_c": 18, "conditions": "sunny", "wind":  {"speed_kmh": 10, "direction": "N"}\n}
```

**Behavioral note (pre-existing, not strict-spec):** with the tool grammar
on, the union lets the first output token pick JSON instead of `<tool_call>`,
and at temperature 0 the model always picks JSON — it answers the schema
directly from priors (sunny/10/N) and never calls `get_weather`. With the tool
grammar off (earlier boots) turn 1 called the tool 5/5. Identical in `U-off`
and `U-on`, so the lever does not change the choice; it only speeds the
decode that follows. Whether the union should bias toward the tool call is a
separate question for the tool-grammar path.

`Done:` lines for the probe's (single-turn) answers:

```
U-on : Done: 80 tokens (stop) 50.0 tok/s, TTFT=164.3ms, serial=0.03 mtp=0.97 p1=0.543 mean_na=1.229 tok_step=2.229 depth=k4
U-on : Done: 80 tokens (stop) 50.0 tok/s, TTFT=161.3ms, serial=0.03 mtp=0.97 p1=0.543 mean_na=1.229 tok_step=2.229 depth=k4
U-off: Done: 80 tokens (stop) 36.6 tok/s, TTFT=164.4ms, serial=1.00 mtp=0.00 p1=0.000 mean_na=0.000 tok_step=1.000 depth=k5
U-off: Done: 80 tokens (stop) 36.6 tok/s, TTFT=162.7ms, serial=1.00 mtp=0.00 p1=0.000 mean_na=0.000 tok_step=1.000 depth=k5
```

`U-on` shows the expected `serial=0.03` (first union token decoded serially)
then `mtp=0.97`; `U-off` is fully serial. 50.0 vs 36.6 tok/s on the same
80-token reply.

`strict_spec_ab.py compare union-off.json union-on.json`:

| case | U-off tok/s | U-on tok/s | on/off | invalid off | invalid on | identical |
|---|---|---|---|---|---|---|
| schema@c1 | 36.6 | 70.8 | 1.93 | 0/3 | 0/3 | 3/3 |
| schema@c4 | 86.8 | 145.0 | 1.67 | 0/12 | 0/12 | 12/12 |
| object@c1 | 36.5 | 69.8 | 1.91 | 0/3 | 0/3 | 3/3 |
| plain@c1 | 50.2 | 50.3 | 1.00 | 0/3 | 0/3 | 3/3 |

0 request errors in both records. Pass criteria met: every probe exits 0 in
`U-on`, final texts match `U-off`, zero invalid replies, zero errors. Rates
with the tool grammar on match the `--disable-tool-grammar` A/B within noise
(70.8 / 145.0 / 69.8 / 50.3 vs 70.8 / 149.0 / 68.9 / 50.0).
