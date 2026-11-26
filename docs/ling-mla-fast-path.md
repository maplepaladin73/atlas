// SPDX-License-Identifier: AGPL-3.0-only

# Ling-3.2-air: MLA prefill fast path — assessment

Status: assessment / roadmap. The scalar-load vectorization (the first item
below) lands with its own PR; the tensor-core rewrite is explicitly NOT in
this document's scope to land in one session.

## Where prefill time goes (GB10, #130/#131/#139 posture)

At ISL 4096 (the design window; the serve default is 32768 max-prefill-token
chunks) the named deltas in nsys, post-#130 (packed prefill) and post-#139
(cuBLASLt k-chain):

| delta | where | why |
|---|---|---|
| ~3.1 s | ling_mla_prefill_attn | scalar fp32 flash-style kernel; see below |
| ~1.3 s | ling_moe_route | per-route-table scan for the routed MoE worklist |
| — | dense FFN / projections | cuBLASLt k-chain lane (#139) |
| — | KDA halves | regresident pair + packed context (#130, #138) |
| — | corpus MLP / head | GEMM lane |

ling_mla_prefill_attn is the single largest named prefill delta and is the
focus of this document.

## The kernel (kernels/gb10/ling-3.2-air/nvfp4/gated_mla.cu:520)

Shape: 16 query rows per CTA, 16 lanes per row (256 threads). Grid
(heads=32, ceil(seq_len/16), batch). Per query row, the inner loop walks
every kv position [0, kv_end) (causal: kv_base + q_pos + 1), and per
position per lane:

- 68 scalar 2-byte bf16 loads (36 score dims + 32 value dims; the 576/512-dim
  K/V rows are split 36/32 across the row's 16 lanes)
- ~68 fp32 FMAs (score dot + value accumulate) + a 5-level half-warp shuffle
  reduction, expf, and the softmax running-max update

No tensor cores, no vectorized loads, no cp.async, no mma. At these row
counts the pass is instruction/latency-bound (a scalar pass over an O(n^2)
frontier), not DRAM-bound: per-head K/V for the whole sequence is ~9 MB at
ISL 4096 and stays L2-resident.

Gates today: mla_layer5_prefill_fast_parity (paged + contiguous fast-path
stages, 25 stage checks), mla_layer5_parity (paged path, 95 checks, all
gated-MLA layers), e2e oracle (2/16-token full-stack tables), the packed-path
extensions from #130, serve smoke.

## Options

| option | effort | expected win | risk |
|---|---|---|---|
| A. Vectorize the scalar loads (uint2, 8B = 4 bf16) | ~1 session | ~1.3-1.5x on the kernel | low — bit-identical arithmetic (same per-element FMA order, same lane assignment) |
| B. Tensor-core mma rewrite (m16n8k16 flash-attention) | multi-day | ~3-5x on the kernel | medium — numerics path must stay parity-gated; needs a new parity harness for the packed shape |
| C. Gemm-split (QK^T + softmax + PV through the cuBLASLt lane) | ~2-3 days | ~2-4x | medium — extra score-matrix intermediate (~64 MiB @ 8k rows); changes the numerics structure rather than the layout |

Option A is landed in a follow-up PR (the kernel now loads its 36/32
contiguous elements as 8B uint2 slices, which preserves the scalar FMA order
bit-for-bit). Option B is the one big remaining prefill win and is the
scoping target for the next slice. Option C is a fallback if A under-delivers.

## Recommended sequence

1. Option A (this PR's subject): land the uint2 vectorization, verify
   mla_layer5_prefill_fast_parity green + serve smoke.
2. Option B (multi-day slice): new flash-attention-style kernel alongside the
   existing one, parity harness extension for the packed shape
   (mla_layer5_prefill_fast_parity already exercises the contiguous
   path; a packed-context parity harness is needed for the mma form), then
   e2e oracle in all three arms (eager/captured/staged), kernel audit
   unresolved:0, then enable behind the default flag.
3. Re-measure ISL 4096 prefill after each step; decide whether the next
   target is 2nd-largest hotspot (ling_moe_route ~1.3 s @ 4096) or pool/
   scheduler overheads.

## Risks

- Any further lane change to this kernel must keep the paged/mla parity
  harness AND the packed parity harness green — the kernel is the numerics
  backbone of all 7 gated-MLA layers.
- Do NOT move the lane under a default flip meant to look like a win without
  the e2e oracle passing in the captured/eager arms.
- The 8B uint2 access is 8B-aligned by construction: lane offsets are
  lane * 36 * 2 = lane * 72 bytes (72 = 9 * 8) and lane * 32 * 2 =
  lane * 64 bytes (64 = 8 * 8), both 8-byte aligned for every lane; row
  starts are 1152/1024-byte aligned by the (576/512)-element geometry.
