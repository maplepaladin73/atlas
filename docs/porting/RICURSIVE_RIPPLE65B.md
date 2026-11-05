# `ricursive` — Ripple-65B Preview: port status & local test guide

Status as of 2026-11-05: **complete implementation lives on upstream branch
`Atlas-Inf/atlas@feat/ripple-65b-hip`** (AzeezIsh, 14 commits, head `97b28583`).
This fork's branch `vibe/ricursive-65b-complete-07de47` is that branch plus our
additions (see "Fork delta" below), so **local can build and test from this branch
directly**. No upstream PR is open for it yet.

Target checkpoints (Apache-2.0, ungated, `model_type: "ricursive"`):

| Checkpoint | Size | Notes |
|---|---|---|
| `ricursiveAI/Ripple-65B-Preview-NVFP4` | ~33 GB | ModelOpt NVFP4 (group 16); MLP only — `ignore` keeps `self_attn*`/`embed_tokens`/`lm_head` BF16. **Test first.** |
| `ricursiveAI/Ripple-65B-Preview` | 129.57 GB BF16 | needs TP=2 or a large single GPU (follow-up) |
| `ricursiveAI/Ripple-65B-Base-Preview` | 129.57 GB BF16 | pretrain-only |
| `ricursiveAI/Ripple-65B-Preview-DFlash2` | 8.08 GB | 4.04B qwen3-family draft, 16-token blocks; wired on the branch (drafter ctx window 16384, per-layer NoPE) |

Reference: `modeling_ricursive.py` / `configuration_ricursive.py` in the HF repo
(vLLM `--trust-remote-code` is the parity reference).

## What the branch implements (all verified in-tree)

| Ripple piece | Implementation on `feat/ripple-65b-hip` |
|---|---|
| Config parse (`model_type: "ricursive"`, eos array, hybrid layer pattern, embed_scale, quant block) | `crates/atlas-core/src/config/parsers/ricursive.rs` (+ unit tests); dispatch arm in `config/dispatch.rs` |
| Layer convention | every layer `FullAttention` + `config.sliding_layers: Vec<bool>` (Gemma-4 convention, so model-level `== FullAttention` checks keep working); sliding layers get `set_sliding_window(1024)` + RoPE θ=10000 full head_dim, global layers `set_rope_disabled(true)` (NoPE — no YaRN needed for the 1M ctx claim) |
| PeriNorm (4 RMSNorms/layer, post-norm before residual) | Gemma-4 slot mapping + `set_post_sublayer_norms`; **PeriNorm arm added to multi-seq batched decode + DFlash verify** (head commit `97b28583` — fixes "fibre fibre" C≥2 decode bug) |
| Per-head Q/K RMSNorm | `q_norm`/`k_norm`, `qk_norm_type="per_head"` |
| Elementwise sigmoid gate `g_proj` [H·D, hidden], f32 | folded into `q_proj` as the Qwen3.5 per-head `[q_h \| gate_h]` interleave at load (`interleave_q_gate`), served by the existing gated path (`sigmoid_gate_mul_batched`); sources reclaimed |
| Value residual (V input = `lerp(scaled layer-0 embeddings, normed hidden, sigmoid(λ))`) | new `layers/qwen3_attention/value_residual.rs` + `model/impl_a1.rs` layer-0 shadow; per-layer blend = bf16(sigmoid(λ)) via `set_value_residual_blend` |
| FFN clamp ±20 (`ffn_clamp_threshold`) | `config.ffn_clamp_threshold` + clamped SwiGLU selected by `DenseFfnLayer::new(.., "ricursive")`; kernel constant `RIPPLE_FFN_CLAMP` in `kernels/strix-hip/ripple-65b/nvfp4/ripple_ops.cu`; loader asserts config == kernel constant |
| `embed_scale = √5120` | generic embed_scale path + strix-hip `embed_scale.cu` kernel |
| ModelOpt NVFP4 MLP | `detect_nvfp4_variant` + `load_dense_ffn(.., variant, ..)`; attention stays BF16 (ignore list) |
| Weight loader / factory | `crates/spark-model/src/weight_loader/ricursive.rs` (+ blend unit test), factory arm |
| DFlash2 drafter | per-layer NoPE (`no_rope_layers`) in the drafter's RoPE sites, ctx window 16384, generalized grouped causal conv, strix-hip top-32 selector |
| Kernels | `kernels/strix-hip/ripple-65b/` (MODEL.toml, nvfp4/KERNEL.toml, embed_scale.cu, ripple_ops.cu), HDIM=128 attention build flag |

## Fork delta (this branch vs upstream `feat/ripple-65b-hip`)

Only additive, no conflicts with the upstream work:
- `config/dispatch.rs`: `architectures: ["RicursiveForCausalLM"]` fallback → `"ricursive"`
  (upstream matches `model_type` only).
- `config/parsers/ricursive.rs`: two extra tests — the architectures fallback and the
  NVFP4 `quantization_config` block parse.
- this guide.

Our earlier scaffold (PR #5's standalone parser/loader with per-layer
`layer_types` SWA mapping and separate `elementwise_gate_weight`/`value_res_lambda`
fields) is **superseded** by the upstream implementation (q‖gate interleave +
`sliding_layers` + `value_residual.rs`) and was dropped in the merge.

## Local test guide

Prereqs (QUICKSTART.md): Ubuntu 22.04/24.04, `build-essential pkg-config git cmake
clang libclang-dev`, CUDA 13.0 (`nvcc --version` → 13.0), Rust via `rust-toolchain.toml`.
First `cargo build --release -p spark-server` ≈ 15–30 min (PTX compile dominates).

```bash
git fetch origin && git checkout vibe/ricursive-65b-07de47   # this branch, in maplepaladin73/atlas
hf download ricursiveAI/Ripple-65B-Preview-NVFP4   # ~33 GB into the HF cache
cargo test -p atlas-core ricursive                  # parser tests (3)
# GB10 (default target; kernels/gb10/ripple-65b is on this branch):
cargo build --release -p spark-server
./target/release/spark serve ricursiveAI/Ripple-65B-Preview-NVFP4
# strix-hip (upstream's original target):
ATLAS_TARGET_HW=strix-hip cargo build --release -p spark-server
ATLAS_TARGET_HW=strix-hip ./target/release/spark serve ricursiveAI/Ripple-65B-Preview-NVFP4
```

Verification loop (CONTRIBUTING.md "Pull Request Process"):
1. Reference parity: vLLM `--trust-remote-code` on the same checkpoint, same host;
   gate = KL logit drift **exactly 0** over teacher-forced top-20 + generated chat,
   plus byte-identical greedy/T=0 controls.
2. `cargo test --workspace`, `cargo clippy`, `cargo fmt --check`, 500-LoC file cap.
3. Model gates: `./target/release/spark benchmark run <gate> --pull-request-gate --yes`
   (`bfcl-subset` ST-995 with normalized floor, `agentic-50`, …). Per-model gate files
   live in the kernel dir's `BENCH.toml`.

## Kernel targets on this branch
- **gb10** (default `ATLAS_TARGET_HW`): `kernels/gb10/ripple-65b/` — MODEL.toml with
  `kernel_source = "qwen3.6-27b"` (the gb10 tree that already serves the same dense
  BF16-attention + NVFP4-FFN shapes family), plus a two-file overlay:
  `nvfp4/ripple_ops.cu` (`ripple_silu_mul_clamp`, `ripple_valres_lerp`) and
  `nvfp4/embed_scale.cu` (`bf16_scale_inplace`, same as the gb10 gemma-4 trees).
  `nvfp4/KERNEL.toml` sets `extra_nvcc_flags = ["--fmad=false", "-DHDIM=128"]`
  (gb10's common `paged_decode_attn.cu` defaults `HDIM 256`; head_dim 128 needs the
  flag — same remedy as `gb10/laguna-s-2.1`). Build audit: target
  `(gb10, ripple-65b, nvfp4)` compiles 211 kernels, overlay shadowing verified.
- **strix-hip**: upstream's original `kernels/strix-hip/ripple-65b/` (HDIM=128 flag).

## Remaining work to call the port done
1. **TP=2 for the BF16 checkpoint** (129.6 GB): wire `slice_for_rank` for Q/K/V/O,
   gate/up/down, lm_head, flip `supports_tp()` (reference: `weight_loader/minimax.rs`).
   NVFP4 (~33 GB) serves a single GB10 without this.
2. **Gates for the new model dir** (`BENCH.toml` under `kernels/gb10/ripple-65b/` and
   `kernels/strix-hip/ripple-65b/`) so `spark benchmark run --pull-request-gate`
   covers ricursive.
3. **GPU verification** on the target host: vLLM parity (KL 0), T=0 byte-identical
   controls, then the benchmark gates.
4. Optional: DFlash2 spec-dec end-to-end on the target host (already wired in code).

## Provenance / authorship
- Upstream implementation: AzeezIsh's `feat/ripple-65b-hip` (14 commits, co-authored
  with Claude Opus 5.5 per commit messages). Architecture facts from
  `ricursiveAI/Ripple-65B-Preview`'s public `config.json` / `modeling_ricursive.py`
  (Apache-2.0; reference used for numerics only).
- Fork delta + this guide: drafted by Mistral Vibe; earlier fork scaffold (PR #5)
  superseded as noted above.
