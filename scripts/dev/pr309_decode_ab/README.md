# PR 309 GLM code-decode A/B (thinking settings vs engine)

Question: is the code-decode drop seen after PR 309 (45.0 vs 65.3 tok/s, median of 3)
an engine regression, or did the thinking settings change what the cell decodes?

Known difference between the images:

| | old image (`3439912`) | PR tip (`354da2156`) and current main |
|---|---|---|
| `thinking_default` | false | true |
| server thinking budget | 256 (default) | 0 (none) |
| effort header when the client sends none | Low (thinking off, no tools) | Max |
| template | `jinja-templates/openai/glm5_next.jinja` | checkpoint template, Max unless `low`/`high` |

`scripts/dev/glm53_native/qualification.env` sets `ATLAS_DFLASH_SPEC_THINK=1`. Without it,
DFlash decodes serially inside `<think>` (`scheduler/mtp_gate.rs`, `spec_dispatch_eligible`).

## Arms

1. `old`: the pre-PR image built 2026-10-29 (`3439912`), the same one used in the Nov 2 run.
2. `pr`: the PR-tip image (`354da2156`), the same one used in the Nov 2 run.
3. `main`: a fresh build of current Atlas-Inf/atlas main (record the commit).

Every arm uses the same launch as the Nov 2 run: GLM-5.3-Flash-FP4-Local (pinned rev),
DFlash2 drafter, `--dflash-gamma=8`, TP2, 4 slots, recipe env files,
`--enable-prefix-caching=false`. Add `ATLAS_DFLASH_WIDTH_LOG=1` on rank 0 to every arm.
Change nothing else between arms.

Boot order: old, pr, main, old, pr, main (2 boots per arm). One `thinking_ab.py` run per boot.

## Per boot

```bash
# 1. Fingerprint (paste into --note, and save the full output next to the results)
docker inspect --format '{{.Id}} {{.Created}}' <image>
docker exec <rank0 container> env | grep -E 'ATLAS_DFLASH_SPEC_THINK|ATLAS_MTP_SPEC_THINK|ATLAS_GLM_DFLASH|ATLAS_DFLASH_WIDTH_LOG'
docker exec <rank0 container> cat /proc/1/cmdline | tr '\0' ' '   # the full serve command line
nvidia-smi -q -d CLOCK | grep -A3 'Applications Clocks'          # clock lock, both nodes

# 2. Measure (4 variants x 3 reps, interleaved, after 1 warmup)
python3 scripts/dev/pr309_decode_ab/thinking_ab.py \
  --url http://127.0.0.1:<port> --tag <old|pr|main>-boot<N> --reps 3 \
  --note "<image digest>; <env file>; SPEC_THINK=<value>" \
  --out pr309_ab.jsonl

# 3. Keep the rank-0 server log for the boot (width/accept lines)
docker logs <rank0 container> > rank0-<tag>.log 2>&1
```

If the Nov 2 harness's code prompt is available, run one extra pass per boot with
`--prompt-file <that prompt> --tag <arm>-boot<N>-orig` so the original cell is reproduced exactly.

After all boots:

```bash
python3 scripts/dev/pr309_decode_ab/thinking_ab.py --summarize pr309_ab.jsonl
```

## Variants (same prompt and sampling; only thinking fields differ)

- `default`: no thinking fields, `max_tokens` 512. The original code cell.
- `matched`: `enable_thinking` true, `reasoning_effort` low, `thinking_token_budget` 256,
  `max_tokens` 2048. The same settings on every image.
- `max_nobud`: `enable_thinking` true, no effort, no budget, `max_tokens` 2048. PR 309's default without the 512 cap.
- `sq_shape`: `enable_thinking` true, `reasoning_effort` low, `max_tokens` 384. The shape of upstream's decode cell.

## What to report back

- The `--summarize` table, the fingerprint of every boot, and any request that errored.
- From the rank-0 logs, the DFlash width/accept lines split into inside-think and after-think if possible.
- Answer these:
  1. On `default`, how many tokens of the code cell are reasoning on `old` vs `pr`?
  2. On `matched`, is `answer_tps` on `pr` and `main` within about 3% of `old`?
  3. Is `answer_tps` on `pr`/`main` much higher than `reasoning_tps`? (That would mean the
     original cell was mostly measuring reasoning.)
  4. Was `ATLAS_DFLASH_SPEC_THINK` the same on every arm?

## Reading the results

- `matched` level across arms, `default` low on `pr`: the drop is the thinking settings
  (more reasoning tokens, decoded at the reasoning rate). Not an engine regression.
- `matched` `answer_tps` still well below `old` on `pr`/`main`, with speculation engaged on both:
  regression candidate. Post the serve command line and this table on the PR, as data-angel asked.
- `SPEC_THINK` differs between arms: fix the env and rerun before drawing any conclusion.

Do not post hostnames or LAN details anywhere in the results.
