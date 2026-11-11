# Next steps after the strict-spec work (written 2026-11-10)

State: PR #336 is merged upstream. PR #343 (shared batched verify) is open, CLA passed, CI not yet
started, one reviewer requested. Measured on the pair: `json_schema` 36.5 -> 70.8 tok/s at one
request, 134.5 -> 207.4 aggregate at eight, replies identical to serial decode. Results:
`RESULTS_PR336_strict_spec.md`.

## 1. Compare with TensorFold on the same method (about 7 minutes, production profile)

Not yet measured on this pair. Published sparkDash prose, aggregate tok/s at 1 / 2 / 4 / 8:
TensorFold dual NVFP4 67.4 / 98.9 / 152.2 / 231.2; Atlas FP4-Local 71.2 / 107.5 / 148.4 / 266.4
(different pairs, so not a head-to-head).

- A `decode` mode for `scripts/dev/strict_spec_ab.py` is written but not pushed (a local stash on
  Devin's checkout). It reads the prompts from a sparkDash checkout and applies its protocol:
  thinking off, temperature 0, 400 forced tokens, decode rate from first to last token.
- Before pushing: test it against a mock streaming server. It has only been compiled.
- Then one run on the production profile; put the pair's numbers beside the two rows above.

## 2. PR #343

- When CI runs or a reviewer comments, fix on `devin/strict-spec-batched`.
- Known failures on `main` itself, not caused by the PR: env budget, 500-line cap, benchmark gate.

## 3. A tool call under the lever (the one path not measured on hardware)

With the tool grammar on, the probe's model always took the JSON branch, so no union reply opened
a tool call. Needs a probe request the model cannot answer without the tool, lever off against on,
replies compared. CPU tests cover the branch-pending gate only.

## 4. Decide whether to chase 80 tok/s at one request

Timing: a strict verify step is about 52 ms (forward 43, propose 7.5, grammar 1.2-2.3). Options:

- **Faster verify forward.** Engine work; would lift plain text too. Largest lever, largest job.
- **Grammar drafts** (`find_jump_forward_string`). Measure first: count how often a token the
  grammar forces was drafted wrong. If rare, drop the idea.
- Deeper drafts do not pay: positions 5 and 6 survive 22% and 4% of the time.

## 5. Housekeeping

- PR #330 (MiMo) is open upstream and its description names hosts and LAN addresses: close it and
  edit the description.
- After #343 merges: delete `devin/strict-spec-full-logits` and `devin/strict-spec-batched` on the
  fork; keep this branch for the record.
- Worth reporting upstream: with the tool grammar on, a request with `tools` and `response_format`
  answers in JSON from priors instead of calling the tool (0 of 5), lever on or off. Keep
  `--disable-tool-grammar` in the production profile, and send `tool_choice: "none"` on a final
  turn that must be schema-valid.
