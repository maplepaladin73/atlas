# HANDOFF: follow-up runs for PR #343 (tool calls with JSON, and step timing)

**As of 2026-11-10.** For the agent with access to the two-GB10 pair. Build the head of
`devin/strict-spec-notes` (same Rust code as PR #343; `git diff origin/devin/strict-spec-batched
--stat -- crates` is empty). Results go to `docs/notes/RESULTS_PR336_strict_spec.md` on this
branch, as a new section. Nothing goes to any upstream PR.

The three-arm A/B is done (see the results file). Two things are still unmeasured:

1. **Tool calls together with `response_format`.** With the tool grammar on, such a request
   compiles a tool-call-or-JSON union. The production profile serves with
   `--disable-tool-grammar=true`, so the union never formed in the A/B.
2. **Where a verify step's time goes.** A lone structured request gets 3.63 tokens a step at about
   51 ms a step; a serial step is about 27 ms. The split of those 51 ms decides what to build next.

Rules as before (`HANDOFF_PR336_strict_spec.md` §2 and §3b): idle pair, fresh boot per arm, same
clock cap, no host names, addresses, paths, file names or credentials in anything committed.

## Boots

| boot | profile | added variables |
|---|---|---|
| `T` | production, unchanged | `ATLAS_STRICT_SPEC=1`, `ATLAS_MTP_TIMING=1`, `ATLAS_MTP_ACCEPT_DEBUG=1` |
| `U-off` | production **without** `--disable-tool-grammar` | `ATLAS_MTP_ACCEPT_DEBUG=1` |
| `U-on` | production **without** `--disable-tool-grammar` | `ATLAS_STRICT_SPEC=1`, `ATLAS_MTP_ACCEPT_DEBUG=1` |

## Boot `T`: timing

```bash
python3 scripts/dev/strict_spec_ab.py run --arm on --out timing.json \
  --cases schema,object,plain --concurrency 1 \
  --commit "$(git rev-parse --short HEAD)" --note "<flags and ATLAS_* of this boot>"
```

Report:

- the three printed rates. `schema@c1` was 70.8 without the timing switch; the difference is the
  switch's own overhead;
- from the rank-0 log, the `MTP verify timing [25 steps, seq_len=...]` lines, verbatim: five
  consecutive ones from the middle of `schema@c1`, five from `object@c1`, five from `plain@c1`
  (cases run in that order; `plain` replies are the 538-token `stop` ones);
- one `Done:` line per case.

## Boots `U-off` and `U-on`: tool calls with JSON

In each boot:

```bash
python3 scripts/dev/tools_rfmt_probe.py --runs 5 --final-tool-choice auto
python3 scripts/dev/tools_rfmt_probe.py --runs 5 --final-tool-choice auto --thinking
python3 scripts/dev/tools_rfmt_probe.py --runs 5
python3 scripts/dev/strict_spec_ab.py run --arm <off|on> --out union-<off|on>.json \
  --cases schema,object,plain --concurrency 1,4 \
  --commit "$(git rev-parse --short HEAD)" --note "<flags and ATLAS_* of this boot>"
```

`--final-tool-choice auto` leaves `tool_choice` unset on the final turn, which is the union. The
third command is the pinned form (`tool_choice: "none"`) for reference.

Report, per boot:

- the exit status and the `n/5 schema-valid` line of each probe command;
- whether turn 1 called the tool in every run (the probe prints it);
- the printed `final (...)` text of run 1 of each probe command. At temperature 0 the same command
  should print the same text in `U-off` and `U-on`; say whether it does;
- the `Done:` lines of the probe's final-turn requests (two or three per command). Expected:
  `U-off` reads `serial=1.00`; `U-on` reads `mtp` above 0. The first answer token of a union is
  always decoded serially, so `serial` slightly above 0 in `U-on` is normal;
- then `python3 scripts/dev/strict_spec_ab.py compare union-off.json union-on.json`: the table.

Pass: every probe command exits 0 in `U-on`, final texts match `U-off`, zero invalid replies and
zero request errors in `union-on.json`. Any failure: keep the reply text and the server log lines
around it, and report them instead of a summary.

## If time is short

`T` first (one boot, about five minutes of requests), then `U-on`, then `U-off`.
