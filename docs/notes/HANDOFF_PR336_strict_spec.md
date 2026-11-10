# HANDOFF: PR #336 `ATLAS_STRICT_SPEC` hardware A/B

**As of 2026-11-10.** Test plan for the agent with access to the two-GB10 pair.

This file lives on the fork-only branch `devin/strict-spec-notes`. The code under test is the head
of `devin/strict-spec-full-logits`; the two branches differ only under `docs/notes/`. Results are
committed to this branch (§6). Do not comment on PR #336 and do not push to its branch.

## 0. Where things stand

One single-stream run exists (reported by the local agent, n=4 per arm, temperature 0, replies of
about 220 tokens, upstream `1f6de550` plus the PR commit, a reduced test profile). It did not use
this plan or its harness.

| decode tok/s | plain | `json_object` | `json_schema` |
|---|---|---|---|
| lever off | 63.0 | 37.4 | 37.5 |
| lever on | 63.2 | 57.2 | 59.7 |

The same morning, on the previous image (upstream `a523a07f`) with the production profile and an
idle pair, plain measured 85.9 / 88.8 and `json_object` 35.3 / 37.1.

Two things follow:

- The lever engages and is a single-stream win (`serial=1.00` off, `mtp=1.00 tok_step=2.5-2.7` on).
- Plain lost about 27% between the two runs while serial structured decode did not move. Other
  traffic would have slowed both, so the cause is the test profile or the newer base image. It is
  not known which. §3a settles it and comes first.

The goal is structured output at 80 tok/s or more, at or above the plain rate. 57-60 is not there.

## 1. What is under test

PR #336 adds `ATLAS_STRICT_SPEC=1` (default off). With it, `response_format` requests
(`json_object`, `json_schema`, tool-or-JSON unions) keep speculative decoding on models whose
verify writes full logits, instead of decoding serially. The PR body reports roughly 24 tok/s
(structured, serial) against roughly 32 tok/s (plain) from one live observation.

CONTRIBUTING ("Opt-in switches: a neutral result doesn't merge") requires an A/B showing a win
before an opt-in arm merges, so this run decides the PR. Question 1 is answered (§0); it is
re-measured here on the production profile. Three questions:

1. Does the lever engage, and is structured output faster with it at one stream?
2. Is it still faster, or at least not slower, at 2 / 4 / 8 concurrent structured requests? Review
   found that strict sequences never join the batched verify (`grammar_rides_batch` in
   `scheduler/mtp_step/one_forward.rs`), so each one costs its own verify forward per step. This
   is inferred from code, not measured.
3. Is every reply still valid, and are temperature-0 replies unchanged?

## 2. Rules

Follow `.agents/skills/measurement-discipline/SKILL.md`. In short:

- One variable. Both arms use the same binary, the same serve flags and the same checkpoint. The
  only difference is `ATLAS_STRICT_SPEC` (unset against `1`, on both ranks).
- A fresh boot per arm. `ATLAS_GLM_STRICT_SPEC` stays unset in both.
- Record the fingerprint: build commit, checkpoint id and revision, full serve flags, the env
  difference, harness arguments, date.
- Three repetitions per case (the harness default). Label the result "single session, n=3".
- An idle pair. If other traffic reaches the pair during a run, discard that run and repeat it.
- Nothing committed may contain host names, IP addresses, interface names, user paths, profile or
  image file names, or credentials. Name the `ATLAS_*` variables and serve flags themselves.

## 3. Setup

- Build the head of `devin/strict-spec-full-logits` and note `git rev-parse HEAD`.
- Serve Qwen3.8-Flash-Next on the pair with the **production profile**, unchanged: the same serve
  flags and the same `ATLAS_*` variables as daily use. Not a reduced test profile.
- Add `ATLAS_MTP_ACCEPT_DEBUG=1` in both arms and keep the server log of each boot.
- List every `ATLAS_*` variable the profile sets, with its value, in the results file. Whether
  `ATLAS_MTP_DYNAMIC_DEPTH`, `ATLAS_MTP_DEEP_START` and `ATLAS_QWEN_COPY_DRAFTS` are set matters
  for reading the result.

### 3a. Baseline check, before any A/B

With the lever off, on the new build, measure plain decode speed the way the 85.9 / 88.8 figure
was measured (same script, same prompts, `max_tokens`, idle pair).

- Plain back at about 86-89: the earlier 63 came from the test profile. Continue with §4.
- Plain still near 63: stop. Boot the previous image (upstream `a523a07f`) with the same profile
  and measure again. If that reads 86-89, the newer base lost plain speed; report it (§6) with a
  plain `Done:` line from each image, and do not run the A/B until it is understood. Compare
  `tok_step` and `mean_na` between the two lines: they show whether plain speculates as deeply.

## 4. Steps

Harness: `scripts/dev/strict_spec_ab.py` (stdlib only). One run takes every case below through
three repetitions and writes a JSON record.

| case | request | checks |
|---|---|---|
| `schema@c1/c2/c4/c8` | `json_schema`, thinking off, temperature 0, N at once | speed, validity, identity |
| `object@c1` | `json_object`, thinking off, temperature 0 | speed, validity, identity |
| `plain@c1` | no `response_format` | control: must not move between arms |
| `sampled@c1/c4` | `json_schema`, temperature 0.8, fixed seed | validity, speed |
| `think@c1` | `json_schema`, `reasoning_effort: low` | validity across `</think>`, identity |
| `logprobs@c1` | `json_schema`, `top_logprobs: 3` | validity, identity (scored verify path) |
| `mixed@c4` | one `json_schema` beside three plain requests | validity, speed |

1. Boot with the lever **off**. Run:
   ```bash
   python3 scripts/dev/strict_spec_ab.py run --arm off --out off.json \
     --commit "$(git rev-parse --short HEAD)" --note "<serve flags>; ATLAS_STRICT_SPEC unset"
   python3 scripts/dev/strict_spec_ab.py run --arm off --out off2.json \
     --commit "$(git rev-parse --short HEAD)" --note "<serve flags>; ATLAS_STRICT_SPEC unset"
   python3 scripts/dev/tools_rfmt_probe.py --runs 5
   python3 scripts/dev/tools_rfmt_probe.py --runs 5 --thinking
   ```
   The second record is the noise floor: `compare off.json off2.json` shows how much the rates
   and the temperature-0 replies move with no change at all.
2. Reboot with `ATLAS_STRICT_SPEC=1` on both ranks. Run the same commands with `--arm on
   --out on.json` (one record is enough) and the two probe commands.
3. Compare:
   ```bash
   python3 scripts/dev/strict_spec_ab.py compare off.json off2.json   # noise floor
   python3 scripts/dev/strict_spec_ab.py compare off.json on.json     # the A/B
   ```
4. Prove engagement from the server logs. Each request logs a line
   `Done: <n> tokens (<reason>) <tok/s> tok/s, TTFT=..., serial=... mtp=... mean_na=... tok_step=...`.
   - Lever off, structured requests: expect `serial=1.00 mtp=0.00`.
   - Lever on, structured requests: expect `mtp` above 0 and `tok_step` above 1.
   - Plain requests: speculating in both arms.

   If structured requests still read `serial=1.00` with the lever on, the lever did not engage.
   Report that, with the boot log, instead of a speed result.

## 5. Reading the result

- **Validity (blocking).** With the lever on: zero invalid replies and zero request errors in every
  structured case, and both probe commands exit 0. For any failure keep the reply, the case name
  and the server log lines around it. A strict grammar violation ends the request with an error;
  that counts as a failure here.
- **Identity.** Temperature-0 cases should read `n/n` identical; this checkpoint's speculation is
  documented as exact (`docs/checkpoints/qwen3.8-flash-next-fp4-local-tp2.md` §3a). Judge a
  mismatch against the noise-floor comparison first. A mismatch that the noise floor does not
  show is a finding: report the first differing position the harness prints.
- **Control.** `plain@c1` must be within the noise floor between arms. If it moved, the two boots
  are not comparable; repeat both.
- **Speed.** Report the `on/off` ratio for every row, next to the noise floor.
  - `schema@c1` and `object@c1` are the headline.
  - `schema@c2/c4/c8` answer question 2. A ratio below 1 beyond the noise floor confirms the
    review concern.
  - `sampled` rows may not gain: a draft rarely matches a sampled pick. Report them as measured.
- Cross-check the harness rate (wall time, prefill included) against the `Done:` line rate of the
  same requests. If they disagree in direction, report both and do not pick one.

## 6. Reporting

Write `docs/notes/RESULTS_PR336_strict_spec.md` on this branch, commit it as
Robert Anderson <anderson.rw@gmail.com> and push to `devin/strict-spec-notes` on the fork. Nothing
goes to PR #336: no comment, and no push to `devin/strict-spec-full-logits`. The file contains:

- the §3a baseline numbers and which case applied;

- the fingerprint (§2), with the env difference between arms;
- the two tables printed by `compare` (noise floor and A/B);
- the probe results for both arms;
- two or three `Done:` lines per arm for a structured request, as engagement evidence;
- anything that failed, with its evidence.

Keep `off.json`, `off2.json`, `on.json` and both server logs until the PR is settled.

How the result maps to the PR, per CONTRIBUTING:

- A win at one stream with no loss under concurrency: the PR states the measurement, and either
  makes the lever the default or names the measurement that will flip it.
- A win at one stream but a loss under concurrency: report both. The likely follow-up is letting
  strict sequences ride the batched verify before the lever can become a default.
- Neutral or worse: the numbers go to the job issue and the PR closes.

## 7. Not in scope

Other models, the GLM arm (`ATLAS_GLM_STRICT_SPEC`), and the DFlash lane. The DFlash lane has no
throughput gate, so a model served with `--dflash` needs its own run before the lever is used there.
