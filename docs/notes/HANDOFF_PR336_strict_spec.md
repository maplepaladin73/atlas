# HANDOFF: `ATLAS_STRICT_SPEC` hardware A/B (PR #336 and the batched-verify follow-up)

**As of 2026-11-10.** Test plan for the agent with access to the two-GB10 pair.

## 0. Branches, and where to report

All three are on the fork. Check out and build **`devin/strict-spec-notes`**.

| branch | holds |
|---|---|
| `devin/strict-spec-notes` | this plan, the harness, the results, and the code under test |
| `devin/strict-spec-batched` | the code under test alone (`33977081`), stacked on the PR head |
| `devin/strict-spec-full-logits` | PR #336's head (`078cd56d`); do not push to it |

`git diff origin/devin/strict-spec-batched origin/devin/strict-spec-notes --stat` must list only
`docs/notes/` and `scripts/dev/strict_spec_ab.py`: the Rust code of the two is identical.

Results are committed to `devin/strict-spec-notes` (§6). Nothing goes to PR #336: no comment and
no push.

## 1. What is under test

1. **PR #336**: `ATLAS_STRICT_SPEC=1` (default off) lets `response_format` requests (`json_object`,
   `json_schema`, tool-or-JSON unions) keep speculative decoding on a full-logits verify instead
   of decoding serially.
2. **Follow-up (`33977081`)**: with the lever on, those requests also share the batched MTP verify.
   Before it, each structured request ran a verify forward of its own every step, and a lone one
   was held to 3 drafts. Unit tests only; it has not run on hardware.

The goal is structured output at 80 tok/s or more, at or above the plain rate.

### Measured so far

One single-stream run of PR #336 alone (reported by the local agent; n=4 per arm, temperature 0,
replies of about 220 tokens, a reduced test profile; it did not use this plan or its harness):

| decode tok/s | plain | `json_object` | `json_schema` |
|---|---|---|---|
| lever off | 63.0 | 37.4 | 37.5 |
| lever on | 63.2 | 57.2 | 59.7 |

The same morning, on the previous image (upstream `a523a07f`) with the production profile and an
idle pair, plain measured 85.9 / 88.8 and `json_object` 35.3 / 37.1.

- The lever engages (`serial=1.00` off, `mtp=1.00 tok_step=2.5-2.7` on) and wins at one stream.
- Plain lost about 27% between the two runs while serial structured decode did not move. Other
  traffic would have slowed both. Candidate causes: the test profile, the newer base image, or a
  different thermal cap on the GB10s (§3b). It is not known which. §3a and §3b come first.

### Open questions

1. Does plain reach 86-89 on the new build with the production profile (§3a)?
2. At 2 / 4 / 8 concurrent structured requests, is the lever a win, and does the batched verify
   beat one verify per request?
3. At one stream, does the batched verify add anything? Expected only if the profile sets
   `ATLAS_MTP_DYNAMIC_DEPTH=1`: a lone request's deep window runs through the batched verify.
   Without it, arms `serial` and `on` should read the same at one stream.
4. Is every reply valid in every arm, and are temperature-0 replies unchanged?

All expectations here are read from the code. None is measured.

## 2. Rules

Follow `.agents/skills/measurement-discipline/SKILL.md`. In short:

- One build, one profile, one checkpoint for every arm. Arms differ only in the variables of §4.
- A fresh boot per arm, the variables set on both ranks. `ATLAS_GLM_STRICT_SPEC` stays unset.
- An idle pair. If other traffic reaches the pair during a run, discard that run and repeat it.
- The same thermal cap on both GB10s for every arm, recorded (§3b). A cap changed between arms,
  or thermal throttling in one arm only, voids the comparison.
- Record the fingerprint: build commit, checkpoint id and revision, full serve flags, every
  `ATLAS_*` variable the profile sets, harness arguments, date.
- Three repetitions per case (the harness default). Label the result "single session, n=3".
- Nothing committed may contain host names, IP addresses, interface names, user paths, profile or
  image file names, or credentials. Name the `ATLAS_*` variables and serve flags themselves.

## 3. Setup

- Build the head of `devin/strict-spec-notes` and note `git rev-parse HEAD`.
- Serve Qwen3.8-Flash-Next on the pair with the **production profile**, unchanged: the same serve
  flags and the same `ATLAS_*` variables as daily use. Not a reduced test profile.
- Add `ATLAS_MTP_ACCEPT_DEBUG=1` in every arm and keep the server log of each boot.
- Note whether the profile sets `ATLAS_MTP_DYNAMIC_DEPTH`, `ATLAS_MTP_DEEP_START` and
  `ATLAS_QWEN_COPY_DRAFTS`. They decide how to read the one-stream rows.

### 3a. Baseline check, before any A/B

Record the thermal cap first (§3b, items 1-3). Then, with the lever off, on the new build, measure
plain decode speed the way the 85.9 / 88.8 figure was measured (same script, same prompts,
`max_tokens`, idle pair), with the §3b per-run readings.

- Plain back at about 86-89: the earlier 63 came from the test profile. Continue with §4.
- Plain still near 63: stop. Boot the previous image (upstream `a523a07f`) with the same profile
  and measure again. If that reads 86-89, the newer base lost plain speed; report it (§6) with a
  plain `Done:` line from each image, and do not run the A/B until it is understood. Compare
  `tok_step` and `mean_na` between the two lines: they show whether plain speculates as deeply.

### 3b. Thermal cap (Robert's request: it affects results)

The GB10s run under a cap that keeps them from overheating: a clock limit or a power limit set
through `nvidia-smi`. Report, for each of the two nodes:

1. **Which cap is in force and its value**: a locked clock range (MHz) or a power limit (W), and
   how it is applied (the command or service that sets it, and whether it survives a reboot).
   If nothing is set, say so.
2. **What the driver reports**, taken before the first arm:
   ```bash
   nvidia-smi --query-gpu=power.limit,enforced.power.limit,power.draw,temperature.gpu,clocks.current.sm,clocks.max.sm --format=csv
   nvidia-smi -q -d PERFORMANCE
   ```
   A field that reads `[N/A]` on GB10 is reported as `[N/A]`. If this driver refuses a field name,
   `nvidia-smi --help-query-gpu` lists the ones it has.
3. **Whether the cap was the same** when the 85.9 / 88.8 plain figure was measured and during the
   63 tok/s run, if that is known. If it differed, §3a is re-read with that in mind.

Then, for every arm (and for each §3a measurement), on both nodes:

- Save `nvidia-smi -q -d PERFORMANCE` immediately before and after the harness run. The
  `Clocks Event Reasons Counters` block counts microseconds per reason; the difference is how
  long each was asserted during the arm. Report the four deltas: `SW Power Capping`,
  `SW Thermal Slowdown`, `HW Thermal Slowdown`, `HW Power Braking`.
- Sample during the run:
  ```bash
  nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.current.sm,clocks_throttle_reasons.active --format=csv -l 5
  ```
  Report the median, minimum and maximum of clock, temperature and power per arm.
- Note the GPU temperature when the harness starts. Later arms start hotter; let the pair settle
  to about the first arm's starting temperature before each run, and report the values.

Do not change the cap for this test. Commit the values only, not raw dumps: `nvidia-smi -q`
output carries serial numbers and UUIDs.

## 4. Steps

Three arms, one boot each:

| arm | variables added to the profile | structured requests |
|---|---|---|
| `off` | none | decode serially |
| `serial` | `ATLAS_STRICT_SPEC=1`, `ATLAS_GRAMMAR_SERIAL_VERIFY=1` | speculate, one verify each (PR #336 as it is) |
| `on` | `ATLAS_STRICT_SPEC=1` | speculate, sharing the batched verify |

`ATLAS_GRAMMAR_SERIAL_VERIFY` is a presence switch: leave it unset in `off` and `on`, do not set
it to `0`.

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

1. Boot arm `off`. Run:
   ```bash
   python3 scripts/dev/strict_spec_ab.py run --arm off --out off.json \
     --commit "$(git rev-parse --short HEAD)" --note "<serve flags and ATLAS_* of this arm>"
   python3 scripts/dev/strict_spec_ab.py run --arm off --out off2.json \
     --commit "$(git rev-parse --short HEAD)" --note "<serve flags and ATLAS_* of this arm>"
   python3 scripts/dev/tools_rfmt_probe.py --runs 5
   python3 scripts/dev/tools_rfmt_probe.py --runs 5 --thinking
   ```
   The second record is the noise floor.
2. Boot arm `serial`. Run the harness once with `--arm serial --out serial.json`, then the two
   probe commands.
3. Boot arm `on`. Run the harness once with `--arm on --out on.json`, then the two probe commands.
4. Compare, baseline first:
   ```bash
   python3 scripts/dev/strict_spec_ab.py compare off.json off2.json     # noise floor
   python3 scripts/dev/strict_spec_ab.py compare off.json serial.json   # PR #336 as it is
   python3 scripts/dev/strict_spec_ab.py compare serial.json on.json    # the batched verify
   python3 scripts/dev/strict_spec_ab.py compare off.json on.json       # both together
   ```
5. Prove engagement from the server logs of each arm.
   - Each request logs `Done: <n> tokens (<reason>) <tok/s> tok/s, TTFT=..., serial=... mtp=...
     mean_na=... tok_step=... depth=k<rows> ...`.
     - `off`, structured: `serial=1.00 mtp=0.00`.
     - `serial` and `on`, structured: `mtp` above 0 and `tok_step` above 1.
   - Verifies log `MTP accept n=<sequences in the verify> k_drafts=<drafts> ...`.
     - `serial`, during `schema@c4/c8`: `n=1` lines only.
     - `on`, during `schema@c4/c8`: lines with `n` of 2 or more.
     - `on`, one stream, with `ATLAS_MTP_DYNAMIC_DEPTH=1`: `n=1` lines with `k_drafts` above 3
       once the request's depth is raised, and `depth=k5` or deeper on its `Done:` line.

   If an arm does not show its pattern, the arm did not engage. Report that, with the boot log,
   instead of a speed result.

## 5. Reading the result

- **Validity (blocking).** In `serial` and `on`: zero invalid replies and zero request errors in
  every structured case, and both probe commands exit 0. For any failure keep the reply, the case
  name and the server log lines around it. A strict grammar violation ends the request with an
  error; that counts as a failure here.
- **Identity.** Temperature-0 cases should read `n/n` identical between arms; this checkpoint's
  speculation is documented as exact (`docs/checkpoints/qwen3.8-flash-next-fp4-local-tp2.md` §3a).
  Judge a mismatch against the noise-floor comparison first. A mismatch the noise floor does not
  show is a finding: report the first differing position the harness prints.
- **Control.** `plain@c1` must be within the noise floor in all three arms. If it moved, the boots
  are not comparable; repeat them.
- **Thermals (§3b).** Arms compare only under the same cap, with similar clocks and similar
  throttle deltas.
  - `SW Power Capping` is the normal steady state of a GB10 (the repo's hardware monitor treats it
    as healthy), so a non-zero delta is expected. It should be of similar size in every arm.
  - A `SW Thermal Slowdown`, `HW Thermal Slowdown` or `HW Power Braking` delta in one arm and not
    the others means that arm was throttled: repeat it once the pair has cooled.
  - An arm that speculates draws more power than serial decode may, so a power cap can hold back
    the `serial` and `on` arms more than `off`. If the median clock or power differs between
    arms, report it beside the speed ratios; it is part of the result, not noise.
- **Speed.** Report every ratio next to the noise floor.
  - `schema@c1` and `object@c1`, `on` against `off`: the headline, to hold against the plain rate
    and the 80 tok/s goal.
  - `schema@c2/c4/c8`, `on` against `serial`: what the batched verify is worth. A ratio at or
    below 1 beyond the noise floor means it did not help; report it as measured.
  - `schema@c2/c4/c8`, `serial` against `off`: whether PR #336 alone loses under concurrency.
  - `sampled` rows may not gain: a draft rarely matches a sampled pick.
- Cross-check the harness rate (wall time, prefill included) against the `Done:` line rate of the
  same requests. If they disagree in direction, report both and do not pick one.

If `on` at one stream is still well under the plain rate, add to the report the `Done:` lines of
one plain and one structured request from arm `on` (`tok_step`, `mean_na`, `depth=`). They show
whether structured requests verify as deeply as plain ones.

## 6. Reporting

Write `docs/notes/RESULTS_PR336_strict_spec.md` on this branch, commit it as
Robert Anderson <anderson.rw@gmail.com> and push to `devin/strict-spec-notes` on the fork. Nothing
goes to PR #336: no comment, and no push to `devin/strict-spec-full-logits`. The file contains:

- the §3a baseline numbers and which case applied;
- the thermal cap in force on each node, and per arm the clock, temperature and power summary and
  the four throttle deltas (§3b);
- the fingerprint (§2), with the variables of each arm;
- the four tables printed by `compare`;
- the probe results for each arm;
- two or three `Done:` lines and `MTP accept` lines per arm, as engagement evidence;
- anything that failed, with its evidence.

Keep the JSON records and the server logs until the PR is settled.

How the result maps to the PR, per CONTRIBUTING ("Opt-in switches: a neutral result doesn't
merge"): an opt-in arm merges only on a measured win, and then either becomes the default or names
the measurement that will flip it. A neutral or worse result goes to the job issue and the PR
closes. Robert decides what goes upstream, including whether `33977081` joins PR #336.

## 7. Not in scope

Other models, the GLM arm (`ATLAS_GLM_STRICT_SPEC`), and the DFlash lane: a model served with
`--dflash` keeps one verify per structured request and needs its own run.
