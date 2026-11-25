# atlas-bug-factory

A Mistral agent (Agents SDK, `mistralai-agents`) that acts as a software-factory worker for
`maplepaladin73/atlas`: each invocation scans the repo for bugs and returns a prioritized report.
Bug triage prioritizes the two model areas the owner actively uses:

- **HY4 Flash** (`kernels/gb10/hy4-flash/`, `crates/spark-model/src/{model,layers}/hy4/`,
  `weight_loader/hy4.rs`, `crates/atlas-core/src/{config/hy4,config/parsers/hy4,hy4_reference}`,
  `crates/spark-server` hy4 tokenizer/tool-parser/grammar/preflight, `docs/HY4_TWO_SPARK.md`)
- **Ling 3.2 Air** (`bailing_moe_v3_vl`: `kernels/gb10/ling-3.2-air/`,
  `crates/spark-model/src/model/bailing_c1/`, `weight_loader/bailing/`, `layers/ling_mtp.rs`,
  `crates/atlas-core/src/config/parsers/bailing.rs`, `jinja-templates/ling-3.2-air.jinja`,
  `scripts_debug/ling_e2e_*`)

The exact, ordered prefix rules live in `priority_map.py` and are rendered into the agent's
instructions; the `classify_files` tool applies them, so P0 > P1 (shared infra) > P2 (rest).

## Layout

- `main.py` — the `agent` (model, instructions, `github_app` connector, tools), plus a local CLI runner.
- `priority_map.py` — priority tiers for bug triage (single source of truth) and the per-area P0 prefixes.
- `tests/test_agent.py` — offline tests with `agents.testing.ScriptedModel` (no API key needed).
- `Dockerfile` — managed-deployment worker image (`agents run-worker`).

## Scan job input

JSON or prose: `{repo, branch, mode, area, since, max_commits, max_files, file_issues, labels}`.
- mode `recent`: commits since `since` (default last 24h), default repo `maplepaladin73/atlas@main`.
- mode `prs`: open pull requests and their diffs.
- mode `area`: bounded deep dive on one P0 area — `area: "hy4-flash"` or `"ling-3.2-air"`,
  reading at most `max_files` (default 40) files under that area's prefixes.
- mode `full`: whole tree, P0 first.
- `file_issues=false` (default): read-only; the report is the execution result. `file_issues=true`
  additionally files high-confidence findings as labeled issues (`bug`, `atlas-bug-factory`,
  `model:hy4-flash` / `model:ling-3.2-air`).

## Run locally

```bash
cd agents/atlas-bug-factory
uv sync
uv run python main.py -- --mode area --area hy4-flash --max-files 40
```

The `github_app` connector must be connected on your account (it is, in this workspace).

## Test

```bash
uv sync --group dev
uv run pytest
```

## Deploy (Mistral-managed) and monitor

1. Push this directory to GitHub (it lives in `maplepaladin73/atlas`; build context is this
   directory: `build_directory=agents/atlas-bug-factory`, `dockerfile_path=Dockerfile`).
2. Deploy with `hardened=true` (the agent uses the `github_app` connector on behalf of the user
   who starts the run): deployment name `atlas-bug-factory`, revision = pushed commit SHA.
3. Run one scan: `execute_workflow` with workflow `atlas-bug-factory` and input
   `{"mode": "area", "area": "hy4-flash", "max_files": 40}`; read the report from the execution
   result/events.
4. For unattended monitoring later (schedules, GitHub webhooks): add a **workspace** credential
   on the `github_app` connector (or a workspace secret token) and run the connector with
   `run_as="deployment"`; OBO user credentials cannot run from schedules. Then create a schedule
   (e.g. daily, input `{"mode": "recent"}`) and/or a GitHub push/PR trigger
   (input `{"mode": "prs"}`). Keep `file_issues=false` until you trust the findings — otherwise the
   agent files issues automatically.

## Notes

- The agent never opens pull requests. When a scan finds a high-confidence bug, the report
  includes a concrete patch sketch; apply it on a branch and open the PR with `gh pr create`.
- File/commit budgets (`max_files`, `max_commits`) keep each run bounded so it does not consume
  unbounded resources; the report header states what was covered.
