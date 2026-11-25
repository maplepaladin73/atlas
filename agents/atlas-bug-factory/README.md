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
instructions; `classify_files` applies them, so P0 > P1 (shared infra) > P2 (rest).

## Layout

- `main.py` — the `agent` (model, instructions, tools, secret), plus a local CLI runner.
- `github_tools.py` — stdlib-only GitHub REST tools (commits, PRs, files, trees, code/issue
  search, issue creation). Read-only unless a job sets `file_issues=true`.
- `priority_map.py` — priority tiers for bug triage (single source of truth).
- `tests/test_agent.py` — offline tests with `agents.testing.ScriptedModel` (no API key needed).
- `Dockerfile` — managed-deployment worker image (`agents run-worker`).

## Scan job input

JSON or prose: `{repo, branch, mode: "recent"|"prs"|"full", since, max_commits, file_issues, labels}`.
Defaults: `maplepaladin73/atlas@main`, mode `recent`, `since` = 24h ago, `file_issues=false`.
The agent's final message is the Markdown report (findings with severity, tier, file:line,
evidence, suggested fix, confidence; issue urls when `file_issues=true`).

## Run locally

```bash
cd agents/atlas-bug-factory
uv sync
export GITHUB_TOKEN=ghp_...   # fine-grained PAT, repo scope on maplepaladin73/atlas
uv run python main.py -- --mode recent --since 2026-11-24T00:00:00Z
uv run python main.py -- --mode prs
```

## Test

```bash
uv sync --group dev
uv run pytest
```

## Deploy (Mistral-managed) and monitor

1. Push this project to GitHub (it lives in `maplepaladin73/atlas`; build context is this
   directory: `build_directory=agents/atlas-bug-factory`, `dockerfile_path=Dockerfile`).
2. Create a workspace secret holding a GitHub token (e.g. `atlas-github-token`), then deploy with
   the binding `GITHUB_TOKEN=secret:workspace:atlas-github-token`
   (`mistral agents deploy`, or Studio's managed deployment). The agent uses no OBO connectors, so
   scheduled runs work: it authenticates with the bound token only.
3. Schedule it, e.g. daily at 08:00 UTC with input `{"mode": "recent"}` (window defaults to 24h),
   and/or add a GitHub webhook trigger on `push` / `pull_request` for `maplepaladin73/atlas` with
   input `{"mode": "recent", "since": "<event time>"}` or `{"mode": "prs"}`.
4. Read each run's report from the workflow execution result; set `file_issues=true` in the
   schedule/trigger input to auto-file high-confidence findings as labeled issues
   (`bug`, `atlas-bug-factory`, `model:hy4-flash` / `model:ling-3.2-air`).

## Notes

- The token needs read access to the repo; `issues:write` only when `file_issues=true`.
- Patches are truncated (`MAX_PATCH_CHARS`) and large files are refused (`MAX_FILE_CHARS`) to keep
  turns bounded; use `search_repo_code` for big files.
