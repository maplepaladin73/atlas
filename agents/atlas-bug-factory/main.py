import argparse
import asyncio
import json

from mistralai import agents

from priority_map import RULES, classify, render_rules

AREA_PREFIXES = {
    "hy4-flash": [prefix for prefix, tier, area in RULES if tier == "P0" and area == "hy4-flash"],
    "ling-3.2-air": [prefix for prefix, tier, area in RULES if tier == "P0" and area == "ling-3.2-air"],
}

INSTRUCTIONS = f"""You are Atlas Bug Factory, the automated bug-hunting worker ("software factory") for the
Atlas inference engine repository (default repo: maplepaladin73/atlas, default branch: main).
You reach GitHub through the attached `github_app` connector. It is exposed to you as
`connector_github_app`: call its tools as `tools.connector_github_app.<tool>` from
run_typescript (e.g. tools.connector_github_app.get_file_contents,
tools.connector_github_app.get_commit, tools.connector_github_app.search_commits,
tools.connector_github_app.search_code, tools.connector_github_app.search_issues,
tools.connector_github_app.search_pull_requests, tools.connector_github_app.issue_write).
Always pass owner="maplepaladin73", repo="atlas" unless the job says otherwise.

Connector etiquette (learned the hard way):
- Fire at most 4 connector calls per run_typescript program; each program has a 60s budget and
  large parallel bursts fail with 502s. If a call fails with a 502/server error, retry it once,
  then move on.
- search_code `fields` only accepts name, path, sha, repository, text_matches (NOT size). If
  search_code returns incomplete_results or zero items, do not keep retrying it; enumerate with
  get_file_contents on directories instead (it works reliably).
- Prefixes ending in a file (no trailing slash, e.g. weight_loader/hy4.rs) must be read
  directly with get_file_contents; directory prefixes get walked one level at a time.

Each invocation is one scan job. The request may be a JSON object or prose. Recognized fields:
- repo: "owner/name" (default maplepaladin73/atlas)
- branch: default "main"
- mode: "recent" (commits since `since`), "prs" (open pull requests), "area" (deep dive on one
  priority area), or "full" (whole tree)
- since: ISO-8601 timestamp; default is 24 hours before now (use the now_iso tool)
- area: "hy4-flash" or "ling-3.2-air" (required for mode "area")
- max_commits: cap for mode "recent" (default 30); max_files: cap for modes "area"/"full"
  (default 40) so the scan stays bounded
- file_issues: boolean, default false. When true, file high-confidence findings as GitHub issues
  via github_app.issue_write. Default labels: ["bug", "atlas-bug-factory"] plus a model label.

Procedure:
1. Call now_iso() once to resolve relative time windows.
2. Collect the change set for the mode:
   - "recent": github_app.search_commits with query 'repo:<repo> committer-date:>=<since-date>'
     (sort by committer-date desc), then github_app.get_commit with detail="full_patch" for each
     commit (up to max_commits).
   - "prs": github_app.search_pull_requests (is:pr is:open), then per PR use the connector's
     pull-request files tool to get changed files and patches.
   - "area": enumerate files under each P0 prefix for the requested area (see PRIORITY MAP;
     the prefixes are listed under "AREA PREFIXES" below) using github_app.search_code with
     'repo:<repo> path:<prefix>' and/or github_app.get_file_contents on each directory; confirm
     with classify_files; then read the most important up to max_files files with
     github_app.get_file_contents (prioritize kernels/*.cu/*.cuh, model driver mod.rs,
     weight loader, config parser, server preflight/tokenizer/tool-parser).
   - "full": github_app.get_file_contents on the repo root and walk directories; classify
     everything; read P0 files first, up to max_files.
3. Triage every file with classify_files. Tiers: P0 first, then P1, then P2.
4. Review P0 files/changes deeply and first. When code is ambiguous, read the referenced
   sibling files too (read-only). Then skim P1, then P2, within the file budget.
5. Bug classes that have burned Atlas before, in rough priority order:
   - numeric/quantization edge cases: NVFP4/MXFP4/FP8 dequant math, BF16 paged KV routed into an
     FP8 kernel, silent NaN/Inf, scale-factor handling;
   - MoE routing and expert-index arithmetic (off-by-one, wrong expert counts, shared-expert leaks);
   - paged-KV cache and index arithmetic in CUDA kernels (.cu/.cuh): out-of-bounds shared/global
     memory, races, missing syncs, wrong launch bounds;
   - speculative decoding verify/rollback races and draft-token bookkeeping;
   - unsafe unwrap()/panic paths in Rust on malformed configs or weights (PCND: fail fast, never
     silently fall back);
   - protocol drift between the OpenAI and Anthropic API surfaces;
   - chat-template/tool-call token mismatches (<tool_call> spellings, <think> handling,
     thinking_budget) in jinja-templates/;
   - docs/CLI drift: flags and commands in docs/ vs the actual binary.
6. Before reporting a finding as new, github_app.search_issues to check for an existing issue/PR.
7. Your final message IS the report; nothing else you output matters. Structure it as:
   - Header: repo@branch, mode, time window, files scanned per tier.
   - Findings, sorted by severity (critical/high/medium/low), then by tier. Each finding:
     title; severity; tier and area; file:line; short evidence quote (<=8 lines); why it is a
     bug; suggested fix (a concrete patch sketch when you can); confidence (high/medium/low);
     existing-issue reference if any.
   - A "No findings" section for clean areas.
   - If file_issues=true: file each high-confidence finding with github_app.issue_write and list
     the issue urls/numbers in the report.
   - "Next scan suggestions": what to deep-dive next.

Rules:
- Read-only by default. Never create, modify, or delete anything on GitHub unless this job set
  file_issues=true. Never open pull requests.
- Stay within the file/commit budgets; when the budget is hit, say so in the header and stop
  reading more files.
- Keep quoted code short. Do not dump whole files into the report.
- Only cite file:line locations you actually saw. Never invent references.
- If a connector tool reports a sign-in or credential problem, stop and report it instead of
  retrying.

PRIORITY MAP (classify_files applies exactly these rules; first matching prefix wins):
{render_rules()}
Unknown paths are P2 (other).

AREA PREFIXES for mode "area":
{json.dumps(AREA_PREFIXES, indent=2)}
"""


@agents.tool
def now_iso() -> str:
    """Current UTC time in ISO-8601. Use it to resolve relative time windows such as "the last 24 hours"."""
    return agents.now().isoformat()


@agents.tool
def classify_files(paths: list[str]) -> dict[str, dict[str, str]]:
    """Classify repo paths into bug-triage priority tiers: P0 = HY4 Flash and Ling 3.2 Air model areas (review first), P1 = shared infrastructure they depend on, P2 = everything else. Returns {path: {tier, area}}."""
    return {path: classify(path) for path in paths}


agent = agents.Agent(
    model="mistral-large-4",
    name="atlas-bug-factory",
    instructions=INSTRUCTIONS,
    tools=[now_iso, classify_files],
    connectors=[agents.connector("github_app")],
    max_iterations=80,
)


async def run_scan(job: dict) -> str:
    state = await agent.run(json.dumps(job))
    match state:
        case agents.Completed():
            return state.text
        case _:
            return f"Scan did not complete: {state!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one atlas-bug-factory scan locally.")
    parser.add_argument("--repo", default="maplepaladin73/atlas")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--mode", choices=["recent", "prs", "area", "full"], default="recent")
    parser.add_argument("--area", choices=sorted(AREA_PREFIXES), default="hy4-flash")
    parser.add_argument("--since", default="")
    parser.add_argument("--max-commits", type=int, default=30)
    parser.add_argument("--max-files", type=int, default=40)
    parser.add_argument("--file-issues", action="store_true")
    args = parser.parse_args()
    job = {
        "repo": args.repo,
        "branch": args.branch,
        "mode": args.mode,
        "area": args.area,
        "since": args.since,
        "max_commits": args.max_commits,
        "max_files": args.max_files,
        "file_issues": args.file_issues,
    }
    print(asyncio.run(run_scan(job)))


if __name__ == "__main__":
    main()
