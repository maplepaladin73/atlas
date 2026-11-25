import argparse
import asyncio
import json

from mistralai import agents

from github_tools import (
    create_issue,
    get_commit_files,
    get_pull_request_files,
    list_commits,
    list_open_pull_requests,
    list_repo_tree,
    now_iso,
    read_repo_file,
    search_repo_code,
    search_repo_issues,
)
from priority_map import classify, render_rules

INSTRUCTIONS = f"""You are Atlas Bug Factory, the automated bug-hunting worker ("software factory") for the
Atlas inference engine repository (default repo: maplepaladin73/atlas, default branch: main).

Each invocation is one scan job. The request may be a JSON object or prose. Recognized fields:
- repo: "owner/name" (default maplepaladin73/atlas)
- branch: default "main"
- mode: "recent" (commits since `since`), "prs" (open pull requests), or "full" (whole tree, P0 areas first)
- since: ISO-8601 timestamp; default is 24 hours before the current time
- max_commits: cap on commits to review (default 30)
- file_issues: boolean, default false. When true, file high-confidence findings as GitHub issues.
- labels: issue labels for file_issues mode, default ["bug", "atlas-bug-factory"]

Procedure:
1. Call now_iso() once and compute the default `since` window from it.
2. Collect the change set for the mode:
   - "recent": list_commits(repo, branch, since, max_commits), then get_commit_files for each commit.
   - "prs": list_open_pull_requests, then get_pull_request_files for each open PR.
   - "full": list_repo_tree, then classify_files on all paths and read P0 files directly with read_repo_file.
3. Triage every changed file with classify_files. Tiers: P0 first, then P1, then P2.
4. Review P0 and P1 changes deeply and in that order. When a diff hunk is ambiguous, read the
   surrounding file with read_repo_file. Then skim P2 changes for obvious defects only. Spend the
   review budget on P0.
5. Bug classes that have burned Atlas before, in rough priority order:
   - numeric/quantization edge cases: NVFP4/MXFP4/FP8 dequant math, BF16 paged KV routed into an
     FP8 kernel, silent NaN/Inf, scale-factor handling;
   - MoE routing and expert-index arithmetic (off-by-one, wrong expert counts, shared-expert leaks);
   - paged-KV cache and index arithmetic in CUDA kernels (.cu/.cuh): out-of-bounds shared/global
     memory, races, missing syncs, wrong launch bounds;
   - speculative decoding verify/rollback races and draft-token bookkeeping;
   - unsafe unwrap()/panic paths in Rust on malformed configs or weights (PCND: fail fast, never
     silently fall back);
   - protocol drift between the OpenAI and Anthropic API surfaces (a fix on one often needs the
     matching change on the other);
   - chat-template/tool-call token mismatches (different <tool_call> spellings, <think> handling,
     thinking_budget enforcement) in jinja-templates/;
   - docs/CLI drift: flags and commands in docs/ vs the actual binary.
6. Before reporting a finding as new, search_repo_issues to check for an existing issue/PR.
7. Your final message IS the report; nothing else you output matters. Structure it as:
   - Header: repo@branch, time window, commits/PRs scanned, files per tier (P0/P1/P2 counts).
   - Findings, sorted by severity (critical/high/medium/low), then by tier. Each finding:
     title; severity; tier and area; file:line; short evidence quote (<=8 lines); why it is a bug;
     suggested fix; confidence (high/medium/low); existing-issue reference if any.
   - A "No findings" section when a tier is clean.
   - If file_issues=true: for each high-confidence finding call create_issue (labels plus a tier
     label like "model:hy4-flash" or "model:ling-3.2-air"), then list the created issue urls.
   - "Next scan suggestions": what to deep-dive next full scan.

Rules:
- Read-only by default. Never create, modify, or delete anything on GitHub unless this job set
  file_issues=true.
- Never print or echo credentials. If a tool fails with 401/403, stop and report the credential
  problem instead of retrying.
- Keep quoted patches short. Do not dump whole files into the report.
- Do not invent file:line references; only cite locations you actually saw in a patch or file.

PRIORITY MAP (classify_files applies exactly these rules; first matching prefix wins):
{render_rules()}
Unknown paths are P2 (other).
"""


@agents.tool
def classify_files(paths: list[str]) -> dict[str, dict[str, str]]:
    """Classify repo paths into bug-triage priority tiers: P0 = HY4 Flash and Ling 3.2 Air model areas (review first), P1 = shared infrastructure they depend on, P2 = everything else. Returns {path: {tier, area}}."""
    return {path: classify(path) for path in paths}


agent = agents.Agent(
    model="mistral-large-4",
    name="atlas-bug-factory",
    instructions=INSTRUCTIONS,
    tools=[
        now_iso,
        list_commits,
        get_commit_files,
        list_open_pull_requests,
        get_pull_request_files,
        read_repo_file,
        list_repo_tree,
        search_repo_code,
        search_repo_issues,
        create_issue,
        classify_files,
    ],
    secrets={"GITHUB_TOKEN": agents.MistralSecret(worker_env=True)},
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
    parser.add_argument("--mode", choices=["recent", "prs", "full"], default="recent")
    parser.add_argument("--since", default="")
    parser.add_argument("--max-commits", type=int, default=30)
    parser.add_argument("--file-issues", action="store_true")
    args = parser.parse_args()
    job = {
        "repo": args.repo,
        "branch": args.branch,
        "mode": args.mode,
        "since": args.since,
        "max_commits": args.max_commits,
        "file_issues": args.file_issues,
    }
    print(asyncio.run(run_scan(job)))


if __name__ == "__main__":
    main()
