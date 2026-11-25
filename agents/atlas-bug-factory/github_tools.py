import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from mistralai import agents

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "maplepaladin73/atlas"
MAX_PATCH_CHARS = 20000
MAX_FILE_CHARS = 120000
MAX_TREE_ENTRIES = 4000
SOURCE_SUFFIXES = (
    ".rs",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".cpp",
    ".toml",
    ".jinja",
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".json",
    ".sh",
)
SKIP_TREE_PREFIXES = ("vendor/", "target/", "test_data/", "book/output/")


def _token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise agents.ToolError(
            "GITHUB_TOKEN is not set. Locally, export it; on a deployed worker, bind a "
            "workspace secret as GITHUB_TOKEN (secret:workspace:<name>)."
        )
    return token


def _repo(repo: str) -> str:
    return repo or os.environ.get("ATLAS_WATCH_REPO", DEFAULT_REPO)


def _request(method: str, path: str, params: dict | None = None, body: dict | None = None) -> object:
    url = GITHUB_API + path
    if params:
        clean = {k: v for k, v in params.items() if v not in (None, "")}
        if clean:
            url += "?" + urllib.parse.urlencode(clean)
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {_token()}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "atlas-bug-factory",
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        raise agents.ToolError(f"GitHub API error {exc.code} on {method} {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise agents.ToolError(f"GitHub API unreachable on {method} {path}: {exc.reason}") from exc
    return json.loads(payload) if payload else None


def _truncate_patch(patch: str, budget: int) -> tuple[str, int]:
    if len(patch) <= budget:
        return patch, len(patch)
    return patch[:budget] + "\n... [patch truncated]", budget


@agents.tool
def now_iso() -> str:
    """Current UTC time in ISO-8601. Use it to resolve relative time windows such as "the last 24 hours"."""
    return agents.now().isoformat()


@agents.tool
def list_commits(repo: str = "", branch: str = "main", since: str = "", per_page: int = 30) -> list[dict]:
    """List commits on a branch, newest first. `since` is an optional ISO-8601 timestamp filter. Returns sha, first-line message, author, date."""
    items = _request(
        "GET",
        f"/repos/{_repo(repo)}/commits",
        {"sha": branch, "since": since, "per_page": min(max(per_page, 1), 100)},
    )
    return [
        {
            "sha": c["sha"],
            "message": c["commit"]["message"].splitlines()[0] if c["commit"]["message"] else "",
            "author": c["commit"]["author"]["name"],
            "date": c["commit"]["author"]["date"],
        }
        for c in items
    ]


@agents.tool
def get_commit_files(repo: str = "", sha: str = "", max_files: int = 100) -> list[dict]:
    """Files changed by one commit, with unified diff patches (truncated to a total budget)."""
    data = _request("GET", f"/repos/{_repo(repo)}/commits/{sha}")
    out = []
    total = 0
    for f in data.get("files", [])[:max_files]:
        patch, used = _truncate_patch(f.get("patch") or "", MAX_PATCH_CHARS - total)
        total += used
        out.append(
            {
                "filename": f["filename"],
                "status": f["status"],
                "additions": f["additions"],
                "deletions": f["deletions"],
                "patch": patch,
            }
        )
    return out


@agents.tool
def list_open_pull_requests(repo: str = "", per_page: int = 20) -> list[dict]:
    """Open pull requests: number, title, author, head branch, url, updated_at."""
    items = _request("GET", f"/repos/{_repo(repo)}/pulls", {"state": "open", "per_page": min(max(per_page, 1), 100)})
    return [
        {
            "number": p["number"],
            "title": p["title"],
            "author": p["user"]["login"],
            "head": p["head"]["ref"],
            "url": p["html_url"],
            "updated_at": p["updated_at"],
        }
        for p in items
    ]


@agents.tool
def get_pull_request_files(repo: str = "", pull_number: int = 0, max_files: int = 100) -> list[dict]:
    """Files changed by a pull request, with unified diff patches (truncated to a total budget)."""
    out = []
    total = 0
    page = 1
    while page <= 3 and len(out) < max_files:
        items = _request(
            "GET",
            f"/repos/{_repo(repo)}/pulls/{pull_number}/files",
            {"per_page": 100, "page": page},
        )
        if not items:
            break
        for f in items:
            if len(out) >= max_files:
                break
            patch, used = _truncate_patch(f.get("patch") or "", MAX_PATCH_CHARS - total)
            total += used
            out.append(
                {
                    "filename": f["filename"],
                    "status": f["status"],
                    "additions": f["additions"],
                    "deletions": f["deletions"],
                    "patch": patch,
                }
            )
        page += 1
    return out


@agents.tool
def read_repo_file(repo: str = "", path: str = "", ref: str = "main") -> str:
    """Read one file from the repo at a ref. Files over ~120k chars are refused; use search_repo_code for those."""
    data = _request("GET", f"/repos/{_repo(repo)}/contents/{path.lstrip('/')}", {"ref": ref})
    if not isinstance(data, dict) or data.get("encoding") != "base64" or "content" not in data:
        kind = data.get("type") if isinstance(data, dict) else "unknown"
        raise agents.ToolError(f"{path} is not a regular file (type={kind}).")
    text = base64.b64decode(data["content"]).decode(errors="replace")
    if len(text) > MAX_FILE_CHARS:
        raise agents.ToolError(
            f"{path} is {len(text)} chars, too large to read whole; narrow with search_repo_code."
        )
    return text


@agents.tool
def list_repo_tree(repo: str = "", ref: str = "main") -> list[str]:
    """All source/config file paths in the repo at a ref, for full-tree scans."""
    data = _request("GET", f"/repos/{_repo(repo)}/git/trees/{ref}", {"recursive": 1})
    paths = [t["path"] for t in data.get("tree", []) if t.get("type") == "blob"]
    paths = [
        p
        for p in paths
        if p.endswith(SOURCE_SUFFIXES) and not p.startswith(SKIP_TREE_PREFIXES)
    ]
    return paths[:MAX_TREE_ENTRIES]


@agents.tool
def search_repo_code(repo: str = "", query: str = "", per_page: int = 20) -> list[dict]:
    """GitHub code search inside the repo, e.g. query='hy4_mla path:kernels/gb10/hy4-flash'. The repo: qualifier is added automatically."""
    q = query if "repo:" in query else f"repo:{_repo(repo)} {query}"
    data = _request("GET", "/search/code", {"q": q, "per_page": min(max(per_page, 1), 100)})
    return [{"path": i["path"], "name": i["name"]} for i in data.get("items", [])]


@agents.tool
def search_repo_issues(repo: str = "", query: str = "", per_page: int = 20) -> list[dict]:
    """Search issues and PRs in the repo, to check whether a finding is already reported. Natural language plus optional qualifiers like 'is:issue label:bug'."""
    q = query if "repo:" in query else f"repo:{_repo(repo)} {query}"
    data = _request("GET", "/search/issues", {"q": q, "per_page": min(max(per_page, 1), 100)})
    return [
        {
            "number": i["number"],
            "title": i["title"],
            "state": i["state"],
            "url": i["html_url"],
            "labels": [label["name"] for label in i.get("labels", [])],
        }
        for i in data.get("items", [])
    ]


@agents.tool
def create_issue(repo: str = "", title: str = "", body: str = "", labels: list[str] | None = None) -> dict:
    """Create a GitHub issue and return its number and url. Only call this when the scan job explicitly set file_issues=true; otherwise stay read-only."""
    data = _request(
        "POST",
        f"/repos/{_repo(repo)}/issues",
        body={"title": title, "body": body, "labels": labels or []},
    )
    return {"number": data["number"], "url": data["html_url"]}
