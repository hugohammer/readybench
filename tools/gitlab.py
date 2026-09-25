"""Start ReadyBench pipelines on GitLab and pull their results back.

  python tools/gitlab.py run --task probe_runner
  python tools/gitlab.py run --task some_script --runner xl-cloud-runner --args "--shard 3"
  python tools/gitlab.py status [pipeline-id]
  python tools/gitlab.py fetch <pipeline-id>      # -> outputs/runs/<pipeline-id>/<job>/
  python tools/gitlab.py log <job-id> [--tail N]
  python tools/gitlab.py cancel <pipeline-id>

Needs tools/gitlab.env (gitignored, copy from gitlab.env.example) with a
personal access token that has the `api` scope (`read_api` suffices for
status/fetch).
"""

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "tools" / "gitlab.env"


def load_env() -> dict:
    if not ENV_FILE.exists():
        sys.exit(f"Missing {ENV_FILE}. Copy tools/gitlab.env.example and fill in the token.")
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"')
    return env


class GitLab:
    def __init__(self, env: dict):
        self.base = env["GITLAB_URL"].rstrip("/") + "/api/v4"
        self.project = urllib.parse.quote(env["PROJECT"], safe="")
        self.token = env["GITLAB_TOKEN"]

    def request(self, method: str, path: str, body: dict | None = None, raw: bool = False):
        url = f"{self.base}/projects/{self.project}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "PRIVATE-TOKEN": self.token, "Content-Type": "application/json"})
        for attempt in range(5):  # ride out short network drops (e.g. DNS on WSL)
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = resp.read()
                break
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
        return payload if raw else json.loads(payload)


def cmd_run(gl: GitLab, args) -> None:
    variables = [{"key": "TASK", "value": args.task}, {"key": "RUNNER_TAG", "value": args.runner},
                 {"key": "TASK_ARGS", "value": args.args}]
    pipeline = gl.request("POST", "/pipeline", {"ref": args.ref, "variables": variables})
    print(f"started pipeline {pipeline['id']}: {pipeline['web_url']}")


def cmd_status(gl: GitLab, args) -> None:
    if args.pipeline is None:
        for p in gl.request("GET", "/pipelines?per_page=10"):
            print(f"{p['id']:>8}  {p['status']:<10} {p['created_at'][:16]}  {p['source']}")
        return
    for job in gl.request("GET", f"/pipelines/{args.pipeline}/jobs?per_page=100"):
        runner = (job.get("runner") or {}).get("description", "-")
        print(f"{job['id']:>8}  {job['status']:<10} {job['name']:<40} {job.get('duration') or 0:>8.0f}s  {runner}")


def cmd_fetch(gl: GitLab, args) -> None:
    target = ROOT / "outputs" / "runs" / str(args.pipeline)
    for job in gl.request("GET", f"/pipelines/{args.pipeline}/jobs?per_page=100"):
        if not job.get("artifacts_file"):
            print(f"skip {job['name']} ({job['status']}, no artifacts)")
            continue
        blob = gl.request("GET", f"/jobs/{job['id']}/artifacts", raw=True)
        dest = target / job["name"].replace(" ", "_").replace(":", "_").replace("/", "_")
        zipfile.ZipFile(io.BytesIO(blob)).extractall(dest)
        print(f"{job['name']}: {len(blob) / 1024**2:.1f} MB -> {dest.relative_to(ROOT)}")


def cmd_cancel(gl: GitLab, args) -> None:
    for job in gl.request("GET", f"/pipelines/{args.pipeline}/jobs?per_page=100"):
        if job["status"] in ("created", "pending", "running"):
            gl.request("POST", f"/jobs/{job['id']}/cancel")
            print(f"canceled job {job['id']} ({job['name']})")


def cmd_log(gl: GitLab, args) -> None:
    text = gl.request("GET", f"/jobs/{args.job}/trace", raw=True).decode("utf-8", "replace")
    lines = text.splitlines()
    print("\n".join(lines[-args.tail:] if args.tail else lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--task", required=True)
    run.add_argument("--runner", default="xl-cloud-runner")
    run.add_argument("--args", default="")
    run.add_argument("--ref", default="main")
    status = sub.add_parser("status")
    status.add_argument("pipeline", nargs="?", type=int)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("pipeline", type=int)
    cancel = sub.add_parser("cancel")
    cancel.add_argument("pipeline", type=int)
    log = sub.add_parser("log")
    log.add_argument("job", type=int)
    log.add_argument("--tail", type=int, default=60, help="last N lines (0 = all)")
    args = parser.parse_args()

    gl = GitLab(load_env())
    {"run": cmd_run, "status": cmd_status, "fetch": cmd_fetch, "log": cmd_log, "cancel": cmd_cancel}[args.command](gl, args)


if __name__ == "__main__":
    main()
