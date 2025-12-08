#!/usr/bin/env python3
"""
Generate a diff from MongoDB suggestions + GitHub issue comments,
apply it to translated-content, and create/update a PR.
"""

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
from pymongo import MongoClient


# Constants
MONGO_COLLECTION = "japanese_typos_suggestions"
MAX_ATTEMPTS = 10
GITHUB_API_BASE = "https://api.github.com"


class GitHubAPI:
    """Helper for GitHub API interactions."""

    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        })

    def get_issue_comments(self, repo: str, issue_number: int) -> List[Dict[str, Any]]:
        """Fetch all comments for an issue."""
        url = f"{GITHUB_API_BASE}/repos/{repo}/issues/{issue_number}/comments"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def list_open_prs(self, repo: str) -> List[Dict[str, Any]]:
        """List all open pull requests."""
        url = f"{GITHUB_API_BASE}/repos/{repo}/pulls"
        params = {"state": "open", "per_page": 100}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_pr_files(self, repo: str, pr_number: int) -> List[str]:
        """Get the list of files changed in a PR."""
        url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}/files"
        response = self.session.get(url)
        response.raise_for_status()
        files = response.json()
        return [f["filename"] for f in files]

    def create_pull_request(
        self,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> Dict[str, Any]:
        """Create a new pull request."""
        url = f"{GITHUB_API_BASE}/repos/{repo}/pulls"
        data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base,
        }
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def create_issue_comment(self, repo: str, issue_number: int, body: str) -> None:
        """Post a comment on an issue."""
        url = f"{GITHUB_API_BASE}/repos/{repo}/issues/{issue_number}/comments"
        response = self.session.post(url, json={"body": body})
        response.raise_for_status()


class OllamaClient:
    """Helper for Ollama API interactions."""

    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint.rstrip("/")
        self.model = model

    def generate(self, prompt: str) -> str:
        """Call Ollama to generate text."""
        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


def fetch_mongo_suggestions(
    mongo_uri: str,
    db_name: str,
    issue_number: int,
) -> List[Dict[str, Any]]:
    """Fetch typo suggestions from MongoDB for a given issue number."""
    client = MongoClient(mongo_uri)
    collection = client[db_name][MONGO_COLLECTION]
    docs = list(collection.find({"issue_number": issue_number}))
    client.close()
    return docs


def build_prompt(
    file_path: str,
    file_content: str,
    suggestions: List[Dict[str, Any]],
    comments: List[Dict[str, Any]],
) -> str:
    """Construct the prompt for Ollama."""
    lines = [
        "You are a Japanese typo correction assistant.",
        "",
        f"Target file: {file_path}",
        "",
        "=== CURRENT FILE CONTENT ===",
        file_content,
        "",
        "=== AUTOMATIC SUGGESTIONS ===",
    ]

    for idx, sug in enumerate(suggestions, 1):
        lines.append(f"Suggestion {idx}:")
        lines.append(f"  Line hint: {sug.get('line_hint', '')}")
        lines.append(f"  Original: {sug.get('original', '')}")
        lines.append(f"  Suggestion: {sug.get('suggestion', '')}")
        lines.append(f"  Reason: {sug.get('reason', '')}")
        lines.append("")

    lines.append("=== USER FEEDBACK (from issue comments) ===")
    if comments:
        for comment in comments:
            # Filter out automated comments from the workflow itself
            body = comment.get("body", "")
            if body.startswith("Created PR:"):
                continue
            lines.append(f"Author: {comment.get('user', {}).get('login', 'unknown')}")
            lines.append(f"Created: {comment.get('created_at', '')}")
            lines.append(f"Comment: {body}")
            lines.append("")
    else:
        lines.append("(No user feedback provided)")
        lines.append("")

    lines.append("=== INSTRUCTIONS ===")
    lines.append("1. Use the automatic suggestions as a starting point.")
    lines.append("2. When user feedback conflicts with suggestions, USER FEEDBACK WINS.")
    lines.append("3. Fix only Japanese typos and related textual issues.")
    lines.append("4. Do NOT modify unrelated text.")
    lines.append("5. Output ONLY a unified diff (git diff / patch format) for the target file.")
    lines.append(f"6. Use the exact file path: {file_path}")
    lines.append("7. Do NOT include markdown code fences, explanations, or extra text.")
    lines.append("8. Output ONLY the raw diff content.")
    lines.append("")
    lines.append("Generate the diff now:")

    return "\n".join(lines)


def normalize_diff(raw_diff: str) -> str:
    """Normalize a diff string by removing code fences and extra whitespace."""
    # Remove markdown code fences
    cleaned = re.sub(r"```(?:diff)?\n?", "", raw_diff)
    cleaned = cleaned.strip()
    return cleaned


def check_diff_applies(diff_content: str, workdir: Path) -> bool:
    """Check if a diff can be applied cleanly."""
    try:
        result = subprocess.run(
            ["git", "apply", "--check"],
            input=diff_content.encode("utf-8"),
            cwd=workdir,
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def _generate_single_diff(
    attempt_number: int,
    ollama: OllamaClient,
    prompt: str,
    workdir: Path,
) -> Tuple[int, Optional[str]]:
    """
    Generate a single diff attempt.

    Returns:
        (attempt_number, diff_content or None)
    """
    try:
        print(f"  Attempt {attempt_number}/{MAX_ATTEMPTS} starting...")

        raw_response = ollama.generate(prompt)
        candidate = normalize_diff(raw_response)

        if not candidate:
            print(f"    Attempt {attempt_number}: Empty response, skipping.")
            return attempt_number, None

        # Check if diff applies
        if not check_diff_applies(candidate, workdir):
            print(f"    Attempt {attempt_number}: Diff does not apply cleanly, skipping.")
            return attempt_number, None

        print(f"    Attempt {attempt_number}: Valid diff generated.")
        print(f"    --- Diff content (attempt {attempt_number}) ---")
        print(candidate)
        print(f"    --- End of diff (attempt {attempt_number}) ---")

        return attempt_number, candidate

    except Exception as e:
        print(f"    Attempt {attempt_number}: Error: {e}")
        return attempt_number, None


def generate_stable_diff(
    ollama: OllamaClient,
    prompt: str,
    workdir: Path,
) -> Tuple[Optional[str], int, Optional[int], Optional[int]]:
    """
    Generate diffs in parallel with stability requirement: 2 identical diffs.

    Returns:
        (diff, total_attempts, stable_attempt_1, stable_attempt_2)
    """
    valid_diffs: List[Tuple[int, str]] = []  # List of (attempt_number, diff_content)

    # Run all attempts in parallel
    with ThreadPoolExecutor(max_workers=MAX_ATTEMPTS) as executor:
        # Submit all attempts
        futures = {
            executor.submit(_generate_single_diff, attempt, ollama, prompt, workdir): attempt
            for attempt in range(1, MAX_ATTEMPTS + 1)
        }

        # Process results as they complete
        for future in as_completed(futures):
            attempt_number, diff_content = future.result()

            if diff_content is None:
                continue

            # Check if this diff matches any previous valid diff
            for prev_attempt, prev_diff in valid_diffs:
                if diff_content == prev_diff:
                    # Found a match!
                    first_attempt = min(prev_attempt, attempt_number)
                    second_attempt = max(prev_attempt, attempt_number)
                    print(f"    STABLE: Attempts {first_attempt} and {second_attempt} produced identical diffs.")

                    # Cancel remaining futures to stop unnecessary work
                    for f in futures:
                        f.cancel()

                    # Return: (diff, total_attempts_when_stable_found, first_attempt, second_attempt)
                    return diff_content, second_attempt, first_attempt, second_attempt

            # No match found, add to list and continue
            valid_diffs.append((attempt_number, diff_content))
            print(f"    Attempt {attempt_number}: New unique diff, continuing...")

    # Exhausted all attempts without finding stability
    print(f"  Completed all {MAX_ATTEMPTS} attempts without finding matching diffs.")
    return None, MAX_ATTEMPTS, None, None


def apply_diff(diff_content: str, workdir: Path) -> bool:
    """Apply a diff to the working directory."""
    try:
        result = subprocess.run(
            ["git", "apply"],
            input=diff_content.encode("utf-8"),
            cwd=workdir,
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error applying diff: {e}")
        return False


def git_has_changes(workdir: Path) -> bool:
    """Check if there are uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def git_commit(
    workdir: Path,
    message: str,
    author_name: str = "GitHub Actions Bot",
    author_email: str = "actions@github.com",
) -> None:
    """Create a git commit."""
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = author_name
    env["GIT_AUTHOR_EMAIL"] = author_email
    env["GIT_COMMITTER_NAME"] = author_name
    env["GIT_COMMITTER_EMAIL"] = author_email

    subprocess.run(
        ["git", "add", "-A"],
        cwd=workdir,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=workdir,
        env=env,
        check=True,
    )


def git_push(workdir: Path, branch: str, force: bool = False) -> None:
    """Push changes to remote."""
    cmd = ["git", "push", "origin", branch]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, cwd=workdir, check=True)


def find_existing_pr_for_file(
    gh: GitHubAPI,
    repo: str,
    file_path: str,
) -> Optional[Dict[str, Any]]:
    """Find an existing open PR that modifies the given file."""
    prs = gh.list_open_prs(repo)
    for pr in prs:
        pr_number = pr["number"]
        files = gh.get_pr_files(repo, pr_number)
        if file_path in files:
            return pr
    return None


def setup_git_branch(
    workdir: Path,
    branch_name: str,
    base_branch: str,
    is_new: bool,
) -> None:
    """Setup the Git branch for committing."""
    if is_new:
        # Create new branch from base
        subprocess.run(
            ["git", "checkout", base_branch],
            cwd=workdir,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=workdir,
            check=True,
        )
    else:
        # Checkout existing branch
        subprocess.run(
            ["git", "fetch", "origin", branch_name],
            cwd=workdir,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", branch_name],
            cwd=workdir,
            check=True,
        )


def main() -> None:
    # Get environment variables
    issue_number = int(os.getenv("ISSUE_NUMBER", "0"))
    issue_repo = os.getenv("ISSUE_REPO", "")
    doc_repo = os.getenv("DOC_REPO", "")
    doc_repo_default_branch = os.getenv("DOC_REPO_DEFAULT_BRANCH", "main")
    mongo_uri = os.getenv("MONGO_URI", "")
    mongo_db = os.getenv("MONGO_DB", "mdn_proofreading")
    ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "")
    gh_pat = os.getenv("GH_PAT", "")
    docs_workdir = Path(os.getenv("DOCS_WORKDIR", ""))

    # Validate inputs
    if not all([issue_number, issue_repo, doc_repo, mongo_uri, ollama_model, gh_pat, docs_workdir]):
        print("Error: Missing required environment variables.")
        sys.exit(1)

    if not docs_workdir.is_dir():
        print(f"Error: DOCS_WORKDIR does not exist: {docs_workdir}")
        sys.exit(1)

    print(f"Processing issue #{issue_number} from {issue_repo}")

    # Initialize clients
    gh = GitHubAPI(gh_pat)
    ollama = OllamaClient(ollama_endpoint, ollama_model)

    # Fetch MongoDB suggestions
    print("Fetching MongoDB suggestions...")
    suggestions = fetch_mongo_suggestions(mongo_uri, mongo_db, issue_number)
    if not suggestions:
        print(f"No suggestions found for issue #{issue_number}")
        sys.exit(0)

    print(f"Found {len(suggestions)} suggestions.")

    # Extract and validate file path
    file_paths = list(set(s.get("file") for s in suggestions if s.get("file")))
    if len(file_paths) != 1:
        print(f"Error: Expected exactly 1 file path, found {len(file_paths)}: {file_paths}")
        sys.exit(1)

    file_path = file_paths[0]
    print(f"Target file: {file_path}")

    # Fetch issue comments
    print("Fetching issue comments...")
    comments = gh.get_issue_comments(issue_repo, issue_number)
    print(f"Found {len(comments)} comments.")

    # Check for existing PR
    print("Checking for existing PR...")
    existing_pr = find_existing_pr_for_file(gh, doc_repo, file_path)

    if existing_pr:
        print(f"Found existing PR #{existing_pr['number']}: {existing_pr['html_url']}")
        branch_name = existing_pr["head"]["ref"]
        is_new_pr = False
    else:
        print("No existing PR found. Will create a new one.")
        branch_name = f"shion/typo-scan-fix-{issue_number}"
        is_new_pr = True

    # Setup branch
    print(f"Setting up branch: {branch_name}")
    setup_git_branch(docs_workdir, branch_name, doc_repo_default_branch, is_new_pr)

    # Read current file content
    target_file = docs_workdir / file_path
    if not target_file.exists():
        print(f"Error: Target file does not exist: {target_file}")
        sys.exit(1)

    file_content = target_file.read_text(encoding="utf-8")

    # Build prompt
    print("Building prompt for Ollama...")
    prompt = build_prompt(file_path, file_content, suggestions, comments)

    # Generate stable diff
    print(f"Generating stable diff (max {MAX_ATTEMPTS} attempts)...")
    diff, total_attempts, stable_1, stable_2 = generate_stable_diff(
        ollama, prompt, docs_workdir
    )

    if diff is None:
        print(f"Failed to generate stable diff after {total_attempts} attempts.")
        sys.exit(1)

    print(f"Successfully generated stable diff (attempts {stable_1} and {stable_2} matched).")

    # Apply diff
    print("Applying diff...")
    if not apply_diff(diff, docs_workdir):
        print("Error: Failed to apply diff.")
        sys.exit(1)

    # Check if there are changes
    if not git_has_changes(docs_workdir):
        print("No changes after applying diff. Exiting without commit.")
        sys.exit(0)

    # Commit changes
    commit_message = f"✏️ Fix typos ({issue_repo}#{issue_number})"
    print(f"Committing changes: {commit_message}")
    git_commit(docs_workdir, commit_message)

    # Push changes
    print(f"Pushing to {branch_name}...")
    git_push(docs_workdir, branch_name)

    # Create or update PR
    pr_url = ""
    if is_new_pr:
        print("Creating new PR...")
        pr_title = f"{file_path}: typoを修正"
        pr_body = f"""Closes {issue_repo}#{issue_number}

Diff was generated with {total_attempts} attempts. Stable identical diffs were produced on attempts {stable_1} and {stable_2}.

Model: {ollama_model}

This PR was automatically generated from MongoDB suggestions and issue comments.
"""
        pr_data = gh.create_pull_request(
            repo=doc_repo,
            title=pr_title,
            body=pr_body,
            head=branch_name,
            base=doc_repo_default_branch,
        )
        pr_url = pr_data["html_url"]
        print(f"Created PR: {pr_url}")
    else:
        pr_url = existing_pr["html_url"]
        print(f"Updated existing PR: {pr_url}")

    # Post comment to issue
    print("Posting comment to issue...")
    comment_body = f"Created PR: {pr_url}"
    gh.create_issue_comment(issue_repo, issue_number, comment_body)

    print("Done!")


if __name__ == "__main__":
    main()
