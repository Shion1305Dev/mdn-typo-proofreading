#!/usr/bin/env python3
"""
Generate a diff from MongoDB suggestions + GitHub issue comments,
apply it to translated-content, and create/update a PR.
"""

import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
from pymongo import MongoClient


# Constants
MONGO_COLLECTION = "japanese_typos_suggestions"
MAX_ATTEMPTS = 2
GITHUB_API_BASE = "https://api.github.com"


def log(message: str, level: str = "INFO") -> None:
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}", flush=True)


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
    lines.append("5. Output ONLY a unified diff (git diff format) for the target file.")
    lines.append(f"6. Use the exact file path: {file_path}")
    lines.append("7. The diff MUST follow this format:")
    lines.append("   --- a/path/to/file")
    lines.append("   +++ b/path/to/file")
    lines.append("   @@ -start,count +start,count @@ optional section header")
    lines.append("    context line (starts with space)")
    lines.append("   -removed line (starts with minus)")
    lines.append("   +added line (starts with plus)")
    lines.append("    context line (starts with space)")
    lines.append("8. Include at least 3 lines of context before and after each change.")
    lines.append("9. Do NOT include markdown code fences, explanations, or extra text.")
    lines.append("10. Output ONLY the raw diff content.")
    lines.append("")
    lines.append("Generate the unified diff now:")

    return "\n".join(lines)


def generate_diff_from_suggestions(
    file_path: str,
    file_content: str,
    suggestions: List[Dict[str, Any]],
) -> str:
    """
    Generate a proper unified diff by applying suggestions to file content.
    This bypasses LLM formatting issues and creates a valid git diff.
    """
    # Start with original content
    modified_content = file_content

    # Apply all suggestions
    for sug in suggestions:
        original = sug.get("original", "")
        suggestion = sug.get("suggestion", "")

        if original and suggestion and original in modified_content:
            # Replace first occurrence
            modified_content = modified_content.replace(original, suggestion, 1)

    # If no changes were made, return empty diff
    if file_content == modified_content:
        return ""

    # Generate unified diff
    # Split into lines without keeping ends first, let difflib handle line endings
    original_lines = file_content.splitlines(keepends=False)
    modified_lines = modified_content.splitlines(keepends=False)

    diff_lines = list(difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    ))

    if not diff_lines:
        return ""

    # Join with newlines to create proper diff format
    return "\n".join(diff_lines) + "\n"


def normalize_diff(raw_diff: str) -> str:
    """Normalize a diff string by removing code fences and extra whitespace."""
    if not raw_diff:
        return ""

    # Remove markdown code fences (both opening and closing)
    # Handle ```diff or ``` at the start and ``` at the end
    cleaned = re.sub(r"^```(?:diff)?\s*\n", "", raw_diff)
    cleaned = re.sub(r"\n```\s*$", "", cleaned)
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


def compute_result_hash(diff_content: str, workdir: Path) -> Optional[str]:
    """
    Apply diff temporarily and compute hash of resulting state.
    Restores working tree after computing hash.

    Returns:
        Hash of the resulting state, or None if application fails.
    """
    try:
        # Apply the diff (use bytes for input, don't use text=True for apply)
        result = subprocess.run(
            ["git", "apply"],
            input=diff_content.encode("utf-8"),
            cwd=workdir,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            log(f"Git apply failed with return code {result.returncode}", "WARN")
            if result.stderr:
                log(f"Git apply stderr: {result.stderr.decode('utf-8', errors='replace')}", "WARN")
            if result.stdout:
                log(f"Git apply stdout: {result.stdout.decode('utf-8', errors='replace')}", "WARN")
            return None

        # Get hash of the resulting working tree state
        # Using git diff to capture all changes, then hash that
        diff_result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if diff_result.returncode != 0:
            log(f"Git diff HEAD failed with return code {diff_result.returncode}", "ERROR")
            return None

        # Compute hash of the diff output (this represents the final state)
        result_hash = hashlib.sha256(diff_result.stdout.encode("utf-8")).hexdigest()

        return result_hash

    except Exception as e:
        log(f"Error computing result hash: {e}", "ERROR")
        import traceback
        log(f"Traceback: {traceback.format_exc()}", "ERROR")
        return None
    finally:
        # Always restore working tree to original state
        try:
            subprocess.run(
                ["git", "restore", "."],
                cwd=workdir,
                capture_output=True,
                timeout=30,
            )
        except Exception as restore_error:
            log(f"Warning: Failed to restore working tree: {restore_error}", "WARN")


def _generate_single_diff(
    attempt_number: int,
    ollama: OllamaClient,
    prompt: str,
    workdir: Path,
) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Generate a single diff attempt.

    Returns:
        (attempt_number, diff_content or None, result_hash or None)
    """
    try:
        log(f"Attempt {attempt_number}/{MAX_ATTEMPTS} starting...")

        # Generate response from Ollama
        raw_response = ollama.generate(prompt)
        log(f"Attempt {attempt_number}: Raw response length: {len(raw_response)} chars")

        # Normalize the diff
        candidate = normalize_diff(raw_response)
        log(f"Attempt {attempt_number}: Normalized diff length: {len(candidate)} chars")

        if not candidate:
            log(f"Attempt {attempt_number}: Empty normalized diff, skipping.", "WARN")
            log(f"Raw response was: {raw_response[:200]}...")
            return attempt_number, None, None

        # Show the normalized diff
        log(f"--- Normalized diff (attempt {attempt_number}) ---")
        log(candidate)
        log(f"--- End of normalized diff (attempt {attempt_number}) ---")

        # Check if diff applies and compute result hash
        result_hash = compute_result_hash(candidate, workdir)
        if result_hash is None:
            log(f"Attempt {attempt_number}: Diff does not apply cleanly, skipping.", "WARN")
            return attempt_number, None, None

        log(f"Attempt {attempt_number}: Valid diff generated! Hash: {result_hash[:16]}...", "SUCCESS")

        return attempt_number, candidate, result_hash

    except Exception as e:
        log(f"Attempt {attempt_number}: Error: {e}", "ERROR")
        import traceback
        log(f"Traceback: {traceback.format_exc()}", "ERROR")
        return attempt_number, None, None


def generate_stable_diff(
    ollama: OllamaClient,
    prompt: str,
    workdir: Path,
) -> Tuple[Optional[str], int, Optional[int], Optional[int]]:
    """
    Generate diffs in parallel with stability requirement: 2 identical results.

    Returns:
        (diff, total_attempts, stable_attempt_1, stable_attempt_2)
    """
    log(f"Prompt: {prompt}")
    # List of (attempt_number, diff_content, result_hash)
    valid_results: List[Tuple[int, str, str]] = []

    # Run all attempts in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all attempts
        futures = {
            executor.submit(_generate_single_diff, attempt, ollama, prompt, workdir): attempt
            for attempt in range(1, MAX_ATTEMPTS + 1)
        }

        # Process results as they complete
        for future in as_completed(futures):
            attempt_number, diff_content, result_hash = future.result()
            log(f"Attempt {attempt_number} completed.")

            if diff_content is None or result_hash is None:
                log(f"Attempt {attempt_number}: Skipping (invalid diff or hash)", "WARN")
                continue

            log("=" * 60)
            log(f"Valid result from attempt {attempt_number}:")
            log(f"Diff Content ({len(diff_content)} chars):")
            log(diff_content)
            log(f"Result Hash: {result_hash}")
            log("=" * 60)

            # Check if this result matches any previous result
            for prev_attempt, _prev_diff, prev_hash in valid_results:
                if result_hash == prev_hash:
                    # Found a match!
                    first_attempt = min(prev_attempt, attempt_number)
                    second_attempt = max(prev_attempt, attempt_number)
                    log(f"STABLE: Attempts {first_attempt} and {second_attempt} produced identical results.", "SUCCESS")

                    # Cancel remaining futures to stop unnecessary work
                    for f in futures:
                        f.cancel()

                    # Return: (diff, total_attempts_when_stable_found, first_attempt, second_attempt)
                    return diff_content, second_attempt, first_attempt, second_attempt

            # No match found, add to list and continue
            valid_results.append((attempt_number, diff_content, result_hash))
            log(f"Attempt {attempt_number}: New unique result, continuing...")

    # Exhausted all attempts without finding stability
    log(f"Completed all {MAX_ATTEMPTS} attempts without finding matching results.", "ERROR")
    return None, MAX_ATTEMPTS, None, None


def apply_diff(diff_content: str, workdir: Path) -> bool:
    """Apply a diff to the working directory."""
    try:
        log("Attempting to apply diff with git apply...")
        log(f"Working directory: {workdir}")
        log(f"Diff content preview (first 500 chars):\n{diff_content[:500]}")

        result = subprocess.run(
            ["git", "apply", "--verbose"],
            input=diff_content.encode("utf-8"),
            cwd=workdir,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            log(f"Failed to apply diff. Git apply returned code {result.returncode}", "ERROR")
            if result.stderr:
                log(f"Git apply stderr: {result.stderr.decode('utf-8', errors='replace')}", "ERROR")
            if result.stdout:
                log(f"Git apply stdout: {result.stdout.decode('utf-8', errors='replace')}", "ERROR")

            # Try with --check to get more diagnostic info
            log("Running git apply --check for diagnostics...")
            check_result = subprocess.run(
                ["git", "apply", "--check", "--verbose"],
                input=diff_content.encode("utf-8"),
                cwd=workdir,
                capture_output=True,
                timeout=30,
            )
            if check_result.stderr:
                log(f"Git apply --check stderr: {check_result.stderr.decode('utf-8', errors='replace')}", "ERROR")

            return False

        log("Diff applied successfully!", "SUCCESS")
        return True
    except Exception as e:
        log(f"Exception while applying diff: {e}", "ERROR")
        import traceback
        log(f"Traceback: {traceback.format_exc()}", "ERROR")
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
) -> None:
    """Create a git commit using default git configuration."""
    subprocess.run(
        ["git", "add", "-A"],
        cwd=workdir,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=workdir,
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
        log("Missing required environment variables.", "ERROR")
        sys.exit(1)

    if not docs_workdir.is_dir():
        log(f"DOCS_WORKDIR does not exist: {docs_workdir}", "ERROR")
        sys.exit(1)

    log(f"Processing issue #{issue_number} from {issue_repo}")

    # Initialize clients
    gh = GitHubAPI(gh_pat)
    ollama = OllamaClient(ollama_endpoint, ollama_model)

    # Fetch MongoDB suggestions
    log("Fetching MongoDB suggestions...")
    suggestions = fetch_mongo_suggestions(mongo_uri, mongo_db, issue_number)
    if not suggestions:
        log(f"No suggestions found for issue #{issue_number}", "WARN")
        sys.exit(0)

    log(f"Found {len(suggestions)} suggestions.")

    # Extract and validate file path
    file_paths = list(set(s.get("file") for s in suggestions if s.get("file")))
    if len(file_paths) != 1:
        log(f"Expected exactly 1 file path, found {len(file_paths)}: {file_paths}", "ERROR")
        sys.exit(1)

    file_path = file_paths[0]
    log(f"Target file: {file_path}")

    # Fetch issue comments
    log("Fetching issue comments...")
    comments = gh.get_issue_comments(issue_repo, issue_number)
    log(f"Found {len(comments)} comments.")

    # Check for existing PR
    log("Checking for existing PR...")
    existing_pr = find_existing_pr_for_file(gh, doc_repo, file_path)

    if existing_pr:
        log(f"Found existing PR #{existing_pr['number']}: {existing_pr['html_url']}")
        branch_name = existing_pr["head"]["ref"]
        is_new_pr = False
    else:
        log("No existing PR found. Will create a new one.")
        branch_name = f"shion/typo-scan-fix-{issue_number}"
        is_new_pr = True

    # Setup branch
    log(f"Setting up branch: {branch_name}")
    setup_git_branch(docs_workdir, branch_name, doc_repo_default_branch, is_new_pr)

    # Read current file content
    target_file = docs_workdir / file_path
    if not target_file.exists():
        log(f"Target file does not exist: {target_file}", "ERROR")
        sys.exit(1)

    file_content = target_file.read_text(encoding="utf-8")

    # Try generating diff directly from suggestions first
    log("Generating diff from suggestions...")
    diff = generate_diff_from_suggestions(file_path, file_content, suggestions)

    if diff:
        log(f"Generated diff ({len(diff)} chars):")
        log("=" * 60)
        log(diff)
        log("=" * 60)

    total_attempts = 0
    stable_1 = None
    stable_2 = None

    # If we have user comments, use Ollama to incorporate them
    if comments and any(not c.get("body", "").startswith("Created PR:") for c in comments):
        log("User comments found. Using Ollama to incorporate feedback...")

        # Build prompt
        log("Building prompt for Ollama...")
        prompt = build_prompt(file_path, file_content, suggestions, comments)

        # Generate stable diff
        log(f"Generating stable diff (max {MAX_ATTEMPTS} attempts)...")
        ollama_diff, total_attempts, stable_1, stable_2 = generate_stable_diff(
            ollama, prompt, docs_workdir
        )

        if ollama_diff is not None:
            log(f"Successfully generated stable diff with Ollama (attempts {stable_1} and {stable_2} matched).", "SUCCESS")
            diff = ollama_diff
        else:
            log(f"Ollama failed after {total_attempts} attempts. Using diff from suggestions only.", "WARN")
            if not diff:
                log("No valid diff available.", "ERROR")
                sys.exit(1)
    else:
        log("No user comments found. Using diff generated from suggestions.")
        if not diff:
            log("No changes to make based on suggestions.", "WARN")
            sys.exit(0)

    log(f"Final diff length: {len(diff)} chars")

    # Apply diff
    log("Applying diff...")
    if not apply_diff(diff, docs_workdir):
        log("Failed to apply diff.", "ERROR")
        sys.exit(1)

    # Check if there are changes
    if not git_has_changes(docs_workdir):
        log("No changes after applying diff. Exiting without commit.", "WARN")
        sys.exit(0)

    # Commit changes
    commit_message = f"✏️ Fix typos ({issue_repo}#{issue_number})"
    log(f"Committing changes: {commit_message}")
    git_commit(docs_workdir, commit_message)

    # Push changes
    log(f"Pushing to {branch_name}...")
    git_push(docs_workdir, branch_name)

    # Create or update PR
    pr_url = ""
    if is_new_pr:
        log("Creating new PR...")
        pr_title = f"{file_path}: typoを修正"

        # Build PR body based on how diff was generated
        pr_body_lines = [f"Closes {issue_repo}#{issue_number}", ""]

        if total_attempts > 0:
            pr_body_lines.append(f"Diff was generated with {total_attempts} Ollama attempts. Stable identical diffs were produced on attempts {stable_1} and {stable_2}.")
            pr_body_lines.append(f"Model: {ollama_model}")
        else:
            pr_body_lines.append("Diff was generated directly from MongoDB suggestions using Python's difflib.")

        pr_body_lines.append("")
        pr_body_lines.append("This PR was automatically generated from MongoDB suggestions" + (" and issue comments." if comments else "."))

        pr_body = "\n".join(pr_body_lines)

        pr_data = gh.create_pull_request(
            repo=doc_repo,
            title=pr_title,
            body=pr_body,
            head=branch_name,
            base=doc_repo_default_branch,
        )
        pr_url = pr_data["html_url"]
        log(f"Created PR: {pr_url}", "SUCCESS")
    else:
        pr_url = existing_pr["html_url"]
        log(f"Updated existing PR: {pr_url}", "SUCCESS")

    # Post comment to issue
    log("Posting comment to issue...")
    comment_body = f"Created PR: {pr_url}"
    gh.create_issue_comment(issue_repo, issue_number, comment_body)

    log("Done!", "SUCCESS")


if __name__ == "__main__":
    main()
