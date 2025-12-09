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
MAX_ATTEMPTS = 4
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
    # Add line numbers to file content
    file_lines = file_content.splitlines()
    numbered_content = "\n".join(
        f"{i:4d} | {line}" for i, line in enumerate(file_lines, start=1)
    )

    lines = [
        "You are a Japanese typo correction assistant.",
        "",
        f"Target file: {file_path}",
        "",
        "=== CURRENT FILE CONTENT (with line numbers) ===",
        numbered_content,
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
    lines.append("1. The file content above is shown with line numbers for reference.")
    lines.append("2. Use the automatic suggestions as a starting point.")
    lines.append("3. When user feedback conflicts with suggestions, USER FEEDBACK WINS.")
    lines.append("4. Fix only Japanese typos and related textual issues.")
    lines.append("5. Do NOT modify unrelated text.")
    lines.append("")
    lines.append("6. Output your changes in JSON format with this structure:")
    lines.append('   {')
    lines.append('     "changes": [')
    lines.append('       {')
    lines.append('         "line_number": <number>,')
    lines.append('         "old_text": "original line content",')
    lines.append('         "new_text": "corrected line content"')
    lines.append('       }')
    lines.append('     ]')
    lines.append('   }')
    lines.append("")
    lines.append("7. Each change must specify:")
    lines.append("   - line_number: The line number from the numbered content above")
    lines.append("   - old_text: The EXACT current text of that line (without line number)")
    lines.append("   - new_text: The corrected text for that line")
    lines.append("")
    lines.append("8. Do NOT include markdown code fences like ```json")
    lines.append("9. Output ONLY valid JSON, nothing else.")
    lines.append("")
    lines.append("Generate the JSON now:")

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

    log(f"Processing {len(suggestions)} suggestions...")

    # Apply all suggestions
    for idx, sug in enumerate(suggestions, 1):
        original = sug.get("original", "")
        suggestion = sug.get("suggestion", "")

        log(f"Suggestion {idx}:")
        log(f"  Original text: '{original}'")
        log(f"  Suggested text: '{suggestion}'")

        if not original or not suggestion:
            log(f"  Skipping: empty original or suggestion", "WARN")
            continue

        if original in modified_content:
            log(f"  Found original text in file. Applying replacement...")
            modified_content = modified_content.replace(original, suggestion, 1)
            log(f"  Replacement applied successfully", "SUCCESS")
        else:
            log(f"  Original text NOT found in file", "WARN")
            # Show a snippet of the file to help debug
            log(f"  File content preview (first 500 chars):\n{file_content[:500]}")

    # If no changes were made, return empty diff
    if file_content == modified_content:
        log("No changes were made to the file content", "WARN")
        return ""

    log(f"File content modified. Changes made: {len(file_content)} -> {len(modified_content)} chars")

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


def apply_json_changes_to_content(
    file_content: str,
    json_response: str,
) -> Optional[str]:
    """
    Parse JSON changes from LLM and apply them to file content.

    Returns:
        Modified file content, or None if parsing fails
    """
    try:
        # Remove markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*\n", "", json_response)
        cleaned = re.sub(r"\n```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        # Parse JSON
        data = json.loads(cleaned)
        changes = data.get("changes", [])

        if not changes:
            log("No changes in JSON response", "WARN")
            return None

        log(f"Parsed {len(changes)} changes from JSON")

        # Split content into lines
        lines = file_content.splitlines()

        # Apply changes (sort by line number descending to avoid index shifts)
        changes_sorted = sorted(changes, key=lambda x: x.get("line_number", 0), reverse=True)

        for change in changes_sorted:
            line_num = change.get("line_number")
            old_text = change.get("old_text")
            new_text = change.get("new_text")

            if not all([line_num, old_text is not None, new_text is not None]):
                log(f"Skipping invalid change: {change}", "WARN")
                continue

            # Convert to 0-based index
            idx = line_num - 1

            if idx < 0 or idx >= len(lines):
                log(f"Line number {line_num} out of range (file has {len(lines)} lines)", "WARN")
                continue

            # Verify old text matches
            if lines[idx] != old_text:
                log(f"Line {line_num} mismatch. Expected: '{old_text}', Got: '{lines[idx]}'", "WARN")
                log(f"Skipping change at line {line_num} due to mismatch", "WARN")
                continue

            log(f"Applying change at line {line_num}: '{old_text}' -> '{new_text}'")
            lines[idx] = new_text

        return "\n".join(lines)

    except json.JSONDecodeError as e:
        log(f"Failed to parse JSON: {e}", "ERROR")
        log(f"Response was: {json_response[:500]}", "ERROR")
        return None
    except Exception as e:
        log(f"Error applying JSON changes: {e}", "ERROR")
        import traceback
        log(f"Traceback: {traceback.format_exc()}", "ERROR")
        return None


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
    file_path: str,
    file_content: str,
    workdir: Path,
) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Generate a single diff attempt using JSON-based approach.

    Returns:
        (attempt_number, diff_content or None, result_hash or None)
    """
    try:
        log(f"Attempt {attempt_number}/{MAX_ATTEMPTS} starting...")

        # Generate JSON response from Ollama
        raw_response = ollama.generate(prompt)
        log(f"Attempt {attempt_number}: Raw response length: {len(raw_response)} chars")
        log(f"--- JSON response (attempt {attempt_number}) ---")
        log(raw_response[:1000])  # Show first 1000 chars
        log(f"--- End of JSON response (attempt {attempt_number}) ---")

        # Parse JSON and apply changes
        modified_content = apply_json_changes_to_content(file_content, raw_response)
        if modified_content is None:
            log(f"Attempt {attempt_number}: Failed to parse or apply JSON changes", "WARN")
            return attempt_number, None, None

        # Check if any changes were made
        if modified_content == file_content:
            log(f"Attempt {attempt_number}: No changes were made", "WARN")
            return attempt_number, None, None

        # Generate diff from original and modified content
        diff = generate_diff_from_suggestions(
            file_path,
            file_content,
            [{"original": file_content, "suggestion": modified_content}]
        )

        # Actually, use difflib directly since we have both versions
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
            log(f"Attempt {attempt_number}: Generated empty diff", "WARN")
            return attempt_number, None, None

        diff = "\n".join(diff_lines) + "\n"

        log(f"--- Generated diff (attempt {attempt_number}) ---")
        log(diff)
        log(f"--- End of diff (attempt {attempt_number}) ---")

        # Check if diff applies and compute result hash
        result_hash = compute_result_hash(diff, workdir)
        if result_hash is None:
            log(f"Attempt {attempt_number}: Diff does not apply cleanly, skipping.", "WARN")
            return attempt_number, None, None

        log(f"Attempt {attempt_number}: Valid diff generated! Hash: {result_hash[:16]}...", "SUCCESS")

        return attempt_number, diff, result_hash

    except Exception as e:
        log(f"Attempt {attempt_number}: Error: {e}", "ERROR")
        import traceback
        log(f"Traceback: {traceback.format_exc()}", "ERROR")
        return attempt_number, None, None


def generate_stable_diff(
    ollama: OllamaClient,
    prompt: str,
    file_path: str,
    file_content: str,
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
            executor.submit(_generate_single_diff, attempt, ollama, prompt, file_path, file_content, workdir): attempt
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
            for prev_attempt, _, prev_hash in valid_results:
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
    doc_repo_pat = os.getenv("DOC_REPO_PAT", "")
    issue_repo_pat = os.getenv("ISSUE_REPO_PAT", "")
    docs_workdir = Path(os.getenv("DOCS_WORKDIR", ""))

    # Validate inputs
    if not all([issue_number, issue_repo, doc_repo, mongo_uri, ollama_model, doc_repo_pat, issue_repo_pat, docs_workdir]):
        log("Missing required environment variables.", "ERROR")
        sys.exit(1)

    if not docs_workdir.is_dir():
        log(f"DOCS_WORKDIR does not exist: {docs_workdir}", "ERROR")
        sys.exit(1)

    log(f"Processing issue #{issue_number} from {issue_repo}")

    # Initialize clients
    gh_doc = GitHubAPI(doc_repo_pat)  # For doc repo operations (PRs)
    gh_issue = GitHubAPI(issue_repo_pat)  # For issue repo operations (comments)
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
    comments = gh_issue.get_issue_comments(issue_repo, issue_number)
    log(f"Found {len(comments)} comments.")

    # Check for existing PR
    log("Checking for existing PR...")
    existing_pr = find_existing_pr_for_file(gh_doc, doc_repo, file_path)

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
            ollama, prompt, file_path, file_content, docs_workdir
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

        pr_data = gh_doc.create_pull_request(
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
    gh_issue.create_issue_comment(issue_repo, issue_number, comment_body)

    log("Done!", "SUCCESS")


if __name__ == "__main__":
    main()
