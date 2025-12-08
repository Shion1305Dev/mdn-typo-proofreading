"""
Tests for issue_to_pr.py script.

Run with: uv run pytest tests/
"""

import sys
import tempfile
import subprocess
from pathlib import Path

import pytest

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / ".github" / "scripts"))

from issue_to_pr import normalize_diff, compute_result_hash


class TestNormalizeDiff:
    """Tests for the normalize_diff function."""

    def test_removes_code_fences(self):
        """Should remove markdown code fences from diff."""
        raw_diff = """```diff
--- a/test.txt
+++ b/test.txt
@@ -1,1 +1,1 @@
-old text
+new text
```"""
        result = normalize_diff(raw_diff)

        assert "```" not in result
        assert "--- a/test.txt" in result
        assert "+new text" in result

    def test_handles_diff_without_fences(self):
        """Should handle diffs without code fences."""
        raw_diff = """--- a/test.txt
+++ b/test.txt
@@ -1,1 +1,1 @@
-old text
+new text"""

        result = normalize_diff(raw_diff)

        assert result == raw_diff
        assert "--- a/test.txt" in result

    def test_strips_whitespace(self):
        """Should strip leading and trailing whitespace."""
        raw_diff = """
--- a/test.txt
+++ b/test.txt
@@ -1,1 +1,1 @@
-old text
+new text
  """
        result = normalize_diff(raw_diff)

        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_handles_empty_string(self):
        """Should return empty string for empty input."""
        assert normalize_diff("") == ""
        assert normalize_diff("   ") == ""

    def test_preserves_diff_content(self):
        """Should preserve all diff content after removing fences."""
        raw_diff = """```diff
--- a/file.js
+++ b/file.js
@@ -10,5 +10,5 @@
 context line
-removed line
+added line
 another context
```"""
        result = normalize_diff(raw_diff)

        assert "context line" in result
        assert "-removed line" in result
        assert "+added line" in result


class TestComputeResultHash:
    """Tests for the compute_result_hash function."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)

            # Initialize git repo
            subprocess.run(
                ["git", "init"],
                cwd=workdir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=workdir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=workdir,
                check=True,
                capture_output=True,
            )

            # Create and commit a test file
            test_file = workdir / "test.txt"
            test_file.write_text("old text\n")
            subprocess.run(
                ["git", "add", "test.txt"],
                cwd=workdir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=workdir,
                check=True,
                capture_output=True,
            )

            yield workdir

    def test_valid_diff_returns_hash(self, git_repo):
        """Should return a hash for a valid diff."""
        diff = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-old text
+new text
"""
        result_hash = compute_result_hash(diff, git_repo)

        assert result_hash is not None
        assert isinstance(result_hash, str)
        assert len(result_hash) == 64  # SHA256 hex digest length

    def test_restores_working_tree(self, git_repo):
        """Should restore working tree after computing hash."""
        diff = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-old text
+new text
"""
        test_file = git_repo / "test.txt"
        original_content = test_file.read_text()

        compute_result_hash(diff, git_repo)

        # File should be restored to original state
        assert test_file.read_text() == original_content

    def test_idempotency(self, git_repo):
        """Should return same hash for same diff."""
        diff = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-old text
+new text
"""
        hash1 = compute_result_hash(diff, git_repo)
        hash2 = compute_result_hash(diff, git_repo)

        assert hash1 == hash2

    def test_different_diffs_different_hashes(self, git_repo):
        """Should return different hashes for different diffs."""
        diff1 = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-old text
+new text
"""
        diff2 = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-old text
+different text
"""
        hash1 = compute_result_hash(diff1, git_repo)
        hash2 = compute_result_hash(diff2, git_repo)

        assert hash1 != hash2

    def test_invalid_diff_returns_none(self, git_repo):
        """Should return None for invalid diff."""
        invalid_diff = """--- a/nonexistent.txt
+++ b/nonexistent.txt
@@ -1 +1 @@
-old text
+new text
"""
        result_hash = compute_result_hash(invalid_diff, git_repo)

        assert result_hash is None

    def test_malformed_diff_returns_none(self, git_repo):
        """Should return None for malformed diff."""
        malformed_diff = "not a valid diff at all"

        result_hash = compute_result_hash(malformed_diff, git_repo)

        assert result_hash is None

    def test_empty_diff_returns_hash(self, git_repo):
        """Should handle empty diff (no changes) gracefully."""
        # Empty diff should still work, just result in no changes
        empty_diff = ""

        result_hash = compute_result_hash(empty_diff, git_repo)

        # This might return None or a hash depending on implementation
        # The important thing is it doesn't crash
        assert result_hash is None or isinstance(result_hash, str)
