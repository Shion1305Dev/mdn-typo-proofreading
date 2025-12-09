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

from issue_to_pr import (
    normalize_diff,
    compute_result_hash,
    generate_diff_from_suggestions,
    apply_json_changes_to_content,
)


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


class TestGenerateDiffFromSuggestions:
    """Tests for the generate_diff_from_suggestions function."""

    def test_generates_valid_unified_diff(self):
        """Should generate a valid unified diff."""
        file_path = "test.txt"
        file_content = "This is old text\nAnother line\n"
        suggestions = [
            {"original": "old", "suggestion": "new"},
        ]

        result = generate_diff_from_suggestions(file_path, file_content, suggestions)

        assert "--- a/test.txt" in result
        assert "+++ b/test.txt" in result
        assert "-This is old text" in result
        assert "+This is new text" in result

    def test_applies_multiple_suggestions(self):
        """Should apply multiple suggestions."""
        file_path = "test.txt"
        file_content = "First typo here\nSecond typo there\n"
        suggestions = [
            {"original": "typo", "suggestion": "correction"},
            {"original": "there", "suggestion": "here"},
        ]

        result = generate_diff_from_suggestions(file_path, file_content, suggestions)

        # Should contain both changes
        assert "correction" in result
        assert "here" in result

    def test_returns_empty_for_no_changes(self):
        """Should return empty string when no changes are made."""
        file_path = "test.txt"
        file_content = "No changes needed\n"
        suggestions = [
            {"original": "nonexistent", "suggestion": "something"},
        ]

        result = generate_diff_from_suggestions(file_path, file_content, suggestions)

        assert result == ""

    def test_handles_empty_suggestions(self):
        """Should handle empty suggestions list."""
        file_path = "test.txt"
        file_content = "Some content\n"
        suggestions = []

        result = generate_diff_from_suggestions(file_path, file_content, suggestions)

        assert result == ""

    def test_handles_japanese_text(self):
        """Should handle Japanese text properly."""
        file_path = "ja/test.md"
        file_content = "これはテストです。\nアクセスをアクセスを提供する。\n"
        suggestions = [
            {"original": "アクセスをアクセスを", "suggestion": "アクセスを"},
        ]

        result = generate_diff_from_suggestions(file_path, file_content, suggestions)

        assert "--- a/ja/test.md" in result
        assert "+++ b/ja/test.md" in result
        assert "-アクセスをアクセスを提供する。" in result
        assert "+アクセスを提供する。" in result

    def test_generated_diff_can_be_applied_by_git(self):
        """Should generate a diff that git can apply."""
        import tempfile
        import subprocess

        file_path = "test.txt"
        original_content = "Line 1\nOld text here\nLine 3\n"
        suggestions = [
            {"original": "Old text", "suggestion": "New text"},
        ]

        diff = generate_diff_from_suggestions(file_path, original_content, suggestions)

        # Create a temporary git repo
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=workdir, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=workdir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=workdir,
                check=True,
                capture_output=True,
            )

            # Create and commit the original file
            test_file = workdir / file_path
            test_file.write_text(original_content)
            subprocess.run(["git", "add", "."], cwd=workdir, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial"],
                cwd=workdir,
                check=True,
                capture_output=True,
            )

            # Try to apply the diff
            result = subprocess.run(
                ["git", "apply"],
                input=diff.encode("utf-8"),
                cwd=workdir,
                capture_output=True,
            )

            # Should succeed
            assert result.returncode == 0, f"git apply failed: {result.stderr.decode()}"

            # Verify the change was applied
            new_content = test_file.read_text()
            assert "New text here" in new_content
            assert "Old text" not in new_content


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


class TestApplyJsonChangesToContent:
    """Tests for the apply_json_changes_to_content function."""

    def test_parses_simple_json_change(self):
        """Should parse and apply a simple JSON change."""
        file_content = "Line 1\nLine 2\nLine 3\n"
        json_response = """{
            "changes": [
                {
                    "line_number": 2,
                    "old_text": "Line 2",
                    "new_text": "Modified Line 2"
                }
            ]
        }"""

        result = apply_json_changes_to_content(file_content, json_response)

        assert result is not None
        assert "Modified Line 2" in result
        assert "Line 1" in result
        assert "Line 3" in result
        # Original Line 2 should be replaced
        lines = result.splitlines()
        assert lines[1] == "Modified Line 2"

    def test_parses_multiple_changes(self):
        """Should parse and apply multiple JSON changes."""
        file_content = "First line\nSecond line\nThird line\n"
        json_response = """{
            "changes": [
                {
                    "line_number": 1,
                    "old_text": "First line",
                    "new_text": "Modified first line"
                },
                {
                    "line_number": 3,
                    "old_text": "Third line",
                    "new_text": "Modified third line"
                }
            ]
        }"""

        result = apply_json_changes_to_content(file_content, json_response)

        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "Modified first line"
        assert lines[1] == "Second line"
        assert lines[2] == "Modified third line"

    def test_removes_markdown_code_fences(self):
        """Should remove markdown code fences from JSON response."""
        file_content = "Test line\n"
        json_response = """```json
{
    "changes": [
        {
            "line_number": 1,
            "old_text": "Test line",
            "new_text": "Changed line"
        }
    ]
}
```"""

        result = apply_json_changes_to_content(file_content, json_response)

        assert result is not None
        assert "Changed line" in result

    def test_handles_japanese_text(self):
        """Should handle Japanese text in changes."""
        file_content = "これはテストです。\n日本語のテキストです。\n"
        json_response = """{
            "changes": [
                {
                    "line_number": 1,
                    "old_text": "これはテストです。",
                    "new_text": "これはサンプルです。"
                }
            ]
        }"""

        result = apply_json_changes_to_content(file_content, json_response)

        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "これはサンプルです。"
        assert lines[1] == "日本語のテキストです。"

    def test_returns_none_for_invalid_json(self):
        """Should return None for invalid JSON."""
        file_content = "Test line\n"
        invalid_json = "{ this is not valid json }"

        result = apply_json_changes_to_content(file_content, invalid_json)

        assert result is None

    def test_returns_none_for_empty_changes(self):
        """Should return None when changes array is empty."""
        file_content = "Test line\n"
        json_response = '{"changes": []}'

        result = apply_json_changes_to_content(file_content, json_response)

        assert result is None

    def test_skips_invalid_change_entry(self):
        """Should skip changes with missing required fields."""
        file_content = "Line 1\nLine 2\nLine 3\n"
        json_response = """{
            "changes": [
                {
                    "line_number": 1,
                    "old_text": "Line 1",
                    "new_text": "Modified Line 1"
                },
                {
                    "line_number": 2
                }
            ]
        }"""

        result = apply_json_changes_to_content(file_content, json_response)

        # Should apply the valid change and skip the invalid one
        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "Modified Line 1"
        assert lines[1] == "Line 2"  # Should remain unchanged

    def test_handles_line_number_out_of_range(self):
        """Should skip changes with line numbers out of range."""
        file_content = "Line 1\nLine 2\n"
        json_response = """{
            "changes": [
                {
                    "line_number": 1,
                    "old_text": "Line 1",
                    "new_text": "Modified Line 1"
                },
                {
                    "line_number": 10,
                    "old_text": "Does not exist",
                    "new_text": "Something"
                }
            ]
        }"""

        result = apply_json_changes_to_content(file_content, json_response)

        # Should apply the valid change and skip the out-of-range one
        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "Modified Line 1"
        assert len(lines) == 2

    def test_applies_changes_in_correct_order(self):
        """Should apply changes in reverse line order to avoid index shifts."""
        file_content = "Line 1\nLine 2\nLine 3\n"
        # Give changes in forward order, function should handle reverse processing
        json_response = """{
            "changes": [
                {
                    "line_number": 1,
                    "old_text": "Line 1",
                    "new_text": "New Line 1"
                },
                {
                    "line_number": 2,
                    "old_text": "Line 2",
                    "new_text": "New Line 2"
                },
                {
                    "line_number": 3,
                    "old_text": "Line 3",
                    "new_text": "New Line 3"
                }
            ]
        }"""

        result = apply_json_changes_to_content(file_content, json_response)

        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "New Line 1"
        assert lines[1] == "New Line 2"
        assert lines[2] == "New Line 3"

    def test_skips_change_on_old_text_mismatch(self):
        """Should skip change when old_text doesn't match exactly."""
        file_content = "Line 1\nLine 2\nLine 3\n"
        json_response = """{
            "changes": [
                {
                    "line_number": 2,
                    "old_text": "Wrong old text",
                    "new_text": "New Line 2"
                }
            ]
        }"""

        # Should skip the change due to mismatch
        result = apply_json_changes_to_content(file_content, json_response)

        assert result is not None
        lines = result.splitlines()
        # Line 2 should remain unchanged
        assert lines[1] == "Line 2"

    def test_applies_only_matching_changes_when_mixed(self):
        """Should apply matching changes and skip mismatching ones."""
        file_content = "Line 1\nLine 2\nLine 3\n"
        json_response = """{
            "changes": [
                {
                    "line_number": 1,
                    "old_text": "Line 1",
                    "new_text": "Modified Line 1"
                },
                {
                    "line_number": 2,
                    "old_text": "Wrong text",
                    "new_text": "Should not apply"
                },
                {
                    "line_number": 3,
                    "old_text": "Line 3",
                    "new_text": "Modified Line 3"
                }
            ]
        }"""

        result = apply_json_changes_to_content(file_content, json_response)

        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "Modified Line 1"
        assert lines[1] == "Line 2"  # Should remain unchanged
        assert lines[2] == "Modified Line 3"

    def test_handles_code_fence_variations(self):
        """Should handle different code fence styles."""
        file_content = "Test line\n"

        # Test with ```json prefix
        json_response1 = """```json
{"changes": [{"line_number": 1, "old_text": "Test line", "new_text": "Changed"}]}
```"""
        result1 = apply_json_changes_to_content(file_content, json_response1)
        assert result1 is not None
        assert "Changed" in result1

        # Test with ``` only
        json_response2 = """```
{"changes": [{"line_number": 1, "old_text": "Test line", "new_text": "Changed"}]}
```"""
        result2 = apply_json_changes_to_content(file_content, json_response2)
        assert result2 is not None
        assert "Changed" in result2
