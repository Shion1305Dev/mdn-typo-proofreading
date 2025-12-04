import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import requests
from dotenv import load_dotenv
from pymongo import MongoClient

GITHUB_API_BASE = "https://api.github.com"
TARGET_OWNER = "Shion1305"
TARGET_REPO = "mdn-typo-proofreading"
DEFAULT_SOURCE_REPO_URL = "https://github.com/mdn/translated-content"


def _find_line_and_url(
    repo_root: Path,
    rel_path: str,
    original: str,
    source_repo_url: str,
    context_radius: int = 2,
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    file_path = repo_root / rel_path
    if not file_path.is_file():
        return None, None, None

    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="utf-8", errors="replace")

    lines = text.splitlines()
    matches: List[int] = [
        idx for idx, line in enumerate(lines, start=1) if original in line
    ]

    if not matches:
        return None, None, None

    line_no = matches[0]
    line_text = lines[line_no - 1]
    start = max(1, line_no - context_radius)
    end = min(len(lines), line_no + context_radius)

    url = f"{source_repo_url}/blob/main/{rel_path}#L{start}-L{end}"
    return line_no, url, line_text


def _labels_for_doc(doc: Mapping[str, object]) -> Set[str]:
    labels: Set[str] = set()

    level = doc.get("level")
    if isinstance(level, str):
        labels.add(f"level-{level.lower()}")

    is_japanese_typo = doc.get("is_japanese_typo")
    if is_japanese_typo is True:
        labels.add("japanese-typo")
    elif is_japanese_typo is False:
        labels.add("non-japanese-typo")

    labels.add("auto-generated")

    return labels


def _ensure_labels(
    session: requests.Session,
    owner: str,
    repo: str,
    labels: Iterable[str],
) -> None:
    for name in labels:
        if not name:
            continue
        resp = session.post(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/labels",
            json={"name": name, "color": "ededed"},
        )
        if resp.status_code in (201, 422):
            # 201: created, 422: already exists
            continue
        resp.raise_for_status()


def _build_issue_title(doc: Mapping[str, object]) -> str:
    file_path = str(doc.get("file") or "")
    original = str(doc.get("original") or "")
    suggestion = str(doc.get("suggestion") or "")
    return f"{file_path}: {original} \u2192 {suggestion}"


def _build_issue_body(
    doc: Mapping[str, object],
    url: Optional[str],
    line_no: Optional[int],
    line_text: Optional[str],
    original: str,
    suggestion: str,
) -> str:
    parts: List[str] = []

    parts.append("### Metadata")
    parts.append("")
    parts.append("| key | value |")
    parts.append("| --- | ----- |")
    meta_fields = [
        "file",
        "chunk_index",
        "line_hint",
        "original",
        "suggestion",
        "reason",
        "level",
        "is_japanese_typo",
    ]
    for key in meta_fields:
        if key in doc:
            parts.append(f"| {key} | `{doc[key]}` |")

    if line_no is not None:
        parts.append(f"| line | `{line_no}` |")

    parts.append("")
    parts.append("### Location")
    parts.append("")
    if url:
        parts.append(f"- Source: {url}")
    else:
        parts.append("- Source: (could not determine location)")

    # If we couldn't locate the line in the source file, don't show a diff.
    if line_text is None or line_no is None:
        parts.append("")
        parts.append(
            "> Note: Could not automatically locate the original text in the source file; "
            "please adjust manually."
        )
        return "\n".join(parts)

    parts.append("")
    parts.append("### Suggestion diff")
    parts.append("")

    # Build an actual line-level diff using the source line.
    old_line = line_text
    if original and original in line_text:
        new_line = line_text.replace(original, suggestion, 1)
    else:
        new_line = None

    parts.append("```diff")
    if new_line is not None:
        parts.append(f"- {old_line}")
        parts.append(f"+ {new_line}")
    else:
        # Fallback to simple original/suggestion diff if replacement isn't possible.
        parts.append(f"- {original}")
        parts.append(f"+ {suggestion}")
    parts.append("```")

    return "\n".join(parts)


def _create_issue_for_doc(
    session: requests.Session,
    owner: str,
    repo: str,
    repo_root: Path,
    source_repo_url: str,
    doc: MutableMapping[str, object],
) -> Optional[int]:
    file_path = doc.get("file")
    original = doc.get("original")
    suggestion = doc.get("suggestion")

    if not isinstance(file_path, str) or not isinstance(original, str) or not isinstance(
        suggestion, str
    ):
        return None

    line_no, url, line_text = _find_line_and_url(
        repo_root, file_path, original, source_repo_url
    )

    labels = _labels_for_doc(doc)
    _ensure_labels(session, owner, repo, labels)

    title = _build_issue_title(doc)
    body = _build_issue_body(
        doc=doc,
        url=url,
        line_no=line_no,
        line_text=line_text,
        original=str(original),
        suggestion=str(suggestion),
    )

    resp = session.post(
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues",
        json={
            "title": title,
            "body": body,
            "labels": sorted(labels),
        },
    )
    resp.raise_for_status()
    data = resp.json()
    number = data.get("number")
    return int(number) if isinstance(number, int) else None


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Create GitHub issues from MongoDB typo suggestions."
    )
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017",
        help="MongoDB connection URI (default: %(default)s)",
    )
    parser.add_argument(
        "--db",
        default="mdn_proofreading",
        help="Database name (default: %(default)s)",
    )
    parser.add_argument(
        "--collection",
        default="japanese_typos_suggestions",
        help="Collection name (default: %(default)s)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("translated-content"),
        help="Path to the translated-content repository (default: %(default)s)",
    )
    parser.add_argument(
        "--source-repo-url",
        default=DEFAULT_SOURCE_REPO_URL,
        help="Base GitHub URL for the source content repo (default: %(default)s)",
    )

    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is not set. Please define it in your environment or .env.")

    if not args.repo_root.is_dir():
        raise SystemExit(f"Repo root not found: {args.repo_root}")

    client = MongoClient(args.mongo_uri)
    coll = client[args.db][args.collection]

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
    )

    # Process in a stable order by file path so related suggestions group together.
    pending_docs = coll.find({"issue_number": {"$exists": False}}).sort("file", 1)
    created = 0
    for doc in pending_docs:
        issue_number = _create_issue_for_doc(
            session=session,
            owner=TARGET_OWNER,
            repo=TARGET_REPO,
            repo_root=args.repo_root,
            source_repo_url=args.source_repo_url,
            doc=doc,
        )
        if issue_number is None:
            continue

        coll.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "issue_number": issue_number,
                    "updated_up": datetime.now(timezone.utc),
                }
            },
        )
        created += 1

    print(f"Created {created} issues and updated MongoDB documents.")


if __name__ == "__main__":
    main()
