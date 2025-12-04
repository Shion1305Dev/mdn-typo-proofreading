import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient
from pymongo.errors import BulkWriteError


TRANSLATED_CONTENT_PREFIX = "/Users/shion/workspace/translated-content"


def _normalize_file_path(doc: dict) -> None:
    file_path = doc.get("file")
    if not isinstance(file_path, str):
        return

    if file_path.startswith(TRANSLATED_CONTENT_PREFIX):
        new_path = file_path[len(TRANSLATED_CONTENT_PREFIX) :]
        if new_path.startswith("/"):
            new_path = new_path[1:]
        doc["file"] = new_path


def import_documents(
    mongo_uri: str,
    database: str,
    collection: str,
    json_path: Path,
    drop_first: bool,
) -> int:
    client = MongoClient(mongo_uri)
    db = client[database]
    coll = db[collection]

    if drop_first:
        coll.drop()

    # Ensure unique index on (file, original, suggestion) when those fields exist.
    coll.create_index(
        [("file", 1), ("original", 1), ("suggestion", 1)],
        unique=True,
        partialFilterExpression={
            "original": {"$exists": True},
            "suggestion": {"$exists": True},
        },
    )

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at {json_path}, got {type(data).__name__}")

    # Normalize and deduplicate by (file, original, suggestion) within this batch.
    now = datetime.now(timezone.utc)
    seen_keys: set[tuple[str, str, str]] = set()
    docs_to_insert: list[dict] = []

    for raw in data:
        if not isinstance(raw, dict):
            continue

        doc = dict(raw)
        _normalize_file_path(doc)

        # Timestamps
        doc.setdefault("created_at", now)
        doc["updated_up"] = now

        key = None
        if all(k in doc for k in ("file", "original", "suggestion")):
            key = (str(doc["file"]), str(doc["original"]), str(doc["suggestion"]))

        if key is not None:
            if key in seen_keys:
                continue
            seen_keys.add(key)

        docs_to_insert.append(doc)

    if not docs_to_insert:
        return 0

    try:
        result = coll.insert_many(docs_to_insert, ordered=False)
        return len(result.inserted_ids)
    except BulkWriteError as exc:
        # Allow duplicate-key errors, re-raise anything else.
        details = getattr(exc, "details", None) or {}
        write_errors = details.get("writeErrors", [])
        non_dup_errors = [
            err for err in write_errors if err.get("code") not in (11000, 11001, 12582)
        ]
        if non_dup_errors:
            raise
        return details.get("nInserted", 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import japanese_typos_suggestions.json into MongoDB."
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
        "--file",
        type=Path,
        default=Path("data/japanese_typos_suggestions.json"),
        help="Path to JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--drop-first",
        action="store_true",
        help="Drop the target collection before inserting.",
    )

    args = parser.parse_args()

    count = import_documents(
        mongo_uri=args.mongo_uri,
        database=args.db,
        collection=args.collection,
        json_path=args.file,
        drop_first=args.drop_first,
    )

    print(f"Inserted {count} documents into {args.db}.{args.collection}")


if __name__ == "__main__":
    main()
