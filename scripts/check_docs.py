"""Validate local documentation links, code fences, and index coverage."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
INDEX = DOCS_DIR / "README.md"
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "assets" / "README.md",
        *sorted(DOCS_DIR.glob("*.md")),
    ]


def local_target(document: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    if (
        not target
        or target.startswith(("#", "http://", "https://", "mailto:"))
    ):
        return None
    path_text = unquote(target.split("#", maxsplit=1)[0])
    return (document.parent / path_text).resolve()


def main() -> int:
    issues: list[str] = []
    files = markdown_files()
    index_text = INDEX.read_text(encoding="utf-8")

    for document in files:
        text = document.read_text(encoding="utf-8")
        relative = document.relative_to(ROOT)
        first_content = next(
            (line for line in text.splitlines() if line.strip()),
            "",
        )
        if not first_content.startswith("# "):
            issues.append(f"{relative}: missing one top-level title")
        if text.count("```") % 2:
            issues.append(f"{relative}: unbalanced fenced code block")

        for raw_target in MARKDOWN_LINK.findall(text):
            target = local_target(document, raw_target)
            if target is not None and not target.exists():
                issues.append(f"{relative}: broken link -> {raw_target}")

    for document in sorted(DOCS_DIR.glob("*.md")):
        if document == INDEX:
            continue
        if document.name not in index_text:
            issues.append(f"docs/README.md: missing {document.name}")

    if issues:
        print("Documentation validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"Documentation validation passed ({len(files)} Markdown files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
