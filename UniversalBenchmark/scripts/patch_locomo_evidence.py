#!/usr/bin/env python3
"""Inject ``evidence`` from raw/locomo10.json into LoCoMo-MC10 JSONL rows.

The flattened QA order matches ``transformed/locomo_mc10_with_name.json`` and
``data/locomo_mc10.json`` (1986 lines each): for each sample in raw order,
all ``qa`` entries are appended in order.

Usage (from repo root or UniversalBenchmark/)::

    python scripts/patch_locomo_evidence.py

Overwrites the two JSONL files in place (large files; may take a minute).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def _repo_ub_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "benchmark" / "interfaces" / "evidence.py").is_file():
            return p
    raise FileNotFoundError("Could not find UniversalBenchmark root")


def _collect_evidence_list(raw_path: Path) -> list[list[str]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"{raw_path}: expected top-level list")
    out: list[list[str]] = []
    for si, sample in enumerate(raw):
        if not isinstance(sample, dict):
            raise TypeError(f"{raw_path}: sample[{si}] must be dict")
        qa_list = sample.get("qa") or []
        if not isinstance(qa_list, list):
            raise TypeError(f"{raw_path}: sample[{si}].qa must be list")
        for qi, qa in enumerate(qa_list):
            if not isinstance(qa, dict):
                raise TypeError(
                    f"{raw_path}: sample[{si}].qa[{qi}] must be dict",
                )
            ev = qa.get("evidence", [])
            if not isinstance(ev, list):
                ev = []
            out.append([str(x) for x in ev])
    return out


def _patch_jsonl(jsonl_path: Path, evidence_list: list[list[str]]) -> None:
    row_index = 0
    fd, tmp_path = tempfile.mkstemp(
        suffix=".jsonl",
        prefix="locomo_evidence_",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fout:
            with jsonl_path.open(encoding="utf-8") as fin:
                for line in fin:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    if row_index >= len(evidence_list):
                        raise ValueError(
                            f"{jsonl_path}: more JSONL rows than evidence "
                            f"entries ({row_index + 1} > {len(evidence_list)})",
                        )
                    row = json.loads(line_stripped)
                    if not isinstance(row, dict):
                        raise TypeError(
                            f"{jsonl_path}: non-object at logical row "
                            f"{row_index + 1}",
                        )
                    row["evidence"] = evidence_list[row_index]
                    fout.write(
                        json.dumps(row, ensure_ascii=False) + "\n",
                    )
                    row_index += 1

        if row_index != len(evidence_list):
            raise ValueError(
                f"{jsonl_path}: row count {row_index} != "
                f"evidence count {len(evidence_list)}",
            )
        os.replace(tmp_path, jsonl_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main() -> int:
    root = _repo_ub_root()
    locomo = (
        root
        / "benchmark"
        / "data"
        / "raw"
        / "percena"
        / "Locomo"
        / "locomo-mc10"
    )
    raw_path = locomo / "raw" / "locomo10.json"
    if not raw_path.is_file():
        print(f"Missing {raw_path}", file=sys.stderr)
        return 1

    evidence_list = _collect_evidence_list(raw_path)
    targets = [
        locomo / "transformed" / "locomo_mc10_with_name.json",
        locomo / "data" / "locomo_mc10.json",
    ]
    for p in targets:
        if not p.is_file():
            print(f"Skip missing {p}", file=sys.stderr)
            continue
        print(f"Patching {p} ({len(evidence_list)} rows)...")
        _patch_jsonl(p, evidence_list)
        print("  done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
