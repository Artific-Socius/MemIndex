"""
EverMemBench-Dynamic: each topic directory (``01``..``05``) -> one Scene.

Corpus: ``{topic_id}/dialogue.json`` (daily group dialogues) as ``background_documents``.
Questions: ``{topic_id}/qa_{topic_id}.json`` (array of rows with ``Q``, ``A``, ``R``, ``options``).

``R`` entries resolve to snippet strings via ``date`` + ``group`` + ``message_index`` against
the same topic's dialogue file (see upstream dataset card).

scene_id: topic directory name, e.g. ``"01"``.
question_id: decimal string index into that topic's QA list.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from ....interfaces.benchmark import Benchmark
from ....interfaces.evidence import EvidenceBundle
from ....interfaces.question import Question
from ....interfaces.scene import Scene
from ....interfaces.scoring import ScoringConfig

from .evermembench_static import EVERMEMBENCH_EVAL_PROMPT, _check_not_lfs_pointer

POOL_NAME = "evermembench_dynamic"

RAW_REPO_REL = Path("UniversalBenchmark") / "benchmark" / "data" / "raw" / "EverMind-AI" / "EverMemBench-Dynamic"


def _find_repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    raise FileNotFoundError(
        "Could not find git repository root (.git). "
        "Run from inside the MemIndex clone, or set paths explicitly."
    )


def _raw_root_from_package() -> Path:
    return _find_repo_root(Path(__file__).resolve()) / RAW_REPO_REL


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Clone the submodule (UniversalBenchmark/benchmark/init_raw.py "
            "--only evermind/EverMemBench-Dynamic)."
        )
    _check_not_lfs_pointer(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def discover_topic_entries(raw_root: Path) -> list[tuple[str, Path]]:
    """Topics with ``dialogue.json`` and ``qa_{tid}.json``; sorted by id."""
    if not raw_root.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for p in sorted(raw_root.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        tid = p.name
        if not re.fullmatch(r"\d{2}", tid):
            continue
        dlg = p / "dialogue.json"
        qa = p / f"qa_{tid}.json"
        if dlg.is_file() and qa.is_file():
            found.append((tid, p))
    return found


def _parse_message_index_field(s: str) -> set[int]:
    """Parse HF ``message_index`` like ``'1, 4-6, 8'`` or ``'2-3'``."""
    out: set[int] = set()
    for part in str(s).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(part))
            except ValueError:
                continue
    return out


def _build_message_lookup(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[int, str]]:
    """(date, group_name) -> message_index -> one-line text."""
    lookup: dict[tuple[str, str], dict[int, str]] = {}
    for rec in records:
        date = str(rec.get("date", ""))
        dials = rec.get("dialogues")
        if not isinstance(dials, dict):
            continue
        for group_name, msgs in dials.items():
            if msgs is None:
                continue
            if not isinstance(msgs, list):
                continue
            key = (date, str(group_name))
            inner = lookup.setdefault(key, {})
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                mi = m.get("message_index")
                try:
                    idx = int(mi)
                except (TypeError, ValueError):
                    continue
                sp = str(m.get("speaker", ""))
                txt = str(m.get("dialogue", ""))
                inner[idx] = f"({idx}) {sp}: {txt}"
    return lookup


def _dialogue_records_to_background_parts(records: list[dict[str, Any]]) -> list[str]:
    """One chunk per (date, group) with non-empty messages."""
    parts: list[str] = []
    for rec in records:
        date = str(rec.get("date", ""))
        topic = str(rec.get("topic_id", ""))
        dials = rec.get("dialogues")
        if not isinstance(dials, dict):
            continue
        for group_name in sorted(dials.keys()):
            msgs = dials[group_name]
            if not msgs:
                continue
            if not isinstance(msgs, list):
                continue
            lines = [f"[topic={topic} date={date} group={group_name}]"]
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                mi = m.get("message_index", "")
                sp = m.get("speaker", "")
                txt = m.get("dialogue", "")
                lines.append(f"  msg{mi} {sp}: {txt}")
            parts.append("\n".join(lines))
    return parts


def _resolve_r_to_documents(
    refs: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[int, str]],
) -> list[str]:
    docs: list[str] = []
    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue
        date = str(ref.get("date", ""))
        group = str(ref.get("group", ""))
        mid = str(ref.get("message_index", ""))
        want = _parse_message_index_field(mid)
        inner = lookup.get((date, group), {})
        lines = [inner[j] for j in sorted(want) if j in inner]
        block = "\n".join(lines) if lines else f"(no matching messages: {date!r} {group!r} idx={mid!r})"
        docs.append(f"[ref {i + 1} date={date} group={group} index={mid}]\n{block}")
    return docs


def _load_qa_rows(topic_dir: Path, topic_id: str) -> list[dict[str, Any]]:
    path = topic_dir / f"qa_{topic_id}.json"
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise TypeError(f"{path}: expected top-level JSON array")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise TypeError(f"{path}: items[{i}] must be dict")
        out.append(row)
    return out


def _load_dialogue_bundle(topic_dir: Path) -> tuple[list[str], dict[tuple[str, str], dict[int, str]]]:
    path = topic_dir / "dialogue.json"
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise TypeError(f"{path}: expected top-level JSON array")
    records = [x for x in raw if isinstance(x, dict)]
    parts = _dialogue_records_to_background_parts(records)
    lookup = _build_message_lookup(records)
    return parts, lookup


def _build_question(
    row: dict[str, Any],
    question_id: str,
    *,
    msg_lookup: dict[tuple[str, str], dict[int, str]],
) -> Question:
    qtext = str(row.get("Q", ""))
    answer = str(row.get("A", ""))
    refs = list(row.get("R") or [])
    docs = _resolve_r_to_documents(refs, msg_lookup)
    opts = row.get("options")
    return Question(
        question_id=question_id,
        question_text=qtext,
        ground_truth=answer,
        evidence=EvidenceBundle(
            evidence_type="evermembench.dynamic.qar",
            payload={
                "documents": docs,
                "n_refs": len(refs),
                "references": refs,
                "options": opts,
                "source_id": row.get("id"),
            },
        ),
        scoring=ScoringConfig(
            eval_mode="score",
            eval_prompt_key="evermembench_dynamic_qa",
            max_score=1.0,
        ),
    )


class TopicDialogueScene(Scene):
    """One topic: background_documents from dialogue days; questions from qa JSON."""

    def __init__(
        self,
        scene_id: str,
        scene_name: str,
        bg_parts: list[str],
        msg_lookup: dict[tuple[str, str], dict[int, str]],
        qa_rows: list[dict[str, Any]],
    ) -> None:
        self._scene_id = scene_id
        self._scene_name = scene_name
        self._bg_parts = bg_parts
        self._msg_lookup = msg_lookup
        self._qa_rows = qa_rows

    @property
    def scene_id(self) -> str:
        return self._scene_id

    @property
    def scene_name(self) -> str | None:
        return self._scene_name

    @property
    def task_type(self) -> str | None:
        return "dynamic_topic_reference"

    def background_text(self, *, max_chars: int | None = None, join_sep: str = "\n\n") -> str:
        text = join_sep.join(self._bg_parts)
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars]
        return text

    def background_documents(self) -> list[str]:
        return list(self._bg_parts)

    def question_count(self) -> int:
        return len(self._qa_rows)

    def get_question_by_id(self, question_id: str) -> Question:
        if not question_id.isdigit():
            raise KeyError(f"Unknown question_id {question_id!r}")
        i = int(question_id)
        if i < 0 or i >= len(self._qa_rows):
            raise KeyError(f"Unknown question_id {question_id!r}")
        return _build_question(self._qa_rows[i], question_id, msg_lookup=self._msg_lookup)

    def questions(self) -> Iterable[Question]:
        def _gen() -> Iterator[Question]:
            for i, row in enumerate(self._qa_rows):
                yield _build_question(row, str(i), msg_lookup=self._msg_lookup)

        return _gen()


class EverMemBenchDynamicBenchmark(Benchmark):
    """topic_id ``01``..``05`` as scene_id; QA scoped per topic."""

    def __init__(self, raw_root: Path | None = None) -> None:
        self._raw_root = raw_root if raw_root is not None else _raw_root_from_package()
        self._topics: list[tuple[str, Path]] = discover_topic_entries(self._raw_root)
        self._topic_dirs: dict[str, Path] = dict(self._topics)
        self._qa_by_topic: dict[str, list[dict[str, Any]]] = {}
        self._dialogue_cache: dict[str, tuple[list[str], dict[tuple[str, str], dict[int, str]]]] = {}
        for tid, tdir in self._topics:
            self._qa_by_topic[tid] = _load_qa_rows(tdir, tid)

    def _bundle_for(self, topic_id: str) -> tuple[list[str], dict[tuple[str, str], dict[int, str]]]:
        if topic_id not in self._dialogue_cache:
            tdir = self._topic_dirs.get(topic_id)
            if tdir is None:
                return [], {}
            self._dialogue_cache[topic_id] = _load_dialogue_bundle(tdir)
        return self._dialogue_cache[topic_id]

    @property
    def benchmark_name(self) -> str:
        return "EverMind-AI/EverMemBench-Dynamic"

    @property
    def eval_prompt(self) -> str:
        return EVERMEMBENCH_EVAL_PROMPT

    @property
    def raw_root(self) -> Path:
        return self._raw_root

    def topic_ids(self) -> list[str]:
        return [t[0] for t in self._topics]

    def qar_counts_by_topic(self) -> dict[str, int]:
        return {tid: len(self._qa_by_topic.get(tid, ())) for tid in self.topic_ids()}

    def scene_index_table(self) -> list[dict[str, str]]:
        return [{"scene_id": tid, "scene_name": tid} for tid in self.topic_ids()]

    def scene_dimension_table(self) -> list[dict[str, str | int]]:
        return [
            {"scene_id": tid, "scene_name": tid, "question_count": len(self._qa_by_topic.get(tid, ()))}
            for tid in self.topic_ids()
        ]

    def list_scenes(self) -> list[str]:
        return self.topic_ids()

    def get_scene(self, scene_id: str) -> Scene:
        tdir = self._topic_dirs.get(scene_id)
        if tdir is None:
            raise KeyError(
                f"No scene {scene_id!r} for {self.benchmark_name!r}. "
                f"Valid: {self.topic_ids()!r}"
            )
        parts, lookup = self._bundle_for(scene_id)
        rows = self._qa_by_topic.get(scene_id, [])
        return TopicDialogueScene(
            scene_id=scene_id,
            scene_name=scene_id,
            bg_parts=parts,
            msg_lookup=lookup,
            qa_rows=rows,
        )
