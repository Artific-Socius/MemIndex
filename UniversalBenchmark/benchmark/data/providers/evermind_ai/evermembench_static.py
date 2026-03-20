"""
EverMemBench-Static: each 512K+ scale directory (data.pkl + unique_reference.pkl) -> one Scene.

Questions: **full QAR** from ``qar/test.jsonl`` then ``qar/train_sft.jsonl`` (loaded once per
Benchmark, shared by all scenes). Corpus/haystack still comes from each scale's
``unique_reference.pkl``.

scene_id: "0", "1", ... (per-benchmark index, ascending corpus scale).
scene_name: directory label, e.g. "512K", "1M".
question_id: decimal string index ``0 .. len(qar_rows)-1`` (test rows first, then train).
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from ....interfaces.benchmark import Benchmark
from ....interfaces.evidence import EvidenceBundle
from ....interfaces.question import Question
from ....interfaces.scene import Scene
from ....interfaces.scoring import ScoringConfig

POOL_NAME = "evermembench_static"

RAW_REPO_REL = Path("UniversalBenchmark") / "benchmark" / "data" / "raw" / "EverMind-AI" / "EverMemBench-Static"

_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"

# Default LLM-as-judge style prompt for this benchmark (also exposed as Benchmark.eval_prompt).
EVERMEMBENCH_EVAL_PROMPT = """You are evaluating answers for EverMemBench-Static (long-context / retrieval QA).

You will be given:
1) The user question
2) A reference (gold) answer
3) A model-generated answer

Decide whether the model answer is correct with respect to the reference answer for the question.
Respond with exactly one token: True or False (capitalized), then a short one-line rationale.

Question:
{question}

Reference answer:
{reference}

Model answer:
{model_answer}

Judgment (True/False):"""


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


_SCALE_RE = re.compile(r"^(\d+)([KkMm])$")


def _scale_bytes_approx(name: str) -> int | None:
    m = _SCALE_RE.match(name.strip())
    if not m:
        return None
    n = int(m.group(1))
    u = m.group(2).upper()
    if u == "K":
        return n * 1000
    return n * 1_000_000


def _is_scale_at_least_512k(name: str) -> bool:
    approx = _scale_bytes_approx(name)
    if approx is None:
        return False
    return approx >= 512_000


def _scale_sort_key(name: str) -> tuple[int, str]:
    approx = _scale_bytes_approx(name)
    if approx is None:
        return (10**30, name)
    return (approx, name)


def _check_not_lfs_pointer(path: Path) -> None:
    with path.open("rb") as f:
        head = f.read(min(200, path.stat().st_size))
    if head.startswith(_LFS_PREFIX):
        raise FileNotFoundError(
            f"{path} is a Git LFS pointer, not real data. "
            "Run `git lfs pull` in the repo (see UniversalBenchmark/benchmark/data/TEMP_init_single_benchmark.py)."
        )


def _load_pickle(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Clone the submodule (see UniversalBenchmark/benchmark/data/TEMP_init_single_benchmark.py)."
        )
    _check_not_lfs_pointer(path)
    with path.open("rb") as f:
        return pickle.load(f)


def discover_scale_dirs(raw_root: Path) -> list[tuple[str, Path]]:
    """
    Valid scenes: data/{name}/ with data.pkl + unique_reference.pkl, name is 512K+ scale label.
    Sorted by approximate corpus size ascending.
    """
    data_root = raw_root / "data"
    if not data_root.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for p in data_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not _is_scale_at_least_512k(name):
            continue
        dp = p / "data.pkl"
        up = p / "unique_reference.pkl"
        if dp.is_file() and up.is_file():
            found.append((name, p))
    found.sort(key=lambda t: _scale_sort_key(t[0]))
    return found


def _normalize_qar_rows(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise TypeError(f"data.pkl must be a list, got {type(raw)}")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise TypeError(f"data.pkl[{i}] must be dict, got {type(row)}")
        out.append(row)
    return out


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}: each line must be a JSON object, got {type(obj)}")
            out.append(obj)
    return out


def load_full_qar_rows(raw_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Load all QAR records: test split then train split.

    Returns (concatenated_rows, counts).
    """
    test_p = raw_root / "qar" / "test.jsonl"
    train_p = raw_root / "qar" / "train_sft.jsonl"
    test_rows = _load_jsonl_objects(test_p)
    train_rows = _load_jsonl_objects(train_p)
    merged = test_rows + train_rows
    counts = {
        "qar_test": len(test_rows),
        "qar_train": len(train_rows),
        "qar_total": len(merged),
    }
    return merged, counts


def _normalize_corpus(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        raise TypeError(f"unique_reference.pkl must be list[str], got {type(raw)}")
    return [str(x) for x in raw]


def _join_background(
    docs: list[str],
    *,
    max_chars: int | None,
    join_sep: str,
) -> str:
    if not docs:
        return ""
    if max_chars is None:
        return join_sep.join(docs)
    parts: list[str] = []
    total = 0
    for p in docs:
        if not p:
            continue
        sep_len = len(join_sep) if parts else 0
        if total + sep_len + len(p) <= max_chars:
            parts.append(p)
            total += sep_len + len(p)
            continue
        room = max_chars - total - sep_len
        if room > 0:
            parts.append(p[:room])
        break
    return join_sep.join(parts)


def _build_question(row: dict[str, Any], question_id: str) -> Question:
    refs = list(row.get("reference_list") or [])
    return Question(
        question_id=question_id,
        question_text=str(row.get("query", "")),
        ground_truth=str(row.get("answer", "")),
        evidence=EvidenceBundle(
            evidence_type="evermembench.qar.reference_list",
            payload={"documents": refs, "n_refs": len(refs)},
        ),
        scoring=ScoringConfig(
            eval_mode="score",
            eval_prompt_key="evermembench_qa",
            max_score=1.0,
        ),
    )


class ScaleContextScene(Scene):
    """
    One scale directory: all QAR rows from full ``qar/*.jsonl`` (passed in) + corpus from
    ``unique_reference.pkl`` under ``scale_dir``.
    """

    def __init__(
        self,
        scene_id: str,
        scene_name: str,
        scale_dir: Path,
        qar_rows: list[dict[str, Any]],
        *,
        lazy_corpus: bool = False,
    ) -> None:
        self._scene_id = scene_id
        self._scene_name = scene_name
        self._scale_dir = scale_dir
        self._lazy_corpus = lazy_corpus
        self._corpus_docs: list[str] | None = None
        self._rows = qar_rows
        if not lazy_corpus:
            self._corpus_docs = _normalize_corpus(_load_pickle(scale_dir / "unique_reference.pkl"))

    def _load_corpus_if_needed(self) -> list[str]:
        if self._corpus_docs is None:
            self._corpus_docs = _normalize_corpus(_load_pickle(self._scale_dir / "unique_reference.pkl"))
        return self._corpus_docs

    @property
    def scene_id(self) -> str:
        return self._scene_id

    @property
    def scene_name(self) -> str | None:
        return self._scene_name

    @property
    def task_type(self) -> str | None:
        return "large_scale_reference"

    def background_text(self, *, max_chars: int | None = None, join_sep: str = "\n\n") -> str:
        docs = self._load_corpus_if_needed()
        return _join_background(docs, max_chars=max_chars, join_sep=join_sep)

    def question_count(self) -> int:
        return len(self._rows)

    def get_question_by_id(self, question_id: str) -> Question:
        if not question_id.isdigit():
            raise KeyError(f"Unknown question_id {question_id!r}")
        i = int(question_id)
        if i < 0 or i >= len(self._rows):
            raise KeyError(f"Unknown question_id {question_id!r}")
        return _build_question(self._rows[i], question_id)

    def questions(self) -> Iterable[Question]:
        def _gen() -> Iterator[Question]:
            for i, row in enumerate(self._rows):
                yield _build_question(row, str(i))

        return _gen()


class EverMemBenchStaticBenchmark(Benchmark):
    """512K+ scales only; scene_id is "0".."N-1"."""

    def __init__(
        self,
        raw_root: Path | None = None,
        *,
        lazy_corpus: bool = False,
    ) -> None:
        self._raw_root = raw_root if raw_root is not None else _raw_root_from_package()
        self._lazy_corpus = lazy_corpus
        self._scale_entries: list[tuple[str, Path]] = discover_scale_dirs(self._raw_root)
        self._qar_rows, self._qar_counts = load_full_qar_rows(self._raw_root)

    @property
    def benchmark_name(self) -> str:
        return "EverMind-AI/EverMemBench-Static"

    @property
    def eval_prompt(self) -> str:
        return EVERMEMBENCH_EVAL_PROMPT

    @property
    def raw_root(self) -> Path:
        return self._raw_root

    @property
    def qar_rows(self) -> list[dict[str, Any]]:
        """All QAR records (test then train); shared across scenes."""
        return self._qar_rows

    def qar_counts(self) -> dict[str, int]:
        """How many rows were loaded from ``qar/test.jsonl``, ``train_sft.jsonl``, and total."""
        return dict(self._qar_counts)

    def scene_index_table(self) -> list[dict[str, str]]:
        return [
            {"scene_id": str(i), "scene_name": name}
            for i, (name, _) in enumerate(self._scale_entries)
        ]

    def scene_dimension_table(self) -> list[dict[str, str | int]]:
        """
        Per-scene question counts: same as full QAR row count (does not load unique_reference.pkl).
        """
        n = len(self._qar_rows)
        return [
            {"scene_id": str(i), "scene_name": name, "question_count": n}
            for i, (name, _) in enumerate(self._scale_entries)
        ]

    def list_scenes(self) -> list[str]:
        return [str(i) for i in range(len(self._scale_entries))]

    def get_scene(self, scene_id: str) -> Scene:
        if not scene_id.isdigit():
            raise ValueError(f"scene_id must be a non-negative integer string, got {scene_id!r}")
        idx = int(scene_id)
        if idx < 0 or idx >= len(self._scale_entries):
            raise KeyError(
                f"No scene {scene_id!r} for {self.benchmark_name!r}. "
                f"Valid ids: 0..{max(len(self._scale_entries) - 1, 0)}"
            )
        name, path = self._scale_entries[idx]
        return ScaleContextScene(
            scene_id=str(idx),
            scene_name=name,
            scale_dir=path,
            qar_rows=self._qar_rows,
            lazy_corpus=self._lazy_corpus,
        )

    # split_line_counts kept for optional diagnostics (qar jsonl, if present)
    def split_line_counts(self) -> dict[str, int]:
        def _count_lines(path: Path) -> int:
            if not path.is_file():
                return 0
            n = 0
            with path.open("rb") as f:
                for _ in f:
                    n += 1
            return n

        root = self._raw_root
        return {
            "test": _count_lines(root / "qar" / "test.jsonl"),
            "train": _count_lines(root / "qar" / "train_sft.jsonl"),
        }
