"""
LongMemEval (cleaned) loader for MemIndex UniversalBenchmark data layer.

Upstream HF dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Files (JSON arrays, NOT JSONL):
  - longmemeval_oracle.json   (~15MB, 500 items)
  - longmemeval_s_cleaned.json (~277MB)
  - longmemeval_m_cleaned.json (~2.7GB)

We implement **one item = one Scene** to keep memory bounded and to support huge files.
Scenes provide conversation context via ``conversation_history`` from ``haystack_sessions``.

Important: the upstream files are large JSON arrays; for the big splits we avoid loading
the entire array. We build a byte-offset index by scanning the JSON text once and then
parse individual items on demand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .....interfaces.benchmark import Benchmark
from .....interfaces.evidence import EvidenceBundle
from .....interfaces.question import Question
from .....interfaces.scene import ConversationTurn, Scene
from .....interfaces.scoring import ScoringConfig

POOL_NAME = "longmemeval_cleaned"

RAW_REPO_REL = (
    Path("UniversalBenchmark")
    / "benchmark"
    / "data"
    / "raw"
    / "xiaowu0162"
    / "LongMemEval"
    / "longmemeval-cleaned"
)


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


@dataclass(frozen=True, slots=True)
class SplitSpec:
    split_id: str
    filename: str


SPLITS: tuple[SplitSpec, ...] = (
    SplitSpec("oracle", "longmemeval_oracle.json"),
    SplitSpec("s_cleaned", "longmemeval_s_cleaned.json"),
    SplitSpec("m_cleaned", "longmemeval_m_cleaned.json"),
)


# ---------------------------------------------------------------------------
# JSON array item offset indexing
# ---------------------------------------------------------------------------


def _index_path_for(raw_root: Path, filename: str) -> Path:
    # Keep indexes under raw root (raw/** is gitignored); safe for local.
    return raw_root / ".indexes" / (filename + ".offsets.json")


def _build_offsets_for_json_array(path: Path) -> list[int]:
    """
    Return a list of byte offsets (file positions) where each top-level object starts.

    Assumes file is a JSON array of objects: [ { ... }, { ... }, ... ].
    Works in binary mode and is careful about strings/escapes.
    """
    offsets: list[int] = []
    with path.open("rb") as f:
        # Skip whitespace until '['
        while True:
            b = f.read(1)
            if not b:
                raise ValueError(f"{path}: unexpected EOF while seeking '['")
            if b in b" \t\r\n":
                continue
            if b == b"[":
                break
            raise ValueError(f"{path}: expected '[' at start, got {b!r}")

        in_str = False
        esc = False
        depth = 0
        while True:
            pos = f.tell()
            b = f.read(1)
            if not b:
                break
            c = b[0]

            if in_str:
                if esc:
                    esc = False
                    continue
                if c == 0x5C:  # backslash
                    esc = True
                    continue
                if c == 0x22:  # quote
                    in_str = False
                continue

            if c in (0x20, 0x09, 0x0D, 0x0A):  # whitespace
                continue
            if c == 0x22:  # quote
                in_str = True
                continue

            if c == 0x7B:  # '{'
                if depth == 0:
                    offsets.append(pos - 1)  # object starts at this '{'
                depth += 1
                continue
            if c == 0x7D:  # '}'
                if depth <= 0:
                    raise ValueError(f"{path}: unmatched '}}' at byte {pos}")
                depth -= 1
                continue

            # End of array
            if c == 0x5D and depth == 0:  # ']'
                break

        if depth != 0 or in_str:
            raise ValueError(f"{path}: unterminated JSON (depth={depth}, in_str={in_str})")

    return offsets


def _load_or_build_offsets(raw_root: Path, filename: str) -> list[int]:
    src = raw_root / filename
    if not src.is_file():
        return []
    idx_path = _index_path_for(raw_root, filename)
    try:
        if idx_path.is_file():
            with idx_path.open(encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and obj.get("filename") == filename and isinstance(obj.get("offsets"), list):
                return [int(x) for x in obj["offsets"]]
    except Exception:
        # Fall through to rebuild.
        pass

    offsets = _build_offsets_for_json_array(src)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    with idx_path.open("w", encoding="utf-8") as f:
        json.dump({"filename": filename, "offsets": offsets}, f, ensure_ascii=False)
    return offsets


def _read_json_object_at(path: Path, start_offset: int) -> dict[str, Any]:
    """
    Parse a single JSON object starting at ``start_offset`` (byte position of '{').
    """
    with path.open("rb") as f:
        f.seek(start_offset)
        buf = bytearray()
        in_str = False
        esc = False
        depth = 0

        while True:
            b = f.read(1)
            if not b:
                break
            buf.extend(b)
            c = b[0]

            if in_str:
                if esc:
                    esc = False
                    continue
                if c == 0x5C:
                    esc = True
                    continue
                if c == 0x22:
                    in_str = False
                continue

            if c == 0x22:
                in_str = True
                continue
            if c == 0x7B:
                depth += 1
                continue
            if c == 0x7D:
                depth -= 1
                if depth == 0:
                    break

        obj = json.loads(buf.decode("utf-8"))
        if not isinstance(obj, dict):
            raise TypeError(f"{path}: item at {start_offset} is not an object")
        return obj


# ---------------------------------------------------------------------------
# Scene / Benchmark
# ---------------------------------------------------------------------------


def _haystack_to_conversation_history(hs: Any) -> list[ConversationTurn]:
    """
    haystack_sessions: list[ list[{role, content, has_answer}] ].
    Convert to ConversationTurn by pairing sequential user/assistant.
    """
    if not isinstance(hs, list):
        return []
    flat: list[dict[str, Any]] = []
    for sess in hs:
        if not isinstance(sess, list):
            continue
        for t in sess:
            if isinstance(t, dict):
                flat.append(t)

    out: list[ConversationTurn] = []
    pending_user: list[str] = []
    for t in flat:
        role = str(t.get("role") or "").lower()
        content = str(t.get("content") or "")
        mark = " [has_answer]" if bool(t.get("has_answer")) else ""
        line = content + mark
        if role == "user":
            pending_user.append(line)
            continue
        if pending_user:
            out.append(ConversationTurn(user_message="\n".join(pending_user), assistant_response=line))
            pending_user = []
        else:
            out.append(ConversationTurn(user_message="[prior context]", assistant_response=line))
    if pending_user:
        out.append(ConversationTurn(user_message="\n".join(pending_user), assistant_response=""))
    return out


def _build_question(item: dict[str, Any], question_id: str) -> Question:
    return Question(
        question_id=question_id,
        question_text=str(item.get("question", "")),
        ground_truth=str(item.get("answer", "")),
        evidence=EvidenceBundle(
            evidence_type="longmemeval.cleaned.item",
            payload={
                "question_id": item.get("question_id"),
                "question_type": item.get("question_type"),
                "question_date": item.get("question_date"),
                "haystack_dates": item.get("haystack_dates"),
                "haystack_session_ids": item.get("haystack_session_ids"),
                "answer_session_ids": item.get("answer_session_ids"),
            },
        ),
        scoring=ScoringConfig(eval_mode="score", eval_prompt_key="longmemeval_cleaned_qa", max_score=1.0),
    )


LONGMEMEVAL_EVAL_PROMPT = """You are evaluating answers for LongMemEval (cleaned).\n\nYou will be given:\n1) The user question\n2) A reference (gold) answer\n3) A model-generated answer\n\nDecide whether the model answer is correct with respect to the reference answer for the question.\nRespond with exactly one token: True or False (capitalized), then a short one-line rationale.\n\nQuestion:\n{question}\n\nReference answer:\n{reference}\n\nModel answer:\n{model_answer}\n\nJudgment (True/False):"""


class LongMemEvalItemScene(Scene):
    def __init__(self, *, scene_id: str, item: dict[str, Any]) -> None:
        self._scene_id = scene_id
        self._item = item
        self._q = _build_question(item, "0")

    @property
    def scene_id(self) -> str:
        return self._scene_id

    @property
    def scene_name(self) -> str | None:
        return str(self._item.get("question_type") or "")

    @property
    def task_type(self) -> str | None:
        return "longmemeval_cleaned"

    def conversation_history(self) -> list[ConversationTurn]:
        return _haystack_to_conversation_history(self._item.get("haystack_sessions"))

    def questions(self) -> Iterable[Question]:
        def _one() -> Iterator[Question]:
            yield self._q

        return _one()


class LongMemEvalCleanedBenchmark(Benchmark):
    """
    Split-aware benchmark.

    scene_id format: ``{split}:{idx}``, e.g. ``oracle:0`` or ``m_cleaned:12345``.
    """

    def __init__(
        self,
        raw_root: Path | None = None,
        *,
        split_id: str = "oracle",
    ) -> None:
        self._raw_root = raw_root if raw_root is not None else _raw_root_from_package()
        self._split_id = split_id
        self._split = next((s for s in SPLITS if s.split_id == split_id), None)
        if self._split is None:
            raise ValueError(f"Unknown split_id {split_id!r}. Known: {[s.split_id for s in SPLITS]!r}")
        self._path = self._raw_root / self._split.filename
        self._offsets = _load_or_build_offsets(self._raw_root, self._split.filename)

    @property
    def benchmark_name(self) -> str:
        return f"xiaowu0162/longmemeval-cleaned:{self._split_id}"

    @property
    def eval_prompt(self) -> str:
        return LONGMEMEVAL_EVAL_PROMPT

    @property
    def raw_root(self) -> Path:
        return self._raw_root

    @property
    def split_id(self) -> str:
        return self._split_id

    @property
    def source_path(self) -> Path:
        return self._path

    def row_count(self) -> int:
        return len(self._offsets)

    def list_scenes(self) -> list[str]:
        return [f"{self._split_id}:{i}" for i in range(len(self._offsets))]

    def _load_item(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self._offsets):
            raise KeyError(f"idx out of range: {idx} (0..{max(len(self._offsets)-1,0)})")
        off = self._offsets[idx]
        return _read_json_object_at(self._path, off)

    def get_scene(self, scene_id: str) -> Scene:
        if ":" not in scene_id:
            raise ValueError(f"scene_id must be '{self._split_id}:<idx>', got {scene_id!r}")
        split, _, idx_s = scene_id.partition(":")
        if split != self._split_id:
            raise KeyError(f"scene_id split mismatch: expected {self._split_id!r}, got {split!r}")
        if not idx_s.isdigit():
            raise KeyError(f"scene_id idx must be integer string, got {idx_s!r}")
        idx = int(idx_s)
        item = self._load_item(idx)
        return LongMemEvalItemScene(scene_id=scene_id, item=item)

