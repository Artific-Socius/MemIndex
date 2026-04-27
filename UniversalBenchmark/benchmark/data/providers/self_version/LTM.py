"""
Self-Version / LTM (Long-Term Memory) benchmark.

This dataset is authored as an interleaved sequence of conversation turns and
evaluation questions. Therefore we implement it as a BenchmarkLite (scripted
Scenario.turns) instead of the data-layer Benchmark+Scene interface, which
would force a preload-history + questions() split.

Registry: :data:`benchmark.data.BENCHMARK_LITE` under key ``Self_Version/LTM``;
use :func:`benchmark.data.get_benchmark_lite`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ....interfaces.evidence import EvidenceBundle
from benchmark_lite import AggregateResult, BenchmarkLite, Scenario, Turn, TurnResult, TurnScore, TurnType
from benchmark_lite.evaluators import BaseEvaluator, get_evaluator

POOL_NAME = "ltm_self_version"

RAW_REPO_REL = (
    Path("UniversalBenchmark")
    / "benchmark"
    / "data"
    / "raw"
    / "Self_Version"
    / "LTM"
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


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True, slots=True)
class LTMScaleConfig:
    scale_name: str  # e.g. "10k"
    memory_distance: int
    head_prompts: list[str]
    task_files: dict[str, Path]  # task_name -> json path
    nonsense_list: list[Any]
    memory_distance_level: str


def discover_scale_configs(raw_root: Path) -> list[LTMScaleConfig]:
    cfg_dir = raw_root / "config"
    if not cfg_dir.is_dir():
        return []

    out: list[LTMScaleConfig] = []
    for cfg_path in cfg_dir.glob("*.json"):
        scale_name = cfg_path.stem
        cfg = _load_json(cfg_path)
        files = cfg.get("files") or {}
        if not isinstance(files, dict):
            raise TypeError(f"{cfg_path}: 'files' must be a dict")

        task_files: dict[str, Path] = {}
        for task_name, rel in files.items():
            task_files[str(task_name)] = (cfg_path.parent / str(rel)).resolve()

        memory_distance = int(cfg.get("memory_distance", 0))
        head_prompts = [str(x) for x in (cfg.get("head_prompts") or [])]
        nonsense_list = list(cfg.get("nonsense_list") or [])
        memory_distance_level = str(cfg.get("memory_distance_level", ""))

        out.append(
            LTMScaleConfig(
                scale_name=scale_name,
                memory_distance=memory_distance,
                head_prompts=head_prompts,
                task_files=task_files,
                nonsense_list=nonsense_list,
                memory_distance_level=memory_distance_level,
            )
        )

    out.sort(key=lambda c: (c.memory_distance, c.scale_name))
    return out


def _is_eval_item(item: dict[str, Any]) -> bool:
    return item.get("score") is not None


def _normalize_ref_strings(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw]


def _evidence_from_score(
    task_name: str,
    score_obj: dict[str, Any],
    *,
    refs: list[str],
) -> EvidenceBundle:
    payload: dict[str, Any] = {}
    if isinstance(score_obj.get("binary_items"), list):
        payload["binary_items"] = score_obj.get("binary_items") or []
    allow_missing = len(refs) == 0
    return EvidenceBundle(
        evidence_type=f"self_version.ltm.{task_name}.score",
        payload=payload,
        references=refs if refs else None,
        allow_missing_references=allow_missing,
    )


def _turn_from_item(
    *,
    scale: LTMScaleConfig,
    task_name: str,
    item: dict[str, Any],
) -> Turn:
    ask = str(item.get("ask", ""))
    idx = item.get("index", None)
    depend = item.get("depend", [])
    refs = item.get("refs", [])
    retry = item.get("retry", None)
    post_process = item.get("post_process", None)

    meta: dict[str, Any] = {
        "pool": POOL_NAME,
        "scale": scale.scale_name,
        "memory_distance": scale.memory_distance,
        "memory_distance_level": scale.memory_distance_level,
        "task": task_name,
        "item_index": idx,
        "depend": depend,
        "refs": refs,
        "retry": retry,
        "post_process": post_process,
    }

    if not _is_eval_item(item):
        return Turn(user_input=ask, turn_type=TurnType.CONVERSATION, metadata=meta)

    score_obj = item.get("score") or {}
    if not isinstance(score_obj, dict):
        raise TypeError(
            f"Expected item.score to be dict for eval item; got {type(score_obj)}"
        )

    score_answer = score_obj.get("answer", "")
    ref_strings = _normalize_ref_strings(refs)
    evidence = _evidence_from_score(task_name, score_obj, refs=ref_strings)
    eval_mode = "weighted_binary" if evidence.payload.get("binary_items") else "binary"
    max_score = float(score_obj.get("score", 1.0) or 1.0)

    meta.update(
        {
            "question_text": ask,
            "ground_truth": score_answer,
            "eval_mode": eval_mode,
            "eval_prompt_key": f"{POOL_NAME}.{task_name}",
            "max_score": max_score,
            "evidence": evidence,
            "raw_score": score_obj,
        }
    )

    return Turn(
        user_input=ask,
        turn_type=TurnType.EVALUATION,
        reference=score_answer,
        metadata=meta,
    )


def _load_task_items(path: Path) -> list[dict[str, Any]]:
    raw = _load_json(path)
    items = raw.get("items", [])
    if not isinstance(items, list):
        raise TypeError(f"{path}: 'items' must be a list")
    out: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise TypeError(f"{path}: items[{i}] must be a dict")
        out.append(it)
    return out


class LTMBenchmarkLite(BenchmarkLite):
    """Self-Version LTM as BenchmarkLite (scripted interleaved turns)."""

    def __init__(self, raw_root: Path | None = None) -> None:
        self._raw_root = raw_root if raw_root is not None else _raw_root_from_package()
        self._scales = discover_scale_configs(self._raw_root)
        self._evaluator_cache: dict[str, BaseEvaluator] = {}

    @property
    def name(self) -> str:
        return "Self_Version/LTM"

    @property
    def raw_root(self) -> Path:
        return self._raw_root

    @property
    def scenario_count(self) -> int:
        """(scale, task) 组合数；不加载各 task JSON 内容。"""
        return sum(len(s.task_files) for s in self._scales)

    def list_scenario_ids(self) -> list[str]:
        """与 :meth:`get_scenarios` 中 ``Scenario.id`` 一致，仅由 config 推导。"""
        ids: list[str] = []
        for scale in self._scales:
            for task_name in sorted(scale.task_files.keys()):
                ids.append(f"{scale.scale_name}:{task_name}")
        return ids

    def get_scenarios(self) -> Iterable[Scenario]:
        scenarios: list[Scenario] = []
        for scale in self._scales:
            # Each (scale, task) -> one scenario.
            for task_name in sorted(scale.task_files.keys()):
                task_path = scale.task_files[task_name]
                items = _load_task_items(task_path)
                turns: list[Turn] = []

                # Head prompts as conversation context.
                for hp in scale.head_prompts:
                    turns.append(Turn(user_input=str(hp), turn_type=TurnType.CONVERSATION))

                for it in items:
                    turns.append(_turn_from_item(scale=scale, task_name=task_name, item=it))

                scenario_id = f"{scale.scale_name}:{task_name}"
                scenarios.append(
                    Scenario(
                        id=scenario_id,
                        description=f"LTM {scale.scale_name} / {task_name}",
                        turns=turns,
                        metadata={
                            "pool": POOL_NAME,
                            "source_benchmark": self.name,
                            "scale": scale.scale_name,
                            "memory_distance": scale.memory_distance,
                            "task": task_name,
                            "raw_task_path": str(task_path),
                        },
                    )
                )
        return scenarios

    def _get_evaluator(self, eval_mode: str) -> BaseEvaluator:
        if eval_mode not in self._evaluator_cache:
            # Use built-in evaluator registry; model is supplied at runtime by env/runner
            # via evaluator's own defaults (openrouter/google/gemini-2.5-flash).
            self._evaluator_cache[eval_mode] = get_evaluator(eval_mode)
        return self._evaluator_cache[eval_mode]

    def evaluate(
        self,
        turn: Turn,
        response: str,
        history: list[TurnResult],
    ) -> TurnScore:
        meta = turn.metadata or {}
        eval_mode = str(meta.get("eval_mode", "binary"))
        ground_truth = meta.get("ground_truth", turn.reference)
        max_score = float(meta.get("max_score", 1.0) or 1.0)
        question_text = str(meta.get("question_text", turn.user_input))
        evidence = meta.get("evidence", None)

        evaluator = self._get_evaluator(eval_mode)
        return evaluator.evaluate(
            question_text=question_text,
            ground_truth=ground_truth,
            response=response,
            max_score=max_score,
            evidence=evidence,
        )

    def aggregate(self, scenario_results: list[Any]) -> AggregateResult:
        total_score = 0.0
        total_max = 0.0
        total = 0
        passed = 0

        for sr in scenario_results:
            # sr is ScenarioResult (pydantic), but keep it duck-typed.
            for tr in getattr(sr, "turn_results", []):
                score = getattr(tr, "score", None)
                if score is None:
                    continue
                total_score += float(score.score)
                total_max += 1.0
                total += 1
                if bool(score.passed):
                    passed += 1

        normalized = total_score / total_max if total_max > 0 else 0.0
        return AggregateResult(
            score=normalized,
            total_score=total_score,
            total_max_score=total_max,
            total=total,
            passed=passed,
            extra={"pool": POOL_NAME},
        )

