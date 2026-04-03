"""
初始化 ``benchmark/data/raw`` 下的 Hugging Face 数据集子模块：``git submodule add/update``、
``git lfs pull`` 等。在 MemIndex 仓库根目录执行::

    python UniversalBenchmark/benchmark/init_raw.py
    python UniversalBenchmark/benchmark/init_raw.py --only percena/locomo-mc10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# #region agent log
_AGENT_DEBUG_LOG_NAME = "debug-54d7a2.log"
_AGENT_SESSION = "54d7a2"


def _agent_debug_log(
    repo_root: Path,
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
) -> None:
    line = json.dumps(
        {
            "sessionId": _AGENT_SESSION,
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "runId": os.environ.get("DEBUG_RUN_ID", "run1"),
        },
        ensure_ascii=False,
    )
    with (repo_root / _AGENT_DEBUG_LOG_NAME).open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# #endregion


@dataclass(frozen=True)
class RawDatasetSpec:
    """一条 raw 子模块：HF 数据集 Git URL 与在父仓库中的相对路径（POSIX）。"""

    id: str
    git_url: str
    submodule_rel_posix: str
    """确保存在的 provider 包目录（相对 ``benchmark/data/providers``），无代码时仅占位。"""
    provider_rel_posix: str | None = None


# 在父仓库中的子模块路径前缀（POSIX）
_RAW_PREFIX = "UniversalBenchmark/benchmark/data/raw"

# 新增数据集：在此追加 (id, git_url, submodule 相对 raw 的路径片段, provider 相对 providers 的路径)
_RAW_ENTRIES: tuple[RawDatasetSpec, ...] = (
    RawDatasetSpec(
        id="evermind/EverMemBench-Static",
        git_url="https://huggingface.co/datasets/EverMind-AI/EverMemBench-Static",
        submodule_rel_posix=f"{_RAW_PREFIX}/EverMind-AI/EverMemBench-Static",
        provider_rel_posix="evermind_ai",
    ),
    RawDatasetSpec(
        id="percena/locomo-mc10",
        git_url="https://huggingface.co/datasets/Percena/locomo-mc10",
        submodule_rel_posix=f"{_RAW_PREFIX}/percena/Locomo/locomo-mc10",
        provider_rel_posix="percena/Locomo",
    ),
)


def _find_git_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError(
        "未找到 Git 仓库根目录（向上未见到 .git）。请在 MemIndex 克隆内运行本脚本。"
    )


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    check: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=check,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )


def _lfs_pull_with_retries(cwd: Path, *, attempts: int = 5, delay_sec: float = 4.0) -> None:
    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        if i == 1:
            print("git lfs pull")
        else:
            print(f"git lfs pull 重试 {i}/{attempts}（{delay_sec:.0f}s 后）...", file=sys.stderr)
            time.sleep(delay_sec)
        try:
            _run(["git", "lfs", "pull"], cwd=cwd, check=True)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
    print(
        "git lfs pull 在多次重试后仍失败。可尝试：\n"
        "  - 检查网络 / VPN / 防火墙；Windows 上 IPv6 不稳定时可尝试改用 IPv4 或关闭代理\n"
        "  - 设置 HF_TOKEN 后配置 Git 凭据（若数据集需登录）\n"
        "  - 在子模块目录手动: git lfs pull",
        file=sys.stderr,
    )
    assert last_err is not None
    raise last_err


def _git_common_dir(root: Path) -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=True,
    )
    return Path(out.stdout.strip())


def _submodule_git_module_dir(root: Path, rel_posix: str) -> Path:
    return _git_common_dir(root) / "modules" / Path(*rel_posix.split("/"))


def _submodule_path_is_gitlink(root: Path, rel_posix: str) -> bool:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-s", "--", rel_posix],
            cwd=str(root),
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return False
    line = out.stdout.strip().splitlines()
    if not line:
        return False
    return line[0].startswith("160000 ")


def _ensure_provider_placeholders(repo_root: Path, spec: RawDatasetSpec) -> None:
    if not spec.provider_rel_posix:
        return
    providers_base = (
        repo_root / "UniversalBenchmark" / "benchmark" / "data" / "providers"
    )
    pkg = providers_base / Path(*spec.provider_rel_posix.split("/"))
    pkg.mkdir(parents=True, exist_ok=True)
    init_py = pkg / "__init__.py"
    if not init_py.is_file():
        init_py.write_text('"""Provider package (add loaders next to this file)."""\n', encoding="utf-8")
    # 保证中间包也有 __init__.py（如 percena/Locomo）
    rel_parts = spec.provider_rel_posix.split("/")
    for i in range(1, len(rel_parts)):
        parent = providers_base / Path(*rel_parts[:i])
        p_init = parent / "__init__.py"
        if not p_init.is_file():
            p_init.write_text('"""Provider subpackage."""\n', encoding="utf-8")


def init_one_raw_submodule(repo_root: Path, spec: RawDatasetSpec) -> None:
    rel = spec.submodule_rel_posix.replace("\\", "/")
    target = repo_root / Path(*rel.split("/"))

    # #region agent log
    _agent_debug_log(
        repo_root,
        hypothesis_id="H2",
        location="init_raw.py:init_one_raw_submodule:entry",
        message="init_one_raw_submodule start",
        data={
            "spec_id": spec.id,
            "rel": rel,
            "target_exists": target.exists(),
            "is_gitlink": _submodule_path_is_gitlink(repo_root, rel),
        },
    )
    # #endregion

    _ensure_provider_placeholders(repo_root, spec)

    os.chdir(repo_root)

    if not _submodule_path_is_gitlink(repo_root, rel):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(
                f"错误: 路径已存在但不是子模块 gitlink: {target}\n"
                "请删除该目录后，在仓库根目录执行:\n"
                f"  git submodule add {spec.git_url} {rel}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        mod_dir = _submodule_git_module_dir(repo_root, rel)
        # raw/** 被父仓库 .gitignore 忽略时，必须用 `git submodule add --force`（见 raw/README.md）。
        add_cmd = ["git", "submodule", "add", "--force"]
        if mod_dir.is_dir():
            print("复用已有子模块 Git 目录（git submodule add --force）", file=sys.stderr)
        add_cmd.extend([spec.git_url, rel])

        # #region agent log
        _agent_debug_log(
            repo_root,
            hypothesis_id="H1",
            location="init_raw.py:init_one_raw_submodule:before_submodule_add",
            message="about to git submodule add",
            data={"add_cmd": add_cmd, "mod_dir_exists": mod_dir.is_dir()},
        )
        # #endregion

        print(f"[{spec.id}] 注册子模块: {rel}（LFS smudge 已跳过，稍后单独 git lfs pull）")
        _run(
            add_cmd,
            cwd=repo_root,
            extra_env={"GIT_LFS_SKIP_SMUDGE": "1"},
        )

        # #region agent log
        _agent_debug_log(
            repo_root,
            hypothesis_id="H1",
            location="init_raw.py:init_one_raw_submodule:after_submodule_add",
            message="git submodule add completed",
            data={"spec_id": spec.id, "rel": rel},
        )
        # #endregion

    print(f"[{spec.id}] 初始化 / 更新子模块: {rel}")
    _run(
        ["git", "submodule", "update", "--init", "--recursive", "--", rel],
        cwd=repo_root,
        extra_env={"GIT_LFS_SKIP_SMUDGE": "1"},
    )

    if not (target / ".git").exists():
        print(f"错误: 子模块目录异常（无 .git）: {target}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[{spec.id}] git lfs install (submodule)")
    _run(["git", "lfs", "install", "--local"], cwd=target, check=False)

    _lfs_pull_with_retries(target)

    print(f"[{spec.id}] git pull --ff-only (submodule)")
    _run(["git", "pull", "--ff-only"], cwd=target, check=False)

    print(f"[{spec.id}] 完成。数据目录: {target}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="初始化 benchmark/data/raw 下的 HF 数据集子模块。")
    p.add_argument(
        "--only",
        action="append",
        metavar="ID",
        help="只处理指定数据集 id（可多次指定）。缺省处理全部。已知 id: "
        + ", ".join(s.id for s in _RAW_ENTRIES),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    repo_root = _find_git_root(script_dir)

    want: set[str] | None = None
    if args.only:
        want = set(args.only)

    selected = [s for s in _RAW_ENTRIES if want is None or s.id in want]
    if want is not None:
        unknown = want - {s.id for s in selected}
        if unknown:
            print(f"错误: 未知的 --only id: {sorted(unknown)}", file=sys.stderr)
            raise SystemExit(2)
    if not selected:
        print("错误: 没有匹配的数据集。", file=sys.stderr)
        raise SystemExit(2)

    # #region agent log
    _agent_debug_log(
        repo_root,
        hypothesis_id="H2",
        location="init_raw.py:main:selected",
        message="datasets to init",
        data={"ids": [s.id for s in selected]},
    )
    # #endregion

    for spec in selected:
        init_one_raw_submodule(repo_root, spec)


if __name__ == "__main__":
    main()
