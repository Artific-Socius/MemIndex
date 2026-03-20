
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Relative to repository root (POSIX path for git CLI)
SUBMODULE_REL = "UniversalBenchmark/benchmark/data/raw/EverMind-AI/EverMemBench-Static"
HF_DATASET_GIT_URL = "https://huggingface.co/datasets/EverMind-AI/EverMemBench-Static"


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
    """Where Git stores the submodule's bare repo: <common-git>/modules/<relpath>."""
    return _git_common_dir(root) / "modules" / Path(*rel_posix.split("/"))


def _submodule_path_is_gitlink(root: Path, rel_posix: str) -> bool:
    """True if parent index records this path as a gitlink (mode 160000)."""
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
    # format: <mode> <hash> <stage>\t<path>
    return line[0].startswith("160000 ")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = _find_git_root(script_dir)
    rel = SUBMODULE_REL.replace("\\", "/")
    target = root / Path(*rel.split("/"))

    os.chdir(root)

    if not _submodule_path_is_gitlink(root, rel):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(
                f"错误: 路径已存在但不是子模块 gitlink: {target}\n"
                "请删除该目录后，在仓库根目录执行:\n"
                f"  git submodule add {HF_DATASET_GIT_URL} {rel}",
                file=sys.stderr,
            )
            sys.exit(1)
        mod_dir = _submodule_git_module_dir(root, rel)
        add_cmd = ["git", "submodule", "add"]
        if mod_dir.is_dir():
            print("复用已有子模块 Git 目录（git submodule add --force）", file=sys.stderr)
            add_cmd.append("--force")
        add_cmd.extend([HF_DATASET_GIT_URL, rel])

        print(f"注册子模块: {rel}（LFS smudge 已跳过，稍后单独 git lfs pull）")
        _run(
            add_cmd,
            cwd=root,
            extra_env={"GIT_LFS_SKIP_SMUDGE": "1"},
        )

    print(f"初始化 / 更新子模块: {rel}")
    _run(
        ["git", "submodule", "update", "--init", "--recursive", "--", rel],
        cwd=root,
        extra_env={"GIT_LFS_SKIP_SMUDGE": "1"},
    )

    if not (target / ".git").exists():
        print(f"错误: 子模块目录异常（无 .git）: {target}", file=sys.stderr)
        sys.exit(1)

    print("git lfs install (submodule)")
    _run(["git", "lfs", "install", "--local"], cwd=target, check=False)

    _lfs_pull_with_retries(target)

    print("git pull --ff-only (submodule)")
    _run(["git", "pull", "--ff-only"], cwd=target, check=False)

    print(f"完成。数据目录: {target}")


if __name__ == "__main__":
    main()
