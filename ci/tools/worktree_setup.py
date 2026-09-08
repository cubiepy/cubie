"""Prepare a fresh git worktree of cubie for agent work.

Orca runs this from the new worktree after ``worktree create`` (see
``orca.yaml``); it also runs by hand from any worktree::

    python ci/tools/worktree_setup.py

Every worktree gets its own ``.venv`` so the editable install resolves
``cubie`` from that worktree and never from the main checkout. The
interpreter is the one the main checkout's ``.venv`` was built from, so
worktrees match it. With ``uv`` on PATH the wheels are hardlinked from
its cache (seconds, no duplicated CUDA wheels); otherwise ``venv`` and
``pip`` do the same work.

Environment:

``ORCA_WORKTREE_PATH``
    The worktree to prepare (default: this file's repo root).
``ORCA_ROOT_PATH``
    The main checkout (default: the owner of the shared ``.git``).
``CUBIE_WORKTREE_EXTRAS``
    Extras installed alongside ``-e .`` (default ``dev,cuda13``: the
    mlir dev lane plus the numba-cuda backend, both of which
    ``benchmarks/ab_gate.py`` needs).
"""

import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, **kwargs):
    print("+", " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def worktree_path():
    override = os.environ.get("ORCA_WORKTREE_PATH")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def root_path(worktree):
    override = os.environ.get("ORCA_ROOT_PATH")
    if override:
        return Path(override).resolve()
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=worktree, check=True, capture_output=True, text=True,
    ).stdout.strip()
    return Path(common).resolve().parent


def venv_python(venv):
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def base_interpreter(root):
    """Interpreter the main checkout's .venv was built from."""
    config = root / ".venv" / "pyvenv.cfg"
    if config.is_file():
        parser = configparser.ConfigParser()
        parser.read_string("[venv]\n" + config.read_text(encoding="utf-8"))
        section = parser["venv"]
        executable = section.get("executable")
        if executable and Path(executable).is_file():
            return Path(executable)
        home = section.get("home")
        if home:
            for name in ("python.exe", "python3", "python"):
                candidate = Path(home) / name
                if candidate.is_file():
                    return candidate
    print(f"no usable {config}; falling back to {sys.executable}")
    return Path(sys.executable)


def build_venv(worktree, interpreter, extras):
    venv = worktree / ".venv"
    python = venv_python(venv)
    uv = shutil.which("uv")
    if uv:
        if not python.is_file():
            run([uv, "venv", "--python", str(interpreter), str(venv)],
                cwd=worktree)
        run([uv, "pip", "install", "--python", str(python),
             "-e", f".[{extras}]"], cwd=worktree)
    else:
        if not python.is_file():
            run([str(interpreter), "-m", "venv", str(venv)])
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python), "-m", "pip", "install", "-e", f".[{extras}]"],
            cwd=worktree)
    return python


def copy_local_settings(root, worktree):
    """Gitignored per-repo Claude settings travel with the worktree."""
    source = root / ".claude" / "settings.local.json"
    target = worktree / ".claude" / "settings.local.json"
    if source.is_file() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"copied {source} -> {target}")


def verify(python, worktree):
    probe = ("import cubie, sys; print(cubie.__file__); "
             "print(sys.executable)")
    result = subprocess.run([str(python), "-c", probe], check=True,
                            capture_output=True, text=True, cwd=worktree)
    module_file, executable = result.stdout.split()
    resolved = Path(module_file).resolve()
    if worktree not in resolved.parents:
        raise SystemExit(
            f"cubie resolves to {resolved}, not inside {worktree}")
    print(f"cubie      {resolved}")
    print(f"python     {executable}")


def main():
    worktree = worktree_path()
    root = root_path(worktree)
    extras = os.environ.get("CUBIE_WORKTREE_EXTRAS", "dev,cuda13")
    print(f"worktree   {worktree}")
    print(f"root       {root}")
    interpreter = base_interpreter(root)
    print(f"base       {interpreter}")
    python = build_venv(worktree, interpreter, extras)
    copy_local_settings(root, worktree)
    verify(python, worktree)


if __name__ == "__main__":
    main()
