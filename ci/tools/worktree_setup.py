"""Build a worktree's own ``.venv`` with cubie installed editable.

Uses the main checkout's ``.venv`` interpreter and ``uv``, installing
``uv`` with its official installer when it is not on PATH, and copies
``.claude/settings.local.json``. An existing ``.venv`` built on a
different interpreter is rebuilt. Env: ``ORCA_WORKTREE_PATH`` (default:
this repo root), ``ORCA_ROOT_PATH`` (default: the main checkout),
``CUBIE_WORKTREE_EXTRAS`` (default ``dev,cuda13``), ``UV_INSTALL_DIR``
(where the installer puts ``uv``; default ``~/.local/bin``).
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


def venv_config(venv):
    """The ``pyvenv.cfg`` keys of ``venv``, or ``None`` without one."""
    config = venv / "pyvenv.cfg"
    if not config.is_file():
        return None
    parser = configparser.ConfigParser()
    parser.read_string("[venv]\n" + config.read_text(encoding="utf-8"))
    return parser["venv"]


def base_interpreter(root):
    """Interpreter the main checkout's .venv was built from."""
    section = venv_config(root / ".venv")
    if section is not None:
        executable = section.get("executable")
        if executable and Path(executable).is_file():
            return Path(executable)
        home = section.get("home")
        if home:
            for name in ("python.exe", "python3", "python"):
                candidate = Path(home) / name
                if candidate.is_file():
                    return candidate
    print(f"no usable {root / '.venv' / 'pyvenv.cfg'}; "
          f"falling back to {sys.executable}")
    return Path(sys.executable)


def uv_executable():
    """``uv`` from PATH, else installed with the official installer."""
    found = shutil.which("uv")
    if found:
        return Path(found)
    install_dir = os.environ.get("UV_INSTALL_DIR")
    if install_dir:
        install_dir = Path(install_dir)
    else:
        install_dir = Path.home() / ".local" / "bin"
    uv = install_dir / ("uv.exe" if os.name == "nt" else "uv")
    if uv.is_file():
        return uv
    print(f"uv not found; installing to {install_dir}")
    if os.name == "nt":
        run(["powershell", "-ExecutionPolicy", "ByPass", "-NoProfile", "-c",
             "irm https://astral.sh/uv/install.ps1 | iex"])
    else:
        run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
    if not uv.is_file():
        raise SystemExit(f"uv installer did not produce {uv}")
    return uv


def venv_matches(venv, interpreter):
    """Whether ``venv`` records ``interpreter``'s directory as its home."""
    section = venv_config(venv)
    if section is None:
        return False
    home = section.get("home")
    if not home:
        return False
    return Path(home).resolve() == interpreter.parent.resolve()


def build_venv(worktree, interpreter, extras, uv):
    venv = worktree / ".venv"
    python = venv_python(venv)
    if venv.exists() and not venv_matches(venv, interpreter):
        print(f"rebuilding {venv}: not built on {interpreter}")
        shutil.rmtree(venv)
    if not python.is_file():
        run([str(uv), "venv", "--python", str(interpreter), str(venv)],
            cwd=worktree)
    run([str(uv), "pip", "install", "--python", str(python),
         "-e", f".[{extras}]"], cwd=worktree)
    return python


def copy_local_settings(root, worktree):
    """Copy the gitignored Claude settings into the worktree."""
    source = root / ".claude" / "settings.local.json"
    target = worktree / ".claude" / "settings.local.json"
    if source.is_file() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"copied {source} -> {target}")


def python_version(python):
    result = subprocess.run(
        [str(python), "-c", "import sys; print(sys.version.split()[0])"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def verify(python, interpreter, worktree):
    probe = ("import cubie, sys; print(cubie.__file__); "
             "print(sys.executable)")
    result = subprocess.run([str(python), "-c", probe], check=True,
                            capture_output=True, text=True, cwd=worktree)
    module_file, executable = result.stdout.strip().splitlines()
    resolved = Path(module_file).resolve()
    if worktree not in resolved.parents:
        raise SystemExit(
            f"cubie resolves to {resolved}, not inside {worktree}")
    version = python_version(python)
    base_version = python_version(interpreter)
    if version != base_version:
        raise SystemExit(
            f"venv python is {version}; base {interpreter} is "
            f"{base_version}")
    print(f"cubie      {resolved}")
    print(f"python     {executable} ({version})")


def main():
    worktree = worktree_path()
    root = root_path(worktree)
    extras = os.environ.get("CUBIE_WORKTREE_EXTRAS", "dev,cuda13")
    print(f"worktree   {worktree}")
    print(f"root       {root}")
    interpreter = base_interpreter(root)
    print(f"base       {interpreter}")
    uv = uv_executable()
    print(f"uv         {uv}")
    python = build_venv(worktree, interpreter, extras, uv)
    copy_local_settings(root, worktree)
    verify(python, interpreter, worktree)


if __name__ == "__main__":
    main()
