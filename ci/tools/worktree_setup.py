"""Build a worktree's own ``.venv`` with cubie installed editable.

Uses the main checkout's ``.venv`` interpreter and ``uv``, and copies
``.claude/settings.local.json``. When ``uv`` is not on PATH the pinned
release archive is downloaded from GitHub, checked against its SHA256
recorded here, and unpacked. An existing ``.venv`` built on a different
interpreter is rebuilt. Env: ``ORCA_WORKTREE_PATH`` (default: this repo
root), ``ORCA_ROOT_PATH`` (default: the main checkout),
``CUBIE_WORKTREE_EXTRAS`` (default ``dev,cuda13``), ``UV_INSTALL_DIR``
(where ``uv`` is unpacked; default ``~/.local/bin``).
"""

import configparser
import hashlib
import io
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

UV_VERSION = "0.12.12"
UV_RELEASE = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/"
UV_SHA256 = {
    "x86_64-pc-windows-msvc":
        "3d54912924c36e862c14f427d04f2ed70a99e8001d1c30caa101f6d5711626d5",
    "aarch64-pc-windows-msvc":
        "36559da51ecee83b2b1d80aa1a0ede2f80e2d9e5761fffcbb9e9366a7f3d022a",
    "x86_64-unknown-linux-gnu":
        "ab9b309d4586403f024e100abaceb396616e178a553e2500c36087d180f09509",
    "aarch64-unknown-linux-gnu":
        "fe08db50cc1b56cd1da7801065ed1103d27ed3f9571cd122386cfc7faf1b8df5",
    "x86_64-apple-darwin":
        "0dc8cd6c961582b0d140b5398f96b23502885277fb3464241456a2435e460dfa",
    "aarch64-apple-darwin":
        "46740540b63fdee9a6cb2e19baf3f1f475b850c440a33e63455087a6871263f1",
}


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


def uv_target():
    """Release archive target triple for this machine."""
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64",
            "arm64": "aarch64", "aarch64": "aarch64"}.get(machine)
    if arch is None:
        raise SystemExit(f"no pinned uv build for machine {machine!r}")
    if os.name == "nt":
        return f"{arch}-pc-windows-msvc"
    if sys.platform == "darwin":
        return f"{arch}-apple-darwin"
    return f"{arch}-unknown-linux-gnu"


def install_uv(uv):
    """Download the pinned uv release, verify its SHA256, unpack ``uv``."""
    target = uv_target()
    archive = f"uv-{target}" + (".zip" if os.name == "nt" else ".tar.gz")
    url = UV_RELEASE + archive
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != UV_SHA256[target]:
        raise SystemExit(
            f"{archive} sha256 {digest} != pinned {UV_SHA256[target]}")
    uv.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            uv.write_bytes(bundle.read("uv.exe"))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as bundle:
            member = bundle.extractfile(f"uv-{target}/uv")
            uv.write_bytes(member.read())
        uv.chmod(0o755)
    print(f"installed uv {UV_VERSION} to {uv}")


def uv_executable():
    """``uv`` from PATH, else the pinned release unpacked locally."""
    found = shutil.which("uv")
    if found:
        return Path(found)
    install_dir = os.environ.get("UV_INSTALL_DIR")
    if install_dir:
        install_dir = Path(install_dir)
    else:
        install_dir = Path.home() / ".local" / "bin"
    uv = install_dir / ("uv.exe" if os.name == "nt" else "uv")
    if not uv.is_file():
        install_uv(uv)
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
