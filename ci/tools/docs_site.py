"""Assemble the versioned documentation site on the ``gh-pages`` branch.

The site holds one folder per documentation version: ``main`` for the
development docs, ``vX.Y.Z`` for each release, and ``pr-N`` for the
preview of pull request ``N``. The site root holds ``versions.json``,
which lists the versions the picker offers, the picker script, and an
``index.html`` that redirects to the latest release, or to ``main``
before the first release is built.

Usage
-----
``python docs_site.py inject BUILD_DIR VERSION BASE``
    Add the picker script and the site verification tag to every page
    of a fresh Sphinx build.
``python docs_site.py index SITE_DIR BASE``
    Rewrite the root files of a ``gh-pages`` checkout from the version
    folders it holds, and remove pages left at the root by unversioned
    deploys.
"""

import json
import re
import shutil
import sys
from pathlib import Path

RELEASE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
PREVIEW = re.compile(r"^pr-\d+$")
VERIFICATION = (
    '<meta name="google-site-verification" '
    'content="Fpgkmgu5X-0RRG9UUYo4erbY0wd7VmLIQU4UZv_YaXc" />'
)
SWITCHER = Path(__file__).with_name("version_switcher.js")
ROOT_FILES = {"index.html", "versions.json", "version_switcher.js",
              ".nojekyll", "CNAME"}


def inject(build_dir: Path, version: str, base: str) -> None:
    """Add the picker script and verification tag to every built page.

    Parameters
    ----------
    build_dir
        Output directory of ``sphinx-build``.
    version
        Folder name the build is deployed to, such as ``main``.
    base
        URL path of the site root, such as ``/cubie/``.
    """
    tags = (
        f"{VERIFICATION}\n"
        f'<script src="{base}version_switcher.js" '
        f'data-version="{version}" defer></script>\n'
    )
    for page in build_dir.rglob("*.html"):
        text = page.read_text(encoding="utf8")
        page.write_text(text.replace("</head>", tags + "</head>", 1),
                        encoding="utf8")


def releases(site_dir: Path) -> list:
    """Return the release folder names, newest first."""
    names = [p.name for p in site_dir.iterdir()
             if p.is_dir() and RELEASE.match(p.name)]
    return sorted(
        names,
        key=lambda name: tuple(int(n) for n in RELEASE.match(name).groups()),
        reverse=True,
    )


def index(site_dir: Path, base: str) -> None:
    """Rewrite the root files of the site from its version folders.

    Write the version list and the picker script on every deploy. Leave
    the root pages untouched while the site holds neither ``main`` nor a
    release, so that a preview deploy never removes the current site.

    Parameters
    ----------
    site_dir
        Checkout of the ``gh-pages`` branch.
    base
        URL path of the site root, such as ``/cubie/``.
    """
    released = releases(site_dir)
    has_main = (site_dir / "main").is_dir()

    versions = []
    if released:
        versions.append({"name": released[0],
                         "label": f"{released[0]} (latest)"})
    if has_main:
        versions.append({"name": "main", "label": "main (development)"})
    versions.extend({"name": name, "label": name} for name in released[1:])
    for entry in versions:
        entry["url"] = f"{base}{entry['name']}/"
    (site_dir / "versions.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf8"
    )
    shutil.copyfile(SWITCHER, site_dir / "version_switcher.js")
    (site_dir / ".nojekyll").write_text("", encoding="utf8")

    if not released and not has_main:
        return
    default = released[0] if released else "main"

    # Unversioned deploys put a whole build at the root; remove it once
    # a versioned build can take its place.
    for entry in site_dir.iterdir():
        name = entry.name
        keep = (
            name == ".git"
            or name in ROOT_FILES
            or name == "main"
            or RELEASE.match(name)
            or PREVIEW.match(name)
        )
        if keep:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()

    target = f"{base}{default}/"
    (site_dir / "index.html").write_text(
        "<!DOCTYPE html>\n"
        "<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        f"{VERIFICATION}\n"
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        f'<link rel="canonical" href="{target}">\n'
        "<title>cubie documentation</title>\n"
        "</head>\n<body>\n"
        f'<p>Redirecting to <a href="{target}">{target}</a>.</p>\n'
        "</body>\n</html>\n",
        encoding="utf8",
    )


def main(argv: list) -> int:
    """Run the subcommand named in ``argv``; see the module docstring."""
    if len(argv) == 4 and argv[0] == "inject":
        inject(Path(argv[1]), argv[2], argv[3])
        return 0
    if len(argv) == 3 and argv[0] == "index":
        index(Path(argv[1]), argv[2])
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
