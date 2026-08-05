"""BigModel files bundled with cubie.

Published Attributes
--------------------
:data:`BIGMODEL_DIR`
    Directory holding the bundled ``.bigmodel`` files.

Module-Level Functions
----------------------
:func:`available_bigmodels`
    Names of the bundled models.
:func:`bigmodel_path`
    Path of a bundled model, by name.
"""

from pathlib import Path
from typing import List

BIGMODEL_DIR = Path(__file__).parent

__all__ = ["BIGMODEL_DIR", "available_bigmodels", "bigmodel_path"]


def available_bigmodels() -> List[str]:
    """Return the names of the bundled BigModel files.

    Returns
    -------
    list of str
        Model names, sorted, without the ``.bigmodel`` suffix.
    """
    return sorted(path.stem for path in BIGMODEL_DIR.glob("*.bigmodel"))


def bigmodel_path(model: str) -> Path:
    """Return the path of a bundled BigModel file.

    Parameters
    ----------
    model : str
        Name of a bundled model, as listed by
        :func:`available_bigmodels`.

    Returns
    -------
    pathlib.Path
        Path to the ``.bigmodel`` file.

    Raises
    ------
    ValueError
        When ``model`` does not name a bundled model.
    """
    path = BIGMODEL_DIR / f"{model}.bigmodel"
    if not path.is_file():
        raise ValueError(
            f"{model!r} is not a bundled model; available models are "
            f"{available_bigmodels()}"
        )
    return path
