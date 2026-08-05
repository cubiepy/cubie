"""Opaque ODE systems for blind compile-pipeline stress tests.

Two domain-agnostic, provenance-free systems are exposed as *unbuilt*
cubie systems::

    from blackbox import blackboxsystem_1, blackboxsystem_2

Every state, observable, and constant label is an opaque token
(``cN_vM``); the systems carry no semantics you can fit to. They arrive
unbuilt — cubie compiles the CUDA kernels lazily on the first solve — so
you construct your own solver around them::

    from cubie import solve_ivp

    y0 = blackboxsystem_1.initial_values.values_array
    result = solve_ivp(blackboxsystem_1, y0, method="crank_nicolson",
                       duration=10.0, cache=False)

Pass ``cache=False`` to your Solver / solve_ivp call to disable the
compiled-kernel cache so the compile pipeline runs fresh every time.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from cubie import load_bigmodel_file

_HERE = Path(__file__).parent


def _load(stem, name):
    """Load an extensionless model file through the BigModel loader.

    The loader requires a ``.bigmodel`` path, so the file is staged as a
    temporary ``.bigmodel`` copy only for the duration of the parse.
    """
    source = _HERE / stem
    with tempfile.TemporaryDirectory() as tmp_dir:
        staged = Path(tmp_dir) / f"{name}.bigmodel"
        shutil.copyfile(source, staged)
        return load_bigmodel_file(
            str(staged), name=name, precision=np.float32
        )


blackboxsystem_1 = _load("blackbox1", "blackboxsystem_1")
blackboxsystem_2 = _load("blackbox2", "blackboxsystem_2")

__all__ = ["blackboxsystem_1", "blackboxsystem_2"]
