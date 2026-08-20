"""Summary metric registry and built-in CUDA reductions.

Published Objects
-----------------
:data:`summary_metrics`
    Module-level :class:`SummaryMetrics` singleton that coordinates
    buffer sizing, function dispatch, and combined metric substitution.

:func:`register_metric`
    Decorator that instantiates and registers a metric class with the
    global registry.

See Also
--------
:class:`~cubie.outputhandling.summarymetrics.metrics.SummaryMetrics`
    Registry class backing the :data:`summary_metrics` singleton.
:class:`~cubie.outputhandling.summarymetrics.metrics.SummaryMetric`
    Abstract base class for metric implementations.

Notes
-----
All built-in metrics self-register on import. Third-party metrics
should use :func:`register_metric` to add themselves when their
module is imported.
"""

from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetrics,
    register_metric,
)
from numpy import float32

# This is the only default datatype in the whole game, look here for type
# mismatch (unexpected float32)
summary_metrics: SummaryMetrics = SummaryMetrics(precision=float32)

# Import each metric once, to register it with the summary_metrics object.
from cubie.outputhandling.summarymetrics import mean  # noqa
from cubie.outputhandling.summarymetrics import max   # noqa
from cubie.outputhandling.summarymetrics import rms   # noqa
from cubie.outputhandling.summarymetrics import peaks # noqa
from cubie.outputhandling.summarymetrics import std   # noqa
from cubie.outputhandling.summarymetrics import min   # noqa
from cubie.outputhandling.summarymetrics import max_magnitude  # noqa
from cubie.outputhandling.summarymetrics import extrema  # noqa
from cubie.outputhandling.summarymetrics import negative_peaks  # noqa
from cubie.outputhandling.summarymetrics import mean_std_rms  # noqa
from cubie.outputhandling.summarymetrics import mean_std  # noqa
from cubie.outputhandling.summarymetrics import std_rms  # noqa
from cubie.outputhandling.summarymetrics import dxdt_max  # noqa
from cubie.outputhandling.summarymetrics import dxdt_min  # noqa
from cubie.outputhandling.summarymetrics import dxdt_extrema  # noqa
from cubie.outputhandling.summarymetrics import d2xdt2_max  # noqa
from cubie.outputhandling.summarymetrics import d2xdt2_min  # noqa
from cubie.outputhandling.summarymetrics import d2xdt2_extrema  # noqa

__all__ = ["summary_metrics", "register_metric"]
