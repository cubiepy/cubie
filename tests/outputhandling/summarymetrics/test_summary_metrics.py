"""Tests for cubie.outputhandling.summarymetrics.metrics."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cubie.outputhandling.summarymetrics.metrics import (
    MetricConfig,
    MetricFuncCache,
    SummaryMetric,
    SummaryMetrics,
    register_metric,
)
from cubie.outputhandling import summary_metrics as global_registry
from cubie.CUDAFactory import CUDAFactoryConfig, CUDADispatcherCache


# ── Concrete subclass for testing abstract SummaryMetric ───────────── #


class _ConcreteMetric(SummaryMetric):
    """Minimal concrete SummaryMetric for testing."""

    def __init__(self, precision, **kwargs):
        defaults = dict(
            buffer_size=1,
            output_size=1,
            name="test_concrete",
            unit_modification="[unit]",
            sample_summaries_every=0.01,
        )
        defaults.update(kwargs)
        super().__init__(precision=precision, **defaults)
        self._update_fn = lambda: "update"
        self._save_fn = lambda: "save"

    def build(self):
        return MetricFuncCache(
            update=self._update_fn, save=self._save_fn
        )


# ── Helper to build a fresh SummaryMetrics with test metrics ───────── #


def _make_registry(precision):
    """Create a fresh SummaryMetrics with two test metrics registered."""
    reg = SummaryMetrics(precision=precision)
    m1 = _ConcreteMetric(
        precision=precision,
        name="alpha",
        buffer_size=2,
        output_size=1,
    )
    m2 = _ConcreteMetric(
        precision=precision,
        name="beta",
        buffer_size=3,
        output_size=2,
        unit_modification="V",
    )
    reg.register_metric(m1)
    reg.register_metric(m2)
    return reg


def _make_registry_with_callable(precision):
    """Registry with a callable-sized metric for parameterised tests."""
    reg = SummaryMetrics(precision=precision)
    m = _ConcreteMetric(
        precision=precision,
        name="sized",
        buffer_size=lambda n: 3 + n,
        output_size=lambda n: n,
        unit_modification="s",
    )
    reg.register_metric(m)
    return reg


# ── MetricFuncCache ────────────────────────────────────────────────── #


def test_metric_func_cache_defaults():
    """MetricFuncCache stores update and save with None defaults."""
    cache = MetricFuncCache()
    assert cache.update is None
    assert cache.save is None


def test_metric_func_cache_stores_callables():
    """MetricFuncCache stores provided update and save callables."""
    def fn_u():
        return "u"

    def fn_s():
        return "s"

    cache = MetricFuncCache(update=fn_u, save=fn_s)
    assert cache.update is fn_u
    assert cache.save is fn_s


def test_metric_func_cache_is_cuda_dispatcher_cache():
    """MetricFuncCache inherits from CUDADispatcherCache."""
    # isinstance is justified: the type itself IS the functionality
    assert issubclass(MetricFuncCache, CUDADispatcherCache)


# ── MetricConfig ───────────────────────────────────────────────────── #


def test_metric_config_default_sample_summaries_every():
    """sample_summaries_every defaults to 0.01."""
    cfg = MetricConfig(precision=np.float32)
    assert cfg.sample_summaries_every == pytest.approx(0.01)


def test_metric_config_custom_sample_summaries_every():
    """sample_summaries_every accepts custom positive values."""
    cfg = MetricConfig(precision=np.float32, sample_summaries_every=0.05)
    assert cfg.sample_summaries_every == pytest.approx(0.05)


def test_metric_config_inherits_precision():
    """MetricConfig inherits precision from CUDAFactoryConfig."""
    # isinstance justified: verifying inheritance IS the functionality
    assert issubclass(MetricConfig, CUDAFactoryConfig)
    cfg = MetricConfig(precision=np.float64)
    assert cfg.precision == np.float64


# ── register_metric decorator ─────────────────────────────────────── #


def test_register_metric_instantiates_with_precision():
    """Decorator instantiates the class with registry.precision."""
    reg = SummaryMetrics(precision=np.float32)
    # Register via the decorator

    @register_metric(reg)
    class _TestMetric(SummaryMetric):
        def __init__(self, precision):
            super().__init__(
                buffer_size=1, output_size=1,
                name="_reg_test", precision=precision,
            )

        def build(self):
            return MetricFuncCache()

    assert "_reg_test" in reg.implemented_metrics
    settings = reg._metric_objects["_reg_test"].compile_settings
    assert settings.precision == np.float32


def test_register_metric_returns_class():
    """Decorator returns the original class, not the instance."""
    reg = SummaryMetrics(precision=np.float32)

    @register_metric(reg)
    class _Cls(SummaryMetric):
        def __init__(self, precision):
            super().__init__(
                buffer_size=1, output_size=1,
                name="_cls_test", precision=precision,
            )

        def build(self):
            return MetricFuncCache()

    # The returned value should be the class, not an instance
    assert isinstance(_Cls, type)
    assert issubclass(_Cls, SummaryMetric)


# ── SummaryMetric.__init__ ─────────────────────────────────────────── #


def test_summary_metric_init_stores_attributes():
    """__init__ stores buffer_size, output_size, name, unit_modification."""
    m = _ConcreteMetric(
        precision=np.float32,
        name="test_m",
        buffer_size=5,
        output_size=3,
        unit_modification="mV",
    )
    assert m.buffer_size == 5
    assert m.output_size == 3
    assert m.name == "test_m"
    assert m.unit_modification == "mV"


def test_summary_metric_init_creates_metric_config():
    """__init__ creates MetricConfig with sample_summaries_every and precision.
    """
    m = _ConcreteMetric(
        precision=np.float64,
        sample_summaries_every=0.05,
    )
    cs = m.compile_settings
    assert cs.sample_summaries_every == pytest.approx(0.05)
    assert cs.precision == np.float64


def test_summary_metric_init_sets_up_compile_settings():
    """__init__ calls setup_compile_settings (compile_settings not None)."""
    m = _ConcreteMetric(precision=np.float32)
    assert m.compile_settings is not None
    # Stronger: verify it's a MetricConfig with expected defaults
    assert m.compile_settings.sample_summaries_every == pytest.approx(0.01)


# ── SummaryMetric properties ──────────────────────────────────────── #


def test_summary_metric_update_device_func():
    """update_device_func returns the cached 'update' function."""
    m = _ConcreteMetric(precision=np.float32)
    # Accessing update_device_func triggers build via get_cached_output
    assert m.update_device_func is m._update_fn


def test_summary_metric_save_device_func():
    """save_device_func returns the cached 'save' function."""
    m = _ConcreteMetric(precision=np.float32)
    assert m.save_device_func is m._save_fn


# ── SummaryMetric.update ──────────────────────────────────────────── #


def test_summary_metric_update_delegates():
    """update() delegates to update_compile_settings with silent=True."""
    m = _ConcreteMetric(precision=np.float32, sample_summaries_every=0.01)
    m.update(sample_summaries_every=0.05)
    assert m.compile_settings.sample_summaries_every == pytest.approx(0.05)


# ── SummaryMetric.build abstract ──────────────────────────────────── #


def test_summary_metric_build_is_abstract():
    """build() is abstract; instantiating SummaryMetric directly raises."""
    with pytest.raises(TypeError, match="abstract"):
        SummaryMetric(
            buffer_size=1, output_size=1,
            name="bad", precision=np.float32,
        )


# ── SummaryMetrics.__attrs_post_init__ ─────────────────────────────── #


def test_summary_metrics_post_init_resets_params():
    """__attrs_post_init__ resets _params to empty dict."""
    reg = SummaryMetrics(precision=np.float32)
    assert reg._params == {}


def test_summary_metrics_post_init_defines_combined_metrics():
    """__attrs_post_init__ defines all expected combined metric mappings."""
    reg = SummaryMetrics(precision=np.float32)
    expected_keys = {
        frozenset(["mean", "std", "rms"]),
        frozenset(["mean", "std"]),
        frozenset(["std", "rms"]),
        frozenset(["max", "min"]),
        frozenset(["dxdt_max", "dxdt_min"]),
        frozenset(["d2xdt2_max", "d2xdt2_min"]),
    }
    assert set(reg._combined_metrics.keys()) == expected_keys
    assert (
        reg._combined_metrics[frozenset(["mean", "std", "rms"])]
        == "mean_std_rms"
    )
    assert reg._combined_metrics[frozenset(["mean", "std"])] == "mean_std"
    assert reg._combined_metrics[frozenset(["std", "rms"])] == "std_rms"
    assert reg._combined_metrics[frozenset(["max", "min"])] == "extrema"
    assert (
        reg._combined_metrics[frozenset(["dxdt_max", "dxdt_min"])]
        == "dxdt_extrema"
    )
    assert (
        reg._combined_metrics[frozenset(["d2xdt2_max", "d2xdt2_min"])]
        == "d2xdt2_extrema"
    )


# ── SummaryMetrics.update ──────────────────────────────────────────── #


def test_summary_metrics_update_precision():
    """update() sets self.precision when 'precision' in kwargs."""
    reg = _make_registry(np.float32)
    reg.update(precision=np.float64)
    assert reg.precision == np.float64


def test_summary_metrics_update_propagates_to_all_metrics():
    """update() propagates kwargs to all registered metric objects."""
    reg = _make_registry(np.float32)
    reg.update(sample_summaries_every=0.07)
    for metric in reg._metric_objects.values():
        assert metric.compile_settings.sample_summaries_every == (
            pytest.approx(0.07)
        )


# ── SummaryMetrics.register_metric ─────────────────────────────────── #


def test_register_metric_duplicate_raises():
    """register_metric raises ValueError for duplicate name."""
    reg = SummaryMetrics(precision=np.float32)
    m1 = _ConcreteMetric(precision=np.float32, name="dup")
    m2 = _ConcreteMetric(precision=np.float32, name="dup")
    reg.register_metric(m1)
    with pytest.raises(ValueError, match="Metric 'dup' is already registered"):
        reg.register_metric(m2)


def test_register_metric_stores_all_data():
    """register_metric appends name, stores sizes, object, and default param.
    """
    reg = SummaryMetrics(precision=np.float32)
    m = _ConcreteMetric(
        precision=np.float32, name="mtest",
        buffer_size=7, output_size=4,
    )
    reg.register_metric(m)
    assert "mtest" in reg._names
    assert reg._buffer_sizes["mtest"] == 7
    assert reg._output_sizes["mtest"] == 4
    assert reg._metric_objects["mtest"] is m
    assert reg._params["mtest"] == 0


# ── SummaryMetrics._apply_combined_metrics ─────────────────────────── #


@pytest.mark.parametrize(
    "request_list, expected_combined, expected_len",
    [
        pytest.param(
            ["mean", "std", "rms"], "mean_std_rms", 1,
            id="mean_std_rms",
        ),
        pytest.param(
            ["mean", "std"], "mean_std", 1,
            id="mean_std",
        ),
        pytest.param(
            ["std", "rms"], "std_rms", 1,
            id="std_rms",
        ),
        pytest.param(
            ["max", "min"], "extrema", 1,
            id="extrema",
        ),
        pytest.param(
            ["dxdt_max", "dxdt_min"], "dxdt_extrema", 1,
            id="dxdt_extrema",
        ),
        pytest.param(
            ["d2xdt2_max", "d2xdt2_min"], "d2xdt2_extrema", 1,
            id="d2xdt2_extrema",
        ),
    ],
)
def test_combined_metrics_substitution(
    request_list,
    expected_combined,
    expected_len,
):
    """Each combined metric pattern is substituted correctly."""
    processed = global_registry.preprocess_request(request_list)
    assert expected_combined in processed
    for original in request_list:
        assert original not in processed
    assert len(processed) == expected_len


def test_combined_metrics_prefers_larger_combination():
    """Larger combinations are preferred over smaller ones."""
    # mean+std+rms should use mean_std_rms, not mean_std + rms
    processed = global_registry.preprocess_request(["mean", "std", "rms"])
    assert "mean_std_rms" in processed
    assert "mean_std" not in processed
    assert "std_rms" not in processed
    assert len(processed) == 1


def test_combined_metrics_preserves_order():
    """Metric order is preserved after substitution."""
    processed = global_registry.preprocess_request(
        ["max_magnitude", "mean", "std"]
    )
    # max_magnitude should come first, then mean_std
    assert processed[0] == "max_magnitude"
    assert processed[1] == "mean_std"


def test_combined_metrics_no_substitution_single_metric():
    """Single metrics from a pair are NOT substituted."""
    for metric in ["mean", "std", "rms", "max", "min"]:
        processed = global_registry.preprocess_request([metric])
        assert metric in processed
        assert len(processed) == 1


def test_combined_metrics_skips_unregistered_combined():
    """No substitution when the combined metric is not registered."""
    reg = SummaryMetrics(precision=np.float32)
    # Register mean and std individually but NOT mean_std
    m_mean = _ConcreteMetric(
        precision=np.float32,
        name="mean",
        buffer_size=1,
        output_size=1,
    )
    m_std = _ConcreteMetric(
        precision=np.float32,
        name="std",
        buffer_size=3,
        output_size=1,
    )
    reg.register_metric(m_mean)
    reg.register_metric(m_std)
    # combined_metrics has mean_std mapping but mean_std is not registered
    processed = reg.preprocess_request(["mean", "std"])
    assert "mean" in processed
    assert "std" in processed
    assert "mean_std" not in processed


# ── SummaryMetrics.preprocess_request ──────────────────────────────── #


def test_preprocess_request_warns_unregistered():
    """preprocess_request warns and removes unregistered metrics."""
    reg = _make_registry(np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = reg.preprocess_request(["alpha", "nonexistent"])
    assert len(w) == 1
    assert "nonexistent" in str(w[0].message)
    assert "not registered" in str(w[0].message)
    assert result == ["alpha"]


def test_preprocess_request_parses_params_and_combines():
    """preprocess_request calls parse_string_for_params and
    _apply_combined_metrics.
    """
    # Use global registry which has all metrics
    result = global_registry.preprocess_request(["mean", "peaks[3]"])
    assert "mean" in result
    assert "peaks" in result
    assert len(result) == 2


# ── SummaryMetrics.implemented_metrics ─────────────────────────────── #


def test_implemented_metrics_returns_names():
    """implemented_metrics returns _names list."""
    reg = _make_registry(np.float32)
    assert reg.implemented_metrics == ["alpha", "beta"]


# ── SummaryMetrics.summaries_buffer_height ─────────────────────────── #


def test_summaries_buffer_height():
    """summaries_buffer_height sums buffer sizes for preprocessed metrics."""
    reg = _make_registry(np.float32)
    # alpha=2, beta=3 => 5
    assert reg.summaries_buffer_height(["alpha", "beta"]) == 5


def test_summaries_buffer_height_empty():
    """summaries_buffer_height returns 0 for empty request."""
    reg = _make_registry(np.float32)
    assert reg.summaries_buffer_height([]) == 0


# ── SummaryMetrics.buffer_offsets ──────────────────────────────────── #


def test_buffer_offsets():
    """buffer_offsets returns cumulative offsets for each metric."""
    reg = _make_registry(np.float32)
    # alpha(size=2) at 0, beta(size=3) at 2
    assert reg.buffer_offsets(["alpha", "beta"]) == (0, 2)


def test_buffer_offsets_empty():
    """buffer_offsets returns empty tuple for empty request."""
    reg = _make_registry(np.float32)
    assert reg.buffer_offsets([]) == ()


# ── SummaryMetrics.buffer_sizes ────────────────────────────────────── #


def test_buffer_sizes():
    """buffer_sizes returns tuple of buffer sizes for each metric."""
    reg = _make_registry(np.float32)
    assert reg.buffer_sizes(["alpha", "beta"]) == (2, 3)


# ── SummaryMetrics.output_offsets ──────────────────────────────────── #


def test_output_offsets():
    """output_offsets returns cumulative output offsets."""
    reg = _make_registry(np.float32)
    # alpha(output=1) at 0, beta(output=2) at 1
    assert reg.output_offsets(["alpha", "beta"]) == (0, 1)


# ── SummaryMetrics.summaries_output_height ─────────────────────────── #


def test_summaries_output_height():
    """summaries_output_height sums output sizes for preprocessed metrics."""
    reg = _make_registry(np.float32)
    # alpha=1, beta=2 => 3
    assert reg.summaries_output_height(["alpha", "beta"]) == 3


# ── SummaryMetrics._get_size ──────────────────────────────────────── #


def test_get_size_int():
    """_get_size returns int directly when size is not callable."""
    reg = _make_registry(np.float32)
    assert reg._get_size("alpha", reg._buffer_sizes) == 2


def test_get_size_callable():
    """_get_size calls size(param) when size is callable."""
    reg = _make_registry_with_callable(np.float32)
    reg._params["sized"] = 5
    assert reg._get_size("sized", reg._buffer_sizes) == 8  # 3 + 5


def test_get_size_callable_warns_param_zero():
    """_get_size warns when callable size has param == 0."""
    reg = _make_registry_with_callable(np.float32)
    reg._params["sized"] = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = reg._get_size("sized", reg._buffer_sizes)
    assert len(w) == 1
    assert "callable size" in str(w[0].message)
    assert "parameter is set to 0" in str(w[0].message)
    assert result == 3  # 3 + 0


# ── SummaryMetrics.legend ──────────────────────────────────────────── #


def test_legend_single_element():
    """Single-element metrics get their name as heading."""
    reg = _make_registry(np.float32)
    assert reg.legend(["alpha"]) == ["alpha"]


def test_legend_multi_element():
    """Multi-element metrics get {name}_1, {name}_2, etc."""
    reg = _make_registry(np.float32)
    # beta has output_size=2
    assert reg.legend(["beta"]) == ["beta_1", "beta_2"]


def test_legend_mixed():
    """Mixed single and multi-element metrics produce correct headings."""
    reg = _make_registry(np.float32)
    assert reg.legend(["alpha", "beta"]) == ["alpha", "beta_1", "beta_2"]


# ── SummaryMetrics.unit_modifications ──────────────────────────────── #


def test_unit_modifications_single():
    """Returns one unit modification per output element."""
    reg = _make_registry(np.float32)
    assert reg.unit_modifications(["alpha"]) == ["[unit]"]


def test_unit_modifications_multi():
    """Multi-element outputs repeat the same modification."""
    reg = _make_registry(np.float32)
    # beta has output_size=2, unit_modification="V"
    assert reg.unit_modifications(["beta"]) == ["V", "V"]


# ── SummaryMetrics.output_sizes ────────────────────────────────────── #


def test_output_sizes():
    """output_sizes returns tuple of output sizes."""
    reg = _make_registry(np.float32)
    assert reg.output_sizes(["alpha", "beta"]) == (1, 2)


# ── SummaryMetrics.save_functions / update_functions ───────────────── #


def test_save_functions():
    """save_functions returns tuple of save device funcs from metric objects.
    """
    reg = _make_registry(np.float32)
    fns = reg.save_functions(["alpha", "beta"])
    assert len(fns) == 2
    # save_device_func triggers build; verify identity with metric objects
    assert fns[0] is reg._metric_objects["alpha"].save_device_func
    assert fns[1] is reg._metric_objects["beta"].save_device_func


def test_update_functions():
    """update_functions returns tuple of update device funcs from metric
    objects.
    """
    reg = _make_registry(np.float32)
    fns = reg.update_functions(["alpha", "beta"])
    assert len(fns) == 2
    assert fns[0] is reg._metric_objects["alpha"].update_device_func
    assert fns[1] is reg._metric_objects["beta"].update_device_func


# ── SummaryMetrics.params ──────────────────────────────────────────── #


def test_params_default():
    """params returns default (0) for metrics without [N] suffix."""
    reg = _make_registry(np.float32)
    assert reg.params(["alpha", "beta"]) == (0, 0)


def test_params_with_parsed():
    """params returns parsed parameter values."""
    reg = _make_registry_with_callable(np.float32)
    assert reg.params(["sized[7]"]) == (7,)


# ── SummaryMetrics.parse_string_for_params ─────────────────────────── #


def test_parse_extracts_param():
    """Extracts [N] parameter from metric string."""
    reg = SummaryMetrics(precision=np.float32)
    result = reg.parse_string_for_params(["foo[3]"])
    assert result == ["foo"]
    assert reg._params["foo"] == 3


def test_parse_raises_non_integer():
    """Raises ValueError for non-integer parameter."""
    reg = SummaryMetrics(precision=np.float32)
    with pytest.raises(ValueError, match="must be an integer"):
        reg.parse_string_for_params(["foo[abc]"])


def test_parse_raises_float():
    """Raises ValueError for float parameter."""
    reg = SummaryMetrics(precision=np.float32)
    with pytest.raises(ValueError, match="must be an integer"):
        reg.parse_string_for_params(["foo[3.14]"])


def test_parse_stores_in_params():
    """Parsed params are stored in _params dict."""
    reg = SummaryMetrics(precision=np.float32)
    reg.parse_string_for_params(["a[10]", "b"])
    assert reg._params["a"] == 10
    assert reg._params["b"] == 0


def test_parse_default_param_zero():
    """Metrics without [N] get default param 0."""
    reg = SummaryMetrics(precision=np.float32)
    reg.parse_string_for_params(["plain"])
    assert reg._params["plain"] == 0


def test_parse_resets_params_each_call():
    """_params is reset on each call to parse_string_for_params."""
    reg = SummaryMetrics(precision=np.float32)
    reg.parse_string_for_params(["a[5]"])
    assert "a" in reg._params
    reg.parse_string_for_params(["b[7]"])
    assert "a" not in reg._params
    assert reg._params["b"] == 7


# ── Real global registry integration ──────────────────────────────── #


EXPECTED_METRICS = [
    "mean", "max", "rms", "peaks", "std", "min", "max_magnitude",
    "extrema", "negative_peaks", "mean_std_rms", "mean_std", "std_rms",
    "dxdt_max", "dxdt_min", "dxdt_extrema",
    "d2xdt2_max", "d2xdt2_min", "d2xdt2_extrema",
]


def test_global_registry_has_all_expected_metrics():
    """Global registry has all 18 expected metrics registered."""
    available = global_registry.implemented_metrics
    for name in EXPECTED_METRICS:
        assert name in available, f"Missing metric: {name}"
    assert len(available) == len(EXPECTED_METRICS)


# ── Per-metric buffer_size, output_size, unit_modification table ───── #


@pytest.mark.parametrize(
    "name, buf, out, unit_mod",
    [
        pytest.param("mean", 1, 1, "[unit]", id="mean"),
        pytest.param("max", 1, 1, "[unit]", id="max"),
        pytest.param("min", 1, 1, "[unit]", id="min"),
        pytest.param("rms", 1, 1, "[unit]", id="rms"),
        pytest.param("std", 3, 1, "[unit]", id="std"),
        pytest.param("max_magnitude", 1, 1, "[unit]", id="max_magnitude"),
        pytest.param("extrema", 2, 2, "[unit]", id="extrema"),
        pytest.param("mean_std", 3, 2, "[unit]", id="mean_std"),
        pytest.param("mean_std_rms", 3, 3, "[unit]", id="mean_std_rms"),
        pytest.param("std_rms", 3, 2, "[unit]", id="std_rms"),
        pytest.param("dxdt_max", 2, 1, "[unit]*s^-1", id="dxdt_max"),
        pytest.param("dxdt_min", 2, 1, "[unit]*s^-1", id="dxdt_min"),
        pytest.param("dxdt_extrema", 3, 2, "[unit]*s^-1", id="dxdt_extrema"),
        pytest.param("d2xdt2_max", 3, 1, "[unit]*s^-2", id="d2xdt2_max"),
        pytest.param("d2xdt2_min", 3, 1, "[unit]*s^-2", id="d2xdt2_min"),
        pytest.param(
            "d2xdt2_extrema",
            4,
            2,
            "[unit]*s^-2",
            id="d2xdt2_extrema",
        ),
    ],
)
def test_metric_sizes_and_unit(name, buf, out, unit_mod):
    """Each metric has the expected buffer_size, output_size, and unit."""
    m = global_registry._metric_objects[name]
    assert m.buffer_size == buf
    assert m.output_size == out
    assert m.unit_modification == unit_mod


# ── Parameterised metrics (peaks, negative_peaks) ─────────────────── #


@pytest.mark.parametrize(
    "name, n",
    [
        pytest.param("peaks", 3, id="peaks-3"),
        pytest.param("peaks", 5, id="peaks-5"),
        pytest.param("negative_peaks", 3, id="neg_peaks-3"),
        pytest.param("negative_peaks", 5, id="neg_peaks-5"),
    ],
)
def test_parameterised_metric_sizes(name, n):
    """Parameterised metrics compute correct buffer and output sizes."""
    request = [f"{name}[{n}]"]
    buf = global_registry.buffer_sizes(request)
    out = global_registry.output_sizes(request)
    assert buf == (3 + n,)
    assert out == (n,)


def test_parameterised_metric_unit_modification():
    """peaks and negative_peaks have unit_modification 's'."""
    assert global_registry._metric_objects["peaks"].unit_modification == "s"
    negative_peaks = global_registry._metric_objects["negative_peaks"]
    assert negative_peaks.unit_modification == "s"


# ── Combined metrics buffer efficiency ─────────────────────────────── #


def test_combined_mean_std_rms_buffer_height():
    """mean+std+rms combined uses 3 buffer slots (not 1+3+1=5 separate)."""
    assert global_registry.summaries_buffer_height(["mean", "std", "rms"]) == 3


def test_combined_extrema_buffer_height():
    """max+min combined uses 2 buffer slots (not 1+1=2 separate but as
    extrema).
    """
    assert global_registry.summaries_buffer_height(["max", "min"]) == 2


def test_combined_dxdt_extrema_buffer_height():
    """dxdt_max+dxdt_min combined uses 3 buffer slots."""
    assert global_registry.summaries_buffer_height(
        ["dxdt_max", "dxdt_min"]
    ) == 3


def test_combined_d2xdt2_extrema_buffer_height():
    """d2xdt2_max+d2xdt2_min combined uses 4 buffer slots."""
    assert global_registry.summaries_buffer_height(
        ["d2xdt2_max", "d2xdt2_min"]
    ) == 4


# ── Multiple combinations in one request ───────────────────────────── #


def test_multiple_independent_combinations():
    """Multiple independent combinations are all applied."""
    processed = global_registry.preprocess_request(
        ["mean", "std", "rms", "max", "min"]
    )
    assert "mean_std_rms" in processed
    assert "extrema" in processed
    assert len(processed) == 2


# ── Real registry offset calculations ──────────────────────────────── #


def test_real_registry_offsets():
    """Offset calculations with real metrics are correct."""
    requested = ["mean", "peaks[2]", "max", "rms"]
    buf_offsets = global_registry.buffer_offsets(requested)
    out_offsets = global_registry.output_offsets(requested)
    # mean(buf=1) at 0, peaks[2](buf=5) at 1, max(buf=1) at 6, rms(buf=1) at 7
    assert buf_offsets == (0, 1, 6, 7)
    # mean(out=1) at 0, peaks[2](out=2) at 1, max(out=1) at 3, rms(out=1) at 4
    assert out_offsets == (0, 1, 3, 4)


# ── Legend with real combined metrics ──────────────────────────────── #


def test_legend_combined_metric():
    """Combined mean_std_rms reports its declared constituent names."""
    headings = global_registry.legend(["mean", "std", "rms"])
    assert headings == ["mean", "std", "rms"]


def test_legend_single_output_metrics():
    """Single-output metrics produce their name as heading."""
    headings = global_registry.legend(["mean", "max", "rms"])
    assert headings == ["mean", "max", "rms"]


def test_legend_extrema_combined_metric():
    """Combined extrema reports max/min under their requested names."""
    headings = global_registry.legend(["max", "min"])
    assert headings == ["max", "min"]


def test_legend_dxdt_extrema_combined_metric():
    """Combined dxdt_extrema reports dxdt_max/dxdt_min names."""
    headings = global_registry.legend(["dxdt_max", "dxdt_min"])
    assert headings == ["dxdt_max", "dxdt_min"]


def test_legend_d2xdt2_extrema_combined_metric():
    """Combined d2xdt2_extrema reports d2xdt2_max/d2xdt2_min names."""
    headings = global_registry.legend(["d2xdt2_max", "d2xdt2_min"])
    assert headings == ["d2xdt2_max", "d2xdt2_min"]


def test_legend_std_rms_combined_metric():
    """Combined std_rms reports std/rms names."""
    headings = global_registry.legend(["std", "rms"])
    assert headings == ["std", "rms"]


def test_legend_mean_std_combined_metric():
    """Combined mean_std reports mean/std names."""
    headings = global_registry.legend(["mean", "std"])
    assert headings == ["mean", "std"]


def test_summary_metric_rejects_mismatched_output_names(precision):
    """output_names length must equal a fixed (non-callable) output_size."""
    with pytest.raises(ValueError, match="output_names has"):
        _ConcreteMetric(
            precision=precision,
            output_size=2,
            output_names=["only_one"],
        )


def test_every_registered_metric_builds():
    """Every built-in metric exposes populated cached device functions.

    The cached properties route through ``CUDAFactory`` to each
    metric's build(), which only assembles closures with
    ``@cuda.jit(device=True)`` decorators; without a kernel launch
    that never triggers a real compile, so reaching them for every
    registered metric is cheap and exercises the host-side setup and
    return lines that a single parametrised algorithm test would not.
    """
    for name, metric in global_registry._metric_objects.items():
        assert callable(metric.update_device_func), name
        assert callable(metric.save_device_func), name
