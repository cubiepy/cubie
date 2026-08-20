"""Tests for the time_logger module."""

import time

import pytest
from cubie.cuda_simsafe import cuda

from cubie.time_logger import (
    CUDAEvent,
    TimeLogger,
    TimingEvent,
    default_timelogger,
)


@cuda.jit
def _busy_kernel(out):
    """Trivial kernel that does enough work to register nonzero time."""
    idx = cuda.grid(1)
    if idx < out.size:
        acc = 0.0
        for i in range(2000):
            acc += i * 0.5
        out[idx] = acc


def _run_busy_kernel(stream):
    """Launch a small kernel on ``stream`` so CUDA events see real work."""
    out = cuda.device_array(256, dtype="float64", stream=stream)
    _busy_kernel[2, 128, stream](out)


class TestTimingEvent:
    """Test TimingEvent dataclass."""

    def test_timing_event_creation(self):
        """Test that TimingEvent can be created with required fields."""
        event = TimingEvent(
            name="test_event",
            event_type="start",
            timestamp=123.456,
        )
        assert event.name == "test_event"
        assert event.event_type == "start"
        assert event.timestamp == 123.456
        assert event.metadata == {}

    def test_timing_event_with_metadata(self):
        """Test TimingEvent with optional metadata."""
        event = TimingEvent(
            name="test_event",
            event_type="progress",
            timestamp=123.456,
            metadata={"message": "Test message"},
        )
        assert event.metadata == {"message": "Test message"}


class TestTimeLogger:
    """Test TimeLogger class."""

    def test_initialization_default(self):
        """Test TimeLogger initialization with default verbosity."""
        logger = TimeLogger()
        assert logger.verbosity is None
        assert logger.events == []

    def test_initialization_verbose(self):
        """Test TimeLogger initialization with verbose level."""
        logger = TimeLogger(verbosity="verbose")
        assert logger.verbosity == "verbose"

    def test_initialization_debug(self):
        """Test TimeLogger initialization with debug level."""
        logger = TimeLogger(verbosity="debug")
        assert logger.verbosity == "debug"

    def test_initialization_none(self):
        """Test TimeLogger initialization with None verbosity."""
        logger = TimeLogger(verbosity=None)
        assert logger.verbosity is None

    def test_initialization_string_none(self):
        """Test TimeLogger initialization with string 'None'."""
        logger = TimeLogger(verbosity='None')
        assert logger.verbosity is None

    def test_initialization_invalid_verbosity(self):
        """Test that invalid verbosity raises ValueError."""
        with pytest.raises(ValueError, match="verbosity must be"):
            TimeLogger(verbosity="invalid")

    def test_none_verbosity_no_op(self):
        """Test that None verbosity creates no-op logger."""
        logger = TimeLogger(verbosity=None)
        # Registration still works even with None verbosity
        logger.register_event("test", "runtime", "Test event")
        logger.start_event("test")
        logger.stop_event("test")
        logger.progress("test", "message")
        assert len(logger.events) == 0

    def test_set_verbosity(self):
        """Test changing verbosity level."""
        logger = TimeLogger(verbosity='default')
        logger.set_verbosity('verbose')
        assert logger.verbosity == 'verbose'
        logger.set_verbosity(None)
        assert logger.verbosity is None

    def test_start_event(self):
        """Test recording a start event."""
        logger = TimeLogger(verbosity="default")
        logger.register_event("test_operation", "runtime", "Test operation")
        logger.start_event("test_operation")

        assert len(logger.events) == 1
        assert logger.events[0].name == "test_operation"
        assert logger.events[0].event_type == "start"
        assert logger.events[0].timestamp > 0

    def test_stop_event(self):
        """Test recording a stop event."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("test_operation", "runtime", "Test operation")
        logger.start_event("test_operation")
        time.sleep(0.01)
        logger.stop_event("test_operation")

        assert len(logger.events) == 2
        assert logger.events[1].name == "test_operation"
        assert logger.events[1].event_type == "stop"
        assert logger.events[1].timestamp > logger.events[0].timestamp

    def test_progress_event(self):
        """Test recording a progress event."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("test_operation", "runtime", "Test operation")
        logger.progress("test_operation", "50% complete")

        assert len(logger.events) == 1
        assert logger.events[0].name == "test_operation"
        assert logger.events[0].event_type == "progress"
        assert logger.events[0].metadata["message"] == "50% complete"

    def test_get_event_duration(self):
        """Test calculating duration between start and stop events.

        Durations are bounded using ``time.perf_counter`` readings taken
        immediately outside and inside the ``start_event``/``stop_event``
        calls. The reported duration must lie between the inner interval
        (measured strictly inside the event) and the outer interval
        (measured around the entire call pair), guaranteeing correctness
        on any platform regardless of sleep precision.
        """
        logger = TimeLogger(verbosity='default')
        logger.register_event("test_operation", "runtime", "Test operation")
        outer_start = time.perf_counter()
        logger.start_event("test_operation")
        inner_start = time.perf_counter()
        time.sleep(0.02)
        inner_end = time.perf_counter()
        logger.stop_event("test_operation")
        outer_end = time.perf_counter()

        duration = logger.get_event_duration("test_operation")
        assert duration is not None
        assert duration >= inner_end - inner_start
        assert duration <= outer_end - outer_start

    def test_get_event_duration_no_stop(self):
        """Test get_event_duration returns None when stop event missing."""
        logger = TimeLogger()
        logger.register_event("test_operation", "runtime", "Test operation")
        logger.start_event("test_operation")

        duration = logger.get_event_duration("test_operation")
        assert duration is None

    def test_get_event_duration_no_start(self):
        """Test get_event_duration returns None when start missing."""
        logger = TimeLogger()
        # This should now raise an error since we require registration and
        # start So this test is no longer valid - removing assertion
        duration = logger.get_event_duration("test_operation")
        assert duration is None

    def test_multiple_operations(self):
        """Test tracking multiple operations."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("operation1", "runtime", "Operation 1")
        logger.register_event("operation2", "runtime", "Operation 2")
        logger.start_event("operation1")
        logger.start_event("operation2")
        logger.stop_event("operation1")
        logger.stop_event("operation2")

        assert len(logger.events) == 4
        assert logger.get_event_duration("operation1") is not None
        assert logger.get_event_duration("operation2") is not None

    def test_callbacks_return_none(self):
        """Test that callbacks work but don't affect functionality."""
        logger = TimeLogger()
        logger.register_event("test", "runtime", "Test event")

        # All callbacks should work without errors
        result1 = logger.start_event("test")
        result2 = logger.stop_event("test")
        result3 = logger.progress("test", "message")

        # None of them return values that would affect code flow
        assert result1 is None
        assert result2 is None
        assert result3 is None

    def test_print_summary_default_verbosity(self, capsys):
        """Test summary output at default verbosity.

        Default mode prints one line per category: "codegen completed in xs"
        """
        logger = TimeLogger(verbosity="default")
        logger.register_event("codegen", "codegen", "Code generation")
        logger.start_event("codegen")
        time.sleep(0.01)
        logger.stop_event("codegen")

        logger.print_summary()
        captured = capsys.readouterr()
        assert "codegen completed in" in captured.out

    def test_print_summary_verbose(self, capsys):
        """Test summary output at verbose level.

        Verbose mode prints inline: "Starting [event]..." then "completed in x
        seconds" when the event stops. Summary shows category totals at the
        end.
        """
        logger = TimeLogger(verbosity="verbose")
        logger.register_event("codegen", "codegen", "Code generation")
        logger.register_event("codegen.component1", "codegen", "Component 1")
        logger.start_event("codegen")
        logger.start_event("codegen.component1")
        time.sleep(0.01)
        logger.stop_event("codegen.component1")
        logger.stop_event("codegen")

        logger.print_summary()
        captured = capsys.readouterr()
        # Verbose mode prints "Starting..." during start_event and "completed
        # in..." during stop_event, then category totals
        assert "Starting" in captured.out
        assert "completed in" in captured.out
        assert "Codegen total:" in captured.out

    def test_print_summary_debug(self, capsys):
        """Test summary output at debug level.

        Debug mode prints individual start/stop messages and a
        category summary at the end.
        """
        logger = TimeLogger(verbosity="debug")
        logger.register_event("test", "runtime", "Test event")
        logger.start_event("test")
        logger.progress("test", "halfway")
        logger.stop_event("test")

        logger.print_summary()
        captured = capsys.readouterr()
        # Debug mode prints during events
        assert "DEBUG" in captured.out
        assert "progress" in captured.out.lower()
        # Debug mode also prints summary at end
        assert "Summary" in captured.out

    def test_get_aggregate_durations(self):
        """Test aggregating event durations.

        Each start/stop pair is bracketed with the logger's own clock
        (time.perf_counter); the aggregate must lie between the sum of
        the inner intervals and the sum of the outer intervals, which
        holds regardless of how long the sleeps actually last.
        """
        logger = TimeLogger(verbosity='default')
        logger.register_event("operation1", "runtime", "Operation 1")
        inner_total = 0.0
        outer_total = 0.0
        for _ in range(2):
            outer_start = time.perf_counter()
            logger.start_event("operation1")
            inner_start = time.perf_counter()
            time.sleep(0.01)
            inner_total += time.perf_counter() - inner_start
            logger.stop_event("operation1")
            outer_total += time.perf_counter() - outer_start

        durations = logger.get_aggregate_durations()
        assert "operation1" in durations
        assert durations["operation1"] >= inner_total
        assert durations["operation1"] <= outer_total

    def test_empty_event_name_raises(self):
        """Test that empty event names raise ValueError."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("valid", "runtime", "Valid event")
        with pytest.raises(ValueError, match="event_name cannot be empty"):
            logger.start_event("")
        with pytest.raises(ValueError, match="event_name cannot be empty"):
            logger.stop_event("")
        with pytest.raises(ValueError, match="event_name cannot be empty"):
            logger.progress("", "message")

    def test_register_event(self):
        """Test registering events with metadata."""
        logger = TimeLogger()
        logger.register_event("dxdt_build", "compile", "Build dxdt function")

        assert "dxdt_build" in logger._event_registry
        assert logger._event_registry["dxdt_build"]["category"] == "compile"
        entry = logger._event_registry["dxdt_build"]
        assert entry["description"] == "Build dxdt function"

    def test_register_event_invalid_category(self):
        """Test that invalid category raises ValueError."""
        logger = TimeLogger()
        with pytest.raises(ValueError, match="category must be"):
            logger.register_event("test", "invalid", "description")

    def test_register_event_valid_categories(self):
        """Test all valid categories."""
        logger = TimeLogger()
        logger.register_event("event1", "codegen", "Codegen event")
        logger.register_event("event2", "runtime", "Runtime event")
        logger.register_event("event3", "compile", "Compile event")

        assert len(logger._event_registry) == 3
        assert logger._event_registry["event1"]["category"] == "codegen"
        assert logger._event_registry["event2"]["category"] == "runtime"
        assert logger._event_registry["event3"]["category"] == "compile"

    def test_register_event_compile_category(self):
        """Test that 'compile' category is accepted."""
        logger = TimeLogger()
        logger.register_event("compile_test", "compile", "Compile event")

        assert "compile_test" in logger._event_registry
        assert logger._event_registry["compile_test"]["category"] == "compile"
        entry = logger._event_registry["compile_test"]
        assert entry["description"] == "Compile event"

    def test_unregistered_event_raises(self):
        """Test that unregistered events raise ValueError."""
        logger = TimeLogger(verbosity='default')
        with pytest.raises(ValueError, match="not registered"):
            logger.start_event("unregistered")
        with pytest.raises(ValueError, match="not registered"):
            logger.stop_event("unregistered")
        with pytest.raises(ValueError, match="not registered"):
            logger.progress("unregistered", "message")

    def test_aggregate_durations_by_category(self):
        """Test filtering aggregate durations by category."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("codegen1", "codegen", "Codegen 1")
        logger.register_event("runtime1", "runtime", "Runtime 1")
        logger.register_event("compile1", "compile", "Compile 1")

        logger.start_event("codegen1")
        time.sleep(0.01)
        logger.stop_event("codegen1")

        logger.start_event("runtime1")
        time.sleep(0.01)
        logger.stop_event("runtime1")

        logger.start_event("compile1")
        time.sleep(0.01)
        logger.stop_event("compile1")

        # Test filtering by category
        codegen_durations = logger.get_aggregate_durations(category="codegen")
        assert "codegen1" in codegen_durations
        assert "runtime1" not in codegen_durations
        assert "compile1" not in codegen_durations

        runtime_durations = logger.get_aggregate_durations(category="runtime")
        assert "runtime1" in runtime_durations
        assert "codegen1" not in runtime_durations
        assert "compile1" not in runtime_durations

    def test_aggregate_durations_compile_category(self):
        """Test filtering aggregate durations for compile category."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("compile1", "compile", "Compile 1")
        logger.register_event("runtime1", "runtime", "Runtime 1")

        outer_start = time.perf_counter()
        logger.start_event("compile1")
        inner_start = time.perf_counter()
        time.sleep(0.01)
        inner_end = time.perf_counter()
        logger.stop_event("compile1")
        outer_end = time.perf_counter()

        logger.start_event("runtime1")
        time.sleep(0.01)
        logger.stop_event("runtime1")

        # Test filtering by compile category. The duration is bracketed
        # with the logger's own clock (time.perf_counter), so the bound
        # holds regardless of sleep precision on any platform.
        compile_durations = logger.get_aggregate_durations(category="compile")
        assert "compile1" in compile_durations
        assert "runtime1" not in compile_durations
        assert compile_durations["compile1"] >= inner_end - inner_start
        assert compile_durations["compile1"] <= outer_end - outer_start

    def test_print_summary_by_category(self, capsys):
        """Test printing summary for specific categories.

        In default mode, print_summary prints one line per category:
        "codegen completed in xs" format. Test that only the requested
        category is printed when filtering.
        """
        # Test codegen category summary
        logger = TimeLogger(verbosity="default")
        logger.register_event("codegen1", "codegen", "Codegen event")
        logger.register_event("compile1", "compile", "Compile event")
        logger.register_event("runtime1", "runtime", "Runtime event")
        logger.start_event("codegen1")
        time.sleep(0.01)
        logger.stop_event("codegen1")
        logger.start_event("compile1")
        time.sleep(0.01)
        logger.stop_event("compile1")
        logger.start_event("runtime1")
        time.sleep(0.01)
        logger.stop_event("runtime1")

        logger.print_summary(category="codegen")
        captured = capsys.readouterr()
        assert "codegen completed in" in captured.out
        assert "compile" not in captured.out
        assert "runtime" not in captured.out

        # Test compile category summary (fresh logger)
        logger = TimeLogger(verbosity="default")
        logger.register_event("codegen1", "codegen", "Codegen event")
        logger.register_event("compile1", "compile", "Compile event")
        logger.register_event("runtime1", "runtime", "Runtime event")
        logger.start_event("codegen1")
        time.sleep(0.01)
        logger.stop_event("codegen1")
        logger.start_event("compile1")
        time.sleep(0.01)
        logger.stop_event("compile1")
        logger.start_event("runtime1")
        time.sleep(0.01)
        logger.stop_event("runtime1")

        logger.print_summary(category="compile")
        captured = capsys.readouterr()
        assert "compile completed in" in captured.out
        assert "codegen" not in captured.out
        assert "runtime" not in captured.out

        # Test runtime category summary (fresh logger)
        logger = TimeLogger(verbosity="default")
        logger.register_event("codegen1", "codegen", "Codegen event")
        logger.register_event("compile1", "compile", "Compile event")
        logger.register_event("runtime1", "runtime", "Runtime event")
        logger.start_event("codegen1")
        time.sleep(0.01)
        logger.stop_event("codegen1")
        logger.start_event("compile1")
        time.sleep(0.01)
        logger.stop_event("compile1")
        logger.start_event("runtime1")
        time.sleep(0.01)
        logger.stop_event("runtime1")

        logger.print_summary(category="runtime")
        captured = capsys.readouterr()
        assert "runtime completed in" in captured.out
        assert "codegen" not in captured.out
        assert "compile" not in captured.out

        # Test all categories summary (fresh logger)
        logger = TimeLogger(verbosity="default")
        logger.register_event("codegen1", "codegen", "Codegen event")
        logger.register_event("compile1", "compile", "Compile event")
        logger.register_event("runtime1", "runtime", "Runtime event")
        logger.start_event("codegen1")
        time.sleep(0.01)
        logger.stop_event("codegen1")
        logger.start_event("compile1")
        time.sleep(0.01)
        logger.stop_event("compile1")
        logger.start_event("runtime1")
        time.sleep(0.01)
        logger.stop_event("runtime1")

        logger.print_summary()
        captured = capsys.readouterr()
        assert "codegen completed in" in captured.out
        assert "compile completed in" in captured.out
        assert "runtime completed in" in captured.out


class TestCUDAEvent:
    """Test CUDAEvent on real GPU (record_start/record_end/elapsed_time_ms)."""

    @pytest.mark.nocudasim
    def test_default_timelogger_used_when_none_provided(self):
        """CUDAEvent falls back to default_timelogger.verbosity."""
        event = CUDAEvent(name="default_logger_event")
        assert event._verbosity == default_timelogger.verbosity

    @pytest.mark.nocudasim
    def test_record_start_and_end_noop_when_verbosity_none(self):
        """record_start/record_end are no-ops when verbosity is None."""
        logger = TimeLogger(verbosity=None)
        event = CUDAEvent(name="noop_event", timelogger=logger)
        stream = cuda.stream()
        event.record_start(stream)
        event.record_end(stream)
        stream.synchronize()
        assert event.elapsed_time_ms() == 0.0

    @pytest.mark.nocudasim
    def test_record_start_and_end_real_gpu(self):
        """record_start/record_end record real CUDA events on the GPU."""
        logger = TimeLogger(verbosity="default")
        event = CUDAEvent(name="gpu_op", timelogger=logger)
        stream = cuda.stream()
        event.record_start(stream)
        _run_busy_kernel(stream)
        event.record_end(stream)
        stream.synchronize()
        elapsed = event.elapsed_time_ms()
        assert elapsed >= 0.0

    @pytest.mark.nocudasim
    def test_elapsed_time_ms_zero_when_start_event_missing(self):
        """elapsed_time_ms returns 0.0 when the start event is None."""
        logger = TimeLogger(verbosity="default")
        event = CUDAEvent(name="missing_start", timelogger=logger)
        event._start_event = None
        assert event.elapsed_time_ms() == 0.0

    @pytest.mark.nocudasim
    def test_elapsed_time_ms_zero_when_end_event_missing(self):
        """elapsed_time_ms returns 0.0 when the end event is None."""
        logger = TimeLogger(verbosity="default")
        event = CUDAEvent(name="missing_end", timelogger=logger)
        event._end_event = None
        assert event.elapsed_time_ms() == 0.0


class TestTimeLoggerExtra:
    """Additional TimeLogger coverage: messaging branches and CUDA events."""

    def test_start_event_called_twice_skips_second(self):
        """A second start_event before stop_event is a no-op."""
        logger = TimeLogger(verbosity="default")
        logger.register_event("dup", "runtime", "Dup event")
        logger.start_event("dup")
        logger.start_event("dup")
        assert len(logger.events) == 1

    def test_start_event_debug_skipped(self, capsys):
        """Debug verbosity prints the skipped message on cache hit."""
        logger = TimeLogger(verbosity="debug")
        logger.register_event("cached", "compile", "Cached op")
        logger.start_event("cached", skipped=True)
        captured = capsys.readouterr()
        assert "Skipped (found in cache)" in captured.out

    def test_start_event_debug_custom_message(self, capsys):
        """Debug verbosity prints a registered custom start message."""
        logger = TimeLogger(verbosity="debug")
        logger.register_event(
            "custom", "runtime", "Custom", start_message="Beginning {label}"
        )
        logger.start_event("custom")
        captured = capsys.readouterr()
        assert "Beginning custom" in captured.out

    def test_start_event_verbose_skipped(self, capsys):
        """Verbose verbosity prints the skipped message on cache hit."""
        logger = TimeLogger(verbosity="verbose")
        logger.register_event("cached2", "compile", "Cached op 2")
        logger.start_event("cached2", skipped=True)
        captured = capsys.readouterr()
        assert "Skipped (found in cache)" in captured.out

    def test_start_event_verbose_custom_message(self, capsys):
        """Verbose verbosity prints a registered custom start message."""
        logger = TimeLogger(verbosity="verbose")
        logger.register_event(
            "custom2", "runtime", "Custom2", start_message="Go {label}"
        )
        logger.start_event("custom2")
        captured = capsys.readouterr()
        assert "Go custom2" in captured.out

    def test_start_event_default_custom_message(self, capsys):
        """Default verbosity prints a registered custom start message."""
        logger = TimeLogger(verbosity="default")
        logger.register_event(
            "custom3", "runtime", "Custom3", start_message="Kickoff {label}"
        )
        logger.start_event("custom3")
        captured = capsys.readouterr()
        assert "Kickoff custom3" in captured.out

    def test_stop_event_without_active_start_is_noop(self):
        """stop_event on an event with no active start is a no-op."""
        logger = TimeLogger(verbosity="default")
        logger.register_event("neverstarted", "runtime", "Never started")
        logger.stop_event("neverstarted")
        assert len(logger.events) == 0

    def test_stop_event_debug_custom_message(self, capsys):
        """Debug verbosity prints a registered custom stop message."""
        logger = TimeLogger(verbosity="debug")
        logger.register_event(
            "custom4",
            "runtime",
            "Custom4",
            stop_message="Done {label} in {duration:.2f}s",
        )
        logger.start_event("custom4")
        logger.stop_event("custom4")
        captured = capsys.readouterr()
        assert "Done custom4" in captured.out

    def test_stop_event_verbose_custom_message(self, capsys):
        """Verbose verbosity prints a registered custom stop message."""
        logger = TimeLogger(verbosity="verbose")
        logger.register_event(
            "custom5",
            "runtime",
            "Custom5",
            stop_message="Finished {label} after {duration:.2f}s",
        )
        logger.start_event("custom5")
        logger.stop_event("custom5")
        captured = capsys.readouterr()
        assert "Finished custom5" in captured.out

    def test_stop_event_default_custom_message(self, capsys):
        """Default verbosity prints a registered custom stop message."""
        logger = TimeLogger(verbosity="default")
        logger.register_event(
            "custom6",
            "runtime",
            "Custom6",
            stop_message="Elapsed {label}: {duration:.2f}s",
        )
        logger.start_event("custom6")
        logger.stop_event("custom6")
        captured = capsys.readouterr()
        assert "Elapsed custom6" in captured.out

    def test_set_verbosity_invalid_raises(self):
        """set_verbosity rejects unrecognised verbosity levels."""
        logger = TimeLogger(verbosity="default")
        with pytest.raises(ValueError, match="verbosity must be"):
            logger.set_verbosity("invalid")

    def test_set_verbosity_string_none_normalized(self):
        """set_verbosity normalizes the string 'None' to None."""
        logger = TimeLogger(verbosity="default")
        logger.set_verbosity("None")
        assert logger.verbosity is None

    def test_print_message_prints_when_verbosity_sufficient(self, capsys):
        """print_message prints when verbosity meets the minimum."""
        logger = TimeLogger(verbosity="verbose")
        logger.print_message("hello there")
        captured = capsys.readouterr()
        assert "hello there" in captured.out

    def test_print_message_suppressed_when_verbosity_insufficient(
        self, capsys
    ):
        """print_message is silent when verbosity is below the minimum."""
        logger = TimeLogger(verbosity="default")
        logger.print_message("hidden message")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_message_noop_when_verbosity_none(self, capsys):
        """print_message is a no-op when verbosity is None."""
        logger = TimeLogger(verbosity=None)
        logger.print_message("no output")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_register_cuda_event_adds_to_registry_and_list(self):
        """CUDAEvent construction registers it with the TimeLogger."""
        logger = TimeLogger(verbosity="default")
        event = CUDAEvent(name="gpu_registered", timelogger=logger)
        assert event in logger._cuda_events
        assert "gpu_registered" in logger._event_registry
        assert (
            logger._event_registry["gpu_registered"]["category"]
            == "runtime"
        )

    def test_register_cuda_event_noop_when_verbosity_none(self):
        """CUDAEvent registration is skipped when verbosity is None."""
        logger = TimeLogger(verbosity=None)
        CUDAEvent(name="gpu_unregistered", timelogger=logger)
        assert logger._cuda_events == []
        assert "gpu_unregistered" not in logger._event_registry

    def test_retrieve_cuda_events_noop_when_verbosity_none(self):
        """_retrieve_cuda_events is a no-op when verbosity is None."""
        logger = TimeLogger(verbosity=None)
        logger._retrieve_cuda_events()
        assert logger._cuda_events == []

    def test_retrieve_cuda_events_noop_when_no_events_registered(self):
        """_retrieve_cuda_events returns early with no pending events."""
        logger = TimeLogger(verbosity="default")
        logger._retrieve_cuda_events()
        assert logger.events == []

    def test_get_category_total_skips_unregistered_cuda_metadata_event(
        self,
    ):
        """_get_category_total skips a duration_ms event whose name was
        never registered (event_info is None)."""
        logger = TimeLogger(verbosity="default")
        logger.events.append(
            TimingEvent(
                name="unregistered_gpu_event",
                event_type="stop",
                timestamp=0.0,
                metadata={"duration_ms": 5.0},
            )
        )
        total = logger._get_category_total("runtime")
        assert total == 0.0

    def test_get_category_total_skips_wrong_category_cuda_metadata_event(
        self,
    ):
        """_get_category_total skips a duration_ms event registered
        under a different category than requested."""
        logger = TimeLogger(verbosity="default")
        logger.register_event("compile_gpu_event", "compile", "Compile op")
        logger.events.append(
            TimingEvent(
                name="compile_gpu_event",
                event_type="stop",
                timestamp=0.0,
                metadata={"duration_ms": 5.0},
            )
        )
        total = logger._get_category_total("runtime")
        assert total == 0.0

    def test_print_summary_debug_skips_unregistered_cuda_metadata_event(
        self, capsys
    ):
        """Debug print_summary skips a duration_ms event whose name was
        never registered (event_info is None)."""
        logger = TimeLogger(verbosity="debug")
        logger.register_event("known_op", "runtime", "Known op")
        logger.start_event("known_op")
        logger.stop_event("known_op")
        logger.events.append(
            TimingEvent(
                name="unregistered_gpu_event2",
                event_type="stop",
                timestamp=0.0,
                metadata={"duration_ms": 5.0},
            )
        )
        logger.print_summary()
        captured = capsys.readouterr()
        assert "unregistered_gpu_event2" not in captured.out

    @pytest.mark.nocudasim
    def test_retrieve_cuda_events_converts_to_timing_events(self):
        """_retrieve_cuda_events appends a stop TimingEvent with duration."""
        logger = TimeLogger(verbosity="default")
        event = CUDAEvent(name="gpu_convert", timelogger=logger)
        stream = cuda.stream()
        event.record_start(stream)
        _run_busy_kernel(stream)
        event.record_end(stream)
        stream.synchronize()

        logger._retrieve_cuda_events()

        assert logger._cuda_events == []
        stop_events = [
            e
            for e in logger.events
            if e.name == "gpu_convert" and e.event_type == "stop"
        ]
        assert len(stop_events) == 1
        assert "duration_ms" in stop_events[0].metadata

    @pytest.mark.nocudasim
    def test_get_category_total_includes_cuda_event_duration(self):
        """_get_category_total sums CUDA event durations for a category."""
        logger = TimeLogger(verbosity="default")
        event = CUDAEvent(name="gpu_total", timelogger=logger)
        stream = cuda.stream()
        event.record_start(stream)
        _run_busy_kernel(stream)
        event.record_end(stream)
        stream.synchronize()

        logger._retrieve_cuda_events()
        total = logger._get_category_total("runtime")
        assert total >= 0.0

    @pytest.mark.nocudasim
    def test_print_summary_runtime_includes_cuda_event(self, capsys):
        """print_summary(category='runtime') retrieves CUDA event timing."""
        logger = TimeLogger(verbosity="default")
        event = CUDAEvent(name="gpu_summary", timelogger=logger)
        stream = cuda.stream()
        event.record_start(stream)
        _run_busy_kernel(stream)
        event.record_end(stream)
        stream.synchronize()

        logger.print_summary(category="runtime")
        captured = capsys.readouterr()
        assert "runtime completed in" in captured.out

    @pytest.mark.nocudasim
    def test_print_summary_debug_prints_cuda_event_ms(self, capsys):
        """Debug print_summary prints CUDA event durations in ms."""
        logger = TimeLogger(verbosity="debug")
        event = CUDAEvent(name="gpu_debug", timelogger=logger)
        stream = cuda.stream()
        event.record_start(stream)
        _run_busy_kernel(stream)
        event.record_end(stream)
        stream.synchronize()

        logger.print_summary()
        captured = capsys.readouterr()
        assert "gpu_debug" in captured.out
        assert "ms" in captured.out


def _solve_logger(chunks=1):
    """Build a logger holding one solve's runtime events."""
    logger = TimeLogger(verbosity="verbose")
    for name in ("solve_ivp", "solver_solve", "gpu_workload"):
        logger.register_event(name, "runtime", name)
    for chunk in range(chunks):
        for prefix in ("h2d_transfer", "kernel", "d2h_transfer"):
            logger.register_event(
                f"{prefix}_chunk_{chunk}", "runtime", prefix
            )
    return logger


def _record_device_solve(logger, chunks=1):
    """Record device durations the way _retrieve_cuda_events would."""
    timings = [("gpu_workload", 19.0 * chunks)]
    for chunk in range(chunks):
        timings += [
            (f"h2d_transfer_chunk_{chunk}", 0.25),
            (f"kernel_chunk_{chunk}", 18.0),
            (f"d2h_transfer_chunk_{chunk}", 0.75),
        ]
    for name, duration_ms in timings:
        logger.events.append(
            TimingEvent(
                name=name,
                event_type="stop",
                timestamp=0.0,
                metadata={"duration_ms": duration_ms},
            )
        )


class TestRuntimeSummary:
    """The runtime total is host time; device time is its own line."""

    def test_total_is_the_host_span(self):
        """Device timings inside the span do not add to the total."""
        logger = _solve_logger()
        logger.start_event("solver_solve")
        time.sleep(0.02)
        logger.stop_event("solver_solve")
        _record_device_solve(logger)

        assert logger._get_category_total("runtime") == (
            logger.get_event_duration("solver_solve")
        )

    def test_outermost_recorded_span_wins(self):
        """solve_ivp supersedes solver_solve once both record."""
        logger = _solve_logger()
        logger.start_event("solve_ivp")
        logger.start_event("solver_solve")
        time.sleep(0.02)
        logger.stop_event("solver_solve")
        logger.stop_event("solve_ivp")

        assert logger._get_category_total("runtime") == (
            logger.get_event_duration("solve_ivp")
        )

    def test_device_line_sums_components_across_chunks(self):
        """Three chunks collapse to one entry per component."""
        logger = _solve_logger(chunks=3)
        _record_device_solve(logger, chunks=3)

        assert logger._device_breakdown_line() == (
            "  device: 0.750ms h2d, 54.000ms kernel, 2.250ms d2h"
        )

    @pytest.mark.parametrize(
        "verbosity, total_prefix",
        [("verbose", "Runtime total:"), ("default", "runtime completed")],
    )
    def test_device_line_follows_the_total(
        self, verbosity, total_prefix, capsys
    ):
        """Both printing verbosities put the device line under the total."""
        logger = _solve_logger()
        logger.set_verbosity(verbosity)
        logger.start_event("solver_solve")
        time.sleep(0.02)
        logger.stop_event("solver_solve")
        _record_device_solve(logger)

        logger.print_summary(category="runtime")
        lines = capsys.readouterr().out.splitlines()
        index = next(
            position
            for position, line in enumerate(lines)
            if line.startswith(total_prefix)
        )
        assert lines[index + 1] == (
            "  device: 0.250ms h2d, 18.000ms kernel, 0.750ms d2h"
        )

    def test_summary_defers_to_the_open_outer_span(self, capsys):
        """The inner summary is skipped and its events reach the outer one."""
        logger = _solve_logger()
        logger.start_event("solve_ivp")
        logger.start_event("solver_solve")
        time.sleep(0.02)
        logger.stop_event("solver_solve")
        _record_device_solve(logger)

        logger.print_summary()
        assert "Runtime total:" not in capsys.readouterr().out

        logger.stop_event("solve_ivp")
        logger.print_summary()
        captured = capsys.readouterr().out
        assert "Runtime total:" in captured
        assert "  device: 0.250ms h2d, 18.000ms kernel, 0.750ms d2h" in (
            captured
        )
