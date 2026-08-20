"""Time logging infrastructure for tracking CuBIE compilation and
runtime performance.

Published Classes
-----------------
:class:`TimeLogger`
    Callback-based timing system with configurable verbosity levels.

    >>> logger = TimeLogger(verbosity="default")
    >>> logger.register_event("build", "compile", "Build step")
    >>> logger.start_event("build")
    >>> logger.stop_event("build")  # doctest: +SKIP

:class:`CUDAEvent`
    CUDA event pair for GPU timeline timing with CUDASIM fallback.

Module-Level Instances
----------------------
:data:`default_timelogger`
    Global singleton ``TimeLogger`` (default verbosity ``None``).

See Also
--------
:mod:`cubie.cubie_cache`
    Uses ``default_timelogger`` for compile timing.
:class:`~cubie.batchsolving.BatchSolverKernel.BatchSolverKernel`
    Creates ``CUDAEvent`` instances for kernel timing.
"""

import time
from time import perf_counter
from typing import Optional, Any
import attrs
from cubie.cuda_simsafe import is_cudasim_enabled
from cubie.cuda_simsafe import cuda

RUNTIME_SPANS = ("solve_ivp", "solver_solve")
"""Runtime spans, outermost first. The first one recorded is the total."""

RUNTIME_DEVICE_COMPONENTS = (
    ("h2d_transfer", "h2d"),
    ("kernel", "kernel"),
    ("d2h_transfer", "d2h"),
)
"""Device event name prefixes and the labels they print under."""

RUNTIME_DEVICE_SPAN = "gpu_workload"
"""Device event enclosing the per-chunk components."""


def _is_device_event(name: str) -> bool:
    """Check whether a runtime event name is timed on the device."""
    prefixes = tuple(prefix for prefix, _ in RUNTIME_DEVICE_COMPONENTS)
    return name == RUNTIME_DEVICE_SPAN or name.startswith(prefixes)


@attrs.define(frozen=True)
class TimingEvent:
    """Record of a single timing event.

    Attributes
    ----------
    name : str
        Identifier for the event (e.g., 'dxdt_compilation')
    event_type : str
        Type of event: 'start', 'stop', or 'progress'
    timestamp : float
        Wall-clock time from time.perf_counter()
    metadata : dict
        Optional metadata (file names, sizes, counts, etc.)
    """

    name: str = attrs.field(validator=attrs.validators.instance_of(str))
    event_type: str = attrs.field(
        validator=attrs.validators.in_({"start", "stop", "progress"})
    )
    timestamp: float = attrs.field(
        validator=attrs.validators.instance_of(float)
    )
    metadata: dict = attrs.field(factory=dict)


class CUDAEvent:
    """CUDA event pair for timing measurements with CUDASIM fallback.

    Parameters
    ----------
    name : str, default="unnamed_cuda_event"
        Event identifier (e.g., "kernel_chunk_0")
    timelogger : TimeLogger, optional
        TimeLogger instance for registration. If None, uses default_timelogger.

    Attributes
    ----------
    name : str
        Event identifier
    _start_event : cuda.event or None
        Start event object (CUDA mode)
    _end_event : cuda.event or None
        End event object (CUDA mode)
    _start_time : float or None
        Start timestamp (CUDASIM mode)
    _end_time : float or None
        End timestamp (CUDASIM mode)
    _verbosity : str or None
        TimeLogger verbosity (for no-op when None)
    """

    def __init__(
        self, name: str = "unnamed_cuda_event", timelogger=None
    ) -> None:
        self.name = name

        # Get TimeLogger instance for registration and verbosity check
        if timelogger is None:
            timelogger = default_timelogger
        self._verbosity = timelogger.verbosity

        # Skip driver-event allocation when verbosity is None: every record/
        # elapsed method is a no-op then, so the batch kernel's 4 events/solve
        # would be created and discarded for nothing.
        if self._verbosity is not None and not is_cudasim_enabled():
            self._start_event = cuda.event()
            self._end_event = cuda.event()
        else:  # pragma: no cover - simulated / timing disabled
            self._start_event = None
            self._end_event = None
        self._start_time = None
        self._end_time = None

        # Register with TimeLogger
        timelogger._register_cuda_event(self)

    def record_start(self, stream) -> None:
        """Record start timestamp on given stream.

        Parameters
        ----------
        stream
            CUDA stream on which to record event

        Notes
        -----
        No-op when verbosity is None.
        Must be called before record_end().
        """
        if self._verbosity is None:
            return

        if not is_cudasim_enabled():
            self._start_event.record(stream)
        else:  # pragma: no cover - simulated
            self._start_time = perf_counter()

    def record_end(self, stream) -> None:
        """Record end timestamp on given stream.

        Parameters
        ----------
        stream
            CUDA stream on which to record event

        Notes
        -----
        No-op when verbosity is None.
        Must be called after record_start().
        """
        if self._verbosity is None:
            return

        if not is_cudasim_enabled():
            self._end_event.record(stream)
        else:  # pragma: no cover - simulated
            self._end_time = perf_counter()

    def elapsed_time_ms(self) -> float:
        """Calculate elapsed time in milliseconds.

        Returns
        -------
        float
            Elapsed time in milliseconds

        Notes
        -----
        This method must NOT block or synchronize. It should be called
        AFTER the stream has been synchronized externally.
        In CUDA mode, uses cuda.event_elapsed_time() which returns
        immediately post-sync.
        In CUDASIM mode, calculates from stored timestamps.

        Returns 0.0 if verbosity is None or if both start and end have
        not been recorded.
        """
        if self._verbosity is None:
            return 0.0

        if not is_cudasim_enabled():
            if self._start_event is None or self._end_event is None:
                return 0.0
            return cuda.event_elapsed_time(self._start_event, self._end_event)
        else:  # pragma: no cover - simulated
            if self._start_time is None or self._end_time is None:
                return 0.0
            return (self._end_time - self._start_time) * 1000.0


class TimeLogger:
    """Callback-based timing system for CuBIE operations.

    Parameters
    ----------
    verbosity : str or None, default='default'
        Output verbosity level. Options:
        - 'default': Aggregate times only
        - 'verbose': Component-level breakdown
        - 'debug': All events with start/stop/progress
        - None or 'None': No-op callbacks with zero overhead

    Attributes
    ----------
    verbosity : str or None
        Current verbosity level
    events : list[TimingEvent]
        Chronological list of all recorded events
    _active_starts : dict[str, float]
        Map of event names to their start timestamps (for matching)
    _event_registry : dict[str, dict]
        Registry of event metadata (category, description) by label

    Notes
    -----
    A default instance is available as cubie.time_logger.default_timelogger.
    Use set_verbosity() to configure the global logger level.
    """

    def __init__(self, verbosity: Optional[str] = None) -> None:
        if verbosity not in {"default", "verbose", "debug", None, "None"}:
            raise ValueError(
                f"verbosity must be 'default', 'verbose', 'debug', "
                f"None, or 'None', got '{verbosity}'"
            )
        # Normalize string 'None' to None
        if verbosity == "None":
            verbosity = None
        self.verbosity = verbosity
        self.events: list[TimingEvent] = []
        self._active_starts: dict[str, tuple[float, bool]] = {}
        self._event_registry: dict[str, dict] = {}
        self._cuda_events: list = []  # Stores CUDAEvent instances

    def start_event(
        self, event_name: str, skipped: bool = False, **metadata: Any
    ) -> None:
        """Record the start of a timed operation.

        Parameters
        ----------
        event_name : str
            Unique identifier for this event
        skipped : bool, default=False
            If True, the event was found in cache and skipped
        **metadata : Any
            Optional metadata to store with event

        Raises
        ------
        ValueError
            If event_name is empty, not registered, or already has an
            active start.

        Notes
        -----
        Events must be registered with _register_event before use.
        No-op when verbosity is None.
        """
        if self.verbosity is None:
            return

        if not event_name:
            raise ValueError("event_name cannot be empty")

        if event_name not in self._event_registry:
            raise ValueError(
                f"Event '{event_name}' not registered. "
                "Call _register_event() before using this event."
            )

        # If a job is logged twice for some reason, skip subsequent starts
        # to capture from first start to first return
        if event_name in self._active_starts:
            return

        timestamp = time.perf_counter()
        # Add skipped flag to metadata
        event_metadata = dict(metadata)
        event_metadata["skipped"] = skipped
        event = TimingEvent(
            name=event_name,
            event_type="start",
            timestamp=timestamp,
            metadata=event_metadata,
        )
        self.events.append(event)
        # Store both timestamp and skipped flag for efficient lookup in stop
        self._active_starts[event_name] = (timestamp, skipped)

        # Get custom start message if registered
        event_info = self._event_registry.get(event_name, {})
        custom_start = event_info.get("start_message")

        if self.verbosity == "debug":
            if skipped:
                print(
                    f"TIMELOGGER [DEBUG] Started: {event_name}... "
                    f"Skipped (found in cache)"
                )
            elif custom_start:
                print(
                    f"TIMELOGGER [DEBUG] "
                    f"{custom_start.format(label=event_name)}"
                )
            else:
                print(f"TIMELOGGER [DEBUG] Started: {event_name}")
        elif self.verbosity == "verbose":
            if skipped:
                print(f"Starting {event_name}... Skipped (found in cache)")
            elif custom_start:
                print(
                    custom_start.format(label=event_name), end="", flush=True
                )
            else:
                print(f"Starting {event_name}...", end="", flush=True)
        elif self.verbosity == "default" and custom_start:
            print(custom_start.format(label=event_name))

    def stop_event(self, event_name: str, **metadata: Any) -> None:
        """Record the end of a timed operation.

        Parameters
        ----------
        event_name : str
            Identifier matching a previous start_event call
        **metadata : Any
            Optional metadata to store with event

        Raises
        ------
        ValueError
            If event_name is empty, not registered, or has no active start.

        Notes
        -----
        Events must be registered with _register_event before use.
        A matching start_event must be called before stop_event.
        No-op when verbosity is None.
        """
        if self.verbosity is None:
            return

        if not event_name:
            raise ValueError("event_name cannot be empty")

        if event_name not in self._event_registry:
            raise ValueError(
                f"Event '{event_name}' not registered. "
                "Call _register_event() before using this event."
            )

        if event_name not in self._active_starts:
            return  # skip extra stops if called twice

        start_timestamp, was_skipped = self._active_starts[event_name]
        timestamp = time.perf_counter()
        duration = timestamp - start_timestamp
        del self._active_starts[event_name]

        event = TimingEvent(
            name=event_name,
            event_type="stop",
            timestamp=timestamp,
            metadata=metadata,
        )
        self.events.append(event)

        # Get custom stop message if registered
        event_info = self._event_registry.get(event_name, {})
        custom_stop = event_info.get("stop_message")

        if self.verbosity == "debug":
            if custom_stop:
                stop_message = custom_stop.format(
                    label=event_name, duration=duration
                )
                print(f"TIMELOGGER [DEBUG] {stop_message}")
            else:
                print(
                    f"TIMELOGGER [DEBUG] Stopped: {event_name} "
                    f"({duration:.3f}s)"
                )
        elif self.verbosity == "verbose":
            # Only print completion for non-skipped events
            # (skipped events already printed their status in start_event)
            if not was_skipped:
                if custom_stop:
                    print(
                        custom_stop.format(label=event_name, duration=duration)
                    )
                else:
                    print(f" completed in {duration:.3f}s")
        elif self.verbosity == "default" and custom_stop:
            print(custom_stop.format(label=event_name, duration=duration))

    def progress(self, event_name: str, message: str, **metadata: Any) -> None:
        """Record a progress update within an operation.

        Parameters
        ----------
        event_name : str
            Identifier for the operation in progress
        message : str
            Progress message to log
        **metadata : Any
            Optional metadata to store with event

        Raises
        ------
        ValueError
            If event_name is empty or not registered.

        Notes
        -----
        Events must be registered with _register_event before use.
        Progress events don't require matching start/stop.
        Only printed in debug mode.
        No-op when verbosity is None.
        """
        if self.verbosity is None:
            return

        if not event_name:
            raise ValueError("event_name cannot be empty")

        if event_name not in self._event_registry:
            raise ValueError(
                f"Event '{event_name}' not registered. "
                "Call _register_event() before using this event."
            )

        timestamp = time.perf_counter()
        metadata_with_msg = dict(metadata)
        metadata_with_msg["message"] = message
        event = TimingEvent(
            name=event_name,
            event_type="progress",
            timestamp=timestamp,
            metadata=metadata_with_msg,
        )
        self.events.append(event)

        if self.verbosity == "debug":
            print(f"TIMELOGGER [DEBUG] Progress: {event_name} - {message}")

    def get_event_duration(self, event_name: str) -> Optional[float]:
        """Query duration of a completed event.

        Parameters
        ----------
        event_name : str
            Name of event to query

        Returns
        -------
        float or None
            Duration in seconds, or None if no matching start/stop pair

        Notes
        -----
        Returns duration of most recent completed event with this name.
        """
        start_time = None
        stop_time = None

        # Search backwards for most recent pair
        for event in reversed(self.events):
            if event.name == event_name:
                if event.event_type == "stop" and stop_time is None:
                    stop_time = event.timestamp
                elif event.event_type == "start" and stop_time is not None:
                    start_time = event.timestamp
                    break

        if start_time is not None and stop_time is not None:
            return stop_time - start_time
        return None

    def get_aggregate_durations(
        self, category: Optional[str] = None
    ) -> dict[str, float]:
        """Aggregate event durations by category or all events.

        Parameters
        ----------
        category : str, optional
            If provided, filter events by their registered category
            ('codegen', 'runtime', or 'compile')

        Returns
        -------
        dict[str, float]
            Mapping of event names to total durations

        Notes
        -----
        Sums all durations for events with the same name.
        Uses the event registry to filter by category.
        """
        durations: dict[str, float] = {}
        event_starts: dict[str, float] = {}

        for event in self.events:
            # Filter by category using registry if requested
            if category is not None:
                event_info = self._event_registry.get(event.name)
                if event_info is None or event_info["category"] != category:
                    continue

            if event.event_type == "start":
                event_starts[event.name] = event.timestamp
            elif event.event_type == "stop":
                if event.name in event_starts:
                    duration = event.timestamp - event_starts[event.name]
                    durations[event.name] = (
                        durations.get(event.name, 0.0) + duration
                    )
                    del event_starts[event.name]

        return durations

    def _get_category_durations(self, cat: str) -> dict[str, float]:
        """Collect every recorded duration in a category, in seconds.

        Parameters
        ----------
        cat : str
            Category name ('codegen', 'compile', or 'runtime')

        Returns
        -------
        dict[str, float]
            Event names mapped to durations in seconds, in record order.
        """
        durations = self.get_aggregate_durations(category=cat)

        # Add CUDA event durations from metadata
        for event in self.events:
            if event.event_type == "stop" and "duration_ms" in event.metadata:
                event_info = self._event_registry.get(event.name)
                if event_info is None or event_info["category"] != cat:
                    continue
                # Convert ms to seconds for consistency
                duration_s = event.metadata["duration_ms"] / 1000.0
                durations[event.name] = duration_s

        return durations

    def _get_category_total(self, cat: str) -> float:
        """Calculate total duration for a category.

        Parameters
        ----------
        cat : str
            Category name ('codegen', 'compile', or 'runtime')

        Returns
        -------
        float
            Total duration in seconds for the category

        Notes
        -----
        The runtime total is host time: the outermost recorded span in
        ``RUNTIME_SPANS``, or the host events summed when none recorded.
        """
        durations = self._get_category_durations(cat)

        if cat == "runtime":
            for name in RUNTIME_SPANS:
                if name in durations:
                    return durations[name]
            return sum(
                duration
                for name, duration in durations.items()
                if not _is_device_event(name)
            )

        return sum(durations.values())

    def _device_breakdown_line(self) -> Optional[str]:
        """Render the runtime device components as one line, chunks summed."""
        durations = self._get_category_durations("runtime")

        parts = []
        for prefix, label in RUNTIME_DEVICE_COMPONENTS:
            total = sum(
                duration
                for name, duration in durations.items()
                if name.startswith(prefix)
            )
            if total > 0:
                parts.append(f"{total * 1000.0:.3f}ms {label}")

        if not parts:
            return None
        return "  device: " + ", ".join(parts)

    def print_summary(self, category: Optional[str] = None) -> None:
        """Print timing summary for all events or specific category.

        Parameters
        ----------
        category : str, optional
            If provided, print summary only for events in this category
            ('codegen', 'runtime', or 'compile'). If None, print all events.

        Notes
        -----
        Verbosity modes:
        - 'default': One line per category, e.g., "codegen completed in 1.23s"
        - 'verbose': Inline timing already printed; category summaries at end
        - 'debug': Individual start/stop already printed; category summaries

        A call made while an outer runtime span is open is skipped.
        Automatically retrieves CUDA event timings when category='runtime'.
        Events are cleared after printing to prevent bleeding between calls.
        """
        if self.verbosity is None:
            return

        if any(name in self._active_starts for name in RUNTIME_SPANS):
            return

        # Retrieve CUDA event timings when printing runtime summary
        if category == "runtime" or category is None:
            self._retrieve_cuda_events()

        if category is None:
            categories_to_print = ["codegen", "compile", "runtime"]
        else:
            categories_to_print = [category]

        if self.verbosity == "default":
            # Default mode: print one line per category
            for cat in categories_to_print:
                total = self._get_category_total(cat)
                if total > 0:
                    print(f"{cat} completed in {total:.3f}s")
                    if cat == "runtime":
                        device = self._device_breakdown_line()
                        if device:
                            print(device)

        elif self.verbosity == "verbose":
            # Verbose mode: individual timings printed inline during events
            # Now print category summaries at the end
            for cat in categories_to_print:
                total = self._get_category_total(cat)
                if total > 0:
                    print(f"\n{cat.capitalize()} total: {total:.3f}s")
                    if cat == "runtime":
                        device = self._device_breakdown_line()
                        if device:
                            print(device)

        elif self.verbosity == "debug":
            # Debug mode: individual start/stop already printed inline
            # Now print category summaries at the end
            for cat in categories_to_print:
                events_to_print = []

                # Get standard durations from start/stop pairs
                durations = self.get_aggregate_durations(category=cat)

                for name, duration in durations.items():
                    events_to_print.append((name, duration, False, cat))

                # Add CUDA event durations from metadata
                for event in self.events:
                    if event.event_type == "stop":
                        if "duration_ms" in event.metadata:
                            event_info = self._event_registry.get(event.name)
                            if event_info is None:
                                continue
                            if event_info["category"] != cat:
                                continue
                            duration_ms = event.metadata["duration_ms"]
                            events_to_print.append(
                                (event.name, duration_ms, True, cat)
                            )

                # Print events for this category
                if events_to_print:
                    print(f"\nTIMELOGGER {cat.capitalize()} Summary:")
                    for name, duration, is_cuda, _ in sorted(
                        events_to_print, key=lambda x: x[0]
                    ):
                        if is_cuda:
                            print(f"TIMELOGGER   {name}: {duration:.3f}ms")
                        else:
                            print(f"TIMELOGGER   {name}: {duration:.3f}s")

        # Clear events after printing to prevent bleeding between calls
        self._clear_events()

    def set_verbosity(self, verbosity: Optional[str]) -> None:
        """Set the verbosity level for this logger.

        Parameters
        ----------
        verbosity : str or None
            New verbosity level. Options are 'default', 'verbose',
            'debug', None, or 'None'.

        Notes
        -----
        Changing verbosity does not clear existing events.
        """
        if verbosity not in {"default", "verbose", "debug", None, "None"}:
            raise ValueError(
                f"verbosity must be 'default', 'verbose', 'debug', "
                f"None, or 'None', got '{verbosity}'"
            )
        # Normalize string 'None' to None
        if verbosity == "None":
            verbosity = None
        self.verbosity = verbosity

    def register_event(
        self,
        label: str,
        category: str,
        description: str,
        start_message: Optional[str] = None,
        stop_message: Optional[str] = None,
    ) -> None:
        """Register an event with metadata for tracking and reporting.

        Parameters
        ----------
        label : str
            Event label used in start_event/stop_event calls
        category : str
            Event category: 'codegen', 'runtime', or 'compile'
        description : str
            Human-readable description included in printouts
        start_message : str, optional
            Custom message to print when event starts. Use {label} as
            placeholder. If None, uses default format.
        stop_message : str, optional
            Custom message to print when event stops. Use {label} and
            {duration} as placeholders. If None, uses default format.

        Notes
        -----
        This method is called by CUDAFactory subclasses to register
        timing events they will track. The category helps organize
        timing reports by operation type.
        """
        if category not in {"codegen", "runtime", "compile"}:
            raise ValueError(
                f"category must be 'codegen', 'runtime', or 'compile', "
                f"got '{category}'"
            )
        if label not in self._event_registry:
            self._event_registry[label] = {
                "category": category,
                "description": description,
                "start_message": start_message,
                "stop_message": stop_message,
            }

    def print_message(
        self, message: str, min_verbosity: str = "verbose"
    ) -> None:
        """Print a message if verbosity level is sufficient.

        Parameters
        ----------
        message : str
            Message to print
        min_verbosity : str, default='verbose'
            Minimum verbosity level required to print. Options:
            'default', 'verbose', 'debug'. Lower levels include higher.

        Notes
        -----
        Use this for one-time notifications like cache hit messages.
        This method centralizes all printing logic in TimeLogger.
        """
        if self.verbosity is None:
            return

        verbosity_levels = {"default": 1, "verbose": 2, "debug": 3}
        current_level = verbosity_levels.get(self.verbosity, 0)
        required_level = verbosity_levels.get(min_verbosity, 2)

        if current_level >= required_level:
            print(message)

    def _register_cuda_event(self, event: "CUDAEvent") -> None:
        """Register a CUDA event for later timing retrieval (internal).

        Parameters
        ----------
        event : CUDAEvent
            CUDA event instance to register

        Notes
        -----
        Called internally by CUDAEvent.__init__().
        Events are stored until _retrieve_cuda_events() is called.
        No-op when verbosity is None.
        """
        if self.verbosity is None:
            return

        # Store event for later retrieval
        self._cuda_events.append(event)

        # Register with standard event registry
        self.register_event(event.name, "runtime", f"GPU event: {event.name}")

    def _retrieve_cuda_events(self) -> None:
        """Retrieve timing from all registered CUDA events.

        Notes
        -----
        This method must be called AFTER stream synchronization.
        It converts CUDA event timings to TimeLogger events by calling
        elapsed_time_ms() on each registered CUDAEvent.

        The _cuda_events list is cleared after retrieval to avoid
        memory growth.

        No-op when verbosity is None or no events registered.
        """
        if self.verbosity is None:
            return

        if not self._cuda_events:
            return

        # Process each CUDA event
        for event in self._cuda_events:
            # Get elapsed time (returns immediately, no blocking)
            elapsed_ms = event.elapsed_time_ms()

            # Create a TimingEvent with the duration in metadata
            timing_event = TimingEvent(
                name=event.name,
                event_type="stop",
                timestamp=0.0,  # Not used for CUDA events
                metadata={"duration_ms": elapsed_ms},
            )
            self.events.append(timing_event)

        # Clear list after retrieval
        self._cuda_events.clear()

    def _clear_events(self) -> None:
        """Clear all timing events.

        Notes
        -----
        Called automatically after print_summary to prevent event accumulation
        across multiple solve calls. Does not clear event registry.
        """
        self.events.clear()
        self._active_starts.clear()
        self._cuda_events.clear()


# Default global logger instance
# Use set_verbosity() to configure, or access via cubie.time_logger
default_timelogger = TimeLogger(verbosity=None)
