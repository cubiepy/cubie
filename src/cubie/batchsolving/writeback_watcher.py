"""Watcher thread for async writeback completion.

Published Classes
-----------------
:class:`WritebackTask`
    Container associating a CUDA event with a pinned buffer, host target,
    and pool for release after completion.

:class:`WritebackWatcher`
    Background daemon thread that polls CUDA events and copies completed
    pinned-buffer data to host arrays; :meth:`wait_all` drains
    outstanding tasks in the calling thread.

See Also
--------
:class:`~cubie.batchsolving.arrays.BatchOutputArrays.OutputArrays`
    Consumer that submits writeback tasks during chunked transfers.
:class:`~cubie.memory.chunk_buffer_pool.ChunkBufferPool`
    Pool managing pinned buffer allocation and reuse.
"""

from collections import deque
from threading import Thread, Event, Condition
from typing import Optional
from time import sleep, perf_counter

from attrs import define, field
from attrs.validators import (
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
)
from numpy import ndarray

from cubie.cuda_simsafe import CUDA_SIMULATION
from cubie.memory.chunk_buffer_pool import PinnedBuffer, ChunkBufferPool


@define
class WritebackTask:
    """Container for a pending writeback operation.

    Attributes
    ----------
    event
        CUDA event to query for completion (None in CUDASIM).
    buffer
        Pinned buffer containing data to copy.
    target_array
        Host array to write data into.
    buffer_pool
        Pool to release buffer to after completion.
    array_name
        Name of the array for pool organization.
    data_shape
        Shape of actual data in buffer (may be smaller than buffer size).
    """

    event: object = field()  # cuda.Event or None
    buffer: PinnedBuffer = field(validator=attrsval_instance_of(PinnedBuffer))
    target_array: Optional[ndarray] = field(
        validator=attrsval_optional(attrsval_instance_of(ndarray))
    )
    buffer_pool: ChunkBufferPool = field(
        validator=attrsval_instance_of(ChunkBufferPool)
    )
    array_name: str = field(validator=attrsval_instance_of(str))
    data_shape: tuple = field(
        default=None, validator=attrsval_optional(attrsval_instance_of(tuple))
    )


class WritebackWatcher:
    """Complete staged transfers when their CUDA events fire.

    Pending :class:`WritebackTask` objects live in a shared deque. A
    background daemon polls their events and completes ready tasks;
    :meth:`wait_all` drains the deque in the calling thread, blocking
    on each task's event. A task is owned by whichever thread popped
    it; an incomplete task returns to the deque.

    Attributes
    ----------
    _tasks : deque
        Shared deque of pending WritebackTask objects.
    _cond : Condition
        Guards ``_tasks`` and ``_pending_count``; signalled on every
        completion and re-queue.
    _thread : Thread or None
        Background polling thread.
    _stop_event : Event
        Signal to terminate the polling thread.
    _poll_interval : float
        Seconds the background thread sleeps between event polls.
    _pending_count : int
        Number of submitted tasks not yet completed, including tasks
        currently owned by a processing thread.
    """

    def __init__(self, poll_interval: float = 0.001) -> None:
        """Initialize the watcher.

        Parameters
        ----------
        poll_interval
            Seconds the background thread sleeps between event polls.
            Default 1ms.
        """
        self._tasks: deque = deque()
        self._cond: Condition = Condition()
        self._thread: Optional[Thread] = None
        self._stop_event: Event = Event()
        self._poll_interval: float = poll_interval
        self._pending_count: int = 0

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running
        self._stop_event.clear()
        self._thread = Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _submit_task(self, task: WritebackTask) -> None:
        """Submit a writeback task to the shared deque.

        Parameters
        ----------
        task
            WritebackTask to submit.
        """
        with self._cond:
            self._pending_count += 1
            self._tasks.append(task)
        # Start thread if not running
        self.start()

    def submit(
        self,
        event: object,
        buffer: PinnedBuffer,
        target_array: ndarray,
        buffer_pool: ChunkBufferPool,
        array_name: str,
        data_shape: Optional[tuple] = None,
    ) -> None:
        """Submit a writeback task for async completion.

        Parameters
        ----------
        event
            CUDA event to monitor for completion.
        buffer
            Pinned buffer containing source data.
        target_array
            Host array to write into.
        buffer_pool
            Pool to release buffer to.
        array_name
            Name of the array for pool organization.
        data_shape
            Shape of actual data in buffer. When provided, only this
            portion of the buffer is copied to target. Used when buffer
            is larger than actual data (e.g., last chunk).
        """
        task = WritebackTask(
            event=event,
            buffer=buffer,
            target_array=target_array,
            buffer_pool=buffer_pool,
            array_name=array_name,
            data_shape=data_shape,
        )
        self._submit_task(task)

    def submit_release(
        self,
        event: object,
        buffer: PinnedBuffer,
        buffer_pool: ChunkBufferPool,
        array_name: str,
    ) -> None:
        """Release a staging buffer after an event completes."""
        self._submit_task(
            WritebackTask(
                event=event,
                buffer=buffer,
                target_array=None,
                buffer_pool=buffer_pool,
                array_name=array_name,
            )
        )

    def wait_all(self, timeout: Optional[float] = None) -> None:
        """Block until all pending writebacks complete.

        Drains the task deque in the calling thread, blocking on each
        task's CUDA event. When the deque is empty but tasks are owned
        by the background thread, waits on the condition until they
        resolve.

        Parameters
        ----------
        timeout
            Maximum seconds to wait. None waits indefinitely.

        Raises
        ------
        TimeoutError
            If timeout expires before completion.
        """
        deadline = (
            None if timeout is None else perf_counter() + timeout
        )
        while True:
            with self._cond:
                if self._pending_count == 0:
                    return
                task = self._tasks.popleft() if self._tasks else None
                if task is None:
                    # Remaining tasks are owned by another thread;
                    # wait for their completion or re-queue signal.
                    remaining = None
                    if deadline is not None:
                        remaining = deadline - perf_counter()
                        if remaining <= 0:
                            raise TimeoutError(
                                f"wait_all timed out after {timeout} "
                                "seconds"
                            )
                    if not self._cond.wait(timeout=remaining):
                        raise TimeoutError(
                            f"wait_all timed out after {timeout} seconds"
                        )
                    continue
            try:
                self._finish_task(task, deadline, timeout)
            except BaseException:
                # Return the unfinished task so a later wait or the
                # shutdown drain can complete it.
                with self._cond:
                    self._tasks.append(task)
                    self._cond.notify_all()
                raise
            with self._cond:
                self._pending_count -= 1
                self._cond.notify_all()

    def shutdown(self, timeout: Optional[float] = None) -> None:
        """Drain pending work and stop the polling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError("writeback watcher did not stop")
            self._thread = None

    def _poll_loop(self) -> None:
        """Main polling loop for the background thread."""
        while not self._stop_event.is_set():
            with self._cond:
                task = self._tasks.popleft() if self._tasks else None
            if task is None:
                sleep(self._poll_interval)
                continue
            if self._process_task(task):
                with self._cond:
                    self._pending_count -= 1
                    self._cond.notify_all()
            else:
                with self._cond:
                    self._tasks.append(task)
                    self._cond.notify_all()
                sleep(self._poll_interval)

        # On shutdown, complete remaining tasks synchronously to
        # ensure all data is written.
        while True:
            with self._cond:
                if self._pending_count == 0:
                    return
                task = self._tasks.popleft() if self._tasks else None
            if task is None:
                # A concurrent wait_all owns the remaining task(s).
                sleep(self._poll_interval)
                continue
            if self._process_task(task):
                with self._cond:
                    self._pending_count -= 1
                    self._cond.notify_all()
            else:
                with self._cond:
                    self._tasks.append(task)
                    self._cond.notify_all()
                sleep(self._poll_interval)

    def _finish_task(
        self,
        task: WritebackTask,
        deadline: Optional[float],
        timeout: Optional[float],
    ) -> None:
        """Complete one task, blocking until its event has fired.

        Parameters
        ----------
        task
            Task to complete.
        deadline
            ``perf_counter`` value after which waiting raises, or
            ``None`` to block indefinitely on the event.
        timeout
            Original timeout in seconds, used in the error message.

        Raises
        ------
        TimeoutError
            If the deadline passes before the event fires.
        """
        if not CUDA_SIMULATION and task.event is not None:
            if deadline is None:
                task.event.synchronize()
            else:
                while not task.event.query():
                    if perf_counter() >= deadline:
                        raise TimeoutError(
                            f"wait_all timed out after {timeout} seconds"
                        )
                    sleep(self._poll_interval)
        self._complete_task(task)

    def _complete_task(self, task: WritebackTask) -> None:
        """Copy staged data to its target and release the buffer."""
        # Copy buffer data to target array at specified slice.
        # If data_shape provided, only copy that portion of the buffer.
        if task.target_array is not None:
            if task.data_shape is not None:
                buffer_slice = tuple(
                    slice(0, s) for s in task.data_shape
                )
                task.target_array[:] = task.buffer.array[buffer_slice]
            else:
                task.target_array[:] = task.buffer.array
        # Release buffer back to pool
        task.buffer_pool.release(task.buffer)

    def _process_task(self, task: WritebackTask) -> bool:
        """Process a single writeback task.

        Parameters
        ----------
        task
            Task to process.

        Returns
        -------
        bool
            True if task completed, False if still pending.
        """
        # In CUDASIM mode or event is None, treat as immediately complete
        if CUDA_SIMULATION or task.event is None:
            is_complete = True
        else:
            # Query event for completion (returns True when complete)
            is_complete = task.event.query()

        if is_complete:
            self._complete_task(task)
            return True

        return False
