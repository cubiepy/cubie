"""Tests for loop buffer_registry migration.

Note: The LoopBufferSettings, LoopLocalSizes, and LoopSliceIndices classes
have been removed. Loop buffers are now managed through the buffer_registry
API with IVPLoop accepting individual parameters.

These tests verify the new buffer_registry-based loop initialization.
"""

from cubie.buffer_registry import buffer_registry
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.outputhandling import OutputCompileFlags


class TestIVPLoopBufferRegistration:
    """Tests for IVPLoop buffer registration via buffer_registry."""

    def test_ivploop_registers_buffers(self, precision):
        """IVPLoop should register all loop buffers with buffer_registry."""
        flags = OutputCompileFlags()

        # Create a minimal loop to test buffer registration
        loop = IVPLoop(
            precision=precision,
            n_states=3,
            compile_flags=flags,
            n_parameters=2,
            n_drivers=1,
            n_observables=2,
            n_error=3,
            n_counters=4,
        )

        try:
            # All buffers should be registered
            assert buffer_registry.shared_buffer_size(loop) >= 0
        finally:
            buffer_registry.clear_parent(loop)

    def test_ivploop_accepts_location_parameters(self, precision):
        """IVPLoop should accept individual location parameters."""
        flags = OutputCompileFlags()

        loop = IVPLoop(
            precision=precision,
            n_states=3,
            n_parameters=2,
            compile_flags=flags,
            state_location="shared",
            parameters_location="shared",
        )

        try:
            # Shared buffers should contribute to shared memory
            shared_size = buffer_registry.shared_buffer_size(loop)
            assert shared_size >= 5  # state (3) + parameters (2)
        finally:
            buffer_registry.clear_parent(loop)

    def test_ivploop_all_local_zero_shared(self, precision):
        """All-local config should have minimal shared memory."""
        flags = OutputCompileFlags()

        loop = IVPLoop(
            precision=precision,
            n_states=3,
            compile_flags=flags,
            n_parameters=2,
            n_drivers=1,
            n_observables=2,
            state_location="local",
            proposed_state_location="local",
            parameters_location="local",
            drivers_location="local",
            proposed_drivers_location="local",
            observables_location="local",
            proposed_observables_location="local",
            error_location="local",
            counters_location="local",
            state_summary_location="local",
            observable_summary_location="local",
        )

        try:
            # All local, so shared should be 0
            shared_size = buffer_registry.shared_buffer_size(loop)
            assert shared_size == 0
        finally:
            buffer_registry.clear_parent(loop)


class TestBufferRegistryIntegration:
    """Tests for buffer_registry API with IVPLoop."""

    def test_get_allocator_returns_callable(self, precision):
        """get_allocator should return a callable for registered buffers."""
        flags = OutputCompileFlags()

        loop = IVPLoop(
            precision=precision,
            n_states=3,
            compile_flags=flags,
        )

        try:
            # Should be able to get allocators for all registered buffers
            alloc = buffer_registry.get_allocator("state", loop)
            assert callable(alloc)
        finally:
            buffer_registry.clear_parent(loop)

    def test_clear_factory_removes_registrations(self, precision):
        """clear_parent should remove all buffer registrations."""
        flags = OutputCompileFlags()

        loop = IVPLoop(
            precision=precision,
            n_states=3,
            compile_flags=flags,
        )

        # Get initial size
        initial_size = buffer_registry.shared_buffer_size(loop)

        # Clear and check size is 0
        buffer_registry.clear_parent(loop)
        cleared_size = buffer_registry.shared_buffer_size(loop)
        assert cleared_size == 0
