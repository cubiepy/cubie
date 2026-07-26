"""Pure launch-sizing helpers for the NCU comparison harness."""


DEFAULT_WAVES = 10
MANIFEST_VERSION = 2
SIZING_MODE = "occupancy-waves"
CAPTURE_MODE_DIRECT = "direct"
CAPTURE_MODE_NCU_CLI = "ncu-cli"
CAPTURE_MODES = (CAPTURE_MODE_DIRECT, CAPTURE_MODE_NCU_CLI)


def wave_trajectory_count(
    waves: int,
    multiprocessors: int,
    blocks_per_multiprocessor: int,
    trajectories_per_block: int,
) -> int:
    """Return the trajectory count filling the requested CUDA waves."""

    factors = {
        "waves": waves,
        "multiprocessors": multiprocessors,
        "blocks_per_multiprocessor": blocks_per_multiprocessor,
        "trajectories_per_block": trajectories_per_block,
    }
    for name, value in factors.items():
        if value < 1:
            raise ValueError(f"{name} must be positive; received {value}")
    return (
        waves
        * multiprocessors
        * blocks_per_multiprocessor
        * trajectories_per_block
    )
