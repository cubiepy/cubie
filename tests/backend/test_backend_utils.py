"""Tests for the occupancy and cubin helpers in ``backend.utils``."""

from struct import pack

import pytest

from cubie.backend.utils import (
    DeviceHardware,
    MAX_REGISTERS_PER_THREAD,
    register_limited_threads,
    sass_bytes_from_cubin,
    shared_keeps_occupancy,
    shared_limited_threads,
)


def _hardware(compute_capability=(8, 9)):
    """An RTX 4070 SUPER-shaped device at the given compute capability."""
    return DeviceHardware(
        compute_capability=compute_capability,
        multiprocessor_count=56,
        l2_cache_bytes=48 << 20,
        shared_memory_per_multiprocessor=102400,
        reserved_shared_memory_per_block=1024,
        max_dynamic_shared_memory_per_block=101376,
        instruction_cache_bytes=128 * 1024,
        registers_per_multiprocessor=65536,
        max_threads_per_multiprocessor=1536,
        max_blocks_per_multiprocessor=24,
        warp_size=32,
    )


@pytest.mark.parametrize(
    "registers, threads",
    [(255, 256), (128, 512), (100, 512), (64, 1024), (32, 1536)],
)
def test_register_limited_threads_match_the_occupancy_calculator(
    registers, threads
):
    """Registers allocate per warp in units of 256 from four
    sub-partitions; 32 registers hit the thread limit."""
    assert register_limited_threads(_hardware(), registers) == threads


def test_register_limited_threads_on_two_sub_partitions():
    """Compute capability 6.0 allocates from two sub-partitions, so a
    warp size that does not divide a quarter of the file evenly fits
    more warps."""
    assert register_limited_threads(_hardware((7, 5)), 48) == 1280
    assert register_limited_threads(_hardware((6, 0)), 48) == 1344


def test_shared_limited_threads_take_the_best_block_size():
    """400 B per run fits 7 blocks of 32, 3 of 64, 1 of 128 and no
    256-thread block; the best is 224 threads."""
    assert shared_limited_threads(_hardware(), 400) == 224


def test_shared_limited_threads_compare_dynamic_bytes_to_the_limit():
    """396 B per run fills the dynamic limit exactly at 256 threads; the
    reserve counts against the SM, not the block limit."""
    assert shared_limited_threads(_hardware(), 396) == 256


def test_shared_keeps_occupancy_compares_with_the_register_limit():
    """A register-capped kernel keeps 256 threads; 100 B per run keeps
    768, 400 B keeps 224, which half occupancy still accepts."""
    hardware = _hardware()
    assert register_limited_threads(hardware, MAX_REGISTERS_PER_THREAD) == 256
    assert shared_keeps_occupancy(hardware, 100)
    assert not shared_keeps_occupancy(hardware, 400)
    assert shared_keeps_occupancy(hardware, 400, fraction=2)


def _elf64(sections):
    """A minimal ELF64 image with the named sections and sizes."""
    names = b"\0"
    name_offsets = []
    for name, _ in sections:
        name_offsets.append(len(names))
        names += name.encode() + b"\0"
    strtab_name = len(names)
    names += b".shstrtab\0"
    header_size = 64
    entry_size = 64
    section_offset = header_size + len(names)
    count = len(sections) + 2
    ident = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    header = ident + pack(
        "<HHIQQQIHHHHHH",
        1, 190, 1, 0, 0, section_offset, 0, header_size, 0, 0,
        entry_size, count, count - 1,
    )

    def entry(name, size, offset):
        return pack("<IIQQQQIIQQ", name, 1, 0, 0, offset, size, 0, 0, 1, 0)

    table = bytes(entry_size)
    for (_, size), name in zip(sections, name_offsets):
        table += entry(name, size, 0)
    table += entry(strtab_name, len(names), header_size)
    return header + names + table


def test_sass_bytes_sums_the_text_sections():
    """Only ``.text.`` sections count; ``.nv.info`` and others do not."""
    image = _elf64([
        (".text.kernel", 4864),
        (".nv.info.kernel", 300),
        (".text.helper", 256),
        (".nv.shared.kernel", 64),
    ])
    assert sass_bytes_from_cubin(image) == 4864 + 256


def test_sass_bytes_rejects_a_non_elf_image():
    """Anything but an ELF image raises."""
    with pytest.raises(ValueError, match="ELF"):
        sass_bytes_from_cubin(b"not a cubin")

