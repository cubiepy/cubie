"""Allocate pinned past a job-object commit cap; print outcomes."""

import ctypes
import json
from ctypes import wintypes

import numpy as np

from cubie.cuda_simsafe import is_pinned_array
from cubie.memory import default_memmgr as mm


class JobLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JobIoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "ReadOperationCount", "WriteOperationCount",
        "OtherOperationCount", "ReadTransferCount",
        "WriteTransferCount", "OtherTransferCount")]


class JobExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JobLimits),
        ("IoInfo", JobIoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD)] + [
        (n, ctypes.c_size_t) for n in (
            "a", "b", "c", "d", "e", "f", "g", "h", "PrivateUsage")]


kernel32 = ctypes.windll.kernel32
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.K32GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
kernel32.CreateJobObjectW.restype = wintypes.HANDLE
kernel32.AssignProcessToJobObject.argtypes = [
    wintypes.HANDLE, wintypes.HANDLE]
kernel32.SetInformationJobObject.argtypes = [
    wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]

mm.probe_device()
mm.get_group_stream("default")

counters = PMC(cb=ctypes.sizeof(PMC))
kernel32.K32GetProcessMemoryInfo(
    kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)

big_bytes = 3 * 2**29

job = kernel32.CreateJobObjectW(None, None)
limits = JobExtendedLimits()
limits.BasicLimitInformation.LimitFlags = 0x100
limits.ProcessMemoryLimit = counters.PrivateUsage + (768 << 20)
assert kernel32.SetInformationJobObject(
    job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
assert kernel32.AssignProcessToJobObject(
    job, kernel32.GetCurrentProcess())

big = mm.allocate_pinned_array((big_bytes,), np.uint8)
small = mm.allocate_pinned_array((64 * 2**20,), np.uint8)
print(json.dumps({
    "big_refused": big is None,
    "small_pinned": small is not None and bool(is_pinned_array(small)),
}))
