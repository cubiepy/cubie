"""Time the interpreter's first CUDA context and transfers."""
from time import perf_counter

t0 = perf_counter()
from numpy import zeros  # noqa: E402
from cubie.cuda_simsafe import cuda  # noqa: E402

t1 = perf_counter()
array = cuda.to_device(zeros(1))
t2 = perf_counter()
array.copy_to_host()
t3 = perf_counter()
print(
    f"import={t1 - t0:.2f}s to_device={t2 - t1:.2f}s "
    f"copy_back={t3 - t2:.2f}s"
)
