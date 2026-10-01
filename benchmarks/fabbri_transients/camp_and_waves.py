"""Check cAMP drift along ACh = 0 and report launch waves per grid size."""

import math

import numpy as np

import common
from cubie._cudasim_extensions import cuda


def waves(solver, n_runs):
    """Return launch geometry and waves for ``n_runs`` runs."""
    (kernel,) = solver.kernel.kernel.overloads.values()
    function = kernel._codelibrary.get_cufunc()
    blocksize, dynshared = solver.kernel.launch_geometry(None, runs=n_runs)
    blocks_per_sm = cuda.current_context().get_active_blocks_per_multiprocessor(
        function, blocksize, dynshared
    )
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    runs_per_block = blocksize // solver.kernel.single_integrator.threads_per_step
    blocks = math.ceil(n_runs / runs_per_block)
    return {
        "blocksize": blocksize, "blocks_per_sm": blocks_per_sm, "sms": sms,
        "resident_runs": blocks_per_sm * sms * runs_per_block,
        "waves": blocks / (blocks_per_sm * sms),
    }


def main():
    system = common.build_system()
    data = np.load(common.RESULTS / "steady_state_log96.npz")
    names = list(data["state_names"])
    camp = names.index("cAMP_cAMP")
    side = int(np.sqrt(data["ach"].size))
    # ACh = 1 nM column (lowest), every Iso.
    column = np.arange(side) * side
    state = np.ascontiguousarray(data["final_state"][:, column])
    params = common.parameter_array(
        system, data["ach"][column], data["iso"][column]
    )
    solver = common.make_solver(
        system, output_types=["state"], save_every=1.0
    )
    result = solver.solve(
        initial_values=state, parameters=params, duration=120.0
    )
    camp_trace = np.asarray(result.time_domain_array[:, camp, :])
    del result
    print("Iso (nM)  cAMP t=0     t=60 s      t=120 s     max |d/dt| over"
          " last 60 s (mM/s)")
    for k in range(0, side, 8):
        tail = camp_trace[60:, k]
        print(f"{data['iso'][column[k]]:8.3f}  {camp_trace[0, k]:.6f}  "
              f"{camp_trace[60, k]:.6f}  {camp_trace[120, k]:.6f}  "
              f"{np.ptp(tail) / 60:.2e}")
    for n_runs in (data["ach"].size, 65536):
        print(n_runs, "runs, final state only:", waves(solver, n_runs))
    solver.close()

    solver = common.make_solver(
        system,
        output_types=["peaks[3]"],
        summarise_variables=[common.VOLTAGE_LABEL],
        save_variables=[],
        sample_summaries_every=2.0**-12,
        summarise_every=1.0,
    )
    solver.solve(
        initial_values=np.ascontiguousarray(data["final_state"]),
        parameters=common.parameter_array(system, data["ach"], data["iso"]),
        duration=1.0,
    )
    for n_runs in (data["ach"].size, 65536):
        print(n_runs, "runs, peaks[3]:", waves(solver, n_runs))
    solver.close()


if __name__ == "__main__":
    main()
