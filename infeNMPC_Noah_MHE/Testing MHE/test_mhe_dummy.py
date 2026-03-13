import pyomo.environ as pyo
import pytest

from dummy_mhe_test_data import make_dummy_io_data
from infNMPC_options import _import_settings
from MHE_estimation import solve_mhe_no_arrival_cost
from indexing_tools import _get_variable_key_for_data
from pyomo.contrib.mpc import ScalarData
from run_MPC import _make_plant

num_steps = 100
def test_mhe_dummy_runs():
    # Skip if IPOPT isn't available in the environment
    if not pyo.SolverFactory("ipopt").available():
        pytest.skip("IPOPT solver is not available.")

    options = _import_settings()
    io_data_array = make_dummy_io_data(num_steps)

    result = solve_mhe_no_arrival_cost(
        options=options,
        io_data_array=io_data_array,
        M_desired=options.MHE_window,
        solver_name="ipopt",
        tee=False,
    )

    # With at least one MV applied, we should get a result
    assert result is not None
    assert result.M_eff >= 1
    assert isinstance(result.xhat, dict)


def _extract_state_data_at_time(model, time_point, state_vars):
    data = model.interface.get_data_at_time(time_point)
    out = {}
    for sv in state_vars:
        sv_name = sv.name.split(".")[-1]
        if sv.is_indexed():
            for index in sv.index_set():
                if isinstance(index, tuple) and len(index) > 1:
                    partial_index_str = ",".join(str(i) for i in index[:-1]) + ",*"
                    key = f"{sv_name}[{partial_index_str}]"
                    out[key] = data.get_data_from_key(key)
                else:
                    key = f"{sv_name}[*]"
                    out[key] = data.get_data_from_key(key)
        else:
            key = f"{sv_name}[*]"
            out[key] = data.get_data_from_key(key)
    return out


def _simulate_truth_states(options, io_data_array):
    """
    Simulate the plant forward with the given MV sequence and return true
    Ca, Cb, Cm at each sampling step (aligned to io_data_array rows).
    """
    plant = _make_plant(options)
    solver = pyo.SolverFactory("ipopt")

    n_cv = len(plant.CV_index)
    n_mv = len(plant.MV_index)

    # Initial truth at t=0
    state_vars = plant.state_vars
    truth = []
    t0_data = plant.interface.get_data_at_time(plant.time.first())
    truth.append(
        {
            "Ca": t0_data.get_data_from_key(_get_variable_key_for_data(plant, "Ca")),
            "Cb": t0_data.get_data_from_key(_get_variable_key_for_data(plant, "Cb")),
            "Cm": t0_data.get_data_from_key(_get_variable_key_for_data(plant, "Cm")),
        }
    )

    new_data_time = list(plant.time)[1:]

    for k in range(1, len(io_data_array)):
        row = io_data_array[k]
        mv_vals = {}
        for j, mv in enumerate(plant.MV_index):
            val = row[n_cv + j]
            if val is None:
                raise ValueError(
                    f"MV '{mv}' is None at io_data_array row {k}. "
                    f"Not enough MV history yet."
                )
            mv_vals[_get_variable_key_for_data(plant, mv)] = float(val)

        input_data = ScalarData(mv_vals)
        plant.interface.load_data(input_data, time_points=new_data_time)
        solver.solve(plant, tee=False)

        full_data = plant.interface.get_data_at_time(options.sampling_time)
        truth.append(
            {
                "Ca": full_data.get_data_from_key(_get_variable_key_for_data(plant, "Ca")),
                "Cb": full_data.get_data_from_key(_get_variable_key_for_data(plant, "Cb")),
                "Cm": full_data.get_data_from_key(_get_variable_key_for_data(plant, "Cm")),
            }
        )

        # Shift state values to become the initial condition for the next step
        tf_data = ScalarData(data=_extract_state_data_at_time(plant, options.sampling_time, state_vars))
        plant.interface.load_data(tf_data)

    return truth


if __name__ == "__main__":
    # Simple direct run without pytest
    print(f"Running: {__file__}", flush=True)
    options = _import_settings()
    options.tee_flag = False
    io_data_array = make_dummy_io_data(num_steps)

    result = solve_mhe_no_arrival_cost(
        options=options,
        io_data_array=io_data_array,
        M_desired=options.MHE_window,
        solver_name="ipopt",
        tee=False,
    )

    print("", flush=True)
    print("=== MHE RESULT ===", flush=True)
    if result is None:
        print("MHE did not run (not enough history or M_eff < 1).", flush=True)
    else:
        print("MHE ran successfully.", flush=True)
        print("M_eff:", result.M_eff, flush=True)
        print("MHE estimates at final time (xhat):", flush=True)
        for k, v in result.xhat.items():
            print(f"  {k}: {v}", flush=True)

    print("", flush=True)
    print("=== TRUTH MODEL ===", flush=True)
    print("True states from plant simulation (aligned to io_data_array rows):", flush=True)
    truth = _simulate_truth_states(options, io_data_array)
    for k, vals in enumerate(truth):
        print(f"  k={k}: Ca={vals['Ca']}, Cb={vals['Cb']}, Cm={vals['Cm']}", flush=True)

    print("", flush=True)
    print("=== SUMMARY ===", flush=True)
    if result is None:
        print("MHE: None", flush=True)
    else:
        ca = result.xhat.get("Ca")
        cb = result.xhat.get("Cb")
        cm = result.xhat.get("Cm")
        print(f"MHE xhat: Ca={ca}, Cb={cb}, Cm={cm}", flush=True)
    if truth:
        last = truth[-1]
        print(f"Truth at k={len(truth)-1}: Ca={last['Ca']}, Cb={last['Cb']}, Cm={last['Cm']}", flush=True)
