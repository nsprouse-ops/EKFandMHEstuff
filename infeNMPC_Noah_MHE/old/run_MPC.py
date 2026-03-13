import pyomo.environ as pyo
from make_model import _make_infinite_horizon_model, _make_finite_horizon_model, _remove_non_collocation_values_infinite, _remove_non_collocation_values_finite
import idaes
from idaes.core.solvers import use_idaes_solver_configuration_defaults
from infNMPC_options import _import_settings
from tqdm import tqdm
from data_save_and_plot import _finalize_live_plot, _setup_live_plot, _update_live_plot, _handle_mpc_results
from indexing_tools import _add_time_indexed_expression
import numpy as np
from pyomo.contrib.mpc import ScalarData
from indexing_tools import _get_variable_key_for_data
import time
from math import isclose
from initialization_tools import _assist_initialization_infinite, _assist_initialization_finite
from MHE_estimation import solve_mhe_no_arrival_cost
from controller_factory import _make_infinite_horizon_controller, _make_finite_horizon_controller
 
# Solver Settings
use_idaes_solver_configuration_defaults()
idaes.cfg.ipopt.options.linear_solver = "ma57" # "ma27"
idaes.cfg.ipopt.options.OF_ma57_automatic_scaling = "yes"
idaes.cfg.ipopt.options.max_iter = 6000
idaes.cfg.ipopt.options.halt_on_ampl_error = "yes"
idaes.cfg.ipopt.options.bound_relax_factor = 0


def _make_plant(options):
    """
    Build and solve the dynamic simulation model (plant).

    This function initializes the plant model using finite-horizon settings,
    fixes manipulated variables (MVs), and solves the model to obtain an initial
    steady-state or consistent profile.

    Args:
        options (Namespace): Simulation and solver configuration parameters.

    Returns:
        pyo.ConcreteModel: The initialized and solved plant model.
    """
    m = pyo.ConcreteModel()
    plant_options = _import_settings()
    plant_options.nfe_finite = 1
    plant_options.infinite_horizon = False

    m = _make_finite_horizon_model(m, plant_options, equations_module="model_equations_true")

    print('Generating Plant')
    
    for var_name in m.MV_index:
        var = getattr(m, var_name)
        var.fix()
        if options.ncp_finite > 1:
            getattr(m, f"{var_name}_interpolation_constraints").deactivate()
    
    m.obj = pyo.Objective(expr=1)

    print('Plant Initial Solve')

    solver = pyo.SolverFactory('ipopt')
    solver.solve(m, tee=options.tee_flag)

    return m


def _mpc_loop(options):
    """
    Execute the NMPC closed-loop simulation.

    Runs the nonlinear model predictive control loop using either a finite or infinite
    horizon controller. It interacts with the plant model at each sampling interval,
    solves optimization problems, updates plots, and logs performance data.

    Args:
        options (Namespace): Runtime configuration including control horizon type,
            sampling time, plotting options, and solver verbosity.

    Returns:
        None
    """
    if options.infinite_horizon:
        if options.initialization_assist:
            controller = _assist_initialization_infinite(options)
            t0_controller = controller.finite_block.time.first()
            state_vars = controller.finite_block.state_vars
        else:
            controller = _make_infinite_horizon_controller(options)
            t0_controller = controller.finite_block.time.first()
            state_vars = controller.finite_block.state_vars
    else:
        if options.initialization_assist:
            controller = _assist_initialization_finite(options)
            t0_controller = controller.time.first()
            state_vars = controller.state_vars
        else:
            controller = _make_finite_horizon_controller(options)
            t0_controller = controller.time.first()
            state_vars = controller.state_vars

    if options.infinite_horizon:
        controller_setpoints = controller.finite_block.steady_state_values
    else:
        controller_setpoints = controller.steady_state_values

    plant = _make_plant(options)

    # Get full initial TimeSeriesData
    sim_data = plant.interface.get_data_at_time([plant.time.first()])
    full_initial_data = plant.interface.get_data_at_time([plant.time.first()])

    # Zero out non-state values in initial sim_data
    state_var_keys = set()
    for sv in state_vars:
        sv_name = sv.name.split(".")[-1]
        if sv.is_indexed():
            for index in sv.index_set():
                if isinstance(index, tuple) and len(index) > 1:
                    partial_index_str = ",".join(str(i) for i in index[:-1]) + ",*"
                    key = f"{sv_name}[{partial_index_str}]"
                    state_var_keys.add(key)
                else:
                    key = f"{sv_name}[*]"
                    state_var_keys.add(key)
        else:
            key = f"{sv_name}[*]"
            state_var_keys.add(key)

    state_var_keys_str = {str(k) for k in state_var_keys}

    for key in sim_data._data:
        if str(key) not in state_var_keys_str:
            sim_data._data[key][0] = None  # Nullify non-state values

    new_data_time = list(plant.time)[1:]
    solver = pyo.SolverFactory('ipopt')

    # Prepare plotting data arrays
    io_data_array = []
    io_meas_array = []
    io_live_plot_array = []
    time_series = []
    cpu_time = []
    mhe_state_hist = None
    true_state_hist = None
    mhe_skipped = 0
    prev_mhe_model = None
    prev_mhe_xhat = None
    current_mhe_window = int(options.MHE_window)
    mhe_min_window = int(getattr(options, "mhe_min_window", 1))
    mhe_consecutive_ok = 0
    noise_amplitude = float(getattr(options, "measurement_noise_amplitude", 0.0))
    if noise_amplitude < 0.0 or noise_amplitude > 1.0:
        raise ValueError("measurement_noise_amplitude must be between 0 and 1.")
    noise_vector = np.random.normal(loc=0.0, scale=1.0, size=options.num_horizons)
    noise_vector = noise_vector - float(np.mean(noise_vector))
    max_abs_noise = float(np.max(np.abs(noise_vector)))
    if max_abs_noise > 0.0:
        noise_vector = noise_vector / max_abs_noise
    else:
        noise_vector = np.zeros(options.num_horizons)
    print("Measurement noise vector:", noise_vector.tolist())

    def _as_list(v):
        return v if isinstance(v, list) else [v]

    def _restore_shape(v_ref, v_list):
        if isinstance(v_ref, list):
            return v_list
        return v_list[0]

    def _make_noisy_measurement(y_actual, y_ss, noise_factor):
        ya = [float(x) for x in _as_list(y_actual)]
        ys = [float(x) for x in _as_list(y_ss)]
        if len(ys) == 1 and len(ya) > 1:
            ys = ys * len(ya)
        noisy = [ya_i + ys_i * noise_factor for ya_i, ys_i in zip(ya, ys)]
        return _restore_shape(y_actual, noisy)

    # Get initial CV values only if they correspond to a differential state
    differential_state_keys = set()
    for sv in state_vars:
        sv_name = sv.name.split(".")[-1]
        if sv.is_indexed():
            for index in sv.index_set():
                if isinstance(index, tuple) and len(index) > 1:
                    partial_index_str = ",".join(str(i) for i in index[:-1]) + ",*"
                    key = f"{sv_name}[{partial_index_str}]"
                    differential_state_keys.add(key)
                else:
                    key = f"{sv_name}[*]"
                    differential_state_keys.add(key)
        else:
            key = f"{sv_name}[*]"
            differential_state_keys.add(key)

    initial_cv_row = []
    for var_name in plant.CV_index:
        key = _get_variable_key_for_data(plant, var_name)
        value = full_initial_data.get_data_from_key(key)
        if isinstance(value, list):
            initial_cv_row.extend(value)
        else:
            initial_cv_row.append(value)

    initial_meas_row = []
    measured_index = list(getattr(plant, "Measured_index", plant.CV_index))
    measured_ss = {}
    for var_name in measured_index:
        key = _get_variable_key_for_data(plant, var_name)
        value = full_initial_data.get_data_from_key(key)
        measured_ss[var_name] = value
        if isinstance(value, list):
            initial_meas_row.extend(value)
        else:
            initial_meas_row.append(value)

    initial_mv_row = [None for _ in plant.MV_index]  # No initial MV values
    io_data_array.append(initial_cv_row + initial_mv_row)
    io_meas_array.append(initial_meas_row + initial_mv_row)
    io_live_plot_array.append(initial_cv_row + initial_mv_row)
    time_series.append(0.0)
    # Initialize MHE/Truth history with NaNs for k=0 to align with time_series
    unmeasured_names = list(getattr(plant, "Unmeasured_index", []))
    if unmeasured_names:
        mhe_state_hist = {name: [np.nan] for name in unmeasured_names}
        true_state_hist = {name: [np.nan] for name in unmeasured_names}

    fig, axes = None, None
    if options.live_plot:
        fig, axes = _setup_live_plot(plant)

    loop_iter = tqdm(range(options.num_horizons), desc="Running MPC")

    if options.infinite_horizon:
        terminal_cost_prev = 1
        first_stage_cost_prev = 1

    for i in loop_iter:

        simulation_time = (i + 1) * options.sampling_time
        noise_factor = noise_amplitude * float(noise_vector[i])

        start_time = time.process_time()
        solver.solve(controller, tee=options.tee_flag)
        end_time = time.process_time()

        # if i == 1:  # Toggle to False to disable model display
        #     with open("model_output.txt", "w") as f:
        #         controller.pprint(ostream=f)

        cpu_time.append(end_time - start_time)

        if options.infinite_horizon:

            # Get all time points and finite element boundaries
            t_points = controller.finite_block.time
            fe_points = t_points.get_finite_elements()

            if options.endpoint_constraints:
                terminal_cost_now = pyo.value(
                    controller.infinite_block.phi[controller.infinite_block.time.last()]
                    * options.beta / options.sampling_time
                )
            else:
                terminal_cost_now = pyo.value(
                    controller.infinite_block.phi[controller.infinite_block.time.prev(controller.infinite_block.time.last())]
                )

            c = options.stage_cost_weights

            penultimate_stage_cost_now = sum(
                pyo.value(
                    c[i]
                    * (
                        _add_time_indexed_expression(
                            controller.finite_block,
                            var_name,
                            fe_points[-1]
                        ) - controller.finite_block.steady_state_values[var_name]
                    )**2
                )
                for i, var_name in enumerate(controller.finite_block.stage_cost_index)
            )

            # Optional debug printing
            print("")
            print("Terminal cost (prev):", terminal_cost_prev)
            print("Terminal cost (now):", terminal_cost_now)
            print("First stage cost (prev):", first_stage_cost_prev)
            print("Penultimate stage cost (now):", penultimate_stage_cost_now)

            LHS = -(
                terminal_cost_prev - terminal_cost_now - options.beta * penultimate_stage_cost_now
            ) / first_stage_cost_prev

            

            print("")
            print(f"Min value of epsilon: {LHS}")
            print("")

        if options.infinite_horizon:
            if options.endpoint_constraints:
                terminal_cost_prev = pyo.value(
                    controller.infinite_block.phi[controller.infinite_block.time.last()]
                    * options.beta / options.sampling_time
                )
            else:
                terminal_cost_prev = pyo.value(
                    controller.infinite_block.phi[controller.infinite_block.time.prev(controller.infinite_block.time.last())]
                    * options.beta / options.sampling_time
                )

            first_stage_cost_prev = sum(
                pyo.value(
                    c[i]
                    * (
                        _add_time_indexed_expression(
                            controller.finite_block,
                            var_name,
                            fe_points[1]
                        ) - controller.finite_block.steady_state_values[var_name]
                    )**2
                )
                for i, var_name in enumerate(controller.finite_block.stage_cost_index)
            )

        ts_data = controller.interface.get_data_at_time(options.sampling_time)

        if options.infinite_horizon:
            input_data = ts_data.extract_variables(
                [getattr(controller.finite_block, var_name) for var_name in controller.finite_block.MV_index],
                context=controller.finite_block
            )
        else:
            input_data = ts_data.extract_variables(
                [getattr(controller, var_name) for var_name in controller.MV_index]
            )

        plant.interface.load_data(input_data, time_points=new_data_time)
        solver.solve(plant, tee=options.tee_flag)

        # Get CV and MV values at the sampling time
        full_data = plant.interface.get_data_at_time(options.sampling_time)
        cv_row = []
        for var_name in plant.CV_index:
            val = full_data.get_data_from_key(_get_variable_key_for_data(plant, var_name))
            if isinstance(val, list):
                cv_row.extend(val)
            else:
                cv_row.append(val)

        meas_row = []
        measured_now = {}
        measured_noisy = {}
        for var_name in measured_index:
            val = full_data.get_data_from_key(_get_variable_key_for_data(plant, var_name))
            y_measured = _make_noisy_measurement(val, measured_ss[var_name], noise_factor)
            measured_noisy[var_name] = y_measured
            if isinstance(y_measured, list):
                meas_row.extend(y_measured)
                if len(y_measured) == 1:
                    measured_now[var_name] = float(y_measured[0])
            else:
                meas_row.append(y_measured)
                measured_now[var_name] = float(y_measured)

        mv_row = []
        for var_name in plant.MV_index:
            val = full_data.get_data_from_key(_get_variable_key_for_data(plant, var_name))
            if isinstance(val, list):
                mv_row.extend(val)
            else:
                mv_row.append(val)

        io_row = cv_row + mv_row
        io_data_array.append(io_row)
        io_meas_array.append(meas_row + mv_row)
        live_cv_row = []
        for var_name in plant.CV_index:
            if var_name in measured_noisy:
                y_live = measured_noisy[var_name]
            else:
                y_live = full_data.get_data_from_key(_get_variable_key_for_data(plant, var_name))
            if isinstance(y_live, list):
                live_cv_row.extend(y_live)
            else:
                live_cv_row.append(y_live)
        io_live_plot_array.append(live_cv_row + mv_row)
        time_series.append(simulation_time)

        if options.live_plot:
            _update_live_plot(
                fig,
                axes,
                time_series,
                io_live_plot_array,
                plant,
                mhe_state_hist=mhe_state_hist,
                setpoint_values=controller_setpoints,
            )

        model_data = plant.interface.get_data_at_time(new_data_time)
        model_data.shift_time_points(simulation_time - plant.time.first() - options.sampling_time)
        sim_data.concatenate(model_data)

        # Only load state variables into tf_data
        full_tf_data = plant.interface.get_data_at_time(options.sampling_time)
        tf_data = ScalarData(data={})
        unmeasured_names = list(getattr(plant, "Unmeasured_index", []))
        unmeasured_base = {name.split("[", 1)[0] for name in unmeasured_names}
        measured_state_vars = [v for v in state_vars if v.local_name not in unmeasured_base]
        unmeasured_state_vars = [v for v in state_vars if v.local_name in unmeasured_base]
        for state_var in measured_state_vars:
            sv_name = state_var.name.split(".")[-1]
            if state_var.is_indexed():
                for index in state_var.index_set():
                    if isinstance(index, tuple) and len(index) > 1:
                        partial_index_str = ",".join(str(i) for i in index[:-1]) + ",*"
                        key = f"{sv_name}[{partial_index_str}]"
                        y_actual = full_tf_data.get_data_from_key(key)
                        y_ss = full_initial_data.get_data_from_key(key)
                        tf_data._data[key] = _make_noisy_measurement(y_actual, y_ss, noise_factor)
                    else:
                        key = f"{sv_name}[*]"
                        y_actual = full_tf_data.get_data_from_key(key)
                        y_ss = full_initial_data.get_data_from_key(key)
                        tf_data._data[key] = _make_noisy_measurement(y_actual, y_ss, noise_factor)
            else:
                key = f"{sv_name}[*]"
                y_actual = full_tf_data.get_data_from_key(key)
                y_ss = full_initial_data.get_data_from_key(key)
                tf_data.data[key] = _make_noisy_measurement(y_actual, y_ss, noise_factor)
        # for UNMEASURED state vars in state vars, insert MHE estimates here
        # keep measured states from plant; replace only unmeasured with MHE xhat
        # Build arrival-cost prior from previous MHE model at the current window start
        prior_xhat = None
        prior_d = None
        warm_start_x0 = None
        if prev_mhe_model is not None:
            try:
                prev_fe_times = list(prev_mhe_model.time.get_finite_elements())
                if prev_fe_times:
                    t_prior = prev_fe_times[1] if len(prev_fe_times) > 1 else prev_fe_times[0]
                    prior_xhat = {}
                    for name in unmeasured_names:
                        try:
                            prior_xhat[name] = pyo.value(_add_time_indexed_expression(prev_mhe_model, name, t_prior))
                        except Exception:
                            continue
                if hasattr(prev_mhe_model, "fe_k") and hasattr(prev_mhe_model, "d"):
                    prev_k = list(prev_mhe_model.fe_k)
                    if prev_k:
                        k_prior = prev_k[1] if len(prev_k) > 1 else prev_k[0]
                        prior_d = {}
                        for name in measured_index:
                            try:
                                prior_d[name] = pyo.value(prev_mhe_model.d[name, k_prior])
                            except Exception:
                                continue
            except Exception:
                prior_xhat = None
                prior_d = None
        if prior_xhat is not None and len(prior_xhat) == 0:
            # Keep bootstrap term active until we have at least one valid prior state.
            prior_xhat = None
        if prior_d is not None and len(prior_d) == 0:
            prior_d = None

        try:
            mhe_result = solve_mhe_no_arrival_cost(
                options=options,
                io_data_array=io_meas_array,
                M_desired=current_mhe_window,
                solver_name="ipopt",
                tee=False,
                prior_xhat=prior_xhat,
                prior_d=prior_d,
                warm_start_x0=warm_start_x0,
            )
        except ValueError:
            mhe_result = None
            mhe_skipped += 1
        if mhe_result is not None:
            # Adaptive MHE window reduction:
            # if state-change and output-residual conditions are satisfied
            # for 2 consecutive iterations across all states/outputs.
            state_default_eps = float(getattr(options, "mhe_state_error_default", 1e-2))
            state_eps_map = dict(getattr(options, "mhe_state_error", {}))
            output_default_eps = float(getattr(options, "mhe_output_error_default", 1e-2))
            output_eps_map = dict(getattr(options, "mhe_output_error", {}))

            state_ok = prev_mhe_xhat is not None and bool(unmeasured_names)
            if state_ok:
                for name in unmeasured_names:
                    if name not in mhe_result.xhat or name not in prev_mhe_xhat:
                        state_ok = False
                        break
                    eps_x = float(state_eps_map.get(name, state_default_eps))
                    if abs(float(mhe_result.xhat[name]) - float(prev_mhe_xhat[name])) >= eps_x:
                        state_ok = False
                        break

            output_ok = bool(measured_index)
            tf_mhe = mhe_result.model.time.last()
            if output_ok:
                for name in measured_index:
                    if name not in measured_now:
                        output_ok = False
                        break
                    try:
                        yhat_now = float(pyo.value(_add_time_indexed_expression(mhe_result.model, name, tf_mhe)))
                    except Exception:
                        output_ok = False
                        break
                    eps_y = float(output_eps_map.get(name, output_default_eps))
                    if abs(float(measured_now[name]) - yhat_now) >= eps_y:
                        output_ok = False
                        break

            if state_ok and output_ok:
                mhe_consecutive_ok += 1
            else:
                mhe_consecutive_ok = 0
                if current_mhe_window != 10:
                    current_mhe_window = 10
                    print("Adaptive MHE window reset to: 10")

            if mhe_consecutive_ok >= 1 and current_mhe_window > mhe_min_window:
                current_mhe_window -= 1
                mhe_consecutive_ok = 0
                print(f"Adaptive MHE window reduced to: {current_mhe_window}")

            prev_mhe_model = mhe_result.model
            prev_mhe_xhat = dict(mhe_result.xhat)
            for name in unmeasured_names:
                if name in mhe_result.xhat:
                    key = _get_variable_key_for_data(plant, name)
                    tf_data._data[key] = mhe_result.xhat[name]
                    if mhe_state_hist is not None:
                        mhe_state_hist[name].append(mhe_result.xhat[name])
                else:
                    if mhe_state_hist is not None:
                        mhe_state_hist[name].append(np.nan)
        else:
            mhe_consecutive_ok = 0
            if mhe_state_hist is not None:
                for name in unmeasured_names:
                    mhe_state_hist[name].append(np.nan)
            # Fallback: use plant values if MHE not available
            for name in unmeasured_names:
                key = _get_variable_key_for_data(plant, name)
                tf_data._data[key] = full_tf_data.get_data_from_key(key)
        # Record truth unmeasured states from plant at this sampling time
        if true_state_hist is not None:
            for name in true_state_hist:
                key = _get_variable_key_for_data(plant, name)
                true_state_hist[name].append(full_tf_data.get_data_from_key(key))
        plant.interface.load_data(tf_data)

        controller.interface.shift_values_by_time(options.sampling_time)
        controller.interface.load_data(tf_data, time_points=t0_controller)
        
        if options.remove_collocation:
            if options.infinite_horizon:
                controller = _remove_non_collocation_values_infinite(controller)
            else:
                controller = _remove_non_collocation_values_finite(controller)

    if options.live_plot and fig is not None:
        _finalize_live_plot(fig)

    _handle_mpc_results(
        sim_data,
        time_series,
        io_data_array,
        plant,
        cpu_time,
        options,
        mhe_state_hist=mhe_state_hist,
        true_state_hist=true_state_hist,
        setpoint_values=controller_setpoints,
    )
    print(f"MHE skipped count: {mhe_skipped}")


if __name__ == "__main__":
    options = _import_settings()
    _mpc_loop(options)
