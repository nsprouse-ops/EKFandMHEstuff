# MHE_estimation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import copy
import pyomo.environ as pyo
from pyomo.contrib.mpc import ScalarData

from make_model import _make_finite_horizon_model, _make_steady_state_model, _solve_steady_state_model
from indexing_tools import _get_derivative_and_state_vars, _get_variable_key_for_data, _add_time_indexed_expression

@dataclass
class MHEResult:
    M_eff: int
    xhat: Dict[str, float]                 #estimated state at current time
    model: pyo.ConcreteModel               #MHE model
    solver_result: object                  #solver results


def build_mhe_histories_from_io_data_array(
    io_data_array: Sequence[Sequence[float]], #all data
    measured_index: Sequence[str], #names of measured states
    MV_index: Sequence[str], #inputs
    M_desired: int,  #what we want our horizon to be, may be shorter at early times
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], int]:
    """
    Build per-sampling-step histories for MHE from io_data_array.

    io_data_array format (based on your run_MPC.py):
      - Row k contains measurements y_k (CVs)
      - Row 0 has MV columns = None
      - Row k>=1 has MV columns = u_{k-1} (input applied between k-1 -> k)

    Returns:
      y_hist: {cv: [y_{k-M_eff}, ..., y_k]} length M_eff+1
      u_hist: {mv: [u_{k-M_eff}, ..., u_{k-1}]} length M_eff
      M_eff: effective window length used (min(M_desired, k))
    """
    if M_desired < 0:
        raise ValueError("M_desired must be >= 0") #error checking for negative horizon length

    n = len(io_data_array)  #length of all data
    if n == 0:
        raise ValueError("io_data_array is empty.") #error checking for empty data

    n_cv = len(measured_index) #number of measured states
    n_mv = len(MV_index) #number of inputs

    k = n - 1 #current time index (0-based), so row k is current measurements, row k-1 is last applied input, etc.
    M_eff = min(M_desired, k) #window lenght, using whatever is bigger, what you want or the current time

    # Measurements y: rows k-M_eff ... k (length M_eff+1)
    y_rows = list(range(k - M_eff, k + 1)) #

    u_rows = list(range(k - M_eff + 1, k + 1))

    # Build y histories
    y_hist: Dict[str, List[float]] = {name: [] for name in measured_index} #building the output history that we use for MHE
    for r in y_rows:
        row = io_data_array[r]
        if len(row) < n_cv + n_mv:
            raise ValueError(
                f"io_data_array row {r} has length {len(row)}, expected {n_cv + n_mv}"
            )
        for j, cv in enumerate(measured_index):
            val = row[j]
            if val is None:
                raise ValueError(
                    f"Measured '{cv}' is None at io_data_array row {r}. "
                    f"Not enough measurement history yet."
                )
            y_hist[cv].append(float(val))

    # Build u histories #building the input history that we use for MHE
    u_hist: Dict[str, List[float]] = {name: [] for name in MV_index}
    if M_eff > 0:
        for r in u_rows:
            row = io_data_array[r]
            for j, mv in enumerate(MV_index):
                val = row[n_cv + j]
                if val is None:
                    raise ValueError(
                        f"MV '{mv}' is None at io_data_array row {r}. "
                        f"Not enough MV history yet."
                    )
                u_hist[mv].append(float(val))

    return y_hist, u_hist, M_eff


def _make_options_for_mhe(options, M_eff: int):
    """
    Pulling options for MHE model from the infNMPC options file
    """
    opt = copy.copy(options)
    opt.nfe_finite = int(M_eff)
    return opt


def _get_cached_ss_arrival_weights(options, equations_module: str = "model_equations") -> Dict[str, float]:
    """
    Build (once) and cache arrival weights as 1/(x_ss^2) using solved steady-state values.
    """
    cached = getattr(options, "_mhe_ss_arrival_weights_cache", None)
    if isinstance(cached, dict) and len(cached) > 0:
        return cached

    m_ss = pyo.ConcreteModel()
    m_ss = _make_steady_state_model(m_ss, options, equations_module=equations_module)

    if options.custom_objective:
        m_ss_target = None
    else:
        setpoint_targets = {
            _get_variable_key_for_data(m_ss, var): pyo.value(m_ss.setpoints[var])
            for var in m_ss.setpoints.index_set()
        }
        m_ss_target = ScalarData(setpoint_targets)

    ss_data = _solve_steady_state_model(m_ss, m_ss_target, options)

    eps = float(getattr(options, "mhe_arrival_ss_weight_epsilon", 1e-12))
    ss_weights: Dict[str, float] = {}
    theta_arrival_weight = float(getattr(options, "theta_arrival_weight", 25.0))
    for name in getattr(m_ss, "Unmeasured_index", []):
        try:
            key = _get_variable_key_for_data(m_ss, name)
            val = ss_data.get_data_from_key(key)
            if isinstance(val, list):
                if len(val) != 1:
                    continue
                val = val[0]
            x_ss = float(val)
            denom = max(abs(x_ss), eps)
            ss_weights[name] = theta_arrival_weight / (denom **2)  #arrival weight for states
        except Exception:
            continue

    setattr(options, "_mhe_ss_arrival_weights_cache", ss_weights)
    return ss_weights


def solve_mhe_no_arrival_cost(
    options,
    io_data_array: Sequence[Sequence[float]],
    M_desired: int,
    solver_name: str = "ipopt",
    tee: bool = False,
    prior_xhat: Optional[Dict[str, float]] = None,
    prior_d_ua: Optional[float] = None,
    prior_d_k: Optional[float] = None,
    warm_start_x0: Optional[Dict[str, float]] = None,
    lambda_arrival: Optional[float] = None,
    equations_module: str = "model_equations",
) -> Optional[MHEResult]:
    """
    creates model, declares weights for arrival cost, defines objective, and solves MHE problem to get xhat

    """
    # Need at least one applied MV
    k = len(io_data_array) - 1
    if k < 1:
        return None

    # Build histories
    # Use a temporary model just to read indicies
    tmp_opt = _make_options_for_mhe(options, M_eff=1)
    tmp = pyo.ConcreteModel()
    tmp = _make_finite_horizon_model(tmp, tmp_opt, equations_module=equations_module)
    measured_index = list(getattr(tmp, "Measured_index", tmp.CV_index))
    MV_index = list(tmp.MV_index)

    y_hist, u_hist, M_eff = build_mhe_histories_from_io_data_array(
        io_data_array=io_data_array,
        measured_index=measured_index,
        MV_index=MV_index,
        M_desired=M_desired,
    )

    # If M_eff ends up 0 cant do anything
    if M_eff < 1:
        return None
    
    arrival_active = (bool(prior_xhat) or (prior_d_ua is not None) or (prior_d_k is not None)) and (M_eff >= M_desired) # Arrival should activate only once the full window is available

    # Build an MHE model with M_eff finite elements
    mhe_opt = _make_options_for_mhe(options, M_eff=M_eff)
    m = pyo.ConcreteModel()
    m = _make_finite_horizon_model(m, mhe_opt, equations_module=equations_module)

    # Finite element times: should be [0, Ts, 2Ts, ..., M_eff*Ts]
    fe_times = list(m.time.get_finite_elements())
    if len(fe_times) != M_eff + 1:
        raise RuntimeError(
            f"Expected {M_eff+1} finite elements, got {len(fe_times)}. "
            f"Check discretization."
        )
    
    t0 = fe_times[0]
    for var in m.state_vars:
        for index in var:
            if (isinstance(index, tuple) and index[-1] == t0) or index == t0: #unfixing the initial state variables at the first time point of the MHE horizon, so that they can be estimated
                var[index].unfix()

    # Bootstrap reference for the very first MHE solve
    # Use model-initialized values from model_equations as xhat0
    #not enought data to accurately predict without this, drastically wrong estimates without it on the first solve
    bootstrap_xhat0: Dict[str, float] = {}
    bootstrap_d_ua0: Optional[float] = None
    bootstrap_d_k0: Optional[float] = None
    if not arrival_active:
        for name in getattr(m, "Unmeasured_index", []):
            try:
                bootstrap_xhat0[name] = float(pyo.value(_add_time_indexed_expression(m, name, t0)))
            except Exception:
                continue
        if hasattr(m, "d_UA"):
            try:
                bootstrap_d_ua0 = float(pyo.value(m.d_UA[t0]))
            except Exception:
                bootstrap_d_ua0 = 0.0
        if hasattr(m, "d_k"):
            try:
                bootstrap_d_k0 = float(pyo.value(m.d_k[t0]))
            except Exception:
                bootstrap_d_k0 = 0.0
    #warm start for x0
    if warm_start_x0:
        for name, guess in warm_start_x0.items():
            try:
                x0 = _add_time_indexed_expression(m, name, t0)
                x0.set_value(float(guess))
            except Exception:
                continue

    # --- Measurement parameters y_meas(measured, t_fe) ---
    m.y_meas = pyo.Param(m.Measured_index, m.time, mutable=True, initialize=0.0)
    m.fe_k = pyo.RangeSet(0, M_eff)
    m.fe_k_dyn = pyo.RangeSet(0, M_eff - 1)
    e_ua_bound = float(getattr(options, "mhe_e_ua_bound", 1.0))
    m.e_ua = pyo.Var(m.fe_k_dyn, initialize=0.0, domain=pyo.Reals, bounds=(-e_ua_bound, e_ua_bound))
    e_k_bound = float(getattr(options, "mhe_e_k_bound", 1.0))
    m.e_k = pyo.Var(m.fe_k_dyn, initialize=0.0, domain=pyo.Reals, bounds=(-e_k_bound, e_k_bound))

    if hasattr(m, "d_UA"):
        def _d_ua_dyn_rule(mm, k):
            t_k = fe_times[k]
            t_kp1 = fe_times[k + 1]
            return mm.d_UA[t_kp1] == mm.d_UA[t_k] + mm.e_ua[k]
        m.d_ua_dyn = pyo.Constraint(m.fe_k_dyn, rule=_d_ua_dyn_rule)

        def _d_ua_hold_rule(mm, t):
            if t in fe_times:
                return pyo.Constraint.Skip
            t_left = mm.time.get_lower_element_boundary(t)
            return mm.d_UA[t] == mm.d_UA[t_left]
        m.d_ua_hold = pyo.Constraint(m.time, rule=_d_ua_hold_rule)

    if hasattr(m, "d_k"):
        def _d_k_dyn_rule(mm, k):
            t_k = fe_times[k]
            t_kp1 = fe_times[k + 1]
            return mm.d_k[t_kp1] == mm.d_k[t_k] + mm.e_k[k]
        m.d_k_dyn = pyo.Constraint(m.fe_k_dyn, rule=_d_k_dyn_rule)

        def _d_k_hold_rule(mm, t):
            if t in fe_times:
                return pyo.Constraint.Skip
            t_left = mm.time.get_lower_element_boundary(t)
            return mm.d_k[t] == mm.d_k[t_left]
        m.d_k_hold = pyo.Constraint(m.time, rule=_d_k_hold_rule)

    # Load measurement history only at finite elements
    for cv in m.Measured_index:
        series = y_hist[cv]
        if len(series) != len(fe_times):
            raise ValueError(
                f"y_hist['{cv}'] length {len(series)} != finite elements {len(fe_times)}"
            )
        for t_fe, val in zip(fe_times, series):
            m.y_meas[cv, t_fe] = float(val)

    #Fix MV trajectory (constant over each time step)
    # At final time, hold last input
    for mv in m.MV_index:
        u_series = u_hist[mv]
        if len(u_series) != M_eff:
            raise ValueError(
                f"u_hist['{mv}'] length {len(u_series)} != M_eff {M_eff}"
            )

        for j, t_fe in enumerate(fe_times):
            idx = min(max(j - 1, 0), M_eff - 1)
            getattr(m, mv)[t_fe].fix(float(u_series[idx]))
    # Arrival weights: per-state override, otherwise default
    default_arrival_lambda = float(
        getattr(options, "mhe_arrival_default_lambda", 1) #checks if there is a defualt set in options, if not uses value at end
        if lambda_arrival is None
        else lambda_arrival
    )
    arrival_weight_map = dict(getattr(options, "mhe_arrival_weights", {}))
    if getattr(options, "mhe_arrival_use_steady_state_weights", True):
        try:
            ss_weight_map = _get_cached_ss_arrival_weights(options, equations_module=equations_module)
            for name, val in ss_weight_map.items():
                arrival_weight_map.setdefault(name, float(val))
        except Exception as err:
            print(f"MHE warning: failed to compute steady-state arrival weights ({err})")

    def _arrival_weight(state_name: str) -> float:
        return float(arrival_weight_map.get(state_name, default_arrival_lambda)) #loooking fow specific weight for this state, if not found uses default
    F_state_weight = float(getattr(options, "F_state_weight", 1.0))
    e_ua_weight = float(getattr(options, "mhe_e_ua_weight", 1.0))
    e_k_weight = float(getattr(options, "mhe_e_k_weight", 1.0))
    d_ua_arrival_weight = float(getattr(options, "mhe_d_ua_arrival_weight", 1.0))
    d_k_arrival_weight = float(getattr(options, "mhe_d_k_arrival_weight", 1.0))
    d_ua_max_step = float(getattr(options, "mhe_d_ua_max_step", 100.0))
    d_k_max_step = float(getattr(options, "mhe_d_k_max_step", 100.0))

    # Limit inter-iteration change of initial disturbance estimate.
    # Before arrival is active, leave d_UA(t0) free so the estimator can
    # move the initial disturbance without bootstrap anchoring.
    d_ua_step_ref: Optional[float] = None
    if prior_d_ua is not None:
        d_ua_step_ref = float(prior_d_ua)
    elif (not arrival_active) and (bootstrap_d_ua0 is not None):
        d_ua_step_ref = float(bootstrap_d_ua0)
    if hasattr(m, "d_UA") and (d_ua_step_ref is not None):
        m.d_ua_step_up = pyo.Constraint(expr=m.d_UA[t0] - d_ua_step_ref <= d_ua_max_step)
        m.d_ua_step_dn = pyo.Constraint(expr=d_ua_step_ref - m.d_UA[t0] <= d_ua_max_step)

    d_k_step_ref: Optional[float] = None
    if prior_d_k is not None:
        d_k_step_ref = float(prior_d_k)
    elif (not arrival_active) and (bootstrap_d_k0 is not None):
        d_k_step_ref = float(bootstrap_d_k0)
    if hasattr(m, "d_k") and (d_k_step_ref is not None):
        m.d_k_step_up = pyo.Constraint(expr=m.d_k[t0] - d_k_step_ref <= d_k_max_step)
        m.d_k_step_dn = pyo.Constraint(expr=d_k_step_ref - m.d_k[t0] <= d_k_max_step)

    #Objective: sum of squared measurement errors at finite elements only
    # + arrival cost at the start of the window, and for the solves before the arrival cost is active,
    # bootstrap cost using model intital conditions and after the first the condition from the estimation before
    def _mhe_obj_rule(mm):
        expr = 0
        for k, t_fe in enumerate(fe_times):
            for cv in mm.Measured_index:
                yhat = _add_time_indexed_expression(mm, cv, t_fe)
                ymeas = mm.y_meas[cv, t_fe]
                # Use stage_cost_weights if you have them; else weight=1
                if hasattr(mm, "cv_cost"):
                    w = mm.cv_cost[cv]
                expr += F_state_weight *  (yhat - ymeas) ** 2
        expr += e_ua_weight * sum(mm.e_ua[k] ** 2 for k in mm.fe_k_dyn)
        expr += e_k_weight * sum(mm.e_k[k] ** 2 for k in mm.fe_k_dyn)
        # Arrival cost at start of window
        if arrival_active:
            if prior_xhat:
                for name, prior_val in prior_xhat.items():
                    try:
                        x0 = _add_time_indexed_expression(mm, name, t0)
                    except Exception:
                        continue
                    w_arr = _arrival_weight(name)
                    expr += w_arr * (x0 - float(prior_val)) ** 2 #arrival weight * squared error of initial state from prior
            if (prior_d_ua is not None) and hasattr(mm, "d_UA"):
                expr += d_ua_arrival_weight * (mm.d_UA[t0] - float(prior_d_ua)) ** 2
            if (prior_d_k is not None) and hasattr(mm, "d_k"):
                expr += d_k_arrival_weight * (mm.d_k[t0] - float(prior_d_k)) ** 2
        elif bootstrap_xhat0:
            for name, init_val in bootstrap_xhat0.items():
                try:
                    x0 = _add_time_indexed_expression(mm, name, t0)
                except Exception:
                    continue
                w_arr = _arrival_weight(name)
                expr += w_arr * (x0 - float(init_val)) ** 2 #arrival weight * squared error of initial state from "prior" (initial state condtion)
        if (not arrival_active) and (bootstrap_d_ua0 is not None) and hasattr(mm, "d_UA"):
            expr += d_ua_arrival_weight * (mm.d_UA[t0] - float(bootstrap_d_ua0)) ** 2
        if (not arrival_active) and (bootstrap_d_k0 is not None) and hasattr(mm, "d_k"):
            expr += d_k_arrival_weight * (mm.d_k[t0] - float(bootstrap_d_k0)) ** 2
        return expr

    m.mhe_obj = pyo.Objective(rule=_mhe_obj_rule)

    # --- Solve ---
    print("MHE estimation solving")
    solver = pyo.SolverFactory(solver_name) #same solver as MPC
    res = solver.solve(m, tee=False) #solving model
    if not pyo.check_optimal_termination(res):
        status = getattr(res.solver, "status", "unknown")
        term = getattr(res.solver, "termination_condition", "unknown")
        raise RuntimeError(
            f"MHE solve failed to reach optimal solution. "
            f"status={status}, termination={term}"
        )

    tf = fe_times[-1]

    if hasattr(m, "d_UA"):
        e_ua_vals = [float(pyo.value(m.e_ua[k])) for k in m.fe_k_dyn]
        d_ua_vals = [float(pyo.value(m.d_UA[t_fe])) for t_fe in fe_times]
        print("MHE e_ua over horizon:", e_ua_vals)
        print("MHE d_UA over horizon (FE points):", d_ua_vals)
    if hasattr(m, "d_k"):
        e_k_vals = [float(pyo.value(m.e_k[k])) for k in m.fe_k_dyn]
        d_k_vals = [float(pyo.value(m.d_k[t_fe])) for t_fe in fe_times]
        print("MHE e_k over horizon:", e_k_vals)
        print("MHE d_k over horizon (FE points):", d_k_vals)

    # End-of-window output residuals: y_measured - y_est_from_states
    residual_tf = {}
    for cv in m.Measured_index:
        try:
            y_meas_tf = float(pyo.value(m.y_meas[cv, tf]))
            y_est_tf = float(pyo.value(_add_time_indexed_expression(m, cv, tf)))
            residual_tf[str(cv)] = y_meas_tf - y_est_tf
        except Exception:
            continue
    print("MHE residual at tf (y_measured - y_est_from_states):", residual_tf)

    #getting what the final estimates were in the MHE, which is the final time in the horzion, these are the estimates that we will use for the initial condition of the MPC at the next time step, and also to report how well the MHE is doing in estimating the current state
    unmeasured = set(m.Unmeasured_index)

    xhat = {}
    for v in m.state_vars:
        if v.is_indexed():
            for index in v.index_set():
                if isinstance(index, tuple):
                    if index[-1] != tf:
                        continue
                    base_index = index[:-1]
                    name = f"{v.local_name}[{','.join(str(i) for i in base_index)}]"
                    if name in unmeasured:
                        xhat[name] = pyo.value(v[index])
                else:
                    if index != tf:
                        continue
                    name = v.local_name
                    if name in unmeasured:
                        xhat[name] = pyo.value(v[index])
        else:
            name = v.local_name
            if name in unmeasured:
                xhat[name] = pyo.value(v[tf])

    return MHEResult(M_eff=M_eff, xhat=xhat, model=m, solver_result=res)

