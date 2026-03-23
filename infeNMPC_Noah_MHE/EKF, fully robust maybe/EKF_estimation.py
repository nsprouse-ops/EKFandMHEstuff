# EKF_estimation.py
"""
Augmented Extended Kalman Filter for CSTR state and disturbance estimation.

Uses the Pyomo model from model_equations.py directly via PyNumero.
No equations are duplicated here — all model structure is discovered from
the model's index sets (Measured_index, Unmeasured_index, Disturbance_index, MV_index).

Key trick: DerivativeVars at t0 are fixed to 0, so each balance constraint reduces to:
    0 - f_i(Ca[t0], ..., d_k[t0], ...) = 0
i.e. constraint residual = -f_i.

  evaluate_constraints()  →  negate  →  ODE RHS  (replaces _cstr_ode)
  evaluate_jacobian()     →  negate  →  A_c, B_c  (replaces differentiate())

Augmented state ordering: [Measured_index..., Unmeasured_index..., Disturbance_index...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

from model_equations import variables_initialize, equations_write


# ── NLP construction ──────────────────────────────────────────────────────────

def _build_nlp():
    """
    Build a 2-point Pyomo DAE model via model_equations and wrap with PyomoNLP.

    Variables/constraints at t1 are fixed/deactivated so PyNumero only operates
    on the t0 slice.  DerivativeVars at t0 are fixed to 0, so the active
    constraints take the form:
        0 - f_i(states[t0], disturbances[t0], inputs[t0]) = 0
    making evaluate_constraints() = -f_rhs and evaluate_jacobian() = -[A_c | B_c | ...].
    """
    m = pyo.ConcreteModel()
    m.time = ContinuousSet(bounds=(0, 1))
    m = variables_initialize(m)
    m = equations_write(m)
    m.obj = pyo.Objective(expr=0)

    t0 = min(m.time)
    t1 = max(m.time)

    all_dyn = (list(m.Measured_index) + list(m.Unmeasured_index)
               + list(m.Disturbance_index) + list(m.MV_index))

    # Fix all dynamic variables at t1
    for name in all_dyn:
        comp = getattr(m, name)
        if t1 in comp:
            comp[t1].fix(pyo.value(comp[t1]))

    # Fix all DerivativeVars at both time points
    for dv_comp in m.component_objects(DerivativeVar):
        for t in (t0, t1):
            if t in dv_comp:
                dv_comp[t].fix(0.0)

    # Deactivate constraints at t1
    for con_comp in m.component_objects(pyo.Constraint):
        if t1 in con_comp:
            con_comp[t1].deactivate()

    nlp = PyomoNLP(m)
    return m, nlp, t0


def _build_index_maps(m, nlp, t0):
    """
    Build all index mappings between Pyomo components and PyNumero vectors.

    Returns:
        state_names, dist_names, input_names  — ordered name lists
        state_idx, dist_idx, input_idx        — NLP primal vector indices
        balance_idx                           — NLP constraint indices (same order as state_names)
        xa_lb, xa_ub                          — augmented-state bounds for clipping
    """
    state_names = list(m.Measured_index) + list(m.Unmeasured_index)
    dist_names  = list(m.Disturbance_index)
    input_names = list(m.MV_index)

    pyomo_vars = nlp.get_pyomo_variables()
    var_to_nlp = {id(v): i for i, v in enumerate(pyomo_vars)}

    state_idx = [var_to_nlp[id(getattr(m, n)[t0])] for n in state_names]
    dist_idx  = [var_to_nlp[id(getattr(m, n)[t0])] for n in dist_names]
    input_idx = [var_to_nlp[id(getattr(m, n)[t0])] for n in input_names]

    # Discover balance constraints: constraints whose body references a DerivativeVar
    pyomo_cons = nlp.get_pyomo_constraints()
    con_to_nlp = {id(c): i for i, c in enumerate(pyomo_cons)}

    state_to_balance = {}
    for con in pyomo_cons:
        for v in identify_variables(con.body, include_fixed=True):
            if isinstance(v.parent_component(), DerivativeVar):
                sv_name = v.parent_component().get_state_var().local_name
                state_to_balance[sv_name] = con_to_nlp[id(con)]
                break

    balance_idx = [state_to_balance[n] for n in state_names]

    # Variable bounds for post-update clipping (None → ±inf)
    def _lb(name):
        v = getattr(m, name)[t0]
        return v.lb if v.lb is not None else -np.inf

    def _ub(name):
        v = getattr(m, name)[t0]
        return v.ub if v.ub is not None else np.inf

    xa_lb = np.array([max(0.0, _lb(n)) for n in state_names]
                     + [_lb(n) for n in dist_names])
    xa_ub = np.array([_ub(n) for n in state_names]
                     + [_ub(n) for n in dist_names])

    return (state_names, dist_names, input_names,
            state_idx, dist_idx, input_idx,
            balance_idx, xa_lb, xa_ub)


# ── NLP evaluation helpers ────────────────────────────────────────────────────

def _set_and_eval_rhs(nlp, buf, x, xw, u,
                      state_idx, dist_idx, input_idx, balance_idx):
    """Evaluate ODE RHS at (x, xw, u) via PyNumero. buf is mutated in place."""
    buf[state_idx] = x
    buf[dist_idx]  = xw
    buf[input_idx] = u
    nlp.set_primals(buf)
    return -nlp.evaluate_constraints()[balance_idx]


def _eval_jacobians(nlp, buf, x, xw, u,
                    state_idx, dist_idx, input_idx, balance_idx):
    """Return continuous-time A_c and B_c via PyNumero Jacobian."""
    buf[state_idx] = x
    buf[dist_idx]  = xw
    buf[input_idx] = u
    nlp.set_primals(buf)
    J   = nlp.evaluate_jacobian().toarray()
    A_c = -J[np.ix_(balance_idx, state_idx)]
    B_c = -J[np.ix_(balance_idx, dist_idx)]
    return A_c, B_c


def _discretize_jacobians(A_c: np.ndarray, B_c: np.ndarray, Ts: float):
    """ZOH discretization via augmented matrix exponential."""
    n, p = A_c.shape[0], B_c.shape[1]
    M = np.zeros((n + p, n + p))
    M[:n, :n] = A_c
    M[:n, n:] = B_c
    expM = expm(M * Ts)
    return expM[:n, :n], expM[:n, n:]


# ── EKF ───────────────────────────────────────────────────────────────────────

@dataclass
class EKFResult:
    xhat: Dict[str, float]
    P: np.ndarray


class AugmentedEKF:
    """
    Augmented EKF.  All structure (state names, sizes, C_bar, bounds) is
    discovered from the Pyomo model's index sets — nothing is hard-coded.

    Augmented state: [Measured_index..., Unmeasured_index..., Disturbance_index...]
    """

    def __init__(self, options, x0=None, xw0=None, P0=None):
        self.Ts = float(options.sampling_time)

        m, nlp, t0 = _build_nlp()
        (self._state_names, self._dist_names, self._input_names,
         self._state_idx,   self._dist_idx,   self._input_idx,
         self._balance_idx, self._xa_lb,      self._xa_ub
         ) = _build_index_maps(m, nlp, t0)

        self._nlp = nlp
        self._buf = nlp.get_primals().copy()

        self.N_X     = len(self._state_names)
        self.N_W     = len(self._dist_names)
        self.N_A     = self.N_X + self.N_W
        self._n_meas = len(list(m.Measured_index))

        # C_bar: measured states occupy the first n_meas slots of the augmented state
        self.C_bar = np.zeros((self._n_meas, self.N_A))
        self.C_bar[:, :self._n_meas] = np.eye(self._n_meas)

        # Random walk for all disturbance states
        self.A_w = np.eye(self.N_W)

        Q_x = float(getattr(options, "ekf_Q_process",     1e-4))
        Q_w = float(getattr(options, "ekf_Q_disturbance", 1e-4))
        R_v = float(getattr(options, "ekf_R",             1e-2))

        self.Q_a = np.diag([Q_x] * self.N_X + [Q_w] * self.N_W)
        self.R   = np.eye(self._n_meas) * R_v

        # Default initial values from model initialization
        if x0 is None:
            x0  = np.array([pyo.value(getattr(m, n)[t0]) for n in self._state_names])
        if xw0 is None:
            xw0 = np.array([pyo.value(getattr(m, n)[t0]) for n in self._dist_names])
        if P0 is None:
            s  = float(getattr(options, "ekf_P0_scale",             1.0))
            sd = float(getattr(options, "ekf_P0_scale_disturbance", 1.0))
            P0 = np.diag([s] * self.N_X + [sd] * self.N_W)

        self.xa         = np.concatenate([x0, xw0])
        self.P          = P0.copy()
        self._aug_names = self._state_names + self._dist_names

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def x(self):  return self.xa[:self.N_X]

    @property
    def xw(self): return self.xa[self.N_X:]

    # ── EKF steps ─────────────────────────────────────────────────────────────

    def predict(self, u: np.ndarray):
        x_prev  = self.x.copy()
        xw_prev = self.xw.copy()
        u       = np.asarray(u, dtype=float)

        A_c, B_c = _eval_jacobians(
            self._nlp, self._buf, x_prev, xw_prev, u,
            self._state_idx, self._dist_idx, self._input_idx, self._balance_idx
        )
        A_d, B_d = _discretize_jacobians(A_c, B_c, self.Ts)

        # Augmented discrete Jacobian
        A_bar = np.zeros((self.N_A, self.N_A))
        A_bar[:self.N_X, :self.N_X] = A_d
        A_bar[:self.N_X, self.N_X:] = B_d
        A_bar[self.N_X:, self.N_X:] = self.A_w

        # Propagate states via ODE using PyNumero for the RHS
        sol = solve_ivp(
            fun=lambda t, x: _set_and_eval_rhs(
                self._nlp, self._buf, x, xw_prev, u,
                self._state_idx, self._dist_idx, self._input_idx, self._balance_idx
            ),
            t_span=(0.0, self.Ts),
            y0=x_prev,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        self.xa = np.concatenate([sol.y[:, -1], self.A_w @ xw_prev])
        self.P  = A_bar @ self.P @ A_bar.T + self.Q_a

    def update(self, y_meas: np.ndarray):
        y_meas = np.asarray(y_meas, dtype=float).reshape(-1, 1)
        y_hat  = (self.C_bar @ self.xa).reshape(-1, 1)
        r      = y_meas - y_hat

        S = self.C_bar @ self.P @ self.C_bar.T + self.R
        K = self.P @ self.C_bar.T @ np.linalg.inv(S)

        self.xa = self.xa + (K @ r).flatten()
        self.xa = np.clip(self.xa, self._xa_lb, self._xa_ub)

        I_KC   = np.eye(self.N_A) - K @ self.C_bar
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R @ K.T

    def step(self, u: np.ndarray, y_meas: np.ndarray) -> EKFResult:
        self.predict(u)
        self.update(y_meas)
        return EKFResult(
            xhat={n: float(self.xa[i]) for i, n in enumerate(self._aug_names)},
            P=self.P.copy()
        )

    # ── Convenience accessors ─────────────────────────────────────────────────

    def get_unmeasured_xhat(self) -> Dict[str, float]:
        return {
            n: float(self.xa[self._n_meas + i])
            for i, n in enumerate(self._state_names[self._n_meas:])
        }

    def get_disturbance_estimates(self) -> Dict[str, float]:
        return {
            n: float(self.xa[self.N_X + i])
            for i, n in enumerate(self._dist_names)
        }

    def initialize_from_plant(self, plant_data: dict):
        for i, name in enumerate(self._state_names):
            if name in plant_data:
                self.xa[i] = float(plant_data[name])


# ── Factory ───────────────────────────────────────────────────────────────────

def make_ekf(options, x0: Optional[np.ndarray] = None,
             xw0: Optional[np.ndarray] = None) -> AugmentedEKF:
    return AugmentedEKF(options, x0, xw0)
