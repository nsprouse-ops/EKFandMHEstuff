# EKF_estimation.py
"""
Augmented Extended Kalman Filter for CSTR state and disturbance estimation.

Augmented state  x_a = [Ca, Cb, Cc, Cm, T, Fb0_est, UA_est]   (7 states)
Disturbance model: x_w(k) = A_w * x_w(k-1),   A_w = I_2  (random walk)
Measurement:       y_m = T                                  (scalar)
Cw = I_2  ->  d = C_w * x_w = x_w  (disturbance states map 1:1 to Fb0, UA)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import pyomo.environ as pyo
from pyomo.core.expr import exp as pyo_exp
from pyomo.core.expr.calculus.derivatives import differentiate   # Pyomo >= 6.x


# ============================================================
#  Model constants  (must match model_equations.py)
# ============================================================
_Ta1   = 288.7
_CpW   = 18.0
_T0    = 297.0
_dH    = -20013.0
_V     = 5.0
_Fm0   = 45.4
_k_pre = 1.696e13          # pre-exponential in controller model
_EaR   = 18012.0 / 1.987   # Ea / R


# ============================================================
#  Pure-Python CSTR ODE  (used by scipy for state prediction)
# ============================================================

def _cstr_ode(t, x_state, u, Fb0, UA):
    """
    Continuous-time CSTR ODE RHS.
    x_state : [Ca, Cb, Cc, Cm, T]
    u       : [Fa0, mc]
    Fb0, UA : current disturbance estimates
    """
    Ca, Cb, Cc, Cm, T = x_state
    Fa0 = float(u[0]);  mc  = float(u[1])
    Fb0 = float(Fb0);   UA  = float(UA)

    k       = _k_pre * np.exp(-_EaR / T)
    ra      = -k * Ca;  rb = -k * Ca;  rc = k * Ca

    v0      = Fa0 / 14.8 + Fb0 / 55.3 + _Fm0 / 24.7
    ThetaCp = 35.0 + Fb0 / Fa0 * 18.0 + _Fm0 / Fa0 * 19.5
    Ta2     = T - (T - _Ta1) * np.exp(-UA / (_CpW * mc))
    Ca0     = Fa0 / v0;  Cb0 = Fb0 / v0;  Cm0 = _Fm0 / v0
    tau_tc  = _V  / v0

    Qr1 = Fa0 * ThetaCp * (T - _T0)
    Qr2 = mc  * _CpW    * (Ta2 - _Ta1)
    Qg  = ra  * _V      * _dH

    Na  = Ca * _V;  Nb = Cb * _V;  Nc = Cc * _V;  Nm = Cm * _V
    NCp = Na * 35.0 + Nb * 18.0 + Nc * 46.0 + Nm * 19.5

    dCadt = (1.0 / tau_tc) * (Ca0 - Ca) + ra
    dCbdt = (1.0 / tau_tc) * (Cb0 - Cb) + rb
    dCcdt = (1.0 / tau_tc) * (0.0 - Cc) + rc
    dCmdt = (1.0 / tau_tc) * (Cm0 - Cm)
    dTdt  = (Qg - Qr1 - Qr2) / NCp

    return [dCadt, dCbdt, dCcdt, dCmdt, dTdt]


# ============================================================
#  Pyomo model for Jacobian computation  (built once, reused)
# ============================================================

def _build_jacobian_model():
    """
    Build a scalar (non-time-indexed) Pyomo model whose variable values
    are updated in-place before each symbolic differentiation call.
    The ODE RHS expressions are stored on the model for reuse.
    """
    m = pyo.ConcreteModel()

    # ---- Variables ----
    m.Ca  = pyo.Var(initialize=1.5,   domain=pyo.NonNegativeReals)
    m.Cb  = pyo.Var(initialize=1.5,   domain=pyo.NonNegativeReals)
    m.Cc  = pyo.Var(initialize=1.5,   domain=pyo.NonNegativeReals)
    m.Cm  = pyo.Var(initialize=1.5,   domain=pyo.NonNegativeReals)
    m.T   = pyo.Var(initialize=297.0, domain=pyo.NonNegativeReals)
    m.Fa0 = pyo.Var(initialize=35.0,  bounds=(10.0, 100.0))
    m.mc  = pyo.Var(initialize=450.0, bounds=(250.0, 1000.0))
    m.Fb0 = pyo.Var(initialize=453.6)   # disturbance state 1
    m.UA  = pyo.Var(initialize=7262.0)  # disturbance state 2

    # ---- Algebraic sub-expressions (same structure as model_equations.py) ----
    k_e     = _k_pre * pyo_exp(-_EaR / m.T)
    ra      = -k_e * m.Ca;  rb = -k_e * m.Ca;  rc = k_e * m.Ca

    v0      = m.Fa0 / 14.8 + m.Fb0 / 55.3 + _Fm0 / 24.7
    ThetaCp = 35.0 + m.Fb0 / m.Fa0 * 18.0 + _Fm0 / m.Fa0 * 19.5
    Ta2     = m.T - (m.T - _Ta1) * pyo_exp(-m.UA / (_CpW * m.mc))
    Ca0     = m.Fa0 / v0;  Cb0 = m.Fb0 / v0;  Cm0 = _Fm0 / v0
    tau_tc  = _V / v0

    Qr1 = m.Fa0 * ThetaCp * (m.T - _T0)
    Qr2 = m.mc  * _CpW    * (Ta2 - _Ta1)
    Qg  = ra    * _V      * _dH

    Na  = m.Ca * _V;  Nb = m.Cb * _V;  Nc = m.Cc * _V;  Nm = m.Cm * _V
    NCp = Na * 35.0 + Nb * 18.0 + Nc * 46.0 + Nm * 19.5

    # ---- ODE RHS as Pyomo expression objects (stored for differentiation) ----
    f_Ca = (1.0 / tau_tc) * (Ca0 - m.Ca) + ra
    f_Cb = (1.0 / tau_tc) * (Cb0 - m.Cb) + rb
    f_Cc = (1.0 / tau_tc) * (0.0 - m.Cc) + rc
    f_Cm = (1.0 / tau_tc) * (Cm0 - m.Cm)
    f_T  = (Qg - Qr1 - Qr2) / NCp

    # Underscore prefix bypasses Pyomo component management
    m._rhs        = [f_Ca, f_Cb, f_Cc, f_Cm, f_T]
    m._state_vars = [m.Ca, m.Cb, m.Cc, m.Cm, m.T]
    m._dist_vars  = [m.Fb0, m.UA]

    return m


def _compute_continuous_jacobians(
    jac_model, x: np.ndarray, u: np.ndarray, xw: np.ndarray
):
    """
    Update Pyomo model values then compute A_c (5x5) and B_c (5x2)
    via symbolic differentiation evaluated at (x, u, xw).
    """
    m = jac_model

    for var, val in zip(m._state_vars, x):
        var.set_value(float(val))
    m.Fa0.set_value(float(u[0]))
    m.mc.set_value(float(u[1]))
    for var, val in zip(m._dist_vars, xw):
        var.set_value(float(val))

    A_c = np.zeros((5, 5))
    for i, f in enumerate(m._rhs):
        for j, xv in enumerate(m._state_vars):
            A_c[i, j] = pyo.value(differentiate(f, wrt=xv))

    B_c = np.zeros((5, 2))
    for i, f in enumerate(m._rhs):
        for j, dv in enumerate(m._dist_vars):
            B_c[i, j] = pyo.value(differentiate(f, wrt=dv))

    return A_c, B_c


def _discretize_jacobians(A_c: np.ndarray, B_c: np.ndarray, Ts: float):
    """
    Zero-order-hold (ZOH) discretization via augmented matrix exponential:

        expm([[A_c, B_c], [0, 0]] * Ts) = [[A_d, B_d], [0, I]]

    Returns A_d (5x5) and B_d (5x2).
    """
    n, m = A_c.shape[0], B_c.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_c
    M[:n, n:] = B_c
    expM = expm(M * Ts)
    return expM[:n, :n], expM[:n, n:]


# ============================================================
#  EKF dataclass
# ============================================================

@dataclass
class EKFResult:
    xhat: Dict[str, float]   # augmented state: Ca, Cb, Cc, Cm, T, Fb0_est, UA_est
    P: np.ndarray            # 7x7 error covariance


# ============================================================
#  Augmented EKF class
# ============================================================

class AugmentedEKF:
    """
    Augmented EKF estimating CSTR process states and disturbance states jointly.

    Augmented state ordering:
        [0] Ca   [1] Cb   [2] Cc   [3] Cm   [4] T   [5] Fb0_est   [6] UA_est

    Cw = I_2   (disturbance output maps 1:1 to Fb0, UA inputs of F)
    Aw = I_2   (random-walk disturbance model)
    """

    _STATE_NAMES = ["Ca", "Cb", "Cc", "Cm", "T"]
    _DIST_NAMES  = ["Fb0_est", "UA_est"]
    _AUG_NAMES   = _STATE_NAMES + _DIST_NAMES

    N_X = 5   # process states
    N_W = 2   # disturbance states
    N_A = 7   # augmented state size

    C_w = np.eye(2)   # disturbance output matrix
    A_w = np.eye(2)   # disturbance transition matrix (random walk)

    def __init__(
        self,
        options,
        x0:  np.ndarray,
        xw0: np.ndarray,
        P0:  np.ndarray,
    ):
        """
        Parameters
        ----------
        options : Options
            Reads: sampling_time, ekf_Q_process, ekf_Q_disturbance, ekf_R
        x0  : (5,)  initial process state [Ca, Cb, Cc, Cm, T]
        xw0 : (2,)  initial disturbance estimate [Fb0, UA]
        P0  : (7,7) initial error covariance
        """
        self.Ts = float(options.sampling_time)

        Q_x = float(getattr(options, "ekf_Q_process",     1e-4))
        Q_w = float(getattr(options, "ekf_Q_disturbance", 1e-6))
        R   = float(getattr(options, "ekf_R",             1e-3))

        self.Q_a = np.diag([Q_x] * self.N_X + [Q_w] * self.N_W)   # (7,7)
        self.R   = np.array([[R]])                                   # (1,1)

        self.xa = np.concatenate([x0.copy(), xw0.copy()])           # (7,)
        self.P  = P0.copy()                                          # (7,7)

        self._jac_model = _build_jacobian_model()

    # ---- convenience properties ----

    @property
    def x(self) -> np.ndarray:
        """Current process state estimate [Ca, Cb, Cc, Cm, T]."""
        return self.xa[: self.N_X]

    @property
    def xw(self) -> np.ndarray:
        """Current disturbance state estimate [Fb0_est, UA_est]."""
        return self.xa[self.N_X :]

    # ---- EKF prediction step ----

    def predict(self, u: np.ndarray):
        """
        Prediction step.

        x_hat(k|k-1)  = F( x_hat(k-1|k-1), u(k-1), C_w * x_w_hat(k-1|k-1) )
        x_w_hat(k|k-1) = A_w * x_w_hat(k-1|k-1)
        P(k|k-1)      = A_bar * P(k-1|k-1) * A_bar' + Q_a

        Parameters
        ----------
        u : (2,)  [Fa0, mc] applied at the previous time step
        """
        x_prev  = self.x.copy()
        xw_prev = self.xw.copy()

        # 1. Jacobians linearized at the previous estimate x(k-1|k-1)
        A_c, B_c = _compute_continuous_jacobians(
            self._jac_model, x_prev, u, xw_prev
        )
        A_d, B_d = _discretize_jacobians(A_c, B_c, self.Ts)

        # 2. Augmented Jacobian A_bar (7x7)
        #    A_bar = [[A_d,    B_d * C_w],
        #             [0,      A_w      ]]
        A_bar = np.zeros((self.N_A, self.N_A))
        A_bar[: self.N_X, : self.N_X] = A_d
        A_bar[: self.N_X, self.N_X :] = B_d @ self.C_w   # C_w = I -> just B_d
        A_bar[self.N_X :, self.N_X :] = self.A_w

        # 3. State prediction via numerical integration of CSTR ODEs
        Fb0_est, UA_est = xw_prev
        sol = solve_ivp(
            fun=lambda t, x: _cstr_ode(t, x, u, Fb0_est, UA_est),
            t_span=(0.0, self.Ts),
            y0=x_prev,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        x_pred  = sol.y[:, -1]
        xw_pred = self.A_w @ xw_prev        # random walk: x_w unchanged

        # 4. Covariance prediction
        self.P = A_bar @ self.P @ A_bar.T + self.Q_a

        # 5. Commit predicted augmented state
        self.xa = np.concatenate([x_pred, xw_pred])

    # ---- EKF update step ----

    def update(self, y_meas: float):
        """
        Update step.

        y_hat(k|k-1) = g( x_a_hat(k|k-1) )  =  T_predicted
        r(k)         = y_m(k) - y_hat(k|k-1)
        S(k)         = C_bar * P(k|k-1) * C_bar' + R
        K_f(k)       = P(k|k-1) * C_bar' * S^{-1}
        x_a(k|k)     = x_a(k|k-1) + K_f * r
        P(k|k)       = (I - K_f * C_bar) * P(k|k-1)  [Joseph form]

        Parameters
        ----------
        y_meas : float  measured temperature T_m(k)
        """
        # C_bar (1x7): d(T)/d(x_a).
        # C_hat  = [0,0,0,0,1]  (T is index 4 in process states)
        # D_hat  = [0,0]         (g = T has no direct dependence on Fb0, UA)
        # -> C_bar = [0, 0, 0, 0, 1,  0, 0]
        C_bar = np.zeros((1, self.N_A))
        C_bar[0, 4] = 1.0

        y_hat = float(self.xa[4])                           # predicted T
        r     = np.array([[y_meas - y_hat]])                # (1,1) innovation

        S = C_bar @ self.P @ C_bar.T + self.R               # (1,1)
        K = self.P @ C_bar.T @ np.linalg.inv(S)             # (7,1) Kalman gain

        # State update
        self.xa = self.xa + (K @ r).flatten()

        # Covariance update - Joseph form for numerical stability
        I_KC   = np.eye(self.N_A) - K @ C_bar
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R @ K.T

    # ---- Combined step ----

    def step(self, u: np.ndarray, y_meas: float) -> EKFResult:
        """
        Full predict + update cycle. Call once per MPC sampling interval.

        Parameters
        ----------
        u      : (2,)  [Fa0, mc] applied in the previous interval
        y_meas : float measured temperature T_m(k)

        Returns
        -------
        EKFResult with xhat dict and updated covariance P
        """
        self.predict(u)
        self.update(y_meas)
        xhat = {name: float(self.xa[i]) for i, name in enumerate(self._AUG_NAMES)}
        return EKFResult(xhat=xhat, P=self.P.copy())

    # ---- Helpers for integration with run_MPC.py ----

    def get_unmeasured_xhat(self) -> Dict[str, float]:
        """
        Return unmeasured process state estimates matching Unmeasured_index
        in model_equations.py: {Ca, Cb, Cc, Cm}.
        Use this to replace mhe_result.xhat in run_MPC.py.
        """
        unmeasured = {"Ca", "Cb", "Cc", "Cm"}
        return {
            name: float(self.xa[i])
            for i, name in enumerate(self._STATE_NAMES)
            if name in unmeasured
        }

    def get_disturbance_estimates(self) -> Dict[str, float]:
        """Return current disturbance state estimates {Fb0, UA}."""
        return {
            "Fb0": float(self.xa[5]),
            "UA":  float(self.xa[6]),
        }

    def initialize_from_plant(self, plant_data: dict):
        """
        Override initial state from plant data after plant is built.
        plant_data : dict mapping state name -> float value
                     e.g. {'Ca': 1.5, 'Cb': 1.5, 'Cc': 1.5, 'Cm': 1.5, 'T': 297.0}
        """
        for i, name in enumerate(self._STATE_NAMES):
            if name in plant_data:
                self.xa[i] = float(plant_data[name])


# ============================================================
#  Factory function
# ============================================================

def make_ekf(
    options,
    x0:  Optional[np.ndarray] = None,
    xw0: Optional[np.ndarray] = None,
) -> AugmentedEKF:
    """
    Build and return an AugmentedEKF initialized at model defaults.

    Parameters
    ----------
    options : Options
        infNMPC options. Add to infNMPC_options.py:
            self.ekf_Q_process     = 1e-4   # process noise (states)
            self.ekf_Q_disturbance = 1e-6   # process noise (disturbances)
            self.ekf_R             = 1e-3   # measurement noise (T)
            self.ekf_P0_scale      = 1.0    # initial covariance = I * scale
    x0  : (5,) initial [Ca, Cb, Cc, Cm, T]  -- defaults to model_equations.py values
    xw0 : (2,) initial [Fb0, UA]             -- defaults to nominal parameter values
    """
    if x0 is None:
        x0 = np.array([1.5, 1.5, 1.5, 1.5, 297.0])   # matches model_equations defaults
    if xw0 is None:
        xw0 = np.array([453.6, 7262.0])               # nominal Fb0, UA

    P0_scale = float(getattr(options, "ekf_P0_scale", 1.0))
    P0 = np.eye(AugmentedEKF.N_A) * P0_scale

    return AugmentedEKF(options, x0, xw0, P0)
