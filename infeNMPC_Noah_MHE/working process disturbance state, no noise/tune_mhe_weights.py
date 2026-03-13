import math
from typing import Dict, List, Tuple

import numpy as np
try:
    import cma
except ImportError as err:
    raise ImportError("Missing dependency 'cma'. Install with: pip install cma") from err

from infNMPC_options import _import_settings
from run_MPC import _mpc_loop

def _compute_objective(result: Dict) -> float:
    io_data_array = result.get("io_data_array")
    cv_index = result.get("cv_index")
    setpoint_values = result.get("setpoint_values")
    if not io_data_array or not cv_index or setpoint_values is None:
        return float("inf")

    track_sse = 0.0
    n_used = 0
    for row in io_data_array:
        if not row:
            continue
        for j, var_name in enumerate(cv_index):
            if str(var_name) != "Cc":
                continue
            if j >= len(row):
                continue
            y_actual = row[j]
            y_sp = setpoint_values.get(var_name, None)
            if y_sp is None:
                continue
            try:
                ya = float(y_actual)
                ys = float(y_sp)
            except Exception:
                continue
            if np.isnan(ya) or np.isnan(ys):
                continue
            track_sse += (ys - ya) ** 2
            n_used += 1

    if n_used == 0:
        return float("inf")

    obj = float(track_sse / n_used)
    skip_penalty = 1e3 * float(result.get("mhe_skipped", 0))
    return obj + skip_penalty


def _evaluate(weights: Dict[str, float], seed: int, num_horizons: int) -> Tuple[float, Dict]:
    options = _import_settings()
    options.live_plot = False
    options.plot_end = False
    options.save_data = False
    options.save_figure = False
    options.tee_flag = False
    options.num_horizons = num_horizons

    options.measurement_noise_seeded = True
    options.measurement_noise_seed = int(seed)
    options.rebuild_setpoints_on_d_ua_change = False

    options.theta_arrival_weight = float(weights["theta_arrival_weight"])
    options.F_state_weight = float(weights["F_state_weight"])
    options.mhe_e_ua_weight = float(weights["mhe_e_ua_weight"])
    options.mhe_d_ua_bootstrap_weight = float(weights["mhe_d_ua_bootstrap_weight"])
    options.mhe_d_ua_arrival_weight = float(weights["mhe_d_ua_arrival_weight"])

    result = _mpc_loop(options)
    score = _compute_objective(result)
    return score, result


def main():
    max_iter = 12
    pop_size = 8
    num_horizons = 60
    eval_seed = 12345

    search_space = {
        "theta_arrival_weight": (1e-2, 1e3),
        "F_state_weight": (1e-3, 1e2),
        "mhe_e_ua_weight": (1e-6, 1e1),
        "mhe_d_ua_bootstrap_weight": (1e-4, 1e3),
        "mhe_d_ua_arrival_weight": (1e-4, 1e3),
    }

    names = list(search_space.keys())
    lower_log = [math.log10(search_space[n][0]) for n in names]
    upper_log = [math.log10(search_space[n][1]) for n in names]

    defaults = _import_settings()
    x0 = [
        math.log10(float(getattr(defaults, "theta_arrival_weight", 1.0))),
        math.log10(float(getattr(defaults, "F_state_weight", 1.0))),
        math.log10(float(getattr(defaults, "mhe_e_ua_weight", 1.0))),
        math.log10(float(getattr(defaults, "mhe_d_ua_bootstrap_weight", 1.0))),
        math.log10(float(getattr(defaults, "mhe_d_ua_arrival_weight", 1.0))),
    ]
    sigma0 = 0.6

    opts = {
        "bounds": [lower_log, upper_log],
        "maxiter": max_iter,
        "popsize": pop_size,
        "seed": 20260226,
        "verb_disp": 1,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_score = float("inf")
    best_weights = None
    eval_count = 0
    gen_count = 0
    print(
        f"Tuning start (CMA-ES): max_iter={max_iter}, pop_size={pop_size}, "
        f"horizons={num_horizons}, seed={eval_seed}"
    )
    while not es.stop():
        gen_count += 1
        pct_done = 100.0 * gen_count / max_iter
        print(f"\n=== Generation {gen_count}/{max_iter} ({pct_done:.1f}% complete) ===")
        xs = es.ask()
        fs = []
        for x in xs:
            candidate = {name: float(10 ** x[i]) for i, name in enumerate(names)}
            eval_count += 1
            print(f"\nEvaluation {eval_count}: weights={candidate}")
            score = float("inf")
            try:
                score, result = _evaluate(candidate, eval_seed, num_horizons)
                print(f"score={score:.6g}, skipped={result.get('mhe_skipped', 0)}")
            except Exception as err:
                score = float("inf")
                print(f"failed: {err}")
            fs.append(score)
            if score < best_score:
                best_score = score
                best_weights = dict(candidate)
                print(f"  New best score={best_score:.6g}")
        es.tell(xs, fs)
        es.disp()

    print("\nBest result")
    print(f"score={best_score:.6g}")
    print(f"weights={best_weights}")


if __name__ == "__main__":
    main()
