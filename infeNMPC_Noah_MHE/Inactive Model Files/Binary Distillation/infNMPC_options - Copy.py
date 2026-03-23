class Options:
    """
    A class to store and manage simulation settings for infinite and finite horizon NMPC.

    Attributes:
    -----------
    num_horizons : int
        Number of simulation steps to run.
    finite_horizon : float
        Time length of the finite horizon (in time units, e.g., hours or seconds).
    ncp_finite : int
        Number of collocation points per finite-horizon finite element.
    sampling_time : float
        Time interval between successive measurements or control updates.
    infinite_horizon : bool
        If True, infinite horizon MPC will be used. If False, finite horizon MPC is used.
    nfe_infinite : int
        Number of finite elements for the infinite horizon approximation.
    ncp_infinite : int
        Number of collocation points per infinite-horizon finite element.
    tee_flag : bool
        If True, solver output will be printed to the console.
    finite_terminal_con : bool
        If True, a terminal state constraint is applied in the finite horizon case.
    terminal_cost_riemann : bool
        If True, a Riemann sum is used to approximate the terminal cost.
    """

    def __init__(self):
        # Simulation control
        self.MHE_window = 7
        self.num_horizons = 25
        self.nfe_finite = 2
        self.ncp_finite = 3
        self.sampling_time = 10

        # Infinite horizon settings
        self.infinite_horizon = True
        self.nfe_infinite = 3
        self.ncp_infinite = 3

        # Solver and model options
        self.tee_flag = False
        self.endpoint_constraints = True
        self.custom_objective = False
        self.initialize_with_initial_data = False
        self.terminal_cost_riemann = False
        self.remove_collocation = True
        self.initialization_assist = False
        self.initialization_assist_sampling_time_start = 10

        self.input_suppression = True
        self.input_suppression_factor = 0.5e0 * 1.0E5
        #list(m.CV_index) + list(m.MV_index) is order
        self.stage_cost_weights = [5e1 * 1.0e4, 1e3 * 1.0, 0e-1 * 1.0e5, 0e-1 * 1.0e5]
        self.gamma = 0.05
        self.beta = 1.2
        # MHE arrival cost weights
        self.mhe_arrival_default_lambda = 0.0025
        #self.mhe_arrival_weights = {}
        #for i in range(1, 17):
        #   self.mhe_arrival_weights[f"x[{i}]"] = 4/((.999*(i/42))**2)
        #for i in range(17, 43):
        #   self.mhe_arrival_weights[f"x[{i}]"] = 4/((.36+((.98-.36)/(42-21-1))*(i-21-1))**2)
        #for i in range(2, 22):
        #    self.mhe_arrival_weights[f"M[{i}]"] = 4/((3340+((3772-3340)/(21-2))*(i-2))**2)
        #for i in range(22, 43):
        #    self.mhe_arrival_weights[f"M[{i}]"] = 4/((2890+((4650-2890)/(42-21-1))*(i-21-1))**2)
        #for i in range(1, 2):
        #    self.mhe_arrival_weights[f"M[{i}]"] = 4/(2*105500)
        self.mhe_state_error_default = 1e-3 #x
        self.mhe_state_error = {}
        for i in range(1, 43):
           self.mhe_state_error[f"x[{i}]"] = .002
        for i in range(2, 43):
            self.mhe_state_error[f"M[{i}]"] = 1
        for i in range(1, 2):
            self.mhe_state_error[f"M[{i}]"] = 5
        self.mhe_output_error_default = 1e-3 #y
        self.mhe_output_error = {}
        for i in range(1, 43):
            self.mhe_output_error[f"T[{i}]"] = 352.92248738561443*0.01
        self.mhe_min_window = 3
        # Display/Data Output options   




        # Display/Data Output options
        self.live_plot = False
        self.plot_end = False
        self.save_data = True
        self.save_figure = True

        # Disturbance options
        self.disturb_flag = False
        self.disturb_distribution = 'normal'
        self.disturb_seeded = True


def _import_settings():
    """
    Create and return an Options object containing all NMPC settings.

    Returns:
    --------
    Options
        An instance of the Options class with initialized values.
    """
    return Options()

