# dummy_mhe_test_data.py

def make_dummy_io_data(num_steps=8):
    """
    Returns io_data_array in the same format as run_MPC.py:
      io_data_array[k] = [Cc(k), T(k), Fa0(k-1), mc(k-1)]
    with k=0 having None MVs.
    """
    io = []

    # k=0 measurement (no MV yet)
    Cc0 = 4.90
    T0  = 392.0
    io.append([Cc0, T0, None, None])

    # make some plausible trajectories
    Fa0 = 36.0   # kmol/h
    mc  = 450.0  # kmol/h

    Cc = Cc0
    T  = T0

    for k in range(1, num_steps):
        # pretend the MV changes slowly (this MV value is u_{k-1})
        Fa0 += 0.2 * ((-1) ** k)         # wiggle a bit
        mc  += 2.0 * ((-1) ** (k + 1))   # wiggle a bit

        # pretend measured CVs respond smoothly
        # (these are y_k at time k)
        Cc += 0.06 - 0.01 * (k % 3)      # drifting toward ~5.18-ish
        T  += 1.5 - 0.2 * (k % 4)        # drifting upward

        io.append([round(Cc, 4), round(T, 3), round(Fa0, 3), round(mc, 3)])

    return io


if __name__ == "__main__":
    io_data_array = make_dummy_io_data(num_steps=8)
    for k, row in enumerate(io_data_array):
        print(k, row)
