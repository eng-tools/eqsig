import numpy as np


def single_elastic_response(motion, step, period, xi):
    """
    Perform Duhamels integral to get the displacement.
    http://www.civil.utah.edu/~bartlett/CVEEN7330/Duhamel%27s_integral.pdf
    http://www1.aucegypt.edu/faculty/mharafa/MENG%20475/Forced%20Vibration.pdf
    :param motion: acceleration in m/s2
    :param step: the time step
    :param period: The period of SDOF oscillator
    :param xi: fraction of critical damping (e.g. 0.05)
    :return:
    """
    w_n = (2.0 * np.pi) / period
    w_d = w_n * np.sqrt(1 - xi ** 2)
    x_w_n = xi * w_n
    length = len(motion)

    time = step * np.arange(length + 1)
    disp = np.zeros(length)

    p = motion * step / w_d

    for i in range(length):
        dtn = time[:-i - 1]
        d_new = p[i] * np.exp(-x_w_n * dtn) * np.sin(w_d * dtn)

        disp[i:] += d_new

    return disp

# deprecated
def all_at_once_response_spectra(motion, step, periods, xi):
    """
    Quite slow
    Perform Duhamels integral to get the displacement.
    http://www.civil.utah.edu/~bartlett/CVEEN7330/Duhamel%27s_integral.pdf
    http://www1.aucegypt.edu/faculty/mharafa/MENG%20475/Forced%20Vibration.pdf
    :param motion:
    :param step:
    :param period:
    :param xi:
    :return:
    """

    w_ns = (2.0 * np.pi) / periods
    w_ds = w_ns * np.sqrt(1 - xi ** 2)
    x_w_ns = xi * w_ns
    length = len(motion)

    time = step * np.arange(length + 1)

    s_d = np.zeros(len(periods))

    for j in range(len(periods)):
        disp = np.zeros(length)
        w_d = w_ds[j]
        x_w_n = x_w_ns[j]
        p = motion * step / w_d

        for i in range(length):
            dtn = time[:-i - 1]
            d_new = p[i] * np.exp(-x_w_n * dtn) * np.sin(w_d * dtn)

            disp[i:] += d_new
        s_d[j] = max(disp)
    s_v = s_d * 2 * np.pi / periods
    s_a = s_d * (2 * np.pi / periods) ** 2

    return s_d, s_v, s_a


def slow_response_spectra(motion, step, periods, xis):
    """
    Perform Duhamels integral to get the displacement.
    http://www.civil.utah.edu/~bartlett/CVEEN7330/Duhamel%27s_integral.pdf
    http://www1.aucegypt.edu/faculty/mharafa/MENG%20475/Forced%20Vibration.pdf
    :param motion: acceleration in m/s2
    :param step: the time step
    :param period: The period of SDOF oscilator
    :param xi: fraction of critical damping (e.g. 0.05)
    :return:
    """
    points = len(periods)
    xi = xis[0]
    s_d = np.zeros(points)

    for i in range(points):
        s_d[i] = max(abs(single_elastic_response(motion, step, periods[i], xi)))

    s_v = s_d * 2 * np.pi / periods
    s_a = s_d * (2 * np.pi / periods) ** 2

    return s_d, s_v, s_a


def compute_a_and_b(xi, w, dt):
    """
    From the paper by Nigam and Jennings (1968), computes the two matrices.

    :param xi: critical damping ratio
    :param w: angular frequencies
    :param dt: time step
    :return: matrices A and B
    """

    # Reduce the terms since all is matrix multiplication.
    xi2 = xi * xi
    w2 = w ** 2
    one_ov_w2 = 1. / w2
    sqrt_b2 = np.sqrt(1. - xi2)
    w_sqrt_b2 = w * sqrt_b2

    exp_b = np.exp(-xi * w * dt)
    two_b_ov_w2 = (2 * xi ** 2 - 1) / (w ** 2 * dt)
    two_b_ov_w3 = 2 * xi / (w ** 3 * dt)

    sin_wsqrt = np.sin(w_sqrt_b2 * dt)
    cos_wsqrt = np.cos(w_sqrt_b2 * dt)

    # A matrix
    a_11 = exp_b * (xi / sqrt_b2 * sin_wsqrt + cos_wsqrt)  # Eq 2.7d(1)
    a_12 = exp_b / (w * sqrt_b2) * sin_wsqrt  # Eq 2.7d(2)
    a_21 = -w / sqrt_b2 * exp_b * sin_wsqrt    # Eq 2.7d(3)
    a_22 = exp_b * (cos_wsqrt - xi / sqrt_b2 * sin_wsqrt)  # Eq 2.7d(4)

    a = np.array([[a_11, a_12], [a_21, a_22]])

    # B matrix
    bsqrd_ov_w2_p_xi_ov_w = two_b_ov_w2 + xi / w
    sin_ov_wsqrt = sin_wsqrt / w_sqrt_b2
    xwcos = xi * w * cos_wsqrt
    wsqrtsin = w_sqrt_b2 * sin_wsqrt

    # Eq 2.7e
    b_11 = exp_b * (bsqrd_ov_w2_p_xi_ov_w * sin_ov_wsqrt + (two_b_ov_w3 + one_ov_w2) * cos_wsqrt) - two_b_ov_w3
    b_12 = -exp_b * (two_b_ov_w2 * sin_ov_wsqrt + two_b_ov_w3 * cos_wsqrt) - one_ov_w2 + two_b_ov_w3
    b_21 = exp_b * (bsqrd_ov_w2_p_xi_ov_w * (cos_wsqrt - xi / sqrt_b2 * sin_wsqrt)
                    - (two_b_ov_w3 + one_ov_w2) * (wsqrtsin + xwcos)) + one_ov_w2 / dt
    b_22 = -exp_b * (two_b_ov_w2 * (cos_wsqrt - xi / sqrt_b2 * sin_wsqrt) - two_b_ov_w3 * (wsqrtsin + xwcos)) - one_ov_w2 / dt

    b = np.array([[-b_11, b_12], [-b_21, -b_22]])  # Why are the signs reversed??

    return a, b


def nigam_and_jennings_response(acc, dt, periods, xi):
    """
    Implementation of the response spectrum calculation from Nigam and Jennings (1968).

    Ref: Nigam, N. C., Jennings, P. C. (1968) Digital calculation of response spectra from strong-motion earthquake
    records. National Science Foundation.

    :param acc: acceleration in m/s2
    :param periods: response periods of interest
    :param dt: time step of the acceleration time series
    :param xi: critical damping factor
    :return: response displacement, response velocity, response acceleration
    """

    acc = np.array(acc).astype(np.float)
    periods = np.array(periods).astype(np.float)
    w = 6.2831853 / periods
    dt = np.float(dt)
    xi = np.float(xi)

    # implement: delta_t should be less than period / 20

    a, b = compute_a_and_b(xi, w, dt)

    resp_u = np.zeros([len(w), len(acc)], dtype=np.float)
    resp_v = np.zeros([len(w), len(acc)], dtype=np.float)

    for i in range(len(acc) - 1):
        # x_i+1 = A cross (u, v) + B cross (acc_i, acc_i+1)  # Eq 2.7a
        resp_u[:, i + 1] = (a[0][0] * resp_u[:, i] + a[0][1] * resp_v[:, i] + b[0][0] * acc[i] + b[0][1] * acc[i + 1])
        resp_v[:, i + 1] = (a[1][0] * resp_u[:, i] + a[1][1] * resp_v[:, i] + b[1][0] * acc[i] + b[1][1] * acc[i + 1])

    w2 = w ** 2
    sdof_acc = -2 * xi * w[:, np.newaxis] * resp_v - w2[:, np.newaxis] * resp_u

    return resp_u, resp_v, sdof_acc


def absmax(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return abs(np.where(-amin > amax, amin, amax))


def pseudo_response_spectra(motion, dt, periods, xi):
    """
    Computes the maximum response displacement, pseudo velocity and pseudo acceleration.

    :param motion: array floats, acceleration in m/s2
    :param dt: float, the time step
    :param periods: array floats, The period of SDOF oscilator
    :param xi: float, fraction of critical damping (e.g. 0.05)
    :return: tuple floats, (spectral displacement, pseudo spectral velocity, pseudo spectral acceleration)
    """
    periods = np.array(periods)
    resp_u, resp_v, resp_a = nigam_and_jennings_response(motion, dt, periods, xi)
    sds = absmax(resp_u, axis=1)
    svs = 2 * np.pi / periods * sds
    sas = (2 * np.pi / periods) ** 2 * sds
    sas = np.where(periods < dt * 6, absmax(motion), sas)
    return sds, svs, sas


def response_series(motion, dt, periods, xi):
    """
    Computes the elastic response to the acceleration time series

    :param motion: array floats, acceleration in m/s2
    :param dt: float, the time step
    :param periods: array floats, The period of SDOF oscilator
    :param xi: float, fraction of critical damping (e.g. 0.05)
    :return: tuple of float arrays, (response displacements, response velocities, response accelerations)
    """
    return nigam_and_jennings_response(motion, dt, periods, xi)


def true_response_spectra(motion, dt, periods, xi):
    """
    Computes the actual maximum response values, not the pseudo values

    :param motion: array floats, acceleration in m/s2
    :param dt: float, the time step
    :param periods: array floats, The period of SDOF oscilator
    :param xi: float, fraction of critical damping (e.g. 0.05)
    :return: tuple floats, (spectral displacement, spectral velocity, spectral acceleration)
    """
    resp_u, resp_v, resp_a = nigam_and_jennings_response(motion, dt, periods, xi)
    sas = absmax(resp_a, axis=1)
    svs = absmax(resp_v, axis=1)
    sds = absmax(resp_u, axis=1)
    sas = np.where(periods < dt * 6, absmax(motion), sas)
    return sds, svs, sas


# def plot_response_spectra():
#     import matplotlib.pyplot as plt
#     step = 0.01
#     xis = [0.05]
#     periods = np.arange(1, 5, 0.5)
#     motion = np.sin(0.1 * np.arange(1000)) * 0.01
#     s_d, s_v, s_a = response_spectra(motion, step, periods, xis)
#
#     plt.plot(periods, s_a)
#     plt.show()
#
#
# def time_the_generation_of_response_spectra():
#     step = 0.01
#     xi = 0.05
#     periods = np.linspace(1, 5, 50)
#     periods = np.array([0.01])
#     motion = np.sin(0.1 * np.arange(10000)) * 0.01
#     # s_d, s_v, s_a = all_at_once_response_spectra(values, step, periods, xis)
#     s_d, s_v, s_a = response_spectra(motion, step, periods, xi)


if __name__ == '__main__':

    # time_response_spectra()


    import cProfile
    cProfile.run('time_the_generation_of_response_spectra()')





