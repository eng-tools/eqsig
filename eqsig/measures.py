import numpy as np
import scipy.integrate
from eqsig import duhamels
from eqsig.exceptions import deprecation


def calc_significant_duration(motion, dt, start=0.05, end=0.95):
    """
    Deprecated. Use calc_sig_dur_vals
    Parameters
    ----------
    motion
    dt
    start
    end

    Returns
    -------

    """
    deprecation("Use calc_sig_dur_vals()")
    return calc_sig_dur_vals(motion, dt, start=start, end=end)


def calc_sig_dur_vals(motion, dt, start=0.05, end=0.95):
    """
    Computes the significant duration using cumulative acceleration according to Trifunac and Brady (1975).

    Parameters
    ----------
    motion: array-like
        acceleration time series
    dt: float
        time step
    start: float, default=0.05
        threshold to start the duration
    end: float, default=0.95
        threshold to end the duration

    Returns
    -------
    tuple (start_time, end_time)
    """

    acc2 = motion ** 2
    cum_acc2 = np.cumsum(acc2)
    ind2 = np.where((cum_acc2 > start * cum_acc2[-1]) & (cum_acc2 < end * cum_acc2[-1]))
    start_time = ind2[0][0] * dt
    end_time = ind2[0][-1] * dt

    return start_time, end_time


def calc_sig_dur(asig, start=0.05, end=0.95, im=None):
    """
    Computes the significant duration using cumulative acceleration according to Trifunac and Brady (1975).

    Parameters
    ----------
    motion: array-like
        acceleration time series
    dt: float
        time step
    start: float, default=0.05
        threshold to start the duration
    end: float, default=0.95
        threshold to end the duration
    im: function or None
        A function that calculates a cumulative intensity measure, default Arias Intensity

    Returns
    -------
    tuple (start_time, end_time)
    """
    if im is None:
        im_vals = calc_arias_intensity(asig)
    else:
        im_vals = im(asig)
    ind2 = np.where((im_vals > start * im_vals[-1]) & (im_vals < end * im_vals[-1]))
    start_time = ind2[0][0] * asig.dt
    end_time = ind2[0][-1] * asig.dt

    return start_time, end_time


def calculate_peak(motion):
    """Calculates the peak absolute response"""
    deprecation("Use calc_peak instead of calculate_peak")
    return max(abs(min(motion)), max(motion))


def calc_peak(motion):
    """Calculates the peak absolute response"""
    return max(abs(min(motion)), max(motion))


def calc_sir(acc_sig):
    """
    Calculates the shaking intensity rate

    ref:
    Dashti, S., Bray, J. D., Pestana, J. S., Riemer, M., and Wilson, D., 2010. Centrifuge testing to
    evaluate and mitigate liquefaction-induced building settlement mechanisms,
    ASCE Journal of Geotechnical and Geoenv. Eng. 136, 918-929

    Parameters
    ----------
    acc_sig: eqsig.AccSignal
        acceleration signal

    Returns
    -------
    float
    """
    ai_total = acc_sig.arias_intensity
    t5, t75 = calc_significant_duration(acc_sig.values, acc_sig.dt)
    return 0.7 * ai_total / (t75 - t5)


def _raw_calc_arias_intensity(acc, dt):
    acc2 = acc ** 2
    return np.pi / (2 * 9.81) * scipy.integrate.cumtrapz(acc2, dx=dt, initial=0)


def calc_arias_intensity(acc_sig):
    """
    Calculates the Arias Intensity

    Parameters
    ----------
    acc_sig: eqsig.AccSignal

    Returns
    -------
    array_like
        A time series of the build up of Arias Intensity
    """

    return _raw_calc_arias_intensity(acc_sig.values, acc_sig.dt)


def calc_cav(acc_sig):
    """
    Calculates the Cumulative Absolute velocity

    ref:
    Electrical Power Research Institute. Standardization of the Cumulative
    Absolute Velocity. 1991. EPRI TR-100082-1'2, Palo Alto, California.
    """
    abs_acc = np.abs(acc_sig.values)
    return scipy.integrate.cumtrapz(abs_acc, dx=acc_sig.dt, initial=0)


def calc_cav_dp(asig):
    """
    Calculates standardized cumulative absolute velocity

    ref:
    Campbell KW, Bozorgnia Y. Predictive equations for the horizontal component of standardized
    cumulative absolute velocity as adapted for use in the shutdown of U.S. nuclear power plants.
    Nucl Eng Des 2011;241:2558-69.

    :param asig:
    :return:
    """
    start = 0
    pga_max = 0
    cav_dp = 0
    points_per_sec = (int(1 / asig.dt))
    total_seconds = int(asig.time[-1])
    cav_dp_1_series = []
    acc_in_g = asig.values / 9.81

    for i in range(0, total_seconds):

        end = start + points_per_sec

        interval_total_time = (start * asig.dt) + 1
        interval_time = np.arange(start * asig.dt, interval_total_time, asig.dt)

        acc_interval = []
        for j in range(start, end + 1):
            acc_interval.append(acc_in_g[j])

        acc_interval = np.array(acc_interval)
        abs_acc_interval = abs(acc_interval)

        x_lower = start * asig.dt  # the lower limit of x
        x_upper = end * asig.dt  # the upper limit of x
        x_int = interval_time[np.where((x_lower <= interval_time) * (interval_time <= x_upper))]
        y_int = np.abs(np.array(abs_acc_interval)[np.where((x_lower <= interval_time) * (interval_time <= x_upper))])
        int_acc = scipy.integrate.trapz(y_int, x_int)

        # calculation of pga (g)
        pga = (max(abs_acc_interval))
        if pga > pga_max:
            pga_max = pga

        if (pga - 0.025) < 0:
            h = 0
        elif (pga - 0.025) >= 0:
            h = 1
        else:
            raise ValueError("cannot evaluate pga: {0}".format(pga))

        cav_dp = cav_dp + (h * int_acc)
        cav_dp_1_series.append(cav_dp)
        start = end
    t1s = np.arange(total_seconds)
    cav_dp_time_series = np.interp(asig.time, t1s, cav_dp_1_series)
    return cav_dp_time_series


def calc_isv(acc_sig):
    """
    Calculates the integral of the squared velocity

    See Kokusho (2013)
    :return:
    """
    return scipy.integrate.cumtrapz(acc_sig.velocity ** 2, dx=acc_sig.dt, initial=0)


def cumulative_response_spectra(acc_signal, fun_name, periods=None, xi=None):

    if periods is None:
        periods = acc_signal.response_times
    else:
        periods = np.array(periods)
    if xi is None:
        xi = 0.05
    resp_u, resp_v, resp_a = duhamels.response_series(acc_signal.values, acc_signal.dt, periods, xi)
    if fun_name == "arias_intensity":
        rs = _raw_calc_arias_intensity(resp_a, acc_signal.dt)
    else:
        raise ValueError
    return rs


def calc_max_velocity_period(asig):
    from eqsig import AccSignal
    periods = np.logspace(-1, 0.3, 100)
    new_sig = AccSignal(asig.values, asig.dt)
    new_sig.generate_response_spectrum(response_times=periods, xi=0.15)

    max_index = np.argmax(new_sig.s_v)
    max_period = periods[max_index]
    return max_period


def max_acceleration_period(asig):
    from eqsig import AccSignal
    periods = np.logspace(-1, 1, 100)
    new_sig = AccSignal(asig.values, asig.dt)
    new_sig.generate_response_spectrum(response_times=periods, xi=0)

    max_index = np.argmax(new_sig.s_a)
    max_period = periods[max_index]
    return max_period


def max_fa_period(asig):
    """Calculates the period corresponding to the maximum value in the Fourier amplitude spectrum"""
    max_index = np.argmax(asig.fa_spectrum)
    max_period = 1. / asig.fa_frequencies[max_index]
    return max_period


def calc_bandwidth_f_min(asig, ratio=0.707):
    fas1_smooth = asig.smooth_fa_spectrum
    max_fas1 = max(fas1_smooth)
    lim_fas = max_fas1 * ratio
    ind2 = np.where(fas1_smooth > lim_fas)
    min_freq = asig.smooth_fa_frequencies[ind2[0][0]]
    return min_freq


def calc_bandwidth_f_max(asig, ratio=0.707):
    fas1_smooth = asig.smooth_fa_spectrum
    max_fas1 = max(fas1_smooth)
    lim_fas = max_fas1 * ratio
    ind2 = np.where(fas1_smooth > lim_fas)
    max_freq = asig.smooth_fa_frequencies[ind2[0][-1]]
    return max_freq


def calc_bracketed_duration(asig, threshold):
    """
    Calculates the Bracketed duration based on some threshold

    Parameters
    ----------
    asig: eqsig.AccSignal
        Acceleration time series object
    threshold: float
        acceleration threshold to calculation duration start and end
    Returns
    -------
    float
    """
    abs_motion = abs(asig.values)

    time = np.arange(asig.npts) * asig.dt
    # Bracketed duration
    ind01 = np.where(abs_motion / 9.8 > threshold)
    time2 = time[ind01]
    try:
        t_bracket = time2[-1] - time2[0]
        # rms acceleration in m/s/s
        a_rms01 = np.sqrt(1 / asig.t_b01 * np.trapz((asig.values[ind01[0][0]:ind01[0][-1]]) ** 2, dx=asig.dt))
    except IndexError:
        t_bracket = -1.
        a_rms01 = -1.
    return t_bracket


def calc_integral_of_abs_velocity(asig):
    abs_vel = abs(asig.velocity)
    vel_int = np.cumsum(abs_vel * asig.dt)
    return vel_int


def calc_cumulative_abs_displacement(asig):
    return calc_integral_of_abs_velocity(asig)


def calc_integral_of_abs_acceleration(asig):
    abs_acc = abs(asig.values)
    acc_int = np.cumsum(abs_acc * asig.dt)
    return acc_int
