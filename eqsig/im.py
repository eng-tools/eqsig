import numpy as np

from eqsig import sdof, functions
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


def calc_sig_dur_vals(motion, dt, start=0.05, end=0.95, se=False):
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
    se: bool, default=False
        If true then return the start and end times

    Returns
    -------
    tuple (start_time, end_time)
    """

    cum_acc2 = np.cumsum(motion ** 2)
    ind2 = np.where((cum_acc2 > start * cum_acc2[-1]) & (cum_acc2 < end * cum_acc2[-1]))
    start_time = ind2[0][0] * dt
    end_time = ind2[0][-1] * dt

    if se:
        return start_time, end_time
    return end_time - start_time


def calc_sig_dur(asig, start=0.05, end=0.95, im=None, se=False):
    """
    Computes the significant duration according to Trifunac and Brady (1975).

    Parameters
    ----------
    asig: eqsig.AccSignal
        acceleration time series object
    dt: float
        time step
    start: float, default=0.05
        threshold to start the duration
    end: float, default=0.95
        threshold to end the duration
    im: function or None (default=None)
        A function that calculates a cumulative intensity measure, if =None, then use eqsig.im.calc_arias_intensity
    se: bool, default=False
        If true then return the start and end times

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
    if se:
        return start_time, end_time
    return end_time - start_time


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
    from scipy.integrate import cumtrapz
    return np.pi / (2 * 9.81) * cumtrapz(acc ** 2, dx=dt, initial=0)


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
    from scipy.integrate import cumtrapz
    abs_acc = np.abs(acc_sig.values)
    return cumtrapz(abs_acc, dx=acc_sig.dt, initial=0)


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
    from scipy.integrate import trapz
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
        int_acc = trapz(y_int, x_int)

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
    from scipy.integrate import cumtrapz
    return cumtrapz(acc_sig.velocity ** 2, dx=acc_sig.dt, initial=0)


def cumulative_response_spectra(acc_signal, fun_name, periods=None, xi=None):

    if periods is None:
        periods = acc_signal.response_times
    else:
        periods = np.array(periods)
    if xi is None:
        xi = 0.05
    resp_u, resp_v, resp_a = sdof.response_series(acc_signal.values, acc_signal.dt, periods, xi)
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


def calc_bandwidth_freqs(asig, ratio=0.707):
    """
    Lower and upper frequencies of smooth Fourier spectrum bandwidth

    Parameters
    ----------
    asig: eqsig.AccSignal
        Acceleration time series object
    ratio:
        ratio of maximum value where bandwidth should be computed

    Returns
    -------
    tuple:
        (lower, upper)
    """
    fas1_smooth = asig.smooth_fa_spectrum
    max_fas1 = max(fas1_smooth)
    lim_fas = max_fas1 * ratio
    ind2 = np.where(fas1_smooth > lim_fas)
    min_freq = asig.smooth_fa_frequencies[ind2[0][0]]
    max_freq = asig.smooth_fa_frequencies[ind2[0][-1]]
    return min_freq, max_freq


def calc_bandwidth_f_min(asig, ratio=0.707):
    """
    Lower frequency of smooth Fourier spectrum bandwidth

    Parameters
    ----------
    asig: eqsig.AccSignal
        Acceleration time series object
    ratio: float
        ratio of maximum value where bandwidth should be computed

    Returns
    -------
    float
    """
    fas1_smooth = asig.smooth_fa_spectrum
    max_fas1 = max(fas1_smooth)
    lim_fas = max_fas1 * ratio
    ind2 = np.where(fas1_smooth > lim_fas)
    min_freq = asig.smooth_fa_frequencies[ind2[0][0]]
    return min_freq


def calc_bandwidth_f_max(asig, ratio=0.707):
    """
        Upper frequency of smooth Fourier spectrum bandwidth

        Parameters
        ----------
        asig: eqsig.AccSignal
            Acceleration time series object
        ratio: float
            ratio of maximum value where bandwidth should be computed

        Returns
        -------
        float
        """
    fas1_smooth = asig.smooth_fa_spectrum
    max_fas1 = max(fas1_smooth)
    lim_fas = max_fas1 * ratio
    ind2 = np.where(fas1_smooth > lim_fas)
    max_freq = asig.smooth_fa_frequencies[ind2[0][-1]]
    return max_freq


def calc_bracketed_duration(asig, threshold):
    """DEPRECATED: Use calc_brac_dur"""
    deprecation("Use calc_brac_dur")
    return calc_brac_dur(asig, threshold)


def calc_brac_dur(asig, threshold, se=False):
    """
    Calculates the Bracketed duration based on some threshold

    Parameters
    ----------
    asig: eqsig.AccSignal
        Acceleration time series object
    threshold: float
        acceleration threshold to calculation duration start and end
    se: bool, default=False
        If true then return the start and end times
    Returns
    -------
    float
    """
    abs_motion = abs(asig.values)

    time = np.arange(asig.npts) * asig.dt
    # Bracketed duration
    ind01 = np.where(abs_motion > threshold)
    time2 = time[ind01]
    try:
        if se:
            return time2[0], time2[-1]
        return time2[-1] - time2[0]
    except IndexError:
        if se:
            return None, None
        return 0


def calc_acc_rms(asig, threshold):
    """Root mean squared acceleration"""
    abs_motion = abs(asig.values)
    # Bracketed duration
    ind01 = np.where(abs_motion > threshold)
    try:
        # rms acceleration in m/s/s
        a_rms01 = np.sqrt(1 / asig.t_b01 * np.trapz((asig.values[ind01[0][0]:ind01[0][-1]]) ** 2, dx=asig.dt))
    except IndexError:
        a_rms01 = 0
    return a_rms01


def calc_a_rms(asig, threshold):
    """DEPRECATED"""
    raise ValueError('calc_a_rms has been removed, use calc_acc_rms note that threshold changed to m/s2')


def calc_integral_of_abs_velocity(asig):
    """Integral of absolute velocity - identical to cumulative absolute displacement"""
    abs_vel = abs(asig.velocity)
    vel_int = np.cumsum(abs_vel * asig.dt)
    return vel_int


def calc_cumulative_abs_displacement(asig):
    """Cumulative absolute displacement - identical to integral of absolute velocity"""
    return calc_integral_of_abs_velocity(asig)


def calc_integral_of_abs_acceleration(asig):
    """Integral of absolute acceleration"""
    abs_acc = abs(asig.values)
    acc_int = np.cumsum(abs_acc * asig.dt)
    return acc_int


def calc_n_cyc_array_w_power_law(values, a_ref, b, cut_off=0.01):
    """
    Calculates the equivalent number of uniform amplitude cycles using a power law

    Parameters
    ----------
    values: array_like
        Time series of values
    a_ref: float
        Reference uniform amplitude
    b: float or array_like
        Power law factor
    cut_off: float
        Low amplitude cutoff value

    Returns
    -------
    array_like
    """
    from scipy.interpolate import interp1d
    peak_indices = functions.get_switched_peak_array_indices(values)
    csr_peaks = np.abs(np.take(values, peak_indices))
    csr_peaks = np.where(csr_peaks < cut_off * np.max(abs(values)), 1.0e-14, csr_peaks)
    n_ref = 1
    perc = 0.5 / (n_ref * (a_ref / csr_peaks)[:, np.newaxis] ** (1 / b))
    n_eq = np.cumsum(perc, axis=0)
    n_eq = np.insert(n_eq, 0, 0, axis=0)
    peak_indices = np.insert(peak_indices, 0, 0, axis=0)
    n_eq = np.insert(n_eq, len(n_eq)-1, n_eq[-1], axis=0)
    peak_indices = np.insert(peak_indices, len(n_eq)-1, len(values), axis=0)

    f = interp1d(peak_indices, n_eq, kind='previous', axis=0)
    n_series = f(np.arange(len(values)))
    return n_series


def calc_cyc_amp_array_w_power_law(values, n_cyc, b):
    """
    Calculate the equivalent uniform loading for a given number of cycles and power coefficient (b)

    :param values: array_like
        Time series of values
    :param n_cyc:
    :param b:
    :return:
    """
    a1_peak_inds_end = functions.get_switched_peak_array_indices(values)
    a1_csr_peaks_end = np.abs(np.take(values, a1_peak_inds_end))
    csr_peaks_s1 = np.zeros_like(values)
    np.put(csr_peaks_s1, a1_peak_inds_end, a1_csr_peaks_end)
    csr_n15_series1 = np.cumsum((np.abs(csr_peaks_s1)[:, np.newaxis] ** (1. / b)) / 2 / n_cyc, axis=0) ** b
    if not hasattr(b, '__len__'):
        return np.reshape(csr_n15_series1, len(values))
    return csr_n15_series1


def calc_cyc_amp_gm_arrays_w_power_law(values0, values1, n_cyc, b):
    """
    Calculate the geometric mean equivalent uniform amplitude for a given number of cycles and power coefficient (b)

    :param values0: array_like
        Time series of values
    :param values1: array_like
        Time series of values in orthogonal direction to values0
    :param n_cycles:
    :param b:
    :return:
    """
    csr_n_series0 = calc_cyc_amp_array_w_power_law(values0, n_cyc=n_cyc, b=b)
    csr_n_series1 = calc_cyc_amp_array_w_power_law(values1, n_cyc=n_cyc, b=b)
    csr_n_series = np.sqrt(csr_n_series0 * csr_n_series1)
    return csr_n_series


def calc_cyc_amp_combined_arrays_w_power_law(values0, values1, n_cyc, b):
    """
    Computes the equivalent cyclic amplitude.

    For a set number of cycles using a power law and assuming both components act equally

    Parameters
    ----------
    values0: array_like
        Time series of values
    values1: array_like
        Time series of values in orthogonal direction to values0
    n_cyc: int or float
        Number of cycles
    b: float
        Power law factor

    Returns
    -------
    array_like
    """
    peak_inds_a0 = functions.get_switched_peak_array_indices(values0)
    csr_peaks_a0 = np.abs(np.take(values0, peak_inds_a0))

    peak_inds_a1 = functions.get_switched_peak_array_indices(values1)
    csr_peaks_a1 = np.abs(np.take(values1, peak_inds_a1))

    csr_peaks_s0 = np.zeros_like(values0)
    np.put(csr_peaks_s0, peak_inds_a0, csr_peaks_a0)
    csr_peaks_s1 = np.zeros_like(values1)
    np.put(csr_peaks_s1, peak_inds_a1, csr_peaks_a1)
    csr_n15_series = np.cumsum((np.abs(csr_peaks_s0) ** (1. / b) + np.abs(csr_peaks_s1) ** (1. / b)) / 2 / n_cyc) ** b

    return csr_n15_series


def calc_unit_kinetic_energy(acc_signal):
    """
    Calculates the cumulative absolute change in kinetic energy for a unit volume of soil with unit mass

    Parameters
    ----------
    acc_signal: eqsig.AccSignal

    Returns
    -------

    """
    kin_energy = 0.5 * acc_signal.velocity * np.abs(acc_signal.velocity)
    delta_energy = np.diff(kin_energy)
    delta_energy = np.insert(delta_energy, 0, kin_energy[0])
    cum_delta_energy = np.cumsum(abs(delta_energy))
    return cum_delta_energy
