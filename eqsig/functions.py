import numpy as np
import scipy

import eqsig
from eqsig import exceptions


def time_series_from_motion(motion, dt):
    npts = len(motion)
    return np.linspace(0, dt * (npts + 1), npts)


def determine_indices_of_peaks_for_cleaned(values):
    """
    Determines the position of values that form a local peak in a signal.

    :param values:
    :return:
    """
    diff = np.ediff1d(values, to_begin=0)
    # if negative then direction has switched
    # direction_switch = np.insert(direction_switch, 0, 0)
    peak_indices = np.where(diff[1:] * diff[:-1] < 0)[0]
    peak_indices = np.insert(peak_indices, 0, 0)  # Include first and last value
    peak_indices = np.insert(peak_indices, len(peak_indices), len(values) - 1)

    return peak_indices


# def determine_indices_of_zero_crossings_for_cleaned(values):
#     """
#     Determines the position of values that are equal or have just passed through zero.
#
#     :param values:
#     :return:
#     """
#     diff = np.diff(values)
#     # if negative then direction has switched
#     direction_switch = diff[1:] * diff[:-1]
#     direction_switch = np.insert(direction_switch, 0, 0)
#     peaks = np.where(direction_switch < 0)
#     peak_indices = peaks[0]
#     peak_indices = np.insert(peak_indices, 0, 0)  # Include first and last value
#     peak_indices = np.insert(peak_indices, len(peak_indices), len(values) - 1)
#
#     return peak_indices


def determine_peak_only_series_4_cleaned_data(values):
    """
    Determines the

    Note: array must not contain adjacent repeated values

    :param values:
    :return:
    """
    peak_indices = determine_indices_of_peaks_for_cleaned(values)
    peak_values = np.take(values, peak_indices)
    signs = np.where(np.mod(np.arange(len(peak_values)), 2), -1, 1)
    delta_peaks = np.where(-signs * peak_values < 0, -np.abs(peak_values), np.abs(peak_values))
    delta_peaks_series = np.zeros_like(values)
    np.put(delta_peaks_series, peak_indices, delta_peaks)

    return delta_peaks_series


def determine_peak_only_delta_series_4_cleaned_data(values):
    """
    Determines the

    Note: array must not contain adjacent repeated values

    :param values:
    :return:
    """
    peak_indices = determine_indices_of_peaks_for_cleaned(values)
    peak_values = np.take(values, peak_indices)
    delta_peaks = np.diff(peak_values)
    delta_peaks = np.insert(delta_peaks, 0, 0)
    delta_peaks_series = np.zeros_like(values)
    assert len(delta_peaks) == len(peak_indices)
    np.put(delta_peaks_series, peak_indices, delta_peaks)

    return delta_peaks_series


def clean_out_non_changing(values):
    """
    Takes an array removes all values that are the same as the previous value.

    :param values: array of floats
    :return: cleaned array, indices of clean values in original array
    """
    # diff_values = np.diff(values)
    # diff_values = np.insert(diff_values, 0, values[0])
    diff_values = np.ediff1d(values, to_begin=values[0])
    non_zero_indices = np.where(diff_values != 0)[0]
    non_zero_indices = np.insert(non_zero_indices, 0, 0)

    cleaned_values = np.take(values, non_zero_indices)
    return cleaned_values, non_zero_indices


def get_peak_array_indices(values, ptype='all'):
    """
    Find the indices for all of the local maxima and minima

    Parameters
    ----------
    :param values: array_like, array of values
    :return:

    Examples
    --------
    >>> values = np.array([0, 2, 1, 2, -1, 1, 1, 0.3, -1, 0.2, 1, 0.2])
    np.array([0, 2, 1, 2, -1, 1, 1, 0.3, -1, 0.2, 1, 0.2])
    >>> get_peak_array_indices(values)
    np.array([0, 1, 2, 3, 4, 5, 8, 10, 11])
    """
    # enforce array type
    values = np.array(values, dtype=float)
    # remove all non-changing values
    cleaned_values, non_zero_indices = clean_out_non_changing(values)
    # cleaned_values *= np.sign(cleaned_values[1])  # ensure first value is increasing
    peak_cleaned_indices = determine_indices_of_peaks_for_cleaned(cleaned_values)
    peak_full_indices = np.take(non_zero_indices, peak_cleaned_indices)
    if ptype == 'min':
        if values[1] - values[0] <= 0:
            return peak_full_indices[1::2]
        else:
            return peak_full_indices[::2]
    elif ptype == 'max':
        if values[1] - values[0] > 0:
            return peak_full_indices[1::2]
        else:
            return peak_full_indices[::2]
    return peak_full_indices


def get_n_cyc_array(values, opt='all', start='origin'):
    """
    Given an array, create an array of the same length that numbers the peaks
    Parameters
    ----------
    values
    opt

    Returns
    -------

    """
    if opt == 'all':
        indys = get_peak_array_indices(values)
    elif opt == 'switched':
        indys = get_switched_peak_array_indices(values)
    else:
        raise ValueError('opt must be either "all" or "switched"')
    # each indy corresponds to half a cycle
    if start == 'origin':
        svalue = -0.25
    elif start == 'peak':
        svalue = 0.0
    else:
        raise ValueError('start must be either "origin" or "peak"')
    if indys[0] != 0:
        indys = np.insert(indys, 0, 0)
    n_cycs = 0.5 * np.arange(len(indys))
    n_cycs[1:] += svalue
    return np.interp(np.arange(len(values)), indys, n_cycs)


def get_peak_indices(asig):
    return get_peak_array_indices(asig.values)


def get_zero_crossings_array_indices(values, keep_adj_zeros=False):
    """
    Find the indices for values that are equal to zero or just passed through zero

    Parameters
    ----------
    values: array_like
        array of values
    keep_adj_zeros: bool,
        if false then if adjacent zeros are found, only first is included
    :return:

    Examples
    --------
    >>> values = np.array([0, 2, 1, 2, -1, 1, 0, 0, 1, 0.3, 0, -1, 0.2, 1, 0.2])
    np.array([0, 2, 1, 2, -1, 1, 0, 0, 1, 0.3, 0, -1, 0.2, 1, 0.2])
    >>> get_zero_crossings_array_indices(values, keep_adj_zeros=False)
    np.array([0, 4, 5, 6, 10, 12])
    """
    # enforce array type
    values = np.array(values, dtype=float)
    # get all zero values
    zero_indices = np.where(values == 0)[0]
    if not keep_adj_zeros and len(zero_indices) > 1:
        diff_is = np.ediff1d(zero_indices, to_begin=10)
        no_adj_is = np.where(diff_is > 1)[0]
        zero_indices = np.take(zero_indices, no_adj_is)
    # if negative then sign has switched
    sign_switch = values[1:] * values[:-1]
    sign_switch = np.insert(sign_switch, 0, values[0])
    through_zero_indices = np.where(sign_switch < 0)[0]
    all_zc_indices = np.concatenate((zero_indices, through_zero_indices))
    all_zc_indices.sort()
    if all_zc_indices[0] != 0:
        all_zc_indices = np.insert(all_zc_indices, 0, 0)  # slow
    return all_zc_indices


def get_zero_crossings_indices(asig):
    return get_zero_crossings_array_indices(asig.values)


def determine_peaks_only_delta_series(values):
    """
    Creates an array with the changes between peak values and zeros for non-peak values.

    Parameters
    ----------
    :param values: array_like, array of values
    :return:

    Examples
    --------
    >>> values = np.array([0, 2, 1, 2, 0, 1, 0, -1, 0, 1, 0])
    np.array([0, 2, 1, 2, 0.3, 1, 0.3, -1, 0.4, 1, 0])
    >>> determine_peaks_only_delta_series(values)
    array([0,  2, -1,  1,  0,  -1,  0,  -2,  0,  2,  0])
    """
    # enforce array type
    values = np.array(values)
    # rebase to zero as first value
    values -= values[0]
    # remove all non-changing values
    cleaned_values, non_zero_indices = clean_out_non_changing(values)
    cleaned_values *= np.sign(cleaned_values[1])  # ensure first value is increasing
    # compute delta peaks for cleaned data
    cleaned_delta_peak_series = determine_peak_only_delta_series_4_cleaned_data(cleaned_values)
    # re-index data to uncleaned array
    delta_peaks_series = np.zeros_like(values)
    np.put(delta_peaks_series, non_zero_indices, cleaned_delta_peak_series)
    return delta_peaks_series


def determine_pseudo_cyclic_peak_only_series(values):
    """
    Creates an array with only peak values assuming an alternative sign and zeros for non-peak values.

    Parameters
    ----------
    :param values: array_like, array of values
    :return:

    Examples
    --------
    >>> values = np.array([0, 2, 1, 2, 0, 1, 0, -1, 0, 1, 0])
    np.array([0, 2, 1, 2, 0.3, 1, 0.3, -1, 0.4, 1, 0])
    >>> determine_pseudo_cyclic_peak_only_series(values)
    array([0,  2, -1,  2,  0,  1,  0,  1,  0,  1,  0])
    """
    # enforce array type
    values = np.array(values)
    # rebase to zero as first value
    values -= values[0]
    # remove all non-changing values
    cleaned_values, non_zero_indices = clean_out_non_changing(values)
    cleaned_values *= np.sign(cleaned_values[1])  # ensure first value is increasing
    # compute delta peaks for cleaned data
    cleaned_delta_peak_series = determine_peak_only_series_4_cleaned_data(cleaned_values)
    # re-index data to uncleaned array
    delta_peaks_series = np.zeros_like(values)
    np.put(delta_peaks_series, non_zero_indices, cleaned_delta_peak_series)
    return delta_peaks_series


def fas2signal(fas, dt, stype="signal"):
    """
    Convert a fourier spectrum to time series signal

    :param fas: positive part only
    :param dt: time step of time series
    :return:
    """
    from eqsig.single import Signal, AccSignal
    n = 2 * len(fas)
    a = np.zeros(2 * len(fas), dtype=complex)
    a[1:n // 2] = fas[1:]
    a[n // 2 + 1:] = np.flip(np.conj(fas[1:]), axis=0)
    a /= dt
    s = np.fft.ifft(a)
    npts = int(2 ** (np.log(n) / np.log(2)))
    s = s[:npts]
    if stype == 'signal':
        return Signal(s, dt)
    else:
        return AccSignal(s, dt)


def generate_fa_spectrum(sig, n_pad=True):
    """
    Produces the Fourier amplitude spectrum

    Parameters
    ----------
    sig: eqsig.Signal

    Returns
    -------
    fa_spectrum: complex array_like
        Complex values of the spectrum
    fa_frequencies: array_like
        Frequencies of the spectrum
    """
    npts = sig.npts
    if n_pad:
        n_factor = 2 ** int(np.ceil(np.log2(npts)))
        fa = scipy.fft(sig.values, n=n_factor)
        points = int(n_factor / 2)
        assert len(fa) == n_factor
    else:
        fa = scipy.fft(sig.values)
        points = int(sig.npts / 2)
    fa_spectrum = fa[range(points)] * sig.dt
    fa_frequencies = np.arange(points) / (2 * points * sig.dt)
    return fa_spectrum, fa_frequencies


def interp_array_to_approx_dt(values, dt, target_dt=0.01, even=True):
    factor = dt / target_dt
    if factor == 1:
        pass
    elif factor > 1:
        factor = int(np.ceil(factor))
    else:
        factor = 1 / np.floor(1 / factor)
    t_int = np.arange(len(values))
    new_npts = factor * len(values)
    if even:
        new_npts = 2 * int(new_npts / 2)
    t_db = np.arange(new_npts) / factor
    acc_interp = np.interp(t_db, t_int, values)
    return acc_interp, dt / factor


def interp_to_approx_dt(asig, target_dt=0.01, even=True):
    """
    Interpolate a signal to a new time step

    Only a target time step is provided and the algorithm determines
     what time step is best to minimise loss of data from aliasing

    Parameters
    ----------
    asig: eqsig.AccSignal
        Acceleration time series object
    target_dt: float
        Target time step
    even: bool
        If true then forces the number of time steps to be an even number

    Returns
    -------

    """
    acc_interp, dt_interp = interp_array_to_approx_dt(asig.values, asig.dt, target_dt=target_dt, even=even)
    return eqsig.AccSignal(acc_interp, dt_interp)


def resample_to_approx_dt(asig, target_dt=0.01, even=True):
    """
    Resample a signal assuming periodic to a new time step

    Only a target time step is provided and the algorithm determines
     what time step is best to minimise loss of data from aliasing

    Parameters
    ----------
    asig: eqsig.AccSignal
        Acceleration time series object
    target_dt: float
        Target time step
    even: bool
        If true then forces the number of time steps to be an even number

    Returns
    -------

    """
    factor = asig.dt / target_dt
    if factor == 1:
        pass
    elif factor > 1:
        factor = int(np.ceil(factor))
    else:
        factor = 1 / np.floor(1 / factor)
    new_npts = factor * asig.npts
    if even:
        new_npts = 2 * int(new_npts / 2)
    acc_interp = scipy.signal.resample(asig.values, new_npts)
    return eqsig.AccSignal(acc_interp, asig.dt / factor)


def get_switched_peak_indices(asig):
    """
    Find the indices for largest peak between each zero crossing

    Parameters
    ----------
    asig: eqsig.AccSignal

    Returns
    -------
    array_like
    """
    values = asig
    if hasattr(asig, "values"):
        values = asig.values
    return get_switched_peak_array_indices(values)


def get_switched_peak_array_indices(values):
    """
    Find the indices for largest peak between each zero crossing

    Parameters
    ----------
    values: array_like

    Returns
    -------
    array_like
    """
    peak_indices = eqsig.get_peak_array_indices(values)
    peak_values = np.take(values, peak_indices)

    last = peak_values[0]
    new_peak_indices = []
    peak_values_set = [0]
    peak_indices_set = [0]
    for i in range(1, len(peak_values)):
        if peak_values[i] * last <= 0:  # only add index if sign changes (negative number)
            i_max_set = np.argmax(np.abs(peak_values_set))
            new_peak_indices.append(peak_indices_set[i_max_set])

            last = peak_values[i]
            peak_values_set = []  # reset set
            peak_indices_set = []

        peak_values_set.append(peak_values[i])
        peak_indices_set.append(i)
        
    if len(peak_values_set):  # add last
        i_max_set = np.argmax(np.abs(peak_values_set))
        new_peak_indices.append(peak_indices_set[i_max_set])
        peak_values_set.append(peak_values[i])
        peak_indices_set.append(i)

    switched_peak_indices = np.take(peak_indices, new_peak_indices)
    return switched_peak_indices


def get_sig_array_indexes_range(fas1_smooth, ratio=15):
    max_fas1 = max(fas1_smooth)
    lim_fas = max_fas1 / ratio
    indys = np.where(fas1_smooth > lim_fas)[0]
    return indys[0], indys[-1]

    # min_freq_i = 10000
    # max_freq_i = 10000
    # for i in range(len(fas1_smooth)):
    #     if fas1_smooth[i] > lim_fas:
    #         min_freq_i = i
    #         break
    # for i in range(len(fas1_smooth)):
    #     if fas1_smooth[-1 - i] > lim_fas:
    #         max_freq_i = len(fas1_smooth) - i
    #         break
    # return min_freq_i, max_freq_i


def get_sig_freq_range(asig, ratio=15):
    indices = get_sig_array_indexes_range(asig.smooth_fa_spectrum, ratio=ratio)
    return np.take(asig.smooth_fa_frequencies, indices)


def calc_fourier_moment(asig, n):
    """
    Original source unknown.

    See :cite:`Rathje:2008va`

    Parameters
    ----------
    asig
    n

    Returns
    -------

    """
    return 2 * np.trapz((2 * np.pi * asig.fa_frequencies) ** n * asig.fa_spectrum ** 2, x=asig.fa_frequencies)


def get_bandwidth_boore_2003(asig):
    m0 = calc_fourier_moment(asig, 0)
    m2 = calc_fourier_moment(asig, 2)
    m4 = calc_fourier_moment(asig, 4)
    return np.sqrt(m2 ** 2 / (m0 * m4))


def put_array_in_2d_array(values, shifts, clip='none'):
    """
    Creates a 2D array where values appear on each line, shifted by a set of indices

    Parameters
    ----------
    values: array_like (1d)
        Values to be shifted
    shifts: array_like (int)
        Indices to shift values
    clip: str or none
        if 'end' then returned 2D array trims values that overlap end of input values array

    Returns
    -------
    array_like (2D)
    """
    npts = len(values)
    # assert shifts is integer array
    end_extras = np.max([np.max(shifts), 0])
    start_extras = - np.min([np.min(shifts), 0])
    out = np.zeros((len(shifts), npts + start_extras + end_extras))
    for i, j in enumerate(shifts):
        out[i, start_extras + j:start_extras + npts + j] = values
    if clip in ['end', 'both'] and end_extras > 0:
            out = out[:, :-end_extras]
    if clip in ['start', 'both']:
        return out[:, start_extras:]
    else:
        return out


def join_sig_w_time_shift(sig, time_shifts, jtype='add'):
    """
    Zero pads values of a signal by an array of time shifts and joins it with the original

    Parameters
    ----------
    sig: eqsig.Signal
        signal to be shifted
    time_shifts: array_like
        Time shifts to be performed
    jtype: str (default='add')
        if = 'add' then shifted and original signals are added, if ='sub' then subtracted

    Returns
    -------
    shifted_values: array_like [2D shape(len(sig.values), len(shift))]

    """
    shifts = np.array(time_shifts / sig.dt, dtype=int)
    values = sig.values
    return join_values_w_shifts(values, shifts, jtype=jtype)


def join_values_w_shifts(values, shifts, jtype='add'):
    """
    Zero pads values by an array of shifts and joins it with the original values

    Parameters
    ----------
    values: array_like
        values to be shifted
    shifts: array_like [int]
        Shifts to be performed
    jtype: str (default='add')
        if = 'add' then shifted and original values are added, if ='sub' then subtracted

    Returns
    -------
    shifted_values: array_like [2D shape(len(values), len(shift))]

    """
    a0 = np.pad(values, (0, np.max(shifts)), mode='constant', constant_values=0)  # 1d
    a1 = put_array_in_2d_array(values, shifts)
    if jtype == 'add':
        return a1 + a0
    elif jtype == 'sub':
        return -a1 + a0


def get_section_average(series, start=0, end=-1, index=False):
    """
    Gets the average value of a part of series.

    Common use is so that it can be patched with another record.

    :param series: A TimeSeries object
    :param start: int or float, optional,
        Section start point
    :param end: int or float, optional,
        Section end point
    :param index: bool, optional,
        if False then start and end are considered values in time.
    :return float,
        The mean value of the section.
    """
    s_index, e_index = time_indices(series.npts, series.dt, start, end, index)

    section_average = np.mean(series.values[s_index:e_index])
    return section_average


def time_indices(npts, dt, start, end, index):
    """
    Determine the new start and end indices of the time series.

    :param npts: Number of points in original time series
    :param dt: Time step of original time series
    :param start: int or float, optional, New start point
    :param end: int or float, optional, New end point
    :param index: bool, optional, if False then start and end are considered values in time.
    :return: tuple, start index, end index
    """
    if index is False:  # Convert time values into indices
        if end != -1:
            e_index = int(end / dt) + 1
        else:
            e_index = end
        s_index = int(start / dt)
    else:
        s_index = start
        e_index = end
    if e_index > npts:
        raise exceptions.SignalProcessingWarning("Cut point is greater than time series length")
    return s_index, e_index


def generate_smooth_fa_spectrum(smooth_fa_frequencies, fa_frequencies, fa_spectrum, band=40):
    smooth_fa_spectrum = np.zeros_like(smooth_fa_frequencies)
    for i in range(smooth_fa_frequencies.size):
        f_centre = smooth_fa_frequencies[i]
        amp_array = np.log10((fa_frequencies / f_centre) ** band)

        amp_array[0] = 0

        wb_vals = np.zeros((len(amp_array)))
        for j in range(len(amp_array)):
            if amp_array[j] == 0:
                wb_vals[j] = 1
            else:
                wb_vals[j] = (np.sin(amp_array[j]) / amp_array[j]) ** 4

        smooth_fa_spectrum[i] = (sum(abs(fa_spectrum) * wb_vals) / sum(wb_vals))
    return smooth_fa_spectrum


def calc_step_fn_vals_error(values, pow=1, dir=None):
    """
    Calculates the error function generated by fitting a step function
    to the values

    Note: Assumes minimum error is at the minimum sum of the error,
    regardless of the `pow`. I.e. The best fit is taken as the mean
    of the values.

    Parameters
    ----------
    values: array_like
    pow: int
        The power that the error should be raised to
    dir: str
        Desired direction of the step function
        if 'down', then all upward steps are set to 10x maximum error
        if 'up', then all downward steps are set to 10x maximum error
        else, no modification to error

    Returns
    -------
    array_like (len same as input array)
    """
    values = np.array(values)
    npts = len(values)
    pre_a = np.tril(values, k=0)
    post_a = np.triu(values, k=0)
    pre_n = np.arange(1, len(values) + 1)
    post_n = np.arange(len(values), 0, -1)
    pre_mean = np.sum(pre_a, axis=1) / pre_n
    post_mean = np.sum(post_a, axis=1) / post_n
    err_pre = np.sum(np.abs(pre_a - pre_mean[:, np.newaxis]) ** pow, axis=1) - (npts - pre_n) * pre_mean ** pow
    err_post = np.sum(np.abs(post_a - post_mean[:, np.newaxis]) ** pow, axis=1) - (npts - post_n) * post_mean ** pow
    # case of 0 has been remove, n + 1 options
    # consider cases where it happens in between two steps
    err = np.ones_like(values)
    err[:-1] = err_post[1:] + err_pre[:-1]
    err[-1] = np.sum(np.abs(values - np.mean(values)) ** pow)
    if dir == 'down':  # if step has to be downward, then increase error for upward steps
        max_err = np.max(err)
        err = np.where(pre_mean < post_mean, max_err * 10, err)
    if dir == 'up':
        max_err = np.max(err)
        err = np.where(pre_mean > post_mean, max_err * 10, err)
    return err


def calc_step_fn_steps_vals(values, ind=None):
    if ind is None:
        ind = np.argmin(calc_step_fn_vals_error(values))
    pre = np.mean(values[:ind])
    post = np.mean(values[ind + 1:])
    return pre, post


def calc_roll_av_vals(values, steps, mode='forward'):
    """
    Calculates the rolling average of a series of values

    Parameters
    ----------
    values: array_like
    steps: int
        size of window to average over
    mode: str (default='forward')
        if 'forward' current value at start of window
        if 'backward' current value at end of window
        if 'centre' or 'center' current value in centre of window

    Returns
    -------
    array_like (len same as input array)
    """
    steps = int(steps)
    if mode == 'forward':
        x_ext = np.concatenate([values, values[-1] * np.ones(steps - 1)])
    elif mode == 'backward':
        x_ext = np.concatenate([values[0] * np.ones(steps - 1), values])
    else:
        s = int(np.floor(steps / 2))
        e = steps - s - 1
        x_ext = np.concatenate([values[0] * np.ones(s), values, values[-1] * np.ones(e)])
    csum = np.zeros(len(values) + steps)
    csum[1:] = np.cumsum(x_ext, dtype=float)

    return (csum[steps:] - csum[:-steps]) / steps


if __name__ == '__main__':
    vals = np.arange(4, 6)
    sfs = np.array([-1, 2])
    expected_full = np.array([[4, 5, 0, 0, 0],
                              [0, 0, 0, 4, 5],
                              ])
    out_a = put_array_in_2d_array(vals, sfs, clip='start')
    # out_a = join_values_w_shifts(vals, sfs, jtype='sub')

    print(out_a)
