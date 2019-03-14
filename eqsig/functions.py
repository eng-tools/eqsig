import numpy as np
import scipy

import eqsig
from eqsig.single import Signal, AccSignal


def time_series_from_motion(motion, dt):
    npts = len(motion)
    return np.linspace(0, dt * (npts + 1), npts)


def determine_indices_of_peaks_for_cleaned(values):
    """
    Determines the position of values that form a local peak in a signal.

    :param values:
    :return:
    """
    diff = np.diff(values)
    # if negative then direction has switched
    direction_switch = diff[1:] * diff[:-1]
    direction_switch = np.insert(direction_switch, 0, 0)
    peaks = np.where(direction_switch < 0)
    peak_indices = peaks[0]
    peak_indices = np.insert(peak_indices, 0, 0) # Include first and last value
    peak_indices = np.insert(peak_indices, len(peak_indices), len(values) - 1)

    return peak_indices


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
    diff_values = np.diff(values)
    diff_values = np.insert(diff_values, 0, values[0])
    non_zero_indices = np.where(diff_values != 0)[0]
    non_zero_indices = np.insert(non_zero_indices, 0, 0)

    cleaned_values = np.take(values, non_zero_indices)
    return cleaned_values, non_zero_indices


def get_peak_indices(values):
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
    >>> determine_indices_of_peaks_for_cleaned(values)
    np.array([0, 1, 2, 3, 4, 5, 8, 10, 11])
    """
    # enforce array type
    values = np.array(values, dtype=float)
    # remove all non-changing values
    cleaned_values, non_zero_indices = clean_out_non_changing(values)
    # cleaned_values *= np.sign(cleaned_values[1])  # ensure first value is increasing
    peak_cleaned_indices = determine_indices_of_peaks_for_cleaned(cleaned_values)
    peak_full_indices = np.take(non_zero_indices, peak_cleaned_indices)
    return peak_full_indices


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


def generate_fa_spectrum(sig):
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
    npts = len(sig.values)
    n_factor = 2 ** int(np.ceil(np.log2(npts)))
    fa = scipy.fft(sig.values, n=n_factor)
    points = int(n_factor / 2)
    assert len(fa) == n_factor
    fa_spectrum = fa[range(points)] * sig.dt
    fa_frequencies = np.arange(points) / (2 * points * sig.dt)
    return fa_spectrum, fa_frequencies


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
    factor = asig.dt / target_dt
    if factor == 1:
        pass
    elif factor > 1:
        factor = int(np.ceil(factor))
    else:
        factor = 1 / np.floor(1 / factor)
    t_int = np.arange(len(asig.values))
    new_npts = factor * len(asig.values)
    if even:
        new_npts = 2 * int(new_npts / 2)
    t_db = np.arange(new_npts) / factor
    acc_interp = np.interp(t_db, t_int, asig.values)
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
    peak_indices = eqsig.get_peak_indices(values)
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
    switched_peak_indices = np.take(peak_indices, new_peak_indices)
    return switched_peak_indices
