import numpy as np


def determine_indices_of_peaks_for_cleaned(values):
    """DEPRECATED: Use determine_indices_of_peaks_for_cleaned_array()"""
    return determine_indices_of_peaks_for_cleaned_array(values)


def determine_indices_of_peaks_for_cleaned_array(values):
    """
    Determines the position of values that form a local peak in a signal.

    Warning: data must be cleaned so that adjacent points have the same value

    Parameters
    ----------
    values: array_like
        Array of values that peaks will be found in

    Returns
    -------
    peak_indices: array_like of int
        Array of indices of peaks
    """
    diff = np.ediff1d(values, to_begin=0)
    # if negative then direction has switched
    # direction_switch = np.insert(direction_switch, 0, 0)
    peak_indices = np.where(diff[1:] * diff[:-1] < 0)[0]
    peak_indices = np.insert(peak_indices, 0, 0)  # Include first and last value
    peak_indices = np.insert(peak_indices, len(peak_indices), len(values) - 1)

    return peak_indices


def _determine_peak_only_series_4_cleaned_data(values):
    """
    Determines the

    Note: array must not contain adjacent repeated values

    :param values:
    :return:
    """
    peak_indices = determine_indices_of_peaks_for_cleaned_array(values)
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
    peak_indices = determine_indices_of_peaks_for_cleaned_array(values)
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
    Find the indices for all the local maxima and minima

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
    peak_cleaned_indices = determine_indices_of_peaks_for_cleaned_array(cleaned_values)
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


def get_peak_indices(asig):
    return get_peak_array_indices(asig.values)


def get_zero_crossings_array_indices(values, keep_adj_zeros=False, tol=0.0):
    """
    Find the indices for values that are equal to zero or just passed through zero

    Parameters
    ----------
    values: array_like
        array of values
    keep_adj_zeros: bool,
        if false then if adjacent zeros are found, only first is included
    tol: float,
        if positive tol then has to go ∆tol past zero, if neg, then does not need to cross zero.
    :return:

    Examples
    --------
    >>> values = np.array([0, 2, 1, 2, -1, 1, 0, 0, 1, 0.3, 0, -1, 0.2, 1, 0.2])
    np.array([0, 2, 1, 2, -1, 1, 0, 0, 1, 0.3, 0, -1, 0.2, 1, 0.2])
    >>> get_zero_crossings_array_indices(values, keep_adj_zeros=False)
    np.array([0, 4, 5, 6, 10, 12])
    """
    if tol < 0:
        raise NotImplemented('not implemented')
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
    if len(all_zc_indices) == 0:
        return np.array([0])
    if all_zc_indices[0] != 0:
        all_zc_indices = np.insert(all_zc_indices, 0, 0)  # slow
    if tol > 0:
        rem_i = []
        for k, ind in enumerate(all_zc_indices[:-1]):
            if k in rem_i:
                continue
            ind1 = all_zc_indices[k+1]
            if max(abs(values[ind:ind1])) < tol:
                rem_i += [k, k+1]
        all_zc_indices = np.delete(all_zc_indices, rem_i)

    return all_zc_indices


def get_zero_crossings_indices(asig):
    return get_zero_crossings_array_indices(asig.values)


def get_zero_and_peak_array_indices(pvals, zvals=None, min_step=0):
    """

    :param pvals:
    :param zvals:
    :param min_step: int
        number of extra steps between zero crossing and peak
    :return:
    """
    if zvals is None:
        zvals = pvals
    peak_indices = get_switched_peak_array_indices(pvals)
    ci = get_zero_crossings_array_indices(zvals)
    cc = 0
    new_ci = []
    new_pi = []
    for i in range(1, len(ci)):
        if i - cc + 1 == len(peak_indices):
            break
        p0 = peak_indices[i - 1 - cc]
        p1 = peak_indices[i - cc]
        c = ci[i]

        if c >= p1 - min_step:
            cc -= 1
            continue
        if p0 == c:
            continue
        if p0 < c < p1:
            new_ci.append(c)
            new_pi.append(p1)
            if len(new_pi) > 1:
                assert new_pi[-2] < new_ci[-1]
            assert new_pi[-1] > new_ci[-1], (i, new_pi[-1], new_ci[-1])
        else:
            cc += 1

    ci = np.array(new_ci)
    piz = np.array(new_pi)
    if len(ci) < 2:
        return [], []
    assert min(piz - ci) > 0
    assert max(piz[:-1] - ci[1:]) < 0
    assert min(np.diff(piz)) > 0
    assert min(np.diff(ci)) > 0

    return ci, piz


def get_major_change_indices(y, rtol=1.0e-8, atol=1.0e-5, already_diff=False, dx=1):
    """Get indices where a significant change in slope occurs"""
    if already_diff:
        dydx = y
    else:
        dydx = np.diff(y, prepend=y[0]) / dx
    inds = [0]
    z_cur = 0
    npts = len(dydx)
    i = 1
    while z_cur + i < npts - 1:
        end_z = z_cur + i
        av_dydx = np.mean(dydx[z_cur: end_z])

        if not np.isclose(av_dydx, dydx[end_z + 1], rtol=rtol, atol=atol):
            inds.append(end_z)
            z_cur = end_z + 1
            i = 0

        i += 1
    inds.append(npts - 1)
    return inds


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


def get_switched_peak_array_indices(values, tol=0.0):
    """
    Find the indices for largest peak between each zero crossing

    Parameters
    ----------
    values: array_like
    tol: float,
        if positive tol then has to go ∆tol past zero, if neg, then does not need to cross zero.

    Returns
    -------
    array_like
    """
    peak_indices = get_peak_array_indices(values)
    peak_values = np.take(values, peak_indices)

    last = peak_values[0]
    new_peak_indices = []
    peak_values_set = [0]
    peak_indices_set = [0]
    for i in range(1, len(peak_values)):
        sgn = np.sign(last)
        adj_val = peak_values[i] + tol * sgn  # if val is -ve then this will make value more +ve
        if adj_val * last <= 0:  # only add index if sign changes (negative number)
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
    cleaned_delta_peak_series = _determine_peak_only_series_4_cleaned_data(cleaned_values)
    # re-index data to uncleaned array
    delta_peaks_series = np.zeros_like(values)
    np.put(delta_peaks_series, non_zero_indices, cleaned_delta_peak_series)
    return delta_peaks_series


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
