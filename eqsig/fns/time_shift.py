import numpy as np

from eqsig import exceptions


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
