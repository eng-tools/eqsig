import numpy as np


def time_series_from_motion(motion, dt):
    npts = len(motion)
    return np.linspace(0, dt * (npts + 1), npts)


def determine_indices_of_peaks(values):
    diff = np.diff(values)
    # if negative then direction has switched
    direction_switch = diff[1:] * diff[:-1]
    direction_switch = np.insert(direction_switch, 0, 0)
    peak_indices = np.where(direction_switch < 0)
    return peak_indices[0]


# def determine_peaks_only_series(values):
#     diff = np.diff(values)
#     # if negative then direction has switched
#     direction_switch = diff[1:] * diff[:-1]
#     peaks = np.where(direction_switch < 0, 1, 0)
#     # peaks array is shorter by two, indices refer +1, add zeros at start and end
#     peaks = np.insert(peaks, 0, 0)
#     peaks = np.insert(peaks, len(peaks), 0)  # don't use -1 here
#     return values * peaks


def determine_delta_peak_only_series_4_cleaned_data(values):
    peak_indices = determine_indices_of_peaks(values)
    peak_values = np.take(values, peak_indices)
    signs = np.where(np.mod(np.arange(len(peak_values)), 2), 1, -1)
    delta_peaks = np.where(-signs * peak_values < 0, -np.abs(peak_values), np.abs(peak_values))
    delta_peaks_series = np.zeros_like(values)
    np.put(delta_peaks_series, peak_indices, delta_peaks)

    return delta_peaks_series


def clean_out_non_changing(values):
    diff_values = np.diff(values)
    non_zero_indices = np.where(diff_values != 0)[0]
    cleaned_values = np.take(values, non_zero_indices)
    cleaned_values = np.insert(cleaned_values, len(cleaned_values), values[-1])
    return cleaned_values, non_zero_indices


def determine_delta_peak_only_series(values):
    """
    Creates an array with only the changes in the peak values and zeros for non-peak values.

    Parameters
    ----------
    :param values: array_like, array of values
    :return:

    Examples
    --------
    >>> values = np.array([0, 2, 1, 2, 0, 1, 0, -1, 0, 1, 0])
    np.array([0, 2, 1, 2, 0.3, 1, 0.3, -1, 0.4, 1, 0])
    >>> determine_delta_peak_only_series(values)
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
    cleaned_delta_peak_series = determine_delta_peak_only_series_4_cleaned_data(cleaned_values)
    # re-index data to uncleaned array
    delta_peaks_series = np.zeros_like(values)
    np.put(delta_peaks_series, non_zero_indices, cleaned_delta_peak_series)
    return delta_peaks_series
