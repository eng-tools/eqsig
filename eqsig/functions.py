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


def determine_peaks_only_series(values):
    diff = np.diff(values)
    # if negative then direction has switched
    direction_switch = diff[1:] * diff[:-1]
    peaks = np.where(direction_switch < 0, 1, 0)
    # peaks array is shorter by two, indices refer +1, add zeros at start and end
    peaks = np.insert(peaks, 0, 0)
    peaks = np.insert(peaks, len(peaks), 0)  # don't use -1 here
    return values * peaks


def determine_delta_peak_only_series(values):
    peak_indices = determine_indices_of_peaks(values)
    peak_values = np.take(values, peak_indices)
    # peak_values = np.insert(peak_values, 0, 0)
    # diff_peaks = np.diff(peak_values)
    # diff_peaks = np.insert(diff_peaks, len(diff_peaks), 0)
    # print(np.cumsum(np.abs(diff_peaks)) - np.mean(peak_values))
    # diff_peaks = np.insert(diff_peaks, 0, peak_values[0] - values[0])  # insert first peak from start
    # must switch signs otherwise doesn't count
    signs = np.where(np.mod(np.arange(len(peak_values)), 2), 1, -1)
    delta_peaks = np.where(-signs * peak_values < 0, -np.abs(peak_values), np.abs(peak_values))
    # delta_peaks = delta_peaks[1:]
    # abs_diff_peaks = np.abs(diff_peaks)
    # delta_peaks = abs_diff_peaks[1:] - abs_diff_peaks[:-1]  # remove the restoring part
    # delta_peaks = np.insert(delta_peaks, 0, diff_peaks[0])
    # delta_peaks = np.cumsum(abs_diff_peaks)
    delta_peaks_series = np.zeros_like(values)
    np.put(delta_peaks_series, peak_indices, delta_peaks)

    return delta_peaks_series



def test_determine_peaks_only_series_with_a_double_peak():
    values = np.array([0, 2, 1, 2, 0, 1, 0, -1, 0, 1, 0]) + 0
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = determine_delta_peak_only_series(values)
    cum_peaks = np.sum((peaks_only))
    print(cum_peaks, expected_sum)
    assert np.isclose(cum_peaks, expected_sum)


if __name__ == '__main__':
    test_determine_peaks_only_series_with_a_double_peak()