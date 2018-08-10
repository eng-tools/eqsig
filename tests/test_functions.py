import numpy as np
import eqsig

from tests.conftest import TEST_DATA_DIR


def test_determine_peaks_only_series_with_triangle_series():
    values = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0]
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.determine_delta_peak_only_series(values)
    cum_peaks = np.sum(np.abs(peaks_only))
    print(cum_peaks, expected_sum)
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_series_with_sine_wave():
    time = np.arange(99)
    values = np.sin(time)
    values[-1] = 0
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.determine_delta_peak_only_series(values)
    cum_peaks = np.sum(np.abs(peaks_only))
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_series_with_ground_motion_1():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    rec = np.loadtxt(record_path + record_filename)
    rec = rec[:-1] * -1000
    rec[-1] = rec[0]
    cum_abs_delta_values = np.sum(np.abs(np.diff(rec)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.determine_delta_peak_only_series_cleaned(rec)
    cum_peaks = np.sum(peaks_only)
    print(cum_peaks, expected_sum)
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_series_with_ground_motion():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    rec = np.loadtxt(record_path + record_filename)
    rec = rec[:-1] * -1000
    rec[-1] = rec[0]
    cum_abs_delta_values = np.sum(np.abs(np.diff(rec)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.determine_delta_peak_only_series_cleaned(rec)
    cum_peaks = np.sum(peaks_only)
    print(cum_peaks, expected_sum)
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_series_with_a_double_peak():
    values = np.array([0, 2, 1, 2, 0, 1, 0, -1, 0, 1, 0]) + 4
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.determine_delta_peak_only_series_cleaned(values)
    cum_peaks = np.sum((peaks_only))
    print(cum_peaks, expected_sum)
    assert np.isclose(cum_peaks, expected_sum)


if __name__ == '__main__':
    # test_determine_peaks_only_series_with_sine_wave()
    # test_determine_peaks_only_series_with_triangle_series()
    test_determine_peaks_only_series_with_ground_motion_1()
