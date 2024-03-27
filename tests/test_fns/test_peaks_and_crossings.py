import numpy as np

import eqsig
from tests.conftest import TEST_DATA_DIR


def test_determine_pseudo_cyclic_peak_only_series_with_triangle_series():
    values = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0]
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.fns.peaks_and_crossings.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(np.abs(peaks_only))
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_delta_series_with_triangle_series():
    values = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0]
    peaks_only = eqsig.fns.peaks_and_crossings.determine_peaks_only_delta_series(values)
    cum_peaks = np.sum(np.abs(peaks_only))
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    cum_diff = np.sum(peaks_only)
    assert np.isclose(cum_diff, 0), (cum_diff, 0)
    assert np.isclose(cum_abs_delta_values, 10), (cum_abs_delta_values, 10)


def test_determine_pseudo_cyclic_peak_only_series_with_sine_wave():
    time = np.arange(99)
    values = np.sin(time)
    values[-1] = 0
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.fns.peaks_and_crossings.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(np.abs(peaks_only))
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_delta_series_with_sine_wave():
    time = np.arange(99)
    values = np.sin(time)
    values[-1] = 0
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values
    peaks_only = eqsig.fns.peaks_and_crossings.determine_peaks_only_delta_series(values)
    cum_peaks = np.sum(np.abs(peaks_only))
    assert np.isclose(cum_peaks, expected_sum), (cum_peaks, expected_sum)


def test_determine_pseudo_cyclic_peak_only_series_with_ground_motion():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    rec = np.loadtxt(record_path + record_filename, skiprows=2)
    cum_abs_delta_values = np.sum(np.abs(np.diff(rec)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.fns.peaks_and_crossings.determine_pseudo_cyclic_peak_only_series(rec)
    cum_peaks = np.sum(peaks_only)
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_delta_series_with_ground_motion():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    rec = np.loadtxt(record_path + record_filename, skiprows=2)
    cum_abs_delta_values = np.sum(np.abs(np.diff(rec)))
    expected_sum = cum_abs_delta_values
    delta_peaks_only = eqsig.fns.peaks_and_crossings.determine_peaks_only_delta_series(rec)
    cum_peaks = np.sum(np.abs(delta_peaks_only))
    assert np.isclose(cum_peaks, expected_sum), (cum_peaks, expected_sum)


def test_determine_pseudo_cyclic_peak_only_series_with_a_double_peak_and_offset():
    values = np.array([0, 2, 1, 2, 0, 1, 0, -1, 0, 1, 0]) + 4
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.fns.peaks_and_crossings.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(peaks_only)
    expected_series = np.array([0,  2, -1,  2,  0,  1,  0,  1,  0,  1,  0])
    assert np.sum(np.abs(peaks_only - expected_series)) == 0.0
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_delta_series_with_a_double_peak_and_offset():
    values = np.array([0, 2, 1, 2, 0, 1, 0, -1, 0, 1, 0]) + 4
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values
    delta_peaks_only = eqsig.fns.peaks_and_crossings.determine_peaks_only_delta_series(values)
    cum_peaks = np.sum(np.abs(delta_peaks_only))
    expected_series = np.array([0,  2, -1,  1,  -2,  1,  0,  -2,  0,  2,  -1])
    assert np.sum(np.abs(delta_peaks_only - expected_series)) == 0.0
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_pseudo_cyclic_peak_only_series_with_non_zero_end():
    end_value = 1.
    values = np.array([0, 2, -1, 2, 0, end_value])
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2 + end_value / 2
    peaks_only = eqsig.fns.peaks_and_crossings.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(peaks_only)
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_series_with_non_zero_end():
    end_value = 1.
    values = np.array([0, 2, -1, 2, 0, end_value])
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values
    delta_peaks_only = eqsig.fns.peaks_and_crossings.determine_peaks_only_delta_series(values)
    cum_peaks = np.sum(np.abs(delta_peaks_only))
    assert np.isclose(cum_peaks, expected_sum), (cum_peaks, expected_sum)


def test_determine_peaks_only_series_with_nonchanging_values():
    values = np.array([0, 1, 1, -3, -5, 0])  # constant then reverse
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2

    peaks_only = eqsig.fns.peaks_and_crossings.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(peaks_only)
    assert np.isclose(cum_peaks, expected_sum), cum_peaks

    values = np.array([0, 1, 1, 3, -5, 0])  # constant the no reverse
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.fns.peaks_and_crossings.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(peaks_only)
    assert np.isclose(cum_peaks, expected_sum), cum_peaks


def test_get_peak_indices():
    values = np.array([0, 2, 1, 2, -1, 1, 1, 0.3, -1, 0.2, 1, 0.2])
    peak_indices = eqsig.fns.peaks_and_crossings.get_peak_array_indices(values)
    peaks_series = np.zeros_like(values)
    np.put(peaks_series, peak_indices, values)
    expected = np.array([0, 1, 2, 3, 4, 5, 8, 10, 11])
    assert np.sum(abs(peak_indices - expected)) == 0

    values = np.array([2, 1, -1, 1])
    peak_indices = eqsig.fns.peaks_and_crossings.get_peak_array_indices(values)
    expected = np.array([0, 2, 3])
    assert np.sum(abs(peak_indices - expected)) == 0

    values = np.array([1, 2, -1, 1])
    peak_indices = eqsig.fns.peaks_and_crossings.get_peak_array_indices(values)
    expected = np.array([0, 1, 2, 3])
    assert np.sum(abs(peak_indices - expected)) == 0


def test_get_peak_indices_w_tol():
    values = np.array([0, 2, 1, 2, -0.01, 1, 1, 0.3, -0.009, 0.2, 1.5, 0.2])
    peak_indices = eqsig.fns.peaks_and_crossings.get_switched_peak_array_indices(values)
    expected = np.array([0, 1, 4, 5, 8, 10])
    assert len(peak_indices) == len(expected), peak_indices
    assert np.sum(abs(peak_indices - expected)) == 0, peak_indices

    values = np.array([0, 2, 1, 2, -0.01, 1, 1, 0.3, -0.009, 0.2, 1.5, 0.2])
    peak_indices = eqsig.fns.peaks_and_crossings.get_switched_peak_array_indices(values, tol=0.01)
    expected = np.array([0, 1, 4, 10])
    assert len(peak_indices) == len(expected), peak_indices
    assert np.sum(abs(peak_indices - expected)) == 0, peak_indices

    values = np.array([0, 2, 1, 2, -0.01, 1, 1, 0.3, -0.009, 0.001, -0.009, 0.2, 1.5, 0.2])
    peak_indices = eqsig.fns.peaks_and_crossings.get_switched_peak_array_indices(values, tol=0.01)
    expected = np.array([0, 1, 4, 12])
    assert len(peak_indices) == len(expected), peak_indices
    assert np.sum(abs(peak_indices - expected)) == 0, peak_indices

    values = np.array([0, 2, 1, 2, -0.01, 1, 1, 0.3, -0.009, 0.001, -0.01, 0.2, 1.5, 0.2])
    peak_indices = eqsig.fns.peaks_and_crossings.get_switched_peak_array_indices(values, tol=0.01)
    expected = np.array([0, 1, 4, 5, 10, 12])
    assert len(peak_indices) == len(expected), peak_indices
    assert np.sum(abs(peak_indices - expected)) == 0, peak_indices


def test_get_zero_crossings_array_indices():
    vs = np.array([0, 2, 1, 2, -1, 1, 0, 0, 1, 0.3, 0, -1, 0.2, 1, 0.2])
    zci = eqsig.fns.peaks_and_crossings.get_zero_crossings_array_indices(vs, keep_adj_zeros=True)
    expected = np.array([0, 4, 5, 6, 7, 10, 12])
    assert np.array_equal(zci, expected)
    zci = eqsig.fns.peaks_and_crossings.get_zero_crossings_array_indices(vs, keep_adj_zeros=False)
    expected = np.array([0, 4, 5, 6, 10, 12])
    assert np.array_equal(zci, expected), zci
    # no zeros
    vs = np.array([1, 2, 1, 2, -1, 1, 1, 1, 1, 0.3, 1, -1, 0.2, 1, 0.2])
    zci = eqsig.fns.peaks_and_crossings.get_zero_crossings_array_indices(vs, keep_adj_zeros=False)
    expected = np.array([0, 4, 5, 11, 12])
    assert np.array_equal(zci, expected), zci
    vs = np.array([-1, -2, 1, 2, -1, 1, 1, 1, 1, 0.3, 1, -1, 0.2, 1, 0.2])
    zci = eqsig.fns.peaks_and_crossings.get_zero_crossings_array_indices(vs, keep_adj_zeros=False)
    expected = np.array([0, 2, 4, 5, 11, 12])
    assert np.array_equal(zci, expected), zci


def test_get_zero_crossings_array_indices_w_tol():
    # expect no change at -0.009
    vs = np.array([0, 2, 1, 2, -1, 0.02, -0.02, 0, 1, 0.3, 0, -0.009, 0.2, 1, 0.2])
    zci = eqsig.fns.peaks_and_crossings.get_zero_crossings_array_indices(vs, keep_adj_zeros=True, tol=0.01)
    expected = np.array([0, 4, 5, 6, 7])
    assert np.array_equal(zci, expected)
