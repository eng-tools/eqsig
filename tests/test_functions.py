import numpy as np
import eqsig
import scipy
from eqsig import functions

from tests.conftest import TEST_DATA_DIR


def test_determine_pseudo_cyclic_peak_only_series_with_triangle_series():
    values = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0]
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(np.abs(peaks_only))
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_delta_series_with_triangle_series():
    values = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0]
    peaks_only = eqsig.determine_peaks_only_delta_series(values)
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
    peaks_only = eqsig.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(np.abs(peaks_only))
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_delta_series_with_sine_wave():
    time = np.arange(99)
    values = np.sin(time)
    values[-1] = 0
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values
    peaks_only = eqsig.determine_peaks_only_delta_series(values)
    cum_peaks = np.sum(np.abs(peaks_only))
    assert np.isclose(cum_peaks, expected_sum), (cum_peaks, expected_sum)


def test_determine_pseudo_cyclic_peak_only_series_with_ground_motion():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    rec = np.loadtxt(record_path + record_filename, skiprows=2)
    cum_abs_delta_values = np.sum(np.abs(np.diff(rec)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.determine_pseudo_cyclic_peak_only_series(rec)
    cum_peaks = np.sum(peaks_only)
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_delta_series_with_ground_motion():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    rec = np.loadtxt(record_path + record_filename, skiprows=2)
    cum_abs_delta_values = np.sum(np.abs(np.diff(rec)))
    expected_sum = cum_abs_delta_values
    delta_peaks_only = eqsig.determine_peaks_only_delta_series(rec)
    cum_peaks = np.sum(np.abs(delta_peaks_only))
    assert np.isclose(cum_peaks, expected_sum), (cum_peaks, expected_sum)


def test_determine_pseudo_cyclic_peak_only_series_with_a_double_peak_and_offset():
    values = np.array([0, 2, 1, 2, 0, 1, 0, -1, 0, 1, 0]) + 4
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(peaks_only)
    expected_series = np.array([0,  2, -1,  2,  0,  1,  0,  1,  0,  1,  0])
    assert np.sum(np.abs(peaks_only - expected_series)) == 0.0
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_delta_series_with_a_double_peak_and_offset():
    values = np.array([0, 2, 1, 2, 0, 1, 0, -1, 0, 1, 0]) + 4
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values
    delta_peaks_only = eqsig.determine_peaks_only_delta_series(values)
    cum_peaks = np.sum(np.abs(delta_peaks_only))
    expected_series = np.array([0,  2, -1,  1,  -2,  1,  0,  -2,  0,  2,  -1])
    assert np.sum(np.abs(delta_peaks_only - expected_series)) == 0.0
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_pseudo_cyclic_peak_only_series_with_non_zero_end():
    end_value = 1.
    values = np.array([0, 2, -1, 2, 0, end_value])
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2 + end_value / 2
    peaks_only = eqsig.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(peaks_only)
    assert np.isclose(cum_peaks, expected_sum)


def test_determine_peaks_only_series_with_non_zero_end():
    end_value = 1.
    values = np.array([0, 2, -1, 2, 0, end_value])
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values
    delta_peaks_only = eqsig.determine_peaks_only_delta_series(values)
    cum_peaks = np.sum(np.abs(delta_peaks_only))
    assert np.isclose(cum_peaks, expected_sum), (cum_peaks, expected_sum)


def test_determine_peaks_only_series_with_nonchanging_values():
    values = np.array([0, 1, 1, -3, -5, 0])  # constant then reverse
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2

    peaks_only = eqsig.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(peaks_only)
    assert np.isclose(cum_peaks, expected_sum), cum_peaks

    values = np.array([0, 1, 1, 3, -5, 0])  # constant the no reverse
    cum_abs_delta_values = np.sum(np.abs(np.diff(values)))
    expected_sum = cum_abs_delta_values / 2
    peaks_only = eqsig.determine_pseudo_cyclic_peak_only_series(values)
    cum_peaks = np.sum(peaks_only)
    assert np.isclose(cum_peaks, expected_sum), cum_peaks


def test_fa_spectrum_conversion():
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    dt = 0.01
    values = np.loadtxt(record_path + record_filename, skiprows=2)

    npts = len(values)
    n_factor = 2 ** int(np.ceil(np.log2(npts)))
    fa = scipy.fft(values, n=n_factor)
    points = int(n_factor / 2)
    fas = fa[range(points)] * dt
    faf = np.arange(points) / (2 * points * dt)
    n = 2 * len(fas)
    asig = eqsig.AccSignal(values, dt)
    fas_eqsig, faf_eqsig = functions.generate_fa_spectrum(asig)

    assert np.isclose(fas, fas_eqsig).all()
    assert np.isclose(faf, faf_eqsig).all()

    a = np.zeros(len(fa), dtype=complex)
    a[1:n // 2] = fas[1:]
    a[n // 2 + 1:] = np.flip(np.conj(fas[1:]), axis=0)
    a /= dt
    sig = np.fft.ifft(fa, n=n_factor)
    sig = sig[:len(values)]
    assert np.isclose(np.sum(np.abs(sig)), np.sum(np.abs(values)))
    asig2 = functions.fas2signal(fas_eqsig, dt, stype="signal")
    trimmed = asig2.values[:len(values)]
    assert np.isclose(np.sum(np.abs(trimmed)), np.sum(np.abs(values)))


def test_get_peak_indices():
    values = np.array([0, 2, 1, 2, -1, 1, 1, 0.3, -1, 0.2, 1, 0.2])
    peak_indices = functions.get_peak_array_indices(values)
    peaks_series = np.zeros_like(values)
    np.put(peaks_series, peak_indices, values)
    expected = np.array([0, 1, 2, 3, 4, 5, 8, 10, 11])
    assert np.sum(abs(peak_indices - expected)) == 0


def test_get_zero_crossings_array_indices():
    vs = np.array([0, 2, 1, 2, -1, 1, 0, 0, 1, 0.3, 0, -1, 0.2, 1, 0.2])
    zci = functions.get_zero_crossings_array_indices(vs, keep_adj_zeros=True)
    expected = np.array([0, 4, 5, 6, 7, 10, 12])
    assert np.array_equal(zci, expected)
    zci = functions.get_zero_crossings_array_indices(vs, keep_adj_zeros=False)
    expected = np.array([0, 4, 5, 6, 10, 12])
    assert np.array_equal(zci, expected), zci
    print(zci)


if __name__ == '__main__':
    test_determine_peaks_only_series_with_non_zero_end()
    # test_fa_spectrum_conversion()
    # test_determine_peaks_only_series_with_sine_wave()
    # test_determine_peaks_only_series_with_triangle_series()
    # test_determine_peaks_only_series_with_ground_motion()
    # test_determine_peaks_only_series_with_a_double_peak_and_offset()
    # test_determine_peaks_only_series_with_nonchanging_values()
    # test_determine_peaks_only_series_with_non_zero_end()
