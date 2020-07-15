import numpy as np
import eqsig
from eqsig import functions as fns
import pytest

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
    fa = np.fft.fft(values, n=n_factor)
    points = int(n_factor / 2)
    fas = fa[range(points)] * dt
    faf = np.arange(points) / (2 * points * dt)
    n = 2 * len(fas)
    asig = eqsig.AccSignal(values, dt)
    fas_eqsig, faf_eqsig = fns.generate_fa_spectrum(asig)

    assert np.isclose(fas, fas_eqsig).all()
    assert np.isclose(faf, faf_eqsig).all()

    a = np.zeros(len(fa), dtype=complex)
    a[1:n // 2] = fas[1:]
    a[n // 2 + 1:] = np.flip(np.conj(fas[1:]), axis=0)
    a /= dt
    sig = np.fft.ifft(fa, n=n_factor)
    sig = sig[:len(values)]
    assert np.isclose(np.sum(np.abs(sig)), np.sum(np.abs(values)))
    asig2 = fns.fas2signal(fas_eqsig, dt, stype="signal")
    trimmed = asig2.values[:len(values)]
    assert np.isclose(np.sum(np.abs(trimmed)), np.sum(np.abs(values)))


def test_get_peak_indices():
    values = np.array([0, 2, 1, 2, -1, 1, 1, 0.3, -1, 0.2, 1, 0.2])
    peak_indices = fns.get_peak_array_indices(values)
    peaks_series = np.zeros_like(values)
    np.put(peaks_series, peak_indices, values)
    expected = np.array([0, 1, 2, 3, 4, 5, 8, 10, 11])
    assert np.sum(abs(peak_indices - expected)) == 0

    values = np.array([2, 1, -1, 1])
    peak_indices = fns.get_peak_array_indices(values)
    expected = np.array([0, 2, 3])
    assert np.sum(abs(peak_indices - expected)) == 0

    values = np.array([1, 2, -1, 1])
    peak_indices = fns.get_peak_array_indices(values)
    expected = np.array([0, 1, 2, 3])
    assert np.sum(abs(peak_indices - expected)) == 0



def test_get_zero_crossings_array_indices():
    vs = np.array([0, 2, 1, 2, -1, 1, 0, 0, 1, 0.3, 0, -1, 0.2, 1, 0.2])
    zci = fns.get_zero_crossings_array_indices(vs, keep_adj_zeros=True)
    expected = np.array([0, 4, 5, 6, 7, 10, 12])
    assert np.array_equal(zci, expected)
    zci = fns.get_zero_crossings_array_indices(vs, keep_adj_zeros=False)
    expected = np.array([0, 4, 5, 6, 10, 12])
    assert np.array_equal(zci, expected), zci
    # no zeros
    vs = np.array([1, 2, 1, 2, -1, 1, 1, 1, 1, 0.3, 1, -1, 0.2, 1, 0.2])
    zci = fns.get_zero_crossings_array_indices(vs, keep_adj_zeros=False)
    expected = np.array([0, 4, 5, 11, 12])
    assert np.array_equal(zci, expected), zci
    vs = np.array([-1, -2, 1, 2, -1, 1, 1, 1, 1, 0.3, 1, -1, 0.2, 1, 0.2])
    zci = fns.get_zero_crossings_array_indices(vs, keep_adj_zeros=False)
    expected = np.array([0, 2, 4, 5, 11, 12])
    assert np.array_equal(zci, expected), zci



def test_put_array_in_2d_array():
    vals = np.arange(1, 5)
    sfs = np.array([1, 2, 3])
    expected_full = np.array([[0, 1, 2, 3, 4, 0, 0],
                              [0, 0, 1, 2, 3, 4, 0],
                              [0, 0, 0, 1, 2, 3, 4]])
    out = fns.put_array_in_2d_array(vals, sfs)
    assert np.array_equal(out, expected_full), out

    # expected = np.array([[0, 1, 2, 3],
    #                      [0, 0, 1, 2],
    #                      [0, 0, 0, 1]])
    out = fns.put_array_in_2d_array(vals, sfs, clip='end')
    assert np.array_equal(out, expected_full[:, :-3]), out
    out = fns.put_array_in_2d_array(vals, sfs, clip='start')
    assert np.array_equal(out, expected_full), out
    out = fns.put_array_in_2d_array(vals, sfs, clip='both')
    assert np.array_equal(out, expected_full[:, :-3]), out
    # neg shift
    vals = np.arange(4, 6)
    sfs = np.array([-1, 2])
    expected_full = np.array([[4, 5, 0, 0, 0],
                              [0, 0, 0, 4, 5],
                              ])
    out = fns.put_array_in_2d_array(vals, sfs, clip='none')
    assert np.array_equal(out, expected_full), out
    out = fns.put_array_in_2d_array(vals, sfs, clip='end')
    assert np.array_equal(out, expected_full[:, :-2]), out
    out = fns.put_array_in_2d_array(vals, sfs, clip='start')
    assert np.array_equal(out, expected_full[:, 1:]), out
    out = fns.put_array_in_2d_array(vals, sfs, clip='both')
    assert np.array_equal(out, expected_full[:, 1:-2]), out



def test_join_values_w_shifts():
    vals = np.arange(1, 5)
    sfs = np.array([1, 2, 3])
    expected = np.array([[1, 3, 5, 7, 4, 0, 0],
                         [1, 2, 4, 6, 3, 4, 0],
                         [1, 2, 3, 5, 2, 3, 4]])
    out = fns.join_values_w_shifts(vals, sfs)
    assert np.array_equal(out, expected), out
    expected = np.array([[ 1,  1,  1,  1, -4,  0,  0],
                        [ 1,  2,  2,  2, -3, -4,  0],
                        [ 1,  2,  3,  3, -2, -3, -4]])


def test_calc_step_fn_error():
    assert min(fns.calc_step_fn_vals_error([4, 4, 4, 4, 1, 1, 1, 1])) == 0.0
    assert min(fns.calc_step_fn_vals_error([4, 4, 4, 4, 1, 1, 1, 1], pow=2)) == 0.0

    assert min(fns.calc_step_fn_vals_error([4, 5, 4, 4, 1, 1, 2, 1])) == 3.0
    assert min(fns.calc_step_fn_vals_error([4, 5, 4, 4, 1, 1, 2, 1], pow=2)) == 1.0


def test_calc_step_fn_steps_val():
    vals = [4, 4, 4, 4, 1, 1, 1, 1]
    ind = np.argmin(fns.calc_step_fn_vals_error(vals))
    pre, post = fns.calc_step_fn_steps_vals(vals, ind)
    assert ind == 3
    assert pre == 4
    assert post == 1

    vals = [4, 5, 4, 4, 1, 1, 2, 1]
    ind = np.argmin(fns.calc_step_fn_vals_error(vals))
    pre, post = fns.calc_step_fn_steps_vals(vals, ind)
    assert ind == 3
    assert np.isclose(pre, 4.333333)
    assert post == 1.25


def test_roll_av_vals():
    expected = np.array([4, 4, 3, 2, 1, 1, 1, 1])
    assert np.sum(fns.calc_roll_av_vals([4, 4, 4, 4, 1, 1, 1, 1], steps=3) - expected) == 0
    expected = np.array([4, 4, 4, 4, 3, 2, 1, 1])
    assert np.sum(fns.calc_roll_av_vals([4, 4, 4, 4, 1, 1, 1, 1], steps=3, mode='backward') - expected) == 0
    expected = np.array([4, 4, 4, 3, 2, 1, 1, 1])
    assert np.sum(fns.calc_roll_av_vals([4, 4, 4, 4, 1, 1, 1, 1], steps=3, mode='centre') - expected) == 0


def test_interp2d():
    y = np.linspace(1, 10, 3)
    yf = np.linspace(0, 22, 5)
    f = np.arange(len(yf))[:, np.newaxis] * np.ones((len(yf), 10))
    f_interp = fns.interp2d(y, yf, f)
    assert np.isclose(f_interp[0][0], (y[0] - yf[0]) / (yf[1] - yf[0])), (f_interp[0][0], (y[0] - yf[0]) / (yf[1] - yf[0]))
    assert np.isclose(f_interp[1][0], 1), (f_interp[0][0], 1)
    assert len(f_interp) == 3
    assert len(f_interp[0]) == 10


def test_interp2d_2():
    f = np.array([[0, 0, 0],  # 0
                  [0, 1, 4],  # 5
                  [2, 6, 2],  # 10
                  [10, 10, 10]  # 30
                  ])
    yf = np.array([0, 1, 2, 3])

    y = np.array([0.5, 1, 2.2, 2.5])
    f_interp = fns.interp2d(y, yf, f)
    print(f_interp)
    assert f_interp[0][0] == 0
    assert f_interp[0][1] == 0.5
    assert f_interp[0][2] == 2.0
    assert f_interp[1][0] == f[1][0]
    assert f_interp[1][1] == f[1][1]
    assert f_interp[1][2] == f[1][2]
    assert np.isclose(f_interp[2][0], f[2][0] + 8 * 0.2)
    assert np.isclose(f_interp[3][2], 6.)


def test_interp2d_at_edge():
    f = np.array([[0, 0, 0],  # 0
                  [10, 10, 10]  # 30
                  ])
    xf = np.array([0, 3])

    x = np.array([0.0, 3.0])
    f_interp = fns.interp2d(x, xf, f)
    print(f_interp)
    assert f_interp[0][0] == 0
    assert f_interp[1][0] == 10.


def test_interp_left():
    x0 = [0, 1, 5]
    x = [0, 2, 6]
    y = [1.5, 2.5, 3.5]
    y_new = fns.interp_left(x0, x, y)
    expected = np.array([1.5, 1.5, 2.5])
    assert np.isclose(y_new, expected).all(), y_new

    x0 = [0, 2, 6]
    y_new = fns.interp_left(x0, x, y)
    expected = np.array([1.5, 2.5, 3.5])
    assert np.isclose(y_new, expected).all(), y_new
    x0 = [-1, 2, 6]
    with pytest.raises(AssertionError):
        y_new = fns.interp_left(x0, x, y)


if __name__ == '__main__':

    # test_interp2d()
    test_interp2d_at_edge()
    # test_put_array_in_2d_array()
    # test_fa_spectrum_conversion()
    # test_determine_peaks_only_series_with_sine_wave()
    # test_determine_peaks_only_series_with_triangle_series()
    # test_determine_peaks_only_series_with_ground_motion()
    # test_determine_peaks_only_series_with_a_double_peak_and_offset()
    # test_determine_peaks_only_series_with_nonchanging_values()
    # test_determine_peaks_only_series_with_non_zero_end()
