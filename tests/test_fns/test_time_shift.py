import numpy as np

import eqsig


def test_put_array_in_2d_array():
    vals = np.arange(1, 5)
    sfs = np.array([1, 2, 3])
    expected_full = np.array([[0, 1, 2, 3, 4, 0, 0],
                              [0, 0, 1, 2, 3, 4, 0],
                              [0, 0, 0, 1, 2, 3, 4]])
    out = eqsig.fns.time_shift.put_array_in_2d_array(vals, sfs)
    assert np.array_equal(out, expected_full), out

    # expected = np.array([[0, 1, 2, 3],
    #                      [0, 0, 1, 2],
    #                      [0, 0, 0, 1]])
    out = eqsig.fns.time_shift.put_array_in_2d_array(vals, sfs, clip='end')
    assert np.array_equal(out, expected_full[:, :-3]), out
    out = eqsig.fns.time_shift.put_array_in_2d_array(vals, sfs, clip='start')
    assert np.array_equal(out, expected_full), out
    out = eqsig.fns.time_shift.put_array_in_2d_array(vals, sfs, clip='both')
    assert np.array_equal(out, expected_full[:, :-3]), out
    # neg shift
    vals = np.arange(4, 6)
    sfs = np.array([-1, 2])
    expected_full = np.array([[4, 5, 0, 0, 0],
                              [0, 0, 0, 4, 5],
                              ])
    out = eqsig.fns.time_shift.put_array_in_2d_array(vals, sfs, clip='none')
    assert np.array_equal(out, expected_full), out
    out = eqsig.fns.time_shift.put_array_in_2d_array(vals, sfs, clip='end')
    assert np.array_equal(out, expected_full[:, :-2]), out
    out = eqsig.fns.time_shift.put_array_in_2d_array(vals, sfs, clip='start')
    assert np.array_equal(out, expected_full[:, 1:]), out
    out = eqsig.fns.time_shift.put_array_in_2d_array(vals, sfs, clip='both')
    assert np.array_equal(out, expected_full[:, 1:-2]), out


def test_join_values_w_shifts():
    vals = np.arange(1, 5)
    sfs = np.array([1, 2, 3])
    expected = np.array([[1, 3, 5, 7, 4, 0, 0],
                         [1, 2, 4, 6, 3, 4, 0],
                         [1, 2, 3, 5, 2, 3, 4]])
    out = eqsig.fns.time_shift.join_values_w_shifts(vals, sfs)
    assert np.array_equal(out, expected), out


def test_calc_step_fn_error():
    assert min(eqsig.fns.average.calc_step_fn_vals_error([4, 4, 4, 4, 1, 1, 1, 1])) == 0.0
    assert min(eqsig.fns.average.calc_step_fn_vals_error([4, 4, 4, 4, 1, 1, 1, 1], pow=2)) == 0.0

    assert min(eqsig.fns.average.calc_step_fn_vals_error([4, 5, 4, 4, 1, 1, 2, 1])) == 3.0
    assert min(eqsig.fns.average.calc_step_fn_vals_error([4, 5, 4, 4, 1, 1, 2, 1], pow=2)) == 1.0


def test_calc_step_fn_steps_val():
    vals = [4, 4, 4, 4, 1, 1, 1, 1]
    ind = np.argmin(eqsig.fns.average.calc_step_fn_vals_error(vals))
    pre, post = eqsig.fns.average.calc_step_fn_steps_vals(vals, ind)
    assert ind == 3
    assert pre == 4
    assert post == 1

    vals = [4, 5, 4, 4, 1, 1, 2, 1]
    ind = np.argmin(eqsig.fns.average.calc_step_fn_vals_error(vals))
    pre, post = eqsig.fns.average.calc_step_fn_steps_vals(vals, ind)
    assert ind == 3
    assert np.isclose(pre, 4.333333)
    assert post == 1.25
