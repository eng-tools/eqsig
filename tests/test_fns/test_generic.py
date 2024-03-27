import numpy as np

from eqsig import fns
import pytest


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

