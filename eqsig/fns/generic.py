import numpy as np
import eqsig


def interp2d(x, xf, f):
    """
    Can interpolate a table to get an array of values in 2D

    Parameters
    ----------
    x: array_like
        1d array of values to be interpolated
    xf: 1d array of values
    f: array_like
        2d array of function values size=(len(x), n)

    Returns
    -------

    Examples
    --------
    >>> f = np.array([[0, 0, 0],
    >>>              [0, 1, 4],
    >>>              [2, 6, 2],
    >>>              [10, 10, 10]
    >>>              ])
    >>> xf = np.array([0, 1, 2, 3])

    >>> x = np.array([0.5, 1, 2.2, 2.5])
    >>> f_interp = interp2d(x, xf, f)
    >>> print(f_interp[0][0])
    0.0
    >>> print(f_interp[0][1])
    0.5
    >>> print(f_interp[0][2])
    2.0
    """
    ind = np.argmin(np.abs(x[:, np.newaxis] - xf), axis=1)
    x_ind = xf[ind]
    ind0 = np.where(x_ind > x, ind - 1, ind)
    ind1 = np.where(x_ind > x, ind, ind + 1)
    ind0 = np.clip(ind0, 0, None)
    ind1 = np.clip(ind1, None, len(xf) - 1)
    f0 = f[ind0]
    f1 = f[ind1]
    a0 = xf[ind0]
    a1 = xf[ind1]
    denom = (a1 - a0)
    denom_adj = np.clip(denom, 1e-10, None)  # to avoid divide by zero warning
    s0 = np.where(denom > 0, (x - a0) / denom_adj, 1)  # if denom less than 0, then out of bounds
    s1 = 1 - s0
    return s1[:, np.newaxis] * f0 + s0[:, np.newaxis] * f1


def interp_left(x0, x, y=None):
    """
    Interpolation takes the lower value

    Parameters
    ----------
    x0: array_like
        Values to be interpolated on x-axis
    x: array_like
        Existing values on x-axis
    y: array_like
        Existing y-axis values
    Returns
    -------

    """
    if y is None:
        y = np.arange(len(x))
    else:
        y = np.array(y)
    is_scalar = False
    if not hasattr(x0, '__len__'):
        is_scalar = True
        x0 = [x0]
    assert min(x0) >= x[0], (min(x0), x[0])
    inds = np.searchsorted(x, x0, side='right') - 1
    if is_scalar:
        return y[inds][0]
    return y[inds]


def remove_poly(values, poly_fit=0):
    """
    Calculates best fit polynomial and removes it from the record
    """

    x = np.linspace(0, 1.0, len(values))
    cofs = np.polyfit(x, values, poly_fit)
    y_cor = 0 * x
    for co in range(len(cofs)):
        mods = x ** (poly_fit - co)
        y_cor += cofs[co] * mods

    return values - y_cor


def gen_ricker_wavelet_asig(omega, t0, duration, dt):
    """
    Generates an acceleration time series that is a Ricker wavelet

    Parameters
    ----------
    omega
    t0
    duration: float
        Total duration of motion
    dt: float
        Time step of motion

    Returns
    -------

    """
    t = np.arange(0, duration, dt)

    vel_amp = (2.0 * (np.pi ** 2.0) * (omega ** 2.0) * ((t - t0) ** 2.0) - 1.0) * np.exp(
        - (np.pi ** 2.0) * (omega ** 2.0) * (t - t0) ** 2.0)
    acc = np.zeros_like(vel_amp)
    acc[1:] = np.diff(vel_amp) / dt
    return eqsig.AccSignal(acc, dt)
