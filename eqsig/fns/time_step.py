import numpy as np

import eqsig


def time_series_from_motion(motion, dt):
    npts = len(motion)
    return np.linspace(0, dt * (npts + 1), npts)


def interp_array_to_approx_dt(values, dt, target_dt=0.01, even=True):
    """
    Interpolate an array of values to a new time step

    Similar to ``interp_to_approx_dt``

    Only a target time step is provided and the algorithm determines
     what time step is best to minimise loss of data from aliasing

    Parameters
    ----------
    values: array_like
        values of time series
    dt: float
        Time step
    target_dt: float
        Target time step
    even: bool
        If true then forces the number of time steps to be an even number

    Returns
    -------
    new_values: array_like
        Interpolated value of time series
    new_dt: float
        New time step of interpolate time series
    """
    factor = dt / target_dt
    if factor == 1:
        pass
    elif factor > 1:
        factor = int(np.ceil(factor))
    else:
        factor = 1 / np.floor(1 / factor)
    t_int = np.arange(len(values))
    new_npts = factor * len(values)
    if even:
        new_npts = 2 * int(new_npts / 2)
    t_db = np.arange(new_npts) / factor
    acc_interp = np.interp(t_db, t_int, values)
    return acc_interp, dt / factor


def interp_to_approx_dt(asig, target_dt=0.01, even=True):
    """
    Interpolate a signal to a new time step

    Only a target time step is provided and the algorithm determines
     what time step is best to minimise loss of data from aliasing

    Parameters
    ----------
    asig: eqsig.AccSignal
        Acceleration time series object
    target_dt: float
        Target time step
    even: bool
        If true then forces the number of time steps to be an even number

    Returns
    -------
    new_asig: eqsig.AccSignal
        Acceleration time series object of interpolated time series
    """
    acc_interp, dt_interp = interp_array_to_approx_dt(asig.values, asig.dt, target_dt=target_dt, even=even)
    return eqsig.AccSignal(acc_interp, dt_interp)


def resample_to_approx_dt(asig, target_dt=0.01, even=True):
    """
    Resample a signal assuming periodic to a new time step

    Only a target time step is provided and the algorithm determines
     what time step is best to minimise loss of data from aliasing

    Parameters
    ----------
    asig: eqsig.AccSignal
        Acceleration time series object
    target_dt: float
        Target time step
    even: bool
        If true then forces the number of time steps to be an even number

    Returns
    -------

    """
    from scipy.signal import resample
    factor = asig.dt / target_dt
    if factor == 1:
        pass
    elif factor > 1:
        factor = int(np.ceil(factor))
    else:
        factor = 1 / np.floor(1 / factor)
    new_npts = factor * asig.npts
    if even:
        new_npts = 2 * int(new_npts / 2)
    acc_interp = resample(asig.values, new_npts)
    return eqsig.AccSignal(acc_interp, asig.dt / factor)
