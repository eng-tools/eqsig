import numpy as np
import scipy.integrate


def calc_velo_and_disp_from_accel_arr(acceleration, dt, trap=True):
    """
    Computes the velocity and acceleration of an acceleration time series, using numerical integration.

    Parameters
    ----------
    acceleration: array_like
        acceleration time series
    dt: float
        time step
    trap: bool (default=True)
        if True then uses trapezium integration

    Returns
    -------
    velocity: array_like (len=len(acceleration))
        velocity time series
    displacement: array_like (len=len(acceleration))
        displacement time series
    """

    if trap is False:
        velocity = np.zeros(len(acceleration) + 1)
        velocity[1:] = acceleration * dt  # computes the increments
        np.cumsum(velocity, out=velocity)  # passed into original array for efficiency
        # np.insert(velocity, 0, 0)
        # velocity = velocity[:-1]
        displacement = velocity * dt  # computes the increments
        np.cumsum(displacement, out=displacement)
        velocity = velocity[:-1]
        displacement = displacement[:-1]
    else:

        velocity = scipy.integrate.cumtrapz(acceleration, dx=dt, initial=0)
        displacement = scipy.integrate.cumtrapz(velocity, dx=dt, initial=0)

    return velocity, displacement


def velocity_and_displacement_from_acceleration(acceleration, dt, trap=True):
    """
    Computes the velocity and acceleration of an acceleration time series, using numerical integration.

    Parameters
    ----------
    acceleration: array_like
        acceleration time series
    dt: float
        time step
    trap: bool (default=True)
        if True then uses trapezium integration

    Returns
    -------
    velocity: array_like (len=len(acceleration))
        velocity time series
    displacement: array_like (len=len(acceleration))
        displacement time series
    """
    return calc_velo_and_disp_from_accel_arr(acceleration, dt, trap=trap)