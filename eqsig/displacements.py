import numpy as np
import scipy.integrate


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

    if trap is False:
        vel_inc = acceleration * dt
        velocity = np.cumsum(vel_inc)
        np.insert(velocity, 0, 0)
        velocity = velocity[:-1]
        disp_inc = velocity * dt
        displacement = np.cumsum(disp_inc)
        np.insert(displacement, 0, 0)
        displacement = displacement[:-1]
    else:

        velocity = scipy.integrate.cumtrapz(acceleration, dx=dt, initial=0)
        displacement = scipy.integrate.cumtrapz(velocity, dx=dt, initial=0)

    return velocity, displacement
