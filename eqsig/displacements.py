
import numpy as np
import scipy


def velocity_and_displacement_from_acceleration(acceleration, dt, trap=True):
    """
    Computes the velocity and acceleration of an acceleration time series, using numerical integration.

    :param acceleration: acceleration time series
    :param dt: time step
    :param trap: if True then uses trapezium integration
    :return:
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
