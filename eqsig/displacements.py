
import numpy as np


def velocity_and_displacement_from_acceleration(acceleration, dt, forward=True):
    """
    Computes the velocity and acceleration of an acceleration time series, using numerical integration.
    :param acceleration: acceleration time series
    :param dt: time step
    :param forward: if True then uses forward integration
    :return:
    """
    npts = len(acceleration)
    velocity = np.zeros(npts)
    displacement = np.zeros(npts)

    for i in range(npts - 1):
        velocity[i + 1] = ((acceleration[i + 1] + acceleration[i]) * dt / 2 + velocity[i])
        displacement[i + 1] = ((velocity[i + 1] + velocity[i]) * dt / 2 + displacement[i])
        if forward is False:
            velocity[i + 1] = ((acceleration[i + 1]) * dt + velocity[i])
            displacement[i + 1] = ((velocity[i + 1]) * dt + displacement[i])

    return velocity, displacement
