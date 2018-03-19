import numpy as np


def significant_duration(motion, dt):
    """
    Computes the significant duration using cumulative acceleration according to Trifunac and Brady (1975).
    :param motion:
    :param dt:
    :return:
    """

    acc2 = motion ** 2
    cum_acc2 = np.cumsum(acc2)
    ind2 = np.where((cum_acc2 > 0.05 * cum_acc2[-1]) &
                    (cum_acc2 < 0.95 * cum_acc2[-1]))
    start_time = ind2[0][0] * dt
    end_time = ind2[0][-1] * dt

    return start_time, end_time


def calculate_peak(motion):
    """Calculates the peak absolute response"""
    return max(abs(min(motion)), max(motion))