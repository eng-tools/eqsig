import numpy as np
import scipy
import scipy.fftpack
import scipy.signal


def plot_stock(splot, asig, norm_x=False, norm_all=False, interp=False, cmap=None):
    """
    Plots the Stockwell transform of an acceleration signal

    Parameters
    ----------
    splot: matplotlib.pyplot.subplot
    asig: eqsig.Signal
    norm_x: bool, default=False
        If true then all values at a time step are normalised by the maximum value
    norm_all: bool, default=False
        If true the all values are normalised by the maximum value
    Returns
    -------
        None
    """
    if not hasattr(asig, "swtf"):
        asig.swtf = transform(asig.values, interp=interp)

    b = abs(asig.swtf)
    if interp:
        b = b[int(len(b) / 2):]
    if norm_all:
        b = b / np.max(b)
    if norm_x:
        b = b / np.max(b, axis=0)
    max_freq = 1 / asig.dt / 2
    # if interp:
    #     max_freq /= 2

    extent = (0, asig.time[-1], 0, max_freq)

    kwargs = {}
    if cmap is not None:
        kwargs['cmap'] = cmap  # rainbow, gnuplot2, plasma
    splot.imshow(b, aspect='auto', extent=extent, **kwargs)


def get_stockwell_freqs(asig):
    return (len(asig.swtf) / 2 - np.arange(0, int(len(asig.swtf) / 2))) / (len(asig.values) * asig.dt)


def get_stockwell_times(asig):
    return np.arange(0, len(asig.swtf[0])) * asig.dt / 2


def plot_fas_at_time(splot, asig, time):
    """Plots the Fourier amplitude spectrum at a time based on a Stockwell transform"""
    if not hasattr(asig, "swtf"):
        asig.swtf = transform(asig.values)
    indy = int(time / asig.dt) * 2  # 2 since Stockwell has twice as many points
    points = len(asig.swtf)
    freqs = np.arange(1, points + 1) / (points * asig.dt)
    splot.plot(freqs, np.flipud(abs(asig.swtf[:, indy])))


def plot_windowed_fas_at_time(splot, asig, time, time_window=3):
    """Plots the time averaged Fourier amplitude spectrum at a time based on a Stockwell transform"""
    if not hasattr(asig, "swtf"):
        asig.swtf = transform(asig.values)
    indy = int(time / asig.dt) * 2  # 2 since Stockwell has twice as many points
    window = int(time_window / (asig.time[-1] / asig.npts))
    s_i = max(int(indy - window / 20), 0)
    f_i = max(int(indy + window / 20), len(asig.swtf))
    points = len(asig.swtf)
    freqs = np.arange(1, points + 1) / (points * asig.dt)
    splot.plot(freqs, np.flipud(np.mean(abs(asig.swtf[:, s_i:f_i]), axis=1)))


def generate_gaussian(n_d2):
    """create a gaussian distribution"""
    f_half = np.arange(0, n_d2 + 1, 1) / (2 * n_d2)
    f = np.concatenate((f_half, np.flipud(-f_half[1:-1])))
    p = 2 * np.pi * np.outer(f, 1. / f_half[1:])
    return np.exp(-p ** 2 / 2).transpose()  # * np.exp(1j * p / n_d2)


def transform(acc, interp=False):
    """
    Performs a Stockwell transform on an array

    Assumes m = 1, p = 1

    :param acc: array_like
    :return:
    """
    # Interpolate here because function drops a time step
    if interp:
        t_int = np.arange(len(acc))
        t_db = np.arange(2 * len(acc)) / 2
        acc_db = np.interp(t_db, t_int, acc)
        n_d2 = int(len(acc))
    else:
        acc_db = acc
        n_d2 = int(len(acc) / 2)
    n_factor = 2 * n_d2
    gaussian = generate_gaussian(n_d2)
    # st0 = np.mean(acc) * np.ones(n_factor)

    fa = scipy.fftpack.fft(acc_db, n_factor, overwrite_x=True)
    diag_con = scipy.linalg.toeplitz(np.conj(fa[:n_d2 + 1]), fa)
    diag_con = diag_con[1:n_d2 + 1, :]  # first line is zero frequency
    skip_is = 0  # can skip more low frequencies since they tend to be zero
    stock = np.flipud(scipy.fftpack.ifft(diag_con[skip_is:, :] * gaussian[skip_is:, :], axis=1))

    # stock = np.insert(stock, 0, st0, 0)
    for i in range(skip_is):
        stock = np.insert(stock, 0, 0, 0)

    return stock


def itransform(stock):
    """Performs an inverse Stockwell Transform"""
    return np.real(scipy.fftpack.ifft(np.sum(stock, axis=1)))


if __name__ == '__main__':
    from tests import conftest
    from eqsig import load_signal
    import matplotlib.pyplot as plt
    import eqsig

    asig = load_signal(conftest.TEST_DATA_DIR + "test_motion_dt0p01.txt", astype="acc_sig")
    asig = eqsig.interp_to_approx_dt(asig, 0.05)
    bf, sps = plt.subplots()
    plot_stock(sps, asig, norm_x=True, interp=True, cmap='plasma')
    # , times = (10, 30), freqs = (0, 7)
    plt.show()
