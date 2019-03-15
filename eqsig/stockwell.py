import numpy as np
import scipy
import scipy.fftpack
import scipy.signal


def plot_stock(splot, asig, norm_x=False):
    """
    Plots the Stockwell transform of an acceleration signal

    :param splot:
    :param asig:
    :return:
    """
    import matplotlib.ticker as ticker
    if not hasattr(asig, "stockwell"):
        asig.swtf = transform(asig.values)
    points = int(asig.npts)
    freqs = np.flipud(np.arange(points) / (points * asig.dt))
    b = abs(asig.swtf)
    b = b[int(len(b) / 2):]
    if norm_x:
        b = b / np.max(b, axis=0)
    # max_x = len(b[0])
    # max_y = len(b)
    # print(max_x, max_y)
    # max_x = 2000
    # extent = (-0.5, max_x - 0.5, max_y - 0.5, -0.5)
    splot.imshow(b, aspect='auto')
    fi = np.arange(len(freqs))
    xticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * asig.dt / 2))
    s10 = np.ceil(asig.time[-1] / 10)
    if s10 < 4:
        step = 5
    else:
        step = 10
    x_majors = np.arange(0, 10 * s10, step) * 2 / asig.dt
    splot.xaxis.set_major_formatter(xticks)
    splot.set_xticks(x_majors)

    # y ticks
    yticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format((len(asig.swtf) / 2 - y) / (len(asig.values) * asig.dt)))

    splot.yaxis.set_major_formatter(yticks)
    max_freq = 1 / asig.dt / 2
    s10 = np.ceil(max_freq / 10)
    if s10 < 4:
        step = 5
    else:
        step = 10
    y_majors = np.arange(0, 10 * s10, step)
    print(y_majors)
    splot.set_yticks(np.flipud(y_majors) * len(asig.values) * asig.dt)


def plot_fas_at_time(splot, asig, time):
    """Plots the Fourier amplitude spectrum at a time based on a Stockwell transform"""
    if not hasattr(asig, "stockwell"):
        asig.swtf = transform(asig.values)
    indy = int(time / asig.dt)
    points = len(asig.swtf)
    freqs = np.arange(1, points + 1) / (points * asig.dt)
    splot.plot(freqs, np.flipud(abs(asig.swtf[:, indy])))


def generate_gaussian(n_d2):
    """create a gaussian distribution"""
    f_half = np.arange(0, n_d2 + 1, 1) / (2 * n_d2)
    f = np.concatenate((f_half, np.flipud(-f_half[1:-1])))
    p = 2 * np.pi * np.outer(f, 1. / f_half[1:])
    return np.exp(-p ** 2 / 2).transpose()  # * np.exp(1j * p / n_d2)


def transform(acc):
    """
    Performs a Stockwell transform on an array

    Assumes m = 1, p = 1

    :param acc: array_like
    :return:
    """
    # Interpolate here because function drops a time step
    t_int = np.arange(len(acc))
    t_db = np.arange(2 * len(acc)) / 2
    acc_db = np.interp(t_db, t_int, acc)

    n_d2 = int(len(acc))
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
    plot_stock(sps, asig, norm_x=True)
    plt.show()