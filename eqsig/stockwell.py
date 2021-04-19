import numpy as np
import eqsig.multiple


def plot_stock(splot, asig, norm_x=False, norm_all=False, interp=False, cmap=None, vmin=None, vmax=None):
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
    min_freq = 1.0 / (len(asig.swtf) * asig.dt * 2)

    extent = (0, asig.time[-1], min_freq, max_freq)

    kwargs = {}
    if cmap is not None:
        kwargs['cmap'] = cmap  # rainbow, gnuplot2, plasma
    if vmin is not None:
        kwargs['vmin'] = vmin
    if vmax is not None:
        kwargs['vmax'] = vmax
    return splot.imshow(b, aspect='auto', extent=extent, **kwargs)


def plot_tifq_vals(subplot, tifq_vals, dt, norm_all=False, norm_x=False, cmap=None):
    """
    Plots a time frequency spectrum (e.g. Stockwell transform)

    Same as `plot_stock` but takes in values (tifq_vals) and time step (dt) instead of
    AccSignal object.

    Parameters
    ----------
    subplot: matplotlib.pyplot.subplot
    tifq_vals: 2-d array
        time-frequency values
    dt: float
        time step
    norm_x: bool, default=False
        If true then all values at a time step are normalised by the maximum value
    norm_all: bool, default=False
        If true the all values are normalised by the maximum value
    Returns
    -------
        None
    """

    b = tifq_vals
    if norm_all:
        b = b / np.max(b)
    if norm_x:
        b = b / np.max(b, axis=0)
    max_freq = 1 / dt / 2
    # if interp:
    #     max_freq /= 2
    min_freq = 1.0 / (len(tifq_vals) * dt)

    total_time = len(tifq_vals[0]) * dt
    extent = (0, total_time, min_freq, max_freq)

    kwargs = {}
    if cmap is not None:
        kwargs['cmap'] = cmap  # rainbow, gnuplot2, plasma
    subplot.imshow(b, aspect='auto', extent=extent, **kwargs)


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


def transform_w_scipy_fft(acc, interp=False):
    """
    Performs a Stockwell transform on an array

    Assumes m = 1, p = 1

    :param acc: array_like
    :return:
    """
    from scipy.linalg import toeplitz
    from scipy.fftpack import fft, ifft  # Try use scipy.fft

    acc_db = acc
    n_d2 = int(len(acc) / 2)
    n_factor = 2 * n_d2
    gaussian = generate_gaussian(n_d2)

    fa = fft(acc_db, n_factor, overwrite_x=True)
    diag_con = toeplitz(np.conj(fa[:n_d2 + 1]), fa)
    diag_con = diag_con[1:n_d2 + 1, :]  # first line is zero frequency
    stock = np.flipud(ifft(diag_con * gaussian, axis=1))

    return stock


def transform(acc, interp=False):
    """
    Performs a Stockwell transform on an array

    Assumes m = 1, p = 1

    :param acc: array_like
    :return:
    """
    from scipy.linalg import toeplitz

    acc_db = acc
    n_d2 = int(len(acc) / 2)
    n_factor = 2 * n_d2
    gaussian = generate_gaussian(n_d2)

    fa = np.fft.fft(acc_db, n_factor)
    diag_con = toeplitz(np.conj(fa[:n_d2 + 1]), fa)
    diag_con = diag_con[1:n_d2 + 1, :]  # first line is zero frequency

    stock = np.flipud(np.fft.ifft(diag_con * gaussian, axis=1))

    return stock


def transform_slow(acc, interp=False, ith=0):
    """
    Performs a Stockwell transform on an array

    Assumes m = 1, p = 1

    :param acc: array_like
    :return:
    """
    from scipy.linalg import toeplitz

    acc_db = acc
    n_d2 = int(len(acc) / 2)
    n_factor = 2 * n_d2
    gaussian = generate_gaussian(n_d2)

    fa = np.fft.fft(acc_db, n_factor)
    diag_con = toeplitz(np.conj(fa[:n_d2 + 1]), fa)
    diag_con = diag_con[1:n_d2 + 1, :]  # first line is zero frequency
    skip_is = 0  # can skip more low frequencies since they tend to be zero
    aa = diag_con[skip_is:, :] * gaussian[skip_is:, :]
    upstock = np.zeros_like(aa)
    aa = aa[:-ith, :]
    upstock[:-ith, :] = np.fft.ifft(aa, axis=1)
    stock = np.flipud(upstock)
    return stock


def dep_itransform(stock):
    """Performs an inverse Stockwell Transform"""
    from scipy.fftpack import ifft  # Try use scipy.fft
    return np.real(ifft(np.sum(stock, axis=1)))


def itransform(stock):
    """Performs an inverse Stockwell Transform"""
    ss = np.sum(stock, axis=1)
    n = 2 * len(ss)
    fas_ss = np.zeros(2 * len(ss), dtype=complex)
    fas_ss[1:n // 2] = np.flip(np.conj(ss[1:]), axis=0)
    fas_ss[n // 2 + 1:] = ss[1:]

    acc_new = np.fft.ifft(fas_ss)
    npts = int(np.ceil(2 ** (np.log(n) / np.log(2))))
    return np.real(acc_new[:npts])


def get_max_stockwell_freq(asig):
    if not hasattr(asig, "swtf"):
        asig.swtf = transform(asig.values)
    points = len(asig.swtf)
    freqs = np.arange(1, points + 1) / (2 * points * asig.dt)
    freqs = np.flipud(freqs)
    indy_max = np.argmax(abs(asig.swtf), axis=0)
    max_f = np.take(freqs, indy_max)
    return max_f


def get_max_tifq_vals_freq(tifq_values, dt):
    points = len(tifq_values)
    freqs = np.arange(1, points + 1) / (2 * points * dt)
    freqs = np.flipud(freqs)
    indy_max = np.argmax(abs(tifq_values), axis=0)
    max_f = np.take(freqs, indy_max)
    return max_f


def plot_max_freq_azimuth(splot, asig1, asig2, max_f=None, norm=False, r_steps=90):
    r = np.linspace(0, 180, r_steps)
    theta = np.pi * r / 180
    ys = []
    for i in range(len(r)):
        asigc = eqsig.multiple.combine_at_angle(asig1, asig2, r[i])
        freqs = np.clip(get_max_stockwell_freq(asigc), None, max_f)
        if norm:
            norm_freqs = (freqs - np.min(freqs)) / (np.max(freqs) - np.min(freqs))
            ys.append(norm_freqs)
        else:
            ys.append(freqs)

    ys = np.array(ys).T
    splot.pcolormesh(theta[np.newaxis], asig1.time[:, np.newaxis], ys)
    splot.set_thetamin(0)
    splot.set_thetamax(180)


if __name__ == '__main__':
    from tests import conftest
    from eqsig import load_signal
    import matplotlib.pyplot as plt
    import eqsig

    # asig = load_signal(conftest.TEST_DATA_DIR + "test_motion_dt0p01.txt", astype="acc_sig")
    # asig = eqsig.interp_to_approx_dt(asig, 0.05)
    # bf, sps = plt.subplots(nrows=3)
    # plot_stock(sps[0], asig, norm_x=True, interp=True, cmap='plasma')
    # vals = itransform(asig.swtf)
    # sps[1].plot(asig.time, vals)
    # asig1 = eqsig.AccSignal(vals, asig.dt, lw=1)
    # sps[1].plot(asig.time, asig.values, lw=1)
    # sps[2].plot(asig1.fa_frequencies, asig1.fa_spectrum, lw=1)
    # sps[2].plot(asig.fa_frequencies, asig.fa_spectrum, lw=1)
    # plt.show()
    # f = 5.0
    # t = np.linspace(0, 10, 5001)
    #
    # w = chirp(t, f0=12.5, f1=2.5, t1=10, method='linear')
    #
    # stock = transform(w)
    # fig, ax = plt.subplots(2, 1, sharex=False)
    # ax[0].plot(t, w)
    # ax[0].set(ylabel='amplitude')
    # ax[1].imshow(np.abs(stock), origin='lower')
    # ax[1].axis('tight')
    # ax[1].set(xlabel='samples', ylabel='frequency')
    # plt.show()
