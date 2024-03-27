import numpy as np


def get_sig_freq_range(asig, ratio=15):
    indices = get_sig_array_indexes_range(asig.smooth_fa_spectrum, ratio=ratio)
    return np.take(asig.smooth_fa_frequencies, indices)


def get_sig_array_indexes_range(fas1_smooth, ratio=15):
    max_fas1 = max(fas1_smooth)
    lim_fas = max_fas1 / ratio
    indys = np.where(fas1_smooth > lim_fas)[0]
    return indys[0], indys[-1]


def calc_fourier_moment(asig, n):
    """
    Original source unknown.

    See :cite:`Rathje:2008va`

    Parameters
    ----------
    asig
    n

    Returns
    -------

    """
    return 2 * np.trapz((2 * np.pi * asig.fa_frequencies) ** n * asig.fa_spectrum ** 2, x=asig.fa_frequencies)


def get_bandwidth_boore_2003(asig):
    m0 = calc_fourier_moment(asig, 0)
    m2 = calc_fourier_moment(asig, 2)
    m4 = calc_fourier_moment(asig, 4)
    return np.sqrt(m2 ** 2 / (m0 * m4))


def calc_smooth_fa_spectrum(fa_frequencies, fa_spectrum, smooth_fa_frequencies=None, band=40):
    """
    Calculates the smoothed Fourier Amplitude Spectrum using the method by Konno and Ohmachi (1998)

    Note: different order of inputs than generate_smooth_fa_spectrum

    Parameters
    ----------
    smooth_fa_frequencies: array_like
        Frequencies to compute the smoothed amplitude
    fa_frequencies: array_like
        Frequencies of the Fourier amplitude spectrum
    fa_spectrum: array_like
        Amplitudes of the Fourier amplitude spectrum
    band:
        window parameter

    Returns
    -------
    smoothed_fa_spectrum: array_like
        Amplitudes of smoothed Fourier spectrum at specified frequencies
    """

    if fa_frequencies[0] == 0:
        fa_frequencies = fa_frequencies[1:]
        fa_spectrum = fa_spectrum[1:]
    if smooth_fa_frequencies is None:
        smooth_fa_frequencies = fa_frequencies

    amp_array = band * np.log10(fa_frequencies[:, np.newaxis] / smooth_fa_frequencies[np.newaxis, :])
    wb_vals = (np.sin(amp_array) / amp_array) ** 4
    wb_vals = np.where(amp_array == 0, 1, wb_vals)
    wb_vals /= np.sum(wb_vals, axis=0)

    return np.sum(abs(fa_spectrum)[:, np.newaxis] * wb_vals, axis=0)
    # return np.dot(abs(fa_spectrum), wb_vals)


def generate_smooth_fa_spectrum(smooth_fa_frequencies, fa_frequencies, fa_spectrum, band=40):
    """Deprecated - use calc_smooth_fa_spectrum"""
    return calc_smooth_fa_spectrum(fa_frequencies, fa_spectrum, smooth_fa_frequencies, band=band)


def calc_smoothing_matrix_konno_1998(fa_frequencies, smooth_fa_frequencies=None, band=40):
    """
    Calculates the smoothing matrix for computing the smoothed Fourier Amplitude Spectrum
        using the method by Konno and Ohmachi 1998

    Parameters
    ----------
    fa_frequencies: array_like
        Frequencies of FAS
    smooth_fa_frequencies: array_like
        Frequencies that smooth FAS should be computed at
    band: int
        Bandwidth of smoothing function

    Returns
    -------
    2d-array_like
    """

    if fa_frequencies[0] == 0:
        fa_frequencies = fa_frequencies[1:]

    if smooth_fa_frequencies is None:
        smooth_fa_frequencies = fa_frequencies

    amp_array = band * np.log10(fa_frequencies[:, np.newaxis] / smooth_fa_frequencies[np.newaxis, :])
    wb_vals = (np.sin(amp_array) / amp_array) ** 4
    wb_vals = np.where(amp_array == 0, 1, wb_vals)
    wb_vals /= np.sum(wb_vals, axis=0)
    return wb_vals


def calc_smooth_fa_spectrum_w_custom_matrix(asig, smooth_matrix):
    """
    Calculates the smoothed Fourier Amplitude Spectrum
    using a custom filter

    """
    return np.dot(abs(asig.fa_spectrum[1:]), smooth_matrix)


def generate_fa_spectrum(sig, n_pad=True):
    """
    Produces the Fourier amplitude spectrum

    Parameters
    ----------
    sig: eqsig.Signal

    Returns
    -------
    fa_spectrum: complex array_like
        Complex values of the spectrum
    fa_frequencies: array_like
        Frequencies of the spectrum
    """

    npts = sig.npts
    if n_pad:
        n_factor = 2 ** int(np.ceil(np.log2(npts)))
        fa = np.fft.fft(sig.values, n=n_factor)
        points = int(n_factor / 2)
        assert len(fa) == n_factor
    else:
        fa = np.fft.fft(sig.values)
        points = int(sig.npts / 2)
    fa_spectrum = fa[range(points)] * sig.dt
    fa_frequencies = np.arange(points) / (2 * points * sig.dt)
    return fa_spectrum, fa_frequencies


def calc_fa_spectrum(sig, n=None, p2_plus=None):
    """
    Produces the Fourier amplitude spectrum

    Parameters
    ----------
    sig: eqsig.Signal

    Returns
    -------
    fa_spectrum: complex array_like
        Complex values of the spectrum
    fa_frequencies: array_like
        Frequencies of the spectrum
    """
    npts = sig.npts
    if p2_plus is not None or n is not None:
        if n is not None:
            n_vals = n
        else:
            n_vals = 2 ** int(np.ceil(np.log2(npts)) + p2_plus)
        fa = np.fft.fft(sig.values, n=n_vals)
        points = int(n_vals / 2)
        assert len(fa) == n_vals
    else:
        fa = np.fft.fft(sig.values)
        points = int(sig.npts / 2)
    fa_spectrum = fa[range(points)] * sig.dt
    fa_frequencies = np.arange(points) / (2 * points * sig.dt)
    return fa_spectrum, fa_frequencies


def fas2values(fas, dt):
    """
    Convert a fourier spectrum to time series signal

    Parameters
    ----------
    fas: array_like of img floats
        Positive part only
    dt: float
        time step of time series
    stype: str
        If 'signal' then return Signal, else return AccSignal
    """

    n = 2 * len(fas)
    a = np.zeros(2 * len(fas), dtype=complex)
    a[1:n // 2] = fas[1:]
    a[n // 2 + 1:] = np.flip(np.conj(fas[1:]), axis=0)
    a /= dt
    s = np.fft.ifft(a)
    npts = int(2 ** (np.log(n) / np.log(2)))
    s = s[:npts]
    return s


def fas2signal(fas, dt, stype="signal"):
    """
    Convert a fourier spectrum to time series signal

    Parameters
    ----------
    fas: array_like of img floats
        Positive part only
    dt: float
        time step of time series
    stype: str
        If 'signal' then return Signal, else return AccSignal
    """
    from eqsig.single import Signal, AccSignal
    n = 2 * len(fas)
    a = np.zeros(2 * len(fas), dtype=complex)
    a[1:n // 2] = fas[1:]
    a[n // 2 + 1:] = np.flip(np.conj(fas[1:]), axis=0)
    a /= dt
    s = np.fft.ifft(a)
    npts = int(2 ** (np.log(n) / np.log(2)))
    s = s[:npts]
    if stype == 'signal':
        return Signal(s, dt)
    else:
        return AccSignal(s, dt)
