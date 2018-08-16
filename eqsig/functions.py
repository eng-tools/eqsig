import numpy as np
import scipy
from eqsig.single import Signal, AccSignal


def time_series_from_motion(motion, dt):
    npts = len(motion)
    return np.linspace(0, dt * (npts + 1), npts)


def determine_indices_of_peaks(values):
    diff = np.diff(values)
    # if negative then direction has switched
    direction_switch = diff[1:] * diff[:-1]
    direction_switch = np.insert(direction_switch, 0, 0)
    peak_indices = np.where(direction_switch < 0)
    return peak_indices[0]


# def determine_peaks_only_series(values):
#     diff = np.diff(values)
#     # if negative then direction has switched
#     direction_switch = diff[1:] * diff[:-1]
#     peaks = np.where(direction_switch < 0, 1, 0)
#     # peaks array is shorter by two, indices refer +1, add zeros at start and end
#     peaks = np.insert(peaks, 0, 0)
#     peaks = np.insert(peaks, len(peaks), 0)  # don't use -1 here
#     return values * peaks


def determine_delta_peak_only_series_4_cleaned_data(values):
    peak_indices = determine_indices_of_peaks(values)
    peak_values = np.take(values, peak_indices)
    signs = np.where(np.mod(np.arange(len(peak_values)), 2), 1, -1)
    delta_peaks = np.where(-signs * peak_values < 0, -np.abs(peak_values), np.abs(peak_values))
    delta_peaks_series = np.zeros_like(values)
    np.put(delta_peaks_series, peak_indices, delta_peaks)

    return delta_peaks_series


def clean_out_non_changing(values):
    diff_values = np.diff(values)
    non_zero_indices = np.where(diff_values != 0)[0]
    cleaned_values = np.take(values, non_zero_indices)
    cleaned_values = np.insert(cleaned_values, len(cleaned_values), values[-1])
    return cleaned_values, non_zero_indices


def determine_delta_peak_only_series(values):
    """
    Creates an array with only the changes in the peak values and zeros for non-peak values.

    Parameters
    ----------
    :param values: array_like, array of values
    :return:

    Examples
    --------
    >>> values = np.array([0, 2, 1, 2, 0, 1, 0, -1, 0, 1, 0])
    np.array([0, 2, 1, 2, 0.3, 1, 0.3, -1, 0.4, 1, 0])
    >>> determine_delta_peak_only_series(values)
    array([0,  2, -1,  2,  0,  1,  0,  1,  0,  1,  0])
    """
    # enforce array type
    values = np.array(values)
    # rebase to zero as first value
    values -= values[0]
    # remove all non-changing values
    cleaned_values, non_zero_indices = clean_out_non_changing(values)
    cleaned_values *= np.sign(cleaned_values[1])  # ensure first value is increasing
    # compute delta peaks for cleaned data
    cleaned_delta_peak_series = determine_delta_peak_only_series_4_cleaned_data(cleaned_values)
    # re-index data to uncleaned array
    delta_peaks_series = np.zeros_like(values)
    np.put(delta_peaks_series, non_zero_indices, cleaned_delta_peak_series)
    return delta_peaks_series


def fas2signal(fas, dt, stype="signal"):
    """
    Convert a fourier spectrum to time series signal

    :param fas: positive part only
    :param dt: time step of time series
    :return:
    """
    n = 2 * len(fas)
    # fa = scipy.fft(self.values, n=n)
    # points = int(n_factor / 2)
    a = np.zeros(n)
    a[0:n // 2] = fas
    a[n // 2:] = np.flip(fas, axis=0)
    a /= dt
    a *= 2
    # fa_spectrum = fa[range(points)] * self.dt
    # fa_frequencies = np.arange(points) / (2 * points * self.dt)
    s = np.fft.ifft(a)
    s = s[:int(len(s) / 2)]
    if stype == 'signal':
        return Signal(s, dt)
    else:
        return AccSignal(s, dt)


def generate_fa_spectrum(sig):
    npts = len(sig.values)
    n_factor = 2 ** int(np.ceil(np.log2(npts)))
    fa = scipy.fft(sig.values, n=n_factor)
    points = int(n_factor / 2)
    assert len(fa) == n_factor
    fa_spectrum = fa[range(points)] * sig.dt
    fa_frequencies = np.arange(points) / (2 * points * sig.dt)
    return fa_spectrum, fa_frequencies


def acc2acc(values, dt):
    npts = len(values)
    n_factor = 2 ** int(np.ceil(np.log2(npts)))
    fa = scipy.fft(values, n=n_factor)
    points = int(n_factor / 2)
    # assert len(fa) == n_factor
    fas = fa[range(points)] * dt
    fa_zero = fa[points]
    fa_frequencies = np.arange(points) / (2 * points * dt)
    n = 2 * len(fas)
    # fa = scipy.fft(self.values, n=n)
    # points = int(n_factor / 2)
    a = np.zeros(len(fa), dtype=complex)
    # a[0] = 0
    a[1:n // 2] = fas[1:]
    a[n // 2 + 1:] = np.flip(np.conj(fas[1:]), axis=0)
    a /= dt
    # plt.plot(fa)
    # plt.plot(a)
    plt.plot(np.imag(fa), np.imag(a))
    # plt.plot(fas / dt)
    plt.show()
    # fa_spectrum = fa[range(points)] * self.dt
    # fa_frequencies = np.arange(points) / (2 * points * self.dt)
    s = np.fft.ifft(fa, n=n_factor)
    # s = s[:int(len(s) / 2)]
    s = s[:len(values)]
    plt.plot(values)
    plt.plot(s)
    plt.show()

    # return fa_spectrum, fa_frequencies



if __name__ == '__main__':
    from tests.conftest import TEST_DATA_DIR
    import matplotlib.pyplot as plt
    import eqsig
    record_path = TEST_DATA_DIR
    record_filename = 'test_motion_dt0p01.txt'
    motion_step = 0.01
    rec = np.loadtxt(record_path + record_filename)
    acc2acc(rec, motion_step)
    # acc_signal = eqsig.AccSignal(rec, motion_step)
    # fa_spectrum, fa_frequencies = generate_fa_spectrum(acc_signal)
    # asig2 = fas2signal(fa_spectrum, motion_step, stype="acc-signal")
    # plt.plot(acc_signal.time, acc_signal.values)
    # plt.plot(asig2.time, asig2.values)
    #
    # plt.semilogx(fa_frequencies, abs(fa_spectrum))
    # plt.show()