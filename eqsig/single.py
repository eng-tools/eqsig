import numpy as np
import scipy.signal as ss
import scipy

from eqsig import exceptions
import eqsig.duhamels as dh
import eqsig.displacements as sd
import eqsig.measures as sm
from eqsig import measures
from eqsig.exceptions import deprecation


class Signal(object):
    _npts = None
    _smooth_freq_points = 61
    _fa_spectrum = None
    _fa_frequencies = None
    _cached_fa = False
    _cached_smooth_fa = False

    def __init__(self, values, dt, label='m1', smooth_freq_range=(0.1, 30), verbose=0, ccbox=0):
        """
        A time series object

        Parameters
        ----------
        values: array_like
            values of the time series
        dt: float
            the time step
        label: str, optional
            A name for the signal
        smooth_freq_range: tuple, optional
            The frequency range for computing the smooth FAS
        verbose: int, optional
            Level of console output
        ccbox: int, optional
            colour id for plotting
        :return:
        """
        self.verbose = verbose
        self._dt = dt
        self._values = np.array(values)
        self.label = label
        self._smooth_freq_range = smooth_freq_range
        self._npts = len(self.values)
        self._smooth_fa_spectrum = np.zeros((self.smooth_freq_points))
        self._smooth_fa_frequencies = np.zeros(self.smooth_freq_points)
        self.ccbox = ccbox

    @property
    def values(self):
        return self._values

    def reset_values(self, new_values):
        self._values = new_values
        self._npts = len(new_values)
        self.clear_cache()

    @property
    def dt(self):
        """ The time step """
        return self._dt

    @property
    def npts(self):  # Deliberately no public setter method for this
        """ The number of points in the time series """
        return self._npts

    @property
    def time(self):
        """ An array of time of equal length to the time series """
        return np.arange(0, self.npts) * self.dt

    @property
    def fa_spectrum(self):
        """Generate the one-sided Fourier Amplitude spectrum"""
        if not self._cached_fa:
            self.generate_fa_spectrum()
        return self._fa_spectrum

    def generate_fa_spectrum(self):
        n_factor = 2 ** int(np.ceil(np.log2(self.npts)))
        fa = scipy.fft(self.values, n=n_factor)
        points = int(n_factor / 2)
        self._fa_spectrum = fa[range(points)] * self.dt
        self._fa_frequencies = np.arange(points) / (2 * points * self.dt)
        self._cached_fa = True

    @property
    def fa_frequencies(self):
        if not self._cached_fa:
            self.generate_fa_spectrum()
        return self._fa_frequencies

    @property
    def smooth_freq_range(self):
        return self._smooth_freq_range

    @smooth_freq_range.setter
    def smooth_freq_range(self, limits):
        self._smooth_freq_range = np.array(limits)
        self._cached_smooth_fa = False

    @property
    def smooth_freq_points(self):
        return self._smooth_freq_points

    @smooth_freq_points.setter
    def smooth_freq_points(self, value):
        self._smooth_freq_points = value
        self._cached_smooth_fa = False

    @property
    def smooth_fa_frequencies(self):
        if not self._cached_smooth_fa:
            self.generate_smooth_fa_spectrum()
        return self._smooth_fa_frequencies

    @property
    def smooth_fa_spectrum(self):
        if not self._cached_smooth_fa:
            self.generate_smooth_fa_spectrum()
        return self._smooth_fa_spectrum

    def clear_cache(self):
        """Resets the dynamically calculated properties."""
        self._cached_smooth_fa = False
        self._cached_fa = False

    def generate_smooth_fa_spectrum(self, band=40):
        """
        Calculates the smoothed Fourier Amplitude Spectrum
        using the method by Konno and Ohmachi 1998

        Parameters
        ----------
        band: int
            range to smooth over
        """
        lf = np.log10(self.smooth_freq_range)

        fa_frequencies = self.fa_frequencies
        fa_spectrum = self.fa_spectrum
        smooth_fa_frequencies = np.logspace(lf[0], lf[1], self.smooth_freq_points, base=10)
        self._smooth_fa_spectrum = generate_smooth_fa_spectrum(smooth_fa_frequencies, fa_frequencies,
                                                               fa_spectrum, band=band)
        self._smooth_fa_frequencies = smooth_fa_frequencies
        self._cached_smooth_fa = True

    def butter_pass(self, cut_off=(0.1, 15), **kwargs):
        """
        Performs a Butterworth filter

        Notes
        -----
        Wraps the scipy 'butter' filter

        Parameters
        ----------
        :param cut_off: Tuple, The cut off frequencies for the filter
        :param kwargs:
        :return:
        """
        if isinstance(cut_off, list) or isinstance(cut_off, tuple) or isinstance(cut_off, np.Array):
            pass
        else:
            raise ValueError("cut_off must be list, tuple or array.")
        if len(cut_off) != 2:
            raise ValueError("cut_off must be length 2.")
        if cut_off[0] is not None and cut_off[1] is not None:
            filter_type = "band"
            cut_off = np.array(cut_off)
        elif cut_off[0] is None:
            filter_type = 'low'
            cut_off = cut_off[1]
        else:
            filter_type = 'high'
            cut_off = cut_off[0]

        filter_order = kwargs.get('filter_order', 4)
        remove_gibbs = kwargs.get('remove_gibbs', 0)
        gibbs_extra = kwargs.get('gibbs_extra', 1)
        gibbs_range = kwargs.get('gibbs_range', 50)
        sampling_rate = 1.0 / self.dt
        nyq = sampling_rate * 0.5

        mote = self.values
        org_len = len(mote)

        if remove_gibbs:
            # Pad end of record with extra zeros then cut it off after filtering
            nindex = int(np.ceil(np.log2(len(mote)))) + gibbs_extra
            new_len = 2 ** nindex
            diff_len = new_len - org_len
            if remove_gibbs == 'start':
                s_len = 0
                f_len = s_len + org_len
            elif remove_gibbs == 'end':
                s_len = diff_len
                f_len = s_len + org_len
            else:
                s_len = int(diff_len / 2)
                f_len = s_len + org_len

            end_value = np.mean(mote[-gibbs_range:])
            start_value = np.mean(mote[:gibbs_range])
            temp = start_value * np.ones(new_len)
            temp[f_len:] = end_value
            temp[s_len:f_len] = mote
            mote = temp
        else:
            s_len = 0
            f_len = org_len

        wp = cut_off / nyq
        b, a = ss.butter(filter_order, wp, btype=filter_type)
        mote = ss.filtfilt(b, a, mote)
        # removing extra zeros from gibbs effect
        mote = mote[s_len:f_len]  # TODO: don't use -1

        self.reset_values(mote)

    def remove_average(self, section=-1, verbose=-1):
        """
        Calculates the average and removes it from the record
        """

        if verbose == -1:
            verbose = self.verbose

        average = np.mean(self.values[:section])
        self.reset_values(self.values - average)

        if verbose:
            print('removed av.: ', average)

    def remove_poly(self, poly_fit=0):
        """
        Calculates best fit polynomial and removes it from the record
        """

        x = np.linspace(0, 1.0, self.npts)
        cofs = np.polyfit(x, self.values, poly_fit)
        y_cor = 0 * x
        for co in range(len(cofs)):
            mods = x ** (poly_fit - co)
            y_cor += cofs[co] * mods

        self.reset_values(self.values - y_cor)

    def get_section_average(self, start=0, end=-1, index=False):
        """
        Gets the average value of a part of series.

        Common use is so that it can be patched
        with another record.

        Parameters
        ----------
        start: int or float, optional
            Section start point
        end: int or float, optional
            Section end point
        index: bool, optional
            if False then start and end are considered values in time.

        Returns
        -------
        float
            The mean value of the section.
        """
        return get_section_average(self, start=start, end=end, index=index)

    def add_constant(self, constant):
        """
        Adds a single value from every value in the signal.

        :param constant:
        :return:
        """
        self.reset_values(self.values + constant)

    def add_series(self, series):
        """
        Adds a single value from every value in the signal.

        :param series: A series of values
        :return:
        """
        if len(series) == self.npts:
            self.reset_values(self.values + series)
        else:
            raise exceptions.SignalProcessingError("new series has different length to Signal")

    def add_signal(self, new_signal):
        """
        Combines a signal.

        :param new_signal: Signal object
        :return:
        """
        if isinstance(new_signal, Signal):
            if new_signal.dt == self.dt:
                self.add_series(new_signal.values)
            else:
                raise exceptions.SignalProcessingError("New signal has different time step")
        else:
            raise exceptions.SignalProcessingError("New signal is not a Signal object")

    def running_average(self, width=1):
        """
        Averages the values over a width (chunk of numbers)
        replaces the value in the centre of the chunk.

        :param width: The range over which values are averaged
        """

        mot = self.values

        for i in range(len(mot)):
            if i < width / 2:
                cc = i + int(width / 2) + 1
                self._values[i] = np.mean(mot[:cc])
            elif i > len(mot) - width / 2:
                cc = i - int(width / 2)
                self._values[i] = np.mean(mot[cc:])
            else:
                cc1 = i - int(width / 2)
                cc2 = i + int(width / 2) + 1
                self._values[i] = np.mean(mot[cc1:cc2])

        self.clear_cache()


class AccSignal(Signal):
    _cached_response_spectra = False
    _cached_disp_and_velo = False
    _cached_xi = 0.05
    _cached_params = {}
    _s_a = None
    _s_v = None
    _s_d = None
    t_b01 = 0.0
    t_b05 = 0.0
    t_b10 = 0.0
    a_rms01 = 0.0
    a_rms05 = 0.0
    a_rms10 = 0.0
    t_595 = 0.0  # significant duration
    sd_start = 0.0  # start time of significant duration
    sd_end = 0.0  # end time of significant duration
    arias_intensity_series = None
    arias_intensity = 0.0
    cav_series = None
    cav = 0.0

    def __init__(self, values, dt, label='m1', smooth_freq_range=(0.1, 30), verbose=0, response_times=(0.1, 5), ccbox=0):
        """
        A time series object

        :param values: An acceleration time series, type=acceleration, should be in m/s2
        :param dt: time step
        """
        super(AccSignal, self).__init__(values, dt, label=label, smooth_freq_range=smooth_freq_range, verbose=verbose, ccbox=ccbox)
        if len(response_times) == 2:
            self.response_times = np.linspace(response_times[0], response_times[1], 100)
        else:
            self.response_times = np.array(response_times)
        self._velocity = np.zeros(self.npts)
        self._displacement = np.zeros(self.npts)

    def clear_cache(self):
        self._cached_smooth_fa = False
        self._cached_fa = False
        self._cached_response_spectra = False
        self._cached_disp_and_velo = False
        self.reset_all_motion_stats()

    def generate_response_spectrum(self, response_times=None, xi=-1):
        """
        Generate the response spectrum for the response_times for a given
        damping (xi). default xi = 0.05
        """
        if self.verbose:
            print('Generating response spectra')
        if response_times is not None:
            self.response_times = response_times

        if xi == -1:
            xi = self._cached_xi
        self._s_d, self._s_v, self._s_a = dh.pseudo_response_spectra(self.values, self.dt, self.response_times, xi)
        self._cached_response_spectra = True

    @property
    def s_a(self):
        """
        Pseudo spectral response acceleration of linear SDOF
        """
        if not self._cached_response_spectra:
            self.generate_response_spectrum()
        return self._s_a

    @property
    def s_v(self):
        """
        Pseudo spectral response velocity of linear SDOF
        """
        if not self._cached_response_spectra:
            self.generate_response_spectrum()
        return self._s_v

    @property
    def s_d(self):
        """
        Spectral response displacement of linear SDOF
        """
        if not self._cached_response_spectra:
            self.generate_response_spectrum()
        return self._s_d

    def correct_me(self):
        """
        This provides a correction to an acceleration time series
        """

        # self.remove_average()
        # self.butter_pass([0.1, 10])

        disp = ss.detrend(self.displacement)
        vel = np.zeros(self.npts)
        acc = np.zeros(self.npts)
        for i in range(self.npts - 1):  # MEANS THAT ACC has a pulse at the end.
            vel[i + 1] = (disp[i + 1] - disp[i]) / self.dt
            acc[i + 1] = (vel[i + 1] - vel[i]) / self.dt
        self.reset_values(acc)

    def remove_rolling_average(self, mtype="velocity", freq_window=5):
        """

        :param mtype:
        :param freq_window: float, window for applying the rolling average
        :return:
        """
        if mtype == "velocity":
            mot = self.velocity
        else:
            mot = self.values
        width = int(1. / (freq_window * self.dt))
        if width < 1:
            raise ValueError("freq_window to high")
        roll = np.zeros_like(mot)
        for i in range(len(mot)):
            if i < width / 2:
                cc = i + int(width / 2) + 1
                roll[i] = np.mean(mot[:cc])
            elif i > len(mot) - width / 2:
                cc = i - int(width / 2)
                roll[i] = np.mean(mot[cc:])
            else:
                cc1 = i - int(width / 2)
                cc2 = i + int(width / 2) + 1
                roll[i] = np.mean(mot[cc1:cc2])
        if mtype == "velocity":
            velocity = self.velocity - roll
            acc = np.diff(velocity) / self.dt
            acc = np.insert(acc, 0, velocity[0] / self.dt)
            self._values = acc
        else:
            self._values -= roll
        self.clear_cache()

    def rebase_displacement(self):
        """
        Correction to make the displacement zero at the end of the record
        """

        end_disp = self.displacement[-1]

        acceleration_correction = 2 * end_disp / (self.dt * self.npts)
        self._values -= acceleration_correction
        self.clear_cache()

    def generate_displacement_and_velocity_series(self, trap=True):
        """
        Calculates the displacement and velocity
        """
        self._velocity, self._displacement = sd.velocity_and_displacement_from_acceleration(self.values,
                                                                                          self.dt, trap=trap)
        self._cached_disp_and_velo = True

    @property
    def velocity(self):
        if not self._cached_disp_and_velo:
            self.generate_displacement_and_velocity_series()
        return self._velocity

    @property
    def displacement(self):
        if not self._cached_disp_and_velo:
            self.generate_displacement_and_velocity_series()
        return self._displacement

    def generate_peak_values(self):
        """
        Determines the peak signal values
        """
        deprecation("generate_peak_values() is no longer in use, all peak values are lazy loaded.")

    @property
    def pga(self):
        if "pga" in self._cached_params:
            return self._cached_params["pga"]
        else:
            pga = sm.calc_peak(self.values)
            self._cached_params["pga"] = pga
            return pga

    @property
    def pgv(self):
        if "pgv" in self._cached_params:
            return self._cached_params["pgv"]
        else:
            pgv = sm.calc_peak(self.velocity)
            self._cached_params["pgv"] = pgv
            return pgv

    @property
    def pgd(self):
        if "pgd" in self._cached_params:
            return self._cached_params["pgd"]
        else:
            pgd = sm.calc_peak(self.displacement)
            self._cached_params["pgd"] = pgd
            return pgd

    def generate_duration_stats(self):
        abs_motion = abs(self.values)

        time = np.arange(self.npts) * self.dt
        # Bracketed duration
        ind01 = np.where(abs_motion / 9.8 > 0.01)  # 0.01g
        time2 = time[ind01]
        try:
            self.t_b01 = time2[-1] - time2[0]
            # rms acceleration in m/s/s
            self.a_rms01 = np.sqrt(1 / self.t_b01 * np.trapz((self.values[ind01[0][0]:ind01[0][-1]]) ** 2, dx=self.dt))
        except IndexError:
            self.t_b01 = -1.
            self.a_rms01 = -1.

        ind05 = np.where(abs_motion / 9.8 > 0.05)  # 0.05g
        time05 = time[ind05]
        try:
            self.t_b05 = time05[-1] - time05[0]
            self.a_rms05 = np.sqrt(1 / self.t_b05 * np.trapz((self.values[ind05[0][0]:ind05[0][-1]]) ** 2, dx=self.dt))
        except IndexError:
            self.t_b05 = -1.
            self.a_rms05 = -1.
        ind10 = np.where(abs_motion / 9.8 > 0.1)  # 0.10g
        time10 = time[ind10]
        try:
            self.t_b10 = time10[-1] - time10[0]
            self.a_rms10 = np.sqrt(1 / self.t_b10 * np.trapz((self.values[ind10[0][0]:ind10[0][-1]]) ** 2, dx=self.dt))
        except IndexError:
            self.t_b10 = -1.
            self.a_rms10 = -1.

        # Trifunac and Brady
        self.sd_start, self.sd_end = sm.calc_sig_dur_vals(self.values, self.dt)

        self.t_595 = self.sd_end - self.sd_start

    def generate_cumulative_stats(self):
        """

        CAV: Cumulative Absolute velocity

        ref:
        Electrical Power Research Institute. Standardization of the Cumulative
        Absolute Velocity. 1991. EPRI TR-100082-1'2, Palo Alto, California.
        """
        # Arias intensity in m/s
        self.arias_intensity_series = measures.calc_arias_intensity(self)
        self.arias_intensity = self.arias_intensity_series[-1]
        self.cav_series = measures.calc_cav(self)
        self.cav = self.cav_series[-1]

    @property
    def isv(self):
        """
        Integral of the squared velocity at the end of the motion

        See Kokusho (2013)
        :return:
        """
        if "isv_series" in self._cached_params:
            return self._cached_params["isv_series"][-1]
        else:
            return self.isv_series[-1]

    @property
    def isv_series(self):
        """
        Integral of the squared velocity at each time step

        See Kokusho (2013)
        :return:
        """
        if "isv_series" in self._cached_params:
            return self._cached_params["isv_series"]
        else:
            isv_series = measures.calc_isv(self)
            self._cached_params["isv_series"] = isv_series
            return isv_series

    def generate_all_motion_stats(self):
        """
        This calculates the duration, Arias intensity
        """

        self.generate_peak_values()
        self.generate_duration_stats()
        self.generate_cumulative_stats()

    def reset_all_motion_stats(self):
        self.t_b01 = 0.0
        self.t_b05 = 0.0
        self.t_b10 = 0.0
        self.a_rms01 = 0.0
        self.a_rms05 = 0.0
        self.a_rms10 = 0.0
        self.t_595 = 0.0
        self.sd_start = 0.0
        self.sd_end = 0.0
        self.arias_intensity = 0.0
        self._cached_params = {}

    # deprecated
    def relative_displacement_response(self, period, xi):
        return dh.single_elastic_response(self.values, self.dt, period, xi)

    def response_series(self, response_times=None, xi=-1):
        """
        Generate the response time series for a set of elastic SDOFs for a given
        damping (xi).
        """
        if self.verbose:
            print('Generating response series')
        if response_times is not None:
            self.response_times = response_times
        if xi == -1:
            xi = self._cached_xi
        resp_u, resp_v, resp_a = dh.response_series(self.values, self.dt, self.response_times, xi)
        return resp_u, resp_v, resp_a


def get_section_average(series, start=0, end=-1, index=False):
    """
    Gets the average value of a part of series.

    Common use is so that it can be patched with another record.

    :param series: A TimeSeries object
    :param start: int or float, optional,
        Section start point
    :param end: int or float, optional,
        Section end point
    :param index: bool, optional,
        if False then start and end are considered values in time.
    :return float,
        The mean value of the section.
    """
    s_index, e_index = time_indices(series.npts, series.dt, start, end, index)

    section_average = np.mean(series.values[s_index:e_index])
    return section_average


def time_indices(npts, dt, start, end, index):
    """
    Determine the new start and end indices of the time series.

    :param npts: Number of points in original time series
    :param dt: Time step of original time series
    :param start: int or float, optional, New start point
    :param end: int or float, optional, New end point
    :param index: bool, optional, if False then start and end are considered values in time.
    :return: tuple, start index, end index
    """
    if index is False:  # Convert time values into indices
        if end != -1:
            e_index = int(end / dt) + 1
        else:
            e_index = end
        s_index = int(start / dt)
    else:
        s_index = start
        e_index = end
    if e_index > npts:
        raise exceptions.SignalProcessingWarning("Cut point is greater than time series length")
    return s_index, e_index


def generate_smooth_fa_spectrum(smooth_fa_frequencies, fa_frequencies, fa_spectrum, band=40):
    smooth_fa_spectrum = np.zeros_like(smooth_fa_frequencies)
    for i in range(smooth_fa_frequencies.size):
        f_centre = smooth_fa_frequencies[i]
        amp_array = np.log10((fa_frequencies / f_centre) ** band)

        amp_array[0] = 0

        wb_vals = np.zeros((len(amp_array)))
        for j in range(len(amp_array)):
            if amp_array[j] == 0:
                wb_vals[j] = 1
            else:
                wb_vals[j] = (np.sin(amp_array[j]) / amp_array[j]) ** 4

        smooth_fa_spectrum[i] = (sum(abs(fa_spectrum) * wb_vals) / sum(wb_vals))
    return smooth_fa_spectrum


