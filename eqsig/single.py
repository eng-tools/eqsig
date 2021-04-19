import numpy as np

from eqsig import exceptions
from eqsig.functions import get_section_average, calc_smooth_fa_spectrum, interp_array_to_approx_dt
import eqsig.sdof as dh
import eqsig.displacements as sd
from eqsig import im
from eqsig.exceptions import deprecation


class Signal(object):
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
    """
    _npts = None
    _smooth_freq_points = 61
    _fa_spectrum = None
    _fa_freqs = None
    _cached_fa = False
    _cached_smooth_fa = False
    _smooth_fa_freqs = None
    _smooth_freq_range = None

    def __init__(self, values, dt, label='m1', smooth_freq_range=(0.1, 30), smooth_fa_freqs=None,
                 verbose=0, ccbox=0):
        self.verbose = verbose
        self._dt = dt
        self._values = np.array(values)
        self.label = label
        if smooth_fa_freqs is not None:
            self.smooth_fa_freqs = smooth_fa_freqs
        else:
            self.set_smooth_fa_frequecies_by_range(smooth_freq_range, 50)
        self._smooth_fa_spectrum = np.zeros(len(self.smooth_fa_freqs))
        self._npts = len(self.values)
        # lf = np.log10(smooth_freq_range)
        # self._smooth_fa_frequencies = np.logspace(lf[0], lf[1], 30, base=10)
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
        """The one-sided Fourier Amplitude spectrum"""
        if not self._cached_fa:
            self.generate_fa_spectrum()
        return self._fa_spectrum

    def gen_fa_spectrum(self, p2_plus=0, n=None):
        """
        Sets the one-sided Fourier Amplitude spectrum and frequencies

        Parameters
        ----------
        p2_plus: int (default=0)
            Additional increments of `n` such that length of signal used is `n` and `n=ceil(log2(self.npts) + p2_plus))`.
        n: int (optional)
            If specified the signal length is padded with zeros before computing FAS

        """
        if n is not None:
            n_factor = n
        else:
            n_factor = 2 ** int(np.ceil(np.log2(self.npts)) + p2_plus)
        fa = np.fft.fft(self.values, n=n_factor)
        points = int(n_factor / 2)
        self._fa_spectrum = fa[range(points)] * self.dt
        self._fa_freqs = np.arange(points) / (2 * points * self.dt)
        self._cached_fa = True

    def generate_fa_spectrum(self):
        self.gen_fa_spectrum()

    @property
    def fa_frequencies(self):
        return self.fa_freqs

    @property
    def fa_freqs(self):
        if not self._cached_fa:
            self.gen_fa_spectrum()
        return self._fa_freqs

    @property
    def smooth_freq_range(self):
        deprecation('AccSignal.smooth_freq_range is deprecated. Use AccSignal.smooth_fa_freqs')
        return self.smooth_fa_freqs[0], self.smooth_fa_freqs[-1]

    @smooth_freq_range.setter
    def smooth_freq_range(self, limits):
        deprecation('AccSignal.smooth_freq_range is deprecated. Set AccSignal.smooth_fa_freqs directly')
        lf = np.log10(np.array(limits))
        self.smooth_fa_freqs = np.logspace(lf[0], lf[1], self.smooth_freq_points, base=10)

    def set_smooth_fa_frequecies_by_range(self, limits, n_points):
        lf = np.log10(limits)
        self._smooth_fa_freqs = np.logspace(lf[0], lf[1], n_points, base=10)
        self._smooth_freq_range = np.array(limits)
        self._cached_smooth_fa = False

    @property
    def smooth_freq_points(self):
        deprecation('AccSignal.smooth_freq_points is deprecated. Use len(AccSignal.smooth_fa_freqs)')
        return len(self.smooth_fa_freqs)

    @smooth_freq_points.setter
    def smooth_freq_points(self, value):
        deprecation('AccSignal.smooth_freq_points is deprecated. Set AccSignal.smooth_fa_freqs directly')
        lf = np.log10(self.smooth_freq_range)
        self.smooth_fa_freqs = np.logspace(lf[0], lf[1], int(value), base=10)

    @property
    def smooth_fa_freqs(self):
        return self._smooth_fa_freqs

    @property
    def smooth_fa_frequencies(self):
        return self._smooth_fa_freqs

    @smooth_fa_frequencies.setter
    def smooth_fa_frequencies(self, frequencies):  # backward compatible using 'frequencies'
        self._smooth_fa_freqs = np.array(frequencies, dtype=float)
        self._cached_smooth_fa = False

    @smooth_fa_freqs.setter
    def smooth_fa_freqs(self, freqs):
        self._smooth_fa_freqs = np.array(freqs, dtype=float)
        self._cached_smooth_fa = False

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
        self.gen_smooth_fa_spectrum(band=band)

    def gen_smooth_fa_spectrum(self, smooth_fa_freqs=None, band=40):
        """
        Calculates the smoothed Fourier Amplitude Spectrum
        using the method by Konno and Ohmachi 1998

        Parameters
        ----------
        band: int
            range to smooth over
        """
        if smooth_fa_freqs is not None:
            self._smooth_fa_freqs = smooth_fa_freqs
        self._smooth_fa_spectrum = calc_smooth_fa_spectrum(self.fa_freqs,
                                                               self.fa_spectrum, self.smooth_fa_freqs, band=band)
        self._cached_smooth_fa = True

    def butter_pass(self, cut_off=(0.1, 15), **kwargs):
        """
        Performs a Butterworth filter

        Notes
        -----
        Wraps the scipy 'butter' filter

        Parameters
        ----------
        cut_off: tuple
            Lower and upper cut off frequencies for the filter, if None then no filter.
            e.g. (None, 15) applies a lowpass filter at 15Hz, whereas (0.1, 10) applies
            a bandpass filter at 0.1Hz to 10Hz.
        filter_order: int
            Order of the Butterworth filter
        remove_gibbs: str (default=None)
            Pads with zeros to remove the Gibbs filter effect,
            if = 'start' then pads at start,
            if = 'end' then pads at end,
            if = 'mid' then pads half at start and half at end
        gibbs_extra: int
            each increment of the value doubles the record length using zero padding.
        :return:
        """
        from scipy.signal import butter, filtfilt
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
        remove_gibbs = kwargs.get('remove_gibbs', None)
        gibbs_extra = kwargs.get('gibbs_extra', 1)
        gibbs_range = kwargs.get('gibbs_range', 50)
        sampling_rate = 1.0 / self.dt
        nyq = sampling_rate * 0.5

        mote = self.values
        org_len = len(mote)

        if remove_gibbs is not None:
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
        b, a = butter(filter_order, wp, btype=filter_type)
        mote = filtfilt(b, a, mote)
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
        Adds a single value to every value in the signal.

        Parameters
        ----------
        constant: float
            Value to be added to time series
        """
        self.reset_values(self.values + constant)

    def add_series(self, series):
        """
        Adds a series of values to the values in the signal.

        Parameters
        ----------
        series: array_like
            A series of values
        """
        if len(series) == self.npts:
            self.reset_values(self.values + series)
        else:
            raise exceptions.SignalProcessingError("new series has different length to Signal")

    def add_signal(self, new_signal):
        """
        Combines a signal.

        Parameters
        ----------
        new_signal: Signal object
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

        Parameters
        ----------
        width: int
            The number of points over which values are averaged
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
    """
    A time series object

    Parameters
    ----------
    values: array_like
        Acceleration time series - should be in m/s2
    dt: float
        Time step
    label: str (default='m1')
        Used for plotting
    smooth_freq_range: tuple [floats] (default=(0.1, 30))
        lower and upper bound of frequency range to compute the smoothed Fourier amplitude spectrum
    verbose: int (default=0)
        Level of output verbosity
    response_times: tuple of floats (default=(0.1, 5))
    ccbox: int
        colour index for plotting
    """

    def __init__(self, values, dt, label='m1', smooth_freq_range=(0.1, 30), smooth_fa_freqs=None, verbose=0, response_period_range=(0.1, 5),
                 response_times=None, ccbox=0):

        super(AccSignal, self).__init__(values, dt, label=label, smooth_freq_range=smooth_freq_range,
                                        smooth_fa_freqs=smooth_fa_freqs, verbose=verbose, ccbox=ccbox)
        if response_times is None:
            self.response_times = np.linspace(response_period_range[0], response_period_range[1], 100)
        else:
            self.response_times = np.array(response_times)
        self._velocity = np.zeros(self.npts)
        self._displacement = np.zeros(self.npts)
        self._cached_params = {}
        self._cached_response_spectra = False
        self._cached_disp_and_velo = False
        self._cached_xi = 0.05
        self._s_a = None
        self._s_v = None
        self._s_d = None

    def clear_cache(self):
        self._cached_smooth_fa = False
        self._cached_fa = False
        self._cached_response_spectra = False
        self._cached_disp_and_velo = False
        self.reset_all_motion_stats()

    def gen_response_spectrum(self, response_times=None, xi=-1, min_dt_ratio=4):
        """
        Generate the response spectrum for the response_times for a given
        damping (xi).

        Parameters
        ----------
        response_times: array_like
            Time periods of SDOFs that response should be computed for
        xi: float (default=0.05 or previously set)
            Ratio of critical damping
        min_dt_ratio: float
            Smallest ratio between response period and integration time step,
            acceleration time series will be interpolated to reach appropriate integration time step if
            not achieved using the acceleration time series time step.
        """
        if self.verbose:
            print('Generating response spectra')
        if response_times is not None:
            self.response_times = response_times
        if self.response_times[0] != 0:
            min_non_zero_period = self.response_times[0]
        else:
            min_non_zero_period = self.response_times[1]
        target_dt = max(min_non_zero_period / 20, self.dt / min_dt_ratio)  # limit to ratio of motion time step
        if target_dt < self.dt:
            values_interp, dt_interp = interp_array_to_approx_dt(self.values, self.dt, target_dt, even=False)
        else:
            values_interp = self.values
            dt_interp = self.dt

        if xi == -1:
            xi = self._cached_xi
        try:
            self._s_d, self._s_v, self._s_a = dh.pseudo_response_spectra(values_interp, dt_interp, self.response_times, xi)
        except MemoryError:
            raise MemoryError('Out of memory. Length of acc (%i) and length of response_times (%i) too large, '
                              'set larger min_dt_ratio.' % (len(values_interp), len(self.response_times)))
        self._cached_response_spectra = True

    def generate_response_spectrum(self, response_times=None, xi=-1, min_dt_ratio=4):
        self.gen_response_spectrum(response_times=response_times, xi=xi, min_dt_ratio=min_dt_ratio)

    @property
    def s_a(self):
        """Pseudo maximum response accelerations of linear SDOFs"""
        if not self._cached_response_spectra:
            self.generate_response_spectrum()
        return self._s_a

    @property
    def s_v(self):
        """Pseudo maximum response velocities of linear SDOFs"""
        if not self._cached_response_spectra:
            self.generate_response_spectrum()
        return self._s_v

    @property
    def s_d(self):
        """Maximum response displacements of linear SDOFs"""
        if not self._cached_response_spectra:
            self.generate_response_spectrum()
        return self._s_d

    def correct_me(self):
        """This provides a correction to an acceleration time series"""
        from scipy.signal import detrend

        disp = detrend(self.displacement)
        vel = np.zeros(self.npts)
        acc = np.zeros(self.npts)
        for i in range(self.npts - 1):  # MEANS THAT ACC has a pulse at the end.
            vel[i + 1] = (disp[i + 1] - disp[i]) / self.dt
            acc[i + 1] = (vel[i + 1] - vel[i]) / self.dt
        self.reset_values(acc)

    def remove_rolling_average(self, mtype="velocity", freq_window=5):
        """
        Removes the rolling average

        Parameters
        ----------
        mtype: str
            motion type to apply method to
        freq_window: int
            window for applying the rolling average
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
        """Correction to make the displacement zero at the end of the record"""

        end_disp = self.displacement[-1]

        acceleration_correction = 2 * end_disp / (self.dt * self.npts)
        self._values -= acceleration_correction
        self.clear_cache()

    def generate_displacement_and_velocity_series(self, trap=True):
        """Calculates the displacement and velocity time series"""
        self._velocity, self._displacement = sd.calc_velo_and_disp_from_accel_arr(self.values, self.dt, trap=trap)
        self._cached_disp_and_velo = True

    @property
    def velocity(self):
        """Velocity time series"""
        if not self._cached_disp_and_velo:
            self.generate_displacement_and_velocity_series()
        return self._velocity

    # @property
    # def zeroed_velocity(self):
    #     """Velocity time series with linear correction to have zero velocity at end"""
    #     return self.velocity - np.arange(self.npts) * self.velocity[-1] / self.npts

    @property
    def displacement(self):
        """Displacement time series"""
        if not self._cached_disp_and_velo:
            self.generate_displacement_and_velocity_series()
        return self._displacement

    # @property
    # def zeroed_displacement(self):
    #     """Displacement time series with linear correction to have zero displacement at end"""
    #     return self.displacement - np.arange(self.npts) * self.displacement[-1] / self.npts

    def generate_peak_values(self):
        """
        DEPRECATED: all peak values are lazy loaded - this method does not need to be run.
        """
        deprecation("generate_peak_values() is no longer in use, all peak values are lazy loaded.")

    @property
    def pga(self):
        """Absolute peak ground acceleration"""
        if "pga" in self._cached_params:
            return self._cached_params["pga"]
        else:
            pga = im.calc_peak(self.values)
            self._cached_params["pga"] = pga
            return pga

    @property
    def pgv(self):
        """Absolute peak ground velocity"""
        if "pgv" in self._cached_params:
            return self._cached_params["pgv"]
        else:
            pgv = im.calc_peak(self.velocity)
            self._cached_params["pgv"] = pgv
            return pgv

    @property
    def pgd(self):
        """Absolute peak ground displacement"""
        if "pgd" in self._cached_params:
            return self._cached_params["pgd"]
        else:
            pgd = im.calc_peak(self.displacement)
            self._cached_params["pgd"] = pgd
            return pgd

    def generate_duration_stats(self):
        """
        Deprecated: Use eqsig.im.calc_sig_dur or eqsig.im.calc_sig_dur, eqsig.im.calc_brac_dur or esig.im.calc_a_rms
        """
        deprecation("Use eqsig.im.calc_sig_dur or eqsig.im.calc_sig_dur, eqsig.im.calc_brac_dur or esig.im.calc_a_rms")
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
        self.sd_start, self.sd_end = im.calc_sig_dur_vals(self.values, self.dt, se=True)

        self.t_595 = self.sd_end - self.sd_start

    def generate_cumulative_stats(self):
        """
        Deprecated: Use eqsig.im.calc_arias_intensity, eqsig.im.calc_cav

        """
        deprecation("Use eqsig.im.calc_arias_intensity, eqsig.im.calc_cav")
        # Arias intensity in m/s
        self.arias_intensity_series = im.calc_arias_intensity(self)
        self.arias_intensity = self.arias_intensity_series[-1]
        self.cav_series = im.calc_cav(self)
        self.cav = self.cav_series[-1]

    def generate_all_motion_stats(self):
        """
        Deprecated: Use eqsig.im functions to calculate stats
        """
        deprecation("Use eqsig.im functions to calculate stats")

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
