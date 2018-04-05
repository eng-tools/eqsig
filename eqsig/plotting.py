__author__ = 'maximmillen'

import matplotlib.pyplot as plt
import numpy as np
import engformat.plot as esfp

from eqsig.exceptions import SignalProcessingError

# from bwplot import spectra as cbox
from bwplot import cbox

import warnings


def deprecation(message):
    warnings.warn(message, stacklevel=3)


def plot_response_spectrum(rec, **kwargs):
    deprecation('Deprecated, switch to: plotting functions moved to engformat package')

    rec.generate_response_spectrum()
    plot_on = 1
    legend_off = kwargs.get('legend_off', False)
    ratio = kwargs.get('ratio', False)
    info_str = kwargs.get('info_str', '')

    label = kwargs.get('label', rec.label)

    time = kwargs.get('time', "frequency")
    sub_plot = kwargs.get('sub_plot', 0)
    response_type = kwargs.get('type', 'acceleration')
    colour = kwargs.get('colour', False)
    ccbox = kwargs.get('ccbox', "auto")
    scale = kwargs.get('scale', 1.0)
    log_off = kwargs.get('log_off', False)
    verbose = kwargs.get('verbose', 0)
    title = kwargs.get('title', False)

    if response_type == 'acceleration':
        response = rec.s_a
        y_label = 'Spectral acceleration [$m/s^2$]'
    elif response_type == 'velocity':
        response = rec.s_v
        y_label = 'Spectral velocity [$m/s$]'
    elif response_type == 'displacement':
        response = rec.s_d
        y_label = 'Spectral displacement [$m$]'
    else:
        raise NotImplementedError
    if sub_plot == 0:
        sub_plot = plt.figure().add_subplot(111)
    else:
        plot_on = 0
    if time == "period":
        x_para = rec.response_times
    else:
        x_para = 1 / rec.response_times

    if ccbox == "auto":
            ccbox = len(sub_plot.lines)
    rec.ccbox = ccbox

    sub_plot.plot(x_para, response * scale, label=label, c=cbox(0 + rec.ccbox), lw=0.7)

    if not log_off:
        sub_plot.set_xscale('log')

    if time == "period":
        if verbose:
            print('setting x-axis as period')
        sub_plot.set_xlabel('Time period [s]')
        sub_plot.set_xlim([0, rec.response_times[-1]])
    else:
        sub_plot.set_xscale('log')
        sub_plot.set_xlabel('Frequency [Hz]')
    sub_plot.set_ylabel(y_label)

    if title is not False:
        sub_plot.set_title(title)

    if legend_off is False:
        sub_plot.legend(loc='upper left', prop={'size': 8})
    if plot_on == 1:

        plt.show()
    else:
        return sub_plot


def plot_time_series(rec, **kwargs):
    deprecation('Deprecated, switch to: plotting functions moved to engformat package')
    plot_on = kwargs.get('plot_on', False)
    legend_off = kwargs.get('legend_off', False)
    info_str = kwargs.get('info_str', '')
    motion_type = kwargs.get('motion_type', 'acceleration')

    y_label = kwargs.get('y_label', motion_type.title())
    x_label = kwargs.get('x_label', 'Time [s]')
    y_limits = kwargs.get('y_limits', False)
    label = kwargs.get('label', rec.label)
    sub_plot = kwargs.get('sub_plot', 0)
    window = kwargs.get('window', [0, -1])
    ccbox = kwargs.get('ccbox', "auto")  # else integer
    
    cut_index = np.array([0, len(rec.motion)])
    if window[0] != 0:
        cut_index[0] = int(window[0] / rec.dt)
    if window[1] != -1:
        cut_index[1] = int(window[1] / rec.dt)

    
    if sub_plot == 0:
        sub_plot = plt.figure().add_subplot(111)
    else:
        plot_on = 0
    if ccbox == "auto":
        ccbox = len(sub_plot.lines)

    if motion_type == "acceleration":
        motion = rec.motion[cut_index[0]:cut_index[1]]
        balance = True
    elif motion_type == "velocity":
        rec.calculate_displacements()
        motion = rec.velocity[cut_index[0]:cut_index[1]]
        balance = True
    elif motion_type == "displacement":
        rec.calculate_displacements()
        motion = rec.displacement[cut_index[0]:cut_index[1]]
        balance = False
    elif motion_type == "custom":
        motion = rec.motion[cut_index[0]:cut_index[1]]
        balance = False
    else:
        raise NotImplementedError

    t0 = rec.dt * (np.linspace(0, (len(motion) + 1), (len(motion))) + cut_index[0])
    sub_plot.plot(t0, motion, label=label, c=cbox(0 + ccbox), lw=0.7)
    # sub_plot.plot(rec.dt * cut_index, [0, 0], c="k", lw=0.5, zorder=0)  # TODO: use plot-tricks
    # esfp.time_series(sub_plot)
    esfp.time_series(sub_plot, balance=balance)

    if x_label is not False:
        sub_plot.set_xlabel(x_label)
    sub_plot.set_ylabel(y_label)
    if info_str != '':
        stitle = 'Time series \n' + info_str
        sub_plot.set_title(stitle)
    if y_limits is not False:
        sub_plot.set_ylim(y_limits)

    x_limits = [0, t0[-1]]
    sub_plot.set_xlim(x_limits)
    if not legend_off:
        sub_plot.legend(loc='upper right', prop={'size': 8})
    if plot_on == 1:
        plt.show()
    else:
        return sub_plot


def plot_avd(rec, sub_plots=None, ccbox=0, **kwargs):
    """
    Plot acceleration, velocity and displacement
    :return:
    """
    deprecation('Deprecated, switch to: plotting functions moved to engformat package')
    label = kwargs.get('label', rec.label)
    legend_off = kwargs.get('legend_off', False)
    if sub_plots is None:
        big_fig = plt.figure()
        sub_plots = [big_fig.add_subplot(311),
                    big_fig.add_subplot(312),
                    big_fig.add_subplot(313)]
    plot_time_series(rec, ccbox=ccbox, sub_plot=sub_plots[0], label=label, legend_off=True)
    plot_time_series(rec, ccbox=ccbox, sub_plot=sub_plots[1], motion_type="velocity", legend_off=True)
    plot_time_series(rec, ccbox=ccbox, sub_plot=sub_plots[2], motion_type="displacement", legend_off=True)

    if not legend_off:
        sub_plots[0].legend()
    return sub_plots

    
def plot_fa_spectrum(rec, **kwargs):
    deprecation('Deprecated, switch to: plotting functions moved to engformat package')
    plot_on = kwargs.get('plot_on', False)
    legend_off = kwargs.get('legend_off', False)
    smooth = kwargs.get('smooth', False)

    info_str = kwargs.get('info_str', '')

    label = kwargs.get('label', rec.label)

    log_off = kwargs.get('log_off', False)
    title = kwargs.get('title', 'Fourier amplitude acceleration spectrums \n' + info_str)

    band = kwargs.get('band', 40)
    sub_plot = kwargs.get('sub_plot', 0)
    ccbox = kwargs.get('ccbox', 'auto')


    if smooth is False:
        spectrum = abs(rec.fa_spectrum)
        frequencies = rec.fa_frequencies

    else:

        rec.generate_smooth_fa_spectrum(band=band)
        spectrum = abs(rec.smooth_fa_spectrum)
        frequencies = rec.smooth_fa_frequencies


    if sub_plot == 0:
        sub_plot = plt.figure().add_subplot(111)
    else:
        plot_on = 0
    if ccbox == "auto":
        ccbox = len(sub_plot.lines)
    rec.ccbox = ccbox

    sub_plot.plot(frequencies, spectrum, label=label, c=cbox(ccbox), lw=0.7)

    sub_plot.set_xscale('log')
    if log_off is not True:
        sub_plot.set_yscale('log')
    sub_plot.set_xlabel('Frequency [Hz]')
    sub_plot.set_ylabel('Fourier Amplitude [m/s2]')

    if title is not False:
        sub_plot.set_title(title)
    if legend_off is False:
        sub_plot.legend(loc='upper left', prop={'size': 8})
    if plot_on == 1:
        plt.show()
    else:
        return sub_plot


def plot_transfer_function(base_rec, recs, **kwargs):
    """
    Plots the transfer function between the base values and a list of other motions
    :param base_rec: A Record object
    :param recs: A list of Record objects
    :param kwargs:
    :return:
    """
    deprecation('Deprecated, switch to: plotting functions moved to engformat package')
    plot_on = kwargs.get('plot_on', False)
    legend_off = kwargs.get('legend_off', False)
    smooth = kwargs.get('smooth', False)

    info_str = kwargs.get('info_str', '')

    base_label = kwargs.get('label', base_rec.label)

    log_off = kwargs.get('log_off', False)
    title = kwargs.get('title', 'Transfer function \n' + info_str)

    band = kwargs.get('band', 40)
    sub_plot = kwargs.get('sub_plot', 0)
    ccbox = kwargs.get('ccbox', 'auto')

    if smooth is False:
        base_spectrum = abs(base_rec.fa_spectrum)
        base_frequencies = base_rec.fa_frequencies
    else:
        base_rec.generate_smooth_fa_spectrum(band=band)
        base_spectrum = abs(base_rec.smooth_fa_spectrum)
        base_frequencies = base_rec.smooth_fa_frequencies

    if sub_plot == 0:
        sub_plot = plt.figure().add_subplot(111)
    else:
        plot_on = 0
    if ccbox == "auto":
        ccbox = len(sub_plot.lines)

    for rec in recs:
        if smooth is False:
            if rec.dt != base_rec.dt or len(rec.motion) != len(base_rec.motion):
                raise SignalProcessingError("Motion lengths and timestep do not match. "
                                            "Cannot build non-smooth spectrum")
            spectrum = abs(rec.fa_spectrum)
        else:
            rec.reset_smooth_spectrum(freq_range=base_rec.freq_range)
            rec.generate_smooth_fa_spectrum(band=band)
            spectrum = abs(rec.smooth_fa_spectrum)

        sub_plot.plot(base_frequencies, spectrum / base_spectrum, label=rec.label, c=cbox(ccbox), lw=0.7)

    sub_plot.set_xscale('log')
    if log_off is not True:
        sub_plot.set_yscale('log')
    sub_plot.set_xlabel('Frequency [Hz]')
    sub_plot.set_ylabel('Transfer function from %s' % base_label)

    if title is not False:
        sub_plot.set_title(title)
    if legend_off is False:
        sub_plot.legend(loc='upper left', prop={'size': 8})
    if plot_on == 1:
        plt.show()
    else:
        return sub_plot