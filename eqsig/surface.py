import numpy as np


def trim_to_length(values, npts, surf2depth_travel_times, dt, trim=False, start=False, s2s_travel_time=0.0):
    """
    Trim a 2D array back to a have the same length

    Parameters
    ----------
    values: array_like
        Values to be outputted
    s2s_travel_time: float
        travel time from input motion location to the surface
    surf2depth_travel_times: array_like
        travel time from surface to depths of interest
    trim: bool
        if true then forces array to be same length as values array
    start: bool
        if true then forces array to have the same start time as values array

    Returns
    -------

    """

    surf_to_depth_shifts = np.array(surf2depth_travel_times / dt, dtype=int)
    start_shift = int(s2s_travel_time / dt)
    if start:  # Trim front
        sis = start_shift - surf_to_depth_shifts
        if not trim:
            extras = np.max([np.max(sis), 0]) - np.min([np.min(2 * surf_to_depth_shifts), 0])
            npts = npts + extras  # plus the zero padding in front and back
    else:
        if trim:
            sis = np.zeros_like(surf_to_depth_shifts)
        else:  # no changes required
            return values
    outs = np.zeros((len(surf_to_depth_shifts), npts))
    for i in range(len(surf_to_depth_shifts)):
        if sis[i] < 0:
            outs[i] = values[i, -sis[i]: npts - sis[i]]
        else:
            outs[i, sis[i]:] = values[i, : npts - sis[i]]  # zero padded
    return outs


def calc_surface_energy(asig, travel_times, nodal=True, up_red=1., down_red=1., stt=0.0, trim=False, start=False):
    """
    Calculates the energy at different travel times from a surface

    Parameters
    ----------
    asig: eqsig.AccSignal object
    travel_times: array_like
        Travel times from surface to depth of interest
    nodal: bool
        If true then surface is nodal (minima)
    up_red: float or array_like
        upward wave reduction factors
    down_red: float or array_like
        downward wave reduction factors
    trim: bool
        if true then forces array to be same length as values array
    start: bool
        if true then forces array to have the same start time as values array
    stt: float
        Travel time from input motion location to the surface

    Returns
    -------

    """
    from scipy.integrate import cumtrapz
    # if not hasattr(up_red, '__len__'):
    #     up_red = up_red * np.ones(len(travel_times))
    # if not hasattr(down_red, '__len__'):
    #     down_red = down_red * np.ones(len(travel_times))
    if not hasattr(travel_times, '__len__'):
        travel_times = np.array([travel_times])
    else:
        travel_times = np.array(travel_times)
    shifts = 2 * travel_times / asig.dt
    max_shift = int(np.max(shifts))
    up_wave = np.pad(asig.values, (0, max_shift), mode='constant', constant_values=0)
    dshifted = np.arange(asig.npts + max_shift)[np.newaxis, :] - shifts[:, np.newaxis]  # TODO: not needed if shifts is scalar
    down_waves = np.interp(dshifted, np.arange(asig.npts), asig.values, left=0, right=0)
    if hasattr(up_red, '__len__'):
        up_wave = up_wave[np.newaxis, :] * up_red[:, np.newaxis]  # 1d
        down_waves *= down_red[:, np.newaxis]
    else:
        up_wave = up_wave * up_red  # 1d  # TODO: may need to increase dimensions here
        down_waves *= down_red
    if nodal:
        acc_series = - down_waves + up_wave
    else:
        acc_series = down_waves + up_wave
    velocity = cumtrapz(acc_series, dx=asig.dt, initial=0, axis=1)
    e = 0.5 * velocity * np.abs(velocity)
    e = trim_to_length(e, asig.npts, travel_times, asig.dt, trim=trim, start=start, s2s_travel_time=stt)
    if len(travel_times) == 1:
        return e[0]
    else:
        return e


# def dep_calc_surface_energy(asig, travel_times, nodal=True, up_red=1, down_red=1, stt=0.0, trim=False, start=False):
#     """
#     Calculates the energy at different travel times from a surface
#
#     Parameters
#     ----------
#     asig
#     travel_times: array_like
#         Travel times from surface to depth of interest
#     nodal: bool
#         If true then surface is nodal (minima)
#     up_red: float or array_like
#         upward wave reduction factors
#     down_red: float or array_like
#         downward wave reduction factors
#     trim: bool
#         if true then forces array to be same length as values array
#     start: bool
#         if true then forces array to have the same start time as values array
#     stt: float
#         Travel time from input motion location to the surface
#
#     Returns
#     -------
#
#     """
#
#     shifts = np.array(2 * travel_times / asig.dt, dtype=int)
#     up_wave = np.pad(asig.values, (0, np.max(shifts)), mode='constant', constant_values=0) * up_red  # 1d
#     down_waves = fn.put_array_in_2d_array(asig.values, shifts) * down_red
#     if nodal:
#         acc_series = - down_waves + up_wave
#     else:
#         acc_series = down_waves + up_wave
#     velocity = scipy.integrate.cumtrapz(acc_series, dx=asig.dt, initial=0, axis=1)
#     e = 0.5 * velocity * np.abs(velocity)
#     return trim_to_length(e, asig.npts, travel_times, asig.dt, trim=trim, start=start, s2s_travel_time=stt)


def calc_cum_abs_surface_energy(asig, travel_times, nodal=True, up_red=1, down_red=1, stt=0.0, trim=False, start=False):
    energy = calc_surface_energy(asig, travel_times, nodal=nodal, up_red=up_red, down_red=down_red, stt=stt,
                                 trim=trim, start=start)
    diff = np.diff(energy, axis=-1, prepend=0)
    return np.cumsum(np.abs(diff), axis=-1)

#
# if __name__ == '__main__':
#     a = np.linspace(0, 10, 100)
#     b = np.sin(a)
#     import eqsig
#     accsig = eqsig.AccSignal(b, 0.1)
#     t_times = np.array([0.1, 0.2])
#     ke = calc_cum_abs_surface_energy(accsig, t_times, stt=0.3, nodal=True, up_red=1, down_red=1, trim=True)
#
#     ke = calc_cum_abs_surface_energy(accsig, t_times, stt=0.3, nodal=True, up_red=tshifts, down_red=tshifts, trim=True)

