import numpy as np


def c_h_factor(period, site_class="C"):
    """
    NZS 1170.5 design standard shape factors

    Parameters
    ----------
    period
    site_class

    Returns
    -------

    """

    single = 0
    if isinstance(period, float):
        single = 1
        period = [period]
    c_h_values = np.zeros(len(period))
    for i in range(len(period)):
        tt = period[i]
        if tt < 0:
            print('Structural period is negative')
            raise ValueError
        else:
            if site_class == 'C':
                if tt == 0:
                    ch_factor = 1.33
                elif tt < 0.1:
                    ch_factor = 1.33 + 1.60 * (tt / 0.1)
                elif tt < 0.3:
                    ch_factor = 2.93
                elif tt < 1.5:
                    ch_factor = 2.0 * (0.5 / tt) ** 0.75
                elif tt < 3.0:
                    ch_factor = 1.32 / tt
                else:
                    ch_factor = 3.96 / tt ** 2
            elif site_class == 'D':
                if tt == 0:
                    ch_factor = 1.12
                elif tt < 0.1:
                    ch_factor = 1.12 + 1.88 * (tt / 0.1)
                elif tt < 0.56:
                    ch_factor = 3.0
                elif tt < 1.5:
                    ch_factor = 2.4 * (0.75 / tt) ** 0.75
                elif tt < 3.0:
                    ch_factor = 2.14 / tt
                else:
                    ch_factor = 6.42 / tt ** 2
            elif site_class == 'E':
                if tt == 0:
                    ch_factor = 1.12
                elif tt < 0.1:
                    ch_factor = 1.12 + 1.88 * (tt / 0.1)
                elif tt < 1.0:
                    ch_factor = 3.0
                elif tt < 1.5:
                    ch_factor = 3.0 / tt ** 0.75
                elif tt < 3.0:
                    ch_factor = 3.32 / tt
                else:
                    ch_factor = 9.96 / tt ** 2
            else:
                print('Soil must be type C, D or E')
                raise ValueError
            c_h_values[i] = ch_factor

        if single:
            c_h_values = c_h_values[0]
    return c_h_values


def t_eff(displacement, site_class, z_factor, r_factor, n_factor):
    """
    Returns the effective period based on the displacement and using the NZ design Spectrum

    :param displacement:
    :param site_class:
    :param z_factor:
    :param r_factor:
    :param n_factor:
    """
    # Assume N (near fault factor) is 1.0
    # given C=Ch*R*N*Z
    gravity = 9.81
    if site_class == 'C':
        t_c = 3.0
        d_c = 3.96 * z_factor * r_factor * n_factor / (2 * np.pi) ** 2 * gravity
    elif site_class == 'D':
        t_c = 3.0
        d_c = 6.42 * z_factor * r_factor * n_factor / (2 * np.pi) ** 2 * gravity
    elif site_class == 'E':
        t_c = 3.0
        d_c = 9.96 * z_factor * r_factor * n_factor / (2 * np.pi) ** 2 * gravity
    else:
        raise ValueError("site_class must be C, D or E")

    if displacement > d_c:
        raise ValueError("displacement exceeds corner displacement")
    else:
        time = t_c * displacement / d_c
    return time


def sd_nzs(period, site_class, z_factor, r_factor, n_factor):
    """
    Returns the NZ sd_nzs value for a given Time and soil type

    :param period: float or array
    :param site_class: Either 'C', 'D' or 'E'
    output: sd_nzs: float or array
    """
    if period < 0:
        raise ValueError('Structural period is negative')
    else:
        if site_class == 'C':
            if period == 0:
                c_h = 1.33 * period ** 2
            elif period < 0.1:
                c_h = (1.33 + 1.60 * (period / 0.1)) * period ** 2
            elif period < 0.3:
                c_h = 2.93 * period ** 2
            elif period < 1.5:
                c_h = (2.0 * (0.5 / period) ** 0.75) * period ** 2
            elif period < 3.0:
                c_h = 1.32 / period * period ** 2
            else:
                c_h = 3.96
        elif site_class == 'D':
            if period == 0:
                c_h = 1.12 * period ** 2
            elif period < 0.1:
                c_h = (1.12 + 1.88 * (period / 0.1)) * period ** 2
            elif period < 0.56:
                c_h = 3.0 * period ** 2
            elif period < 1.5:
                c_h = 2.4 * (0.75 / period) ** 0.75 * period ** 2
            elif period < 3.0:
                c_h = 2.14 / period * period ** 2
            else:
                c_h = 6.42
        elif site_class == 'E':
            if period == 0:
                c_h = 1.12 * period ** 2
            elif period < 0.1:
                c_h = (1.12 + 1.88 * (period / 0.1)) * period ** 2
            elif period < 1.0:
                c_h = 3.0 * period ** 2
            elif period < 1.5:
                c_h = 3.0 / period ** 0.75 * period ** 2
            elif period < 3.0:
                c_h = 3.32 / period * period ** 2
            else:
                c_h = 9.96
        else:
            raise ValueError('Soil must be type C, D or E')
    sd = c_h * z_factor * n_factor * r_factor
    return sd


if __name__ == '__main__':
    Time = 0.7
    Soil_type = 'D'
    # Acc=Force(Time,Soil_type)
