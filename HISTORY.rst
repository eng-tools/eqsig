History
=======

1.2.10 (2020-11-24)
-------------------
* Adjusted `eqsig.stockwell.plot_stock`, since min freq was out by factor of 0.5.

1.2.5 (2020-11-24)
-------------------
* Added `gen_ricker_wavelet_asig` to create an acceleration signal that is a Ricker wavelet
* Added `eqsig.sdof.calc_input_energy_spectrum` to compute the input energy into an SDOF
* Can now load a Signal with a scale factor by passing in the keyword `m=<scale factor>`
* The left interpolation function interp_left now returns the same size as x, which can be a scalar, and if `y` is None then assumes index (0,1,2,...,n)

1.2.4 (2020-07-20)
-------------------
* Fixed issue with computation of surface energy spectra
* Support for numpy==1.19

1.2.3 (2020-05-05)
-------------------
* Fixed docs for generation of FAS, changed kwarg `n_plus` to `p2_plus` since this adds to the power of 2.

1.2.2 (2020-05-05)
-------------------
* Switched to numpy for computing the Fourier amplitude spectrum

1.2.1 (2020-05-05)
-------------------

* Added `response_period_range` to AccSignal object initial inputs to define response periods using an upper and lower limit
* Improved speed of surface energy calculation `calc_surface_energy` and returns correct size based on input dimensions
* Removed global import of scipy - done at function level
* Added an `interp_left` function to interpolate an array and take lower value
* Fixed issue with inverse of stockwell transform `stockwell.itransform`, it no longer doubles the time step
* Increased speed of stockwell transform `stockwell.transform`.
* Added `remove_poly` function to remove a polynomial fit from an array
* Added option to access `fa_frequencies` and `smooth_fa_frequencies` as `fa_freqs` and `smooth_fa_freqs`.
* Added option for computing smoothed FAS with extra zero padding
* Added function for computing smoothed fas using a custom smoothing matrix.


1.2.0 (2019-11-03)
-------------------

* Added `interp2d` fast interpolation of a 2D array to obtain a new 2D array
* No longer raises warning when period is 0.0 for computing response spectrum
* Fixed issue with computation of smoothed response spectrum for dealing with zeroth frequency
* Increased speed of`generate_smooth_fa_spectrum`
* Can now directly set `AccSignal.smooth_fa_frequencies`
* Deprecated `AccSignal.smooth_freq_points` and `AccSignal.smooth_freq_range` will be removed in later version

1.1.2 (2019-10-31)
-------------------

* More accuracy in `calc_surface_energy` - now interpolates between time steps. More tests added.


1.1.1 (2019-10-29)
-------------------

* Fixed issue in `get_zero_crossings_array_indices` where it would fail if array did not contain any zeros.
* Added calculation of equivalent number of cycles and equivalent uniform amplitude using power law relationship as intensity measures
* Added function `get_n_cyc_array` to compute number of cycles series from a loading series
* Added intensity measure `im.calc_unit_kinetic_energy()` to compute the cumulative change in kinetic energy according to Millen et al. (2019)
* Added `surface.py` with calculation of surface energy and cumulative change in surface energy time series versus depth from surface


1.1.0 (2019-10-08)
-------------------

* Fixed issue with second order term in sdof response spectrum calculation which effected high frequency response, updated example to show difference

1.0.0 (2019-07-01)
-------------------

* First production release