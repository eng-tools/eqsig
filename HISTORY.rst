=======
History
=======

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