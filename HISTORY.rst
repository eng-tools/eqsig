=======
History
=======

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