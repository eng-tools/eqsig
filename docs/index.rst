.. eqsig documentation master file, created by
   sphinx-quickstart on Tue Jul 10 19:21:51 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to eqsig's documentation!
=================================

Release v\ |version|

The eqsig package is for signal processing of earthquake related data.

It provides two main objects the ``Signal`` and ``AccSignal`` objects that represent time series data of any type
and acceleration time series data.

The objects have common signal processing methods available (e.g. Butterworth filtering) and common
parameters (eg. Fourier Spectra).

Most functions and intensity measures have been written using the signal objects as inputs but have also be seperately
defined to allow inputs as a numpy array and time step, these functions typically have ``_array`` in the function name.



.. toctree::
   :maxdepth: 2

   eqsig
   install

* :ref:`genindex`


