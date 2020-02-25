eqsig package
=============

The eqsig package is based around the ``AccSignal`` and ``Signal`` objects to represent acceleration time series and other
time series data. These objects hold commonly used signal parameters and use caching to avoid expensive recalculations.

Single time series objects
--------------------------

.. automodule:: eqsig.single
    :members:
    :undoc-members:
    :show-inheritance:

Multiple time series objects
----------------------------

.. automodule:: eqsig.multiple
    :members:
    :undoc-members:
    :show-inheritance:

Intensity measure functions
---------------------------

.. automodule:: eqsig.im
    :members:
    :undoc-members:
    :show-inheritance:

Signal processing and analysis functions
----------------------------------------

.. automodule:: eqsig.functions
    :members:
    :undoc-members:
    :show-inheritance:


Loading and saving functions
----------------------------

The eqsig format is:
 - First line: name of signal
 - Second line: Number of points and time steps, separated by a space
 - Remaining lines: Each line contains one value of time series

.. automodule:: eqsig.loader
    :members:
    :undoc-members:
    :show-inheritance:

eqsig.sdof module
-----------------

.. automodule:: eqsig.sdof
    :members:
    :undoc-members:
    :show-inheritance:

eqsig.displacements module
--------------------------

.. automodule:: eqsig.displacements
    :members:
    :undoc-members:
    :show-inheritance:

eqsig.stockwell module
----------------------------

.. automodule:: eqsig.stockwell
    :members:
    :undoc-members:
    :show-inheritance:

eqsig.stockwell module
----------------------------

.. automodule:: eqsig.stockwell
    :members:
    :undoc-members:
    :show-inheritance:

eqsig.exceptions module
-----------------------

.. automodule:: eqsig.exceptions
    :members:
    :undoc-members:
    :show-inheritance:

