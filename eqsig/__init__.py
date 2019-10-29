from eqsig.single import Signal, AccSignal
from eqsig import design_spectra
from eqsig.functions import *
from eqsig.multiple import Cluster, combine_at_angle, compute_rotated
from eqsig.im import calc_significant_duration, calculate_peak  # deprecated load
from eqsig import im as measures
from eqsig import im
from eqsig import stockwell
from eqsig.loader import save_signal, load_signal, save_values_and_dt, load_values_and_dt, load_asig, load_sig
from eqsig import loader
from eqsig import __about__
from eqsig import sdof
from eqsig import surface

__project__ = __about__.__project__
__author__ = __about__.__author__
__version__ = __about__.__version__
__license__ = __about__.__license__
