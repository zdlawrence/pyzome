.. _qglogp:

QG-log(p) diagnostics (``pyzome.qglogp``)
=========================================
Functions for computing diagnostics based on QG-log(p) 
assumptions. Most of the diagnostics require that their input 
data include a log-pressure altitude coordinate (since many 
of the quantities take derivatives as a function of altitude).
The function :func:`pyzome.qglogp.add_logp_altitude` can be used 
to add this coordinate to an xarray Dataset/DataArray.

Functions
---------
.. automodule:: pyzome.qglogp
   :members: