.. _intro:

Introduction
============
``pyzome`` is a Python package that aids with calculating
diagnostics that are commonly used in atmospheric science 
to understand the general circulation and variability 
of the atmosphere. ``pyzome`` includes utilities for 
computing zonal and meridional means, zonal wavenumber 
decompositions, and includes explicit calculations for 
Transformed Eulerian Mean (TEM) and log-pressure 
quasi-geostrophic diagnostics. 

``pyzome`` is built on top of the ``xarray`` package, 
and is designed to work with ``xarray`` DataArrays and 
Datasets. It is generally designed to work with regularly 
gridded atmospheric data, as is commonly output fro many 
models and reanalysis datasets. ``pyzome`` leverages coordinate 
names and attributes to automatically perform operations 
such as means and derivatives across geophysical dimensions. 
For instance::

    >>> # assume ds is an xarray Dataset 
    >>> import pyzome as pzm
    >>> ds_zm = pzm.zonal_mean(ds)
    >>> ds_polar_cap = pzm.meridional_mean(ds_zm, 60, 90)

In the above example, ``ds`` could have dimensions named, 
e.g., ``lon``, ``lons``, or ``longitude``, and ``lat``, 
``lats`` or ``latitude`` (or other reasonably similar names), 
and ``pyzome`` will automatically determine the correct dimensions.

For explicit diagnostic calculations, ``pyzome`` expects that 
quantities are expressed in SI units. This means that data on 
pressure levels should have their pressures in Pascals 
(with an appropriate coordinate name and units attribute). 
These assumptions may be relaxed in the future by building 
on top of other python modules such as `pint <https://pint.readthedocs.io/en/stable/>`_
and `pint-xarray <https://pint-xarray.readthedocs.io/en/latest/>`_.

Please see the other doc pages for more details about the 
included ``pyzome`` modules and functions.