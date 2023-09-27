.. _xarray_accessors:

xarray accessors (``pyzome.accessor``)
=======================================
pyzome provides a set of `xarray <http://xarray.pydata.org/en/stable/>`__ 
accessors that link variable-agnostic pyzome functions to xarray object 
methods. 

pyzome coordinates
------------------
The pyzome coordinate mappings, which identify the relevant coordinates in 
an xarray object, are stored internally as a dictionary. pyzome always uses 
the convention that `lon` refers to longitude, `lat` to latitude, `plev` to 
pressure levels, and `zonal_wavenum` to zonal wavenumbers. Thus, to get the 
coordinate names for a given xarray object, you can use the `coord_map` 
method of the xarray accessor:

    >>> import pyzome as pzm
    >>> # assume ds is an xarray dataset with coordinates called "longitude" and "pressure"
    >>> ds.pzm.coord_map("lon")
    >>> ds.pzm.coord_map("plev")

Alternatively, you can directly refer to `lon`, `lat`, `plev`, and 
`zonal_wavenum` as accessor _atrributes_ to directly obtain the underlying
xarray coordinates:

    >>> ds.pzm.lon
    >>> ds.pzm.plev

If any of these pyzome coordinates are not present or cannot be identified in 
the xarray object, trying to access them will raise a KeyError.

PyzomeAccessor
--------------
.. autoclass:: pyzome.accessor.PyzomeAccessor
    :members:

PyzomeDataArrayAccessor
-----------------------
.. autoclass:: pyzome.accessor.PyzomeDataArrayAccessor
    :members: