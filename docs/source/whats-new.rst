.. _whats-new.2023.09.0:

What's New
==========

v2023.09.0 (26 Sep 2023)
-------------------------

New Features
~~~~~~~~~~~~
- Added xarray accessor classes to use variable-agnostic pyzome functions directly on xarray objects.
- Updated the "scipy" fft backend to use lazy evaluation, consistent with the "xrft" backend 

Breaking Changes
~~~~~~~~~~~~~~~~
- pyzome.basic.zonal_mean and pyzome.basic.meridional_mean now use ``strict = True`` by default 

Bug Fixes
~~~~~~~~~
- Addressed a bug in pyzome.qglogp.plumb_wave_activity_flux that caused the function to not properly check for required coordinates
- Fixed the obtrusive logger from the recipes.zmd module

Documentation
~~~~~~~~~~~~~
- Updated docs to include xarray accessor classes

Internal Changes
~~~~~~~~~~~~~~~~
- switch to `CALVER <https://calver.org/>`_ versioning scheme