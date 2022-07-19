======
pyzome
======

..
        image:: https://img.shields.io/travis/zdlawrence/pyzome.svg
        :target: https://travis-ci.org/zdlawrence/pyzome
..
        image:: https://img.shields.io/pypi/v/pyzome.svg
        :target: https://pypi.python.org/pypi/pyzome


pyzome (pronounced like "rhizome") is a Python package for doing computations and analyses of zonal
mean and related atmospheric variables. It simplifies the process of computing relevant zonal mean
and wave diagnostics for understanding wave-mean flow interactions, wave propagation characteristics,
and zonal mean momentum budgets.

pyzome is in an early alpha stage and currently under active development. While much of the
basic functionality and diagnostic computations work (as of July 2022), all code is subject
to change.

..
        * Free software: 3-clause BSD license
        * Documentation: (COMING SOON!) https://zdlawrence.github.io/pyzome.

Features
--------
- zonal and meridional mean computations
- planetary wave decompositions of fields; wave covariances between 2 fields
- Transformed Eulerian Mean (TEM) diagnostics, such as EP-fluxes and residual velocities
- Quasi-geostrophic diagnostics, such as meridional QGPV gradients and the refractive index
- A "recipes" framework that simplifies the process of computing these diagnostics
- More to come!
