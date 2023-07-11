======
pyzome
======

.. image:: https://github.com/zdlawrence/pyzome/actions/workflows/ci_tests.yml/badge.svg
    :target: https://github.com/zdlawrence/pyzome/actions/workflows/ci_tests.yml

.. image:: https://codecov.io/github/zdlawrence/pyzome/branch/main/graph/badge.svg?token=J5CT0XW4FD
    :target: https://codecov.io/github/zdlawrence/pyzome

.. image:: https://readthedocs.org/projects/pyzome/badge/?version=latest
    :target: https://pyzome.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


pyzome (rhymes with "rhizome") is a Python package for atmospheric sciences. It 
simplifies the process of computing relevant diagnostics commonly used to understand
the general atmospheric circulation of the Earth and other planetary atmospheres. 
It is designed to be used in conjunction with xarray for enabling coordinate- and 
unit-aware computations. 

pyzome is in an early stage and currently under active development. While much of 
the core functionality is in place, the API (e.g., function names and call 
signatures) is still subject to change.


Features
--------
- zonal and meridional mean computations
- zonal wavenumber decompositions of fields; zonal wave covariances between 2 fields
- Transformed Eulerian Mean (TEM) diagnostics, such as EP-fluxes and residual velocities
- quasi-geostrophic diagnostics, such as meridional QGPV gradients and the refractive index
- A "recipes" framework that simplifies the process of computing these diagnostics
- More to come!


Development Roadmap
-------------------
- Build on cf-xarray to streamline coordinate-aware computations
- Build on pint and pint-xarray to streamline unit-aware computations
- Expand documentation
- Diagnostics validation
- Add more core modules:
   - Equivalent Latitude computations
   - ???
- Expand "recipes" to include more diagnostics
   - Annular mode indices
   - Sudden stratospheric warmings
   - Momentum budgets
   - ???


Acknowledgments
---------------
The development of this code was originally supported by the NWS OSTI Weeks 3-4 
Program under NOAA Award NA20NWS4680051. Continued support and development of the 
package is provided on a volunteer basis by the author and contributors.
