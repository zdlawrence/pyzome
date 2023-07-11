from __future__ import annotations
import logging
from typing import Iterable

import xarray as xr

from ..basic import zonal_mean
from ..zonal_waves import zonal_wave_coeffs, zonal_wave_covariance
from ..checks import infer_xr_coord_names

logging.basicConfig(level=logging.NOTSET)

LONG_NAMES = {
    "u": "zonal mean zonal wind",
    "v": "zonal mean meridional wind",
    "w": "zonal mean vertical pressure velocity",
    "T": "zonal mean temperature",
    "Z": "zonal mean geopotential height",
    "uv": "zonal mean eddy momentum flux",
    "vT": "zonal mean eddy heat flux",
    "uw": "zonal mean eddy vertical momentum flux",
    "wT": "zonal mean eddy vertical heat flux",
    "uv_k": "eddy momentum flux due to zonal wavenumber k",
    "vT_k": "eddy heat flux due to zonal wavenumber k",
    "uw_k": "eddy vertical momentum flux due to zonal wavenumber k",
    "wT_k": "eddy vertical heat flux due to zonal wavenumber k",
    "Z_k_real": "real part of zonally fourier transformed geopotential height",
    "Z_k_imag": "imaginary part of zonally fourier transformed geopotential height",
    "T_k_real": "real part of zonally fourier transformed temperature",
    "T_k_imag": "imaginary part of zonally fourier transformed temperature",
}

UNITS = {
    "u": "m s-1",
    "v": "m s-1",
    "w": "Pa s-1",
    "T": "K",
    "Z": "m",
    "uv": "m+2 s-2",
    "vT": "K m s-1",
    "uw": "m Pa s-2",
    "wT": "K Pa s-1",
    "uv_k": "m+2 s-2",
    "vT_k": "K m s-1",
    "uw_k": "m Pa s-2",
    "wT_k": "K Pa s-1",
    "Z_k_real": "m",
    "Z_k_imag": "m",
    "T_k_real": "K",
    "T_k_imag": "K",
}


def create_zonal_mean_dataset(
    ds: xr.Dataset,
    verbose: bool = False,
    include_waves: bool = False,
    waves: None | Iterable[int] = None,
    fftpkg: str = "scipy",
    lon_coord: str = "",
):
    r"""Compiles a "zonal mean dataset".

    Given an xarray dataset containing full fields of basic state
    variables such as velocity components and temperatures, this
    function will compute as many zonal mean diagnostics as possible.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset containing full fields (i.e., containing latitude &
        longitude dimensions) of basic state variables. This function
        currently assumes specific names and units:

        - 'u' = zonal wind component in m/s
        - 'v' = meridional wind component in m/s
        - 'w' = vertical pressure velocity in Pa/s
        - 'T' = temperature in K
        - 'Z' = geopotential height in m

        If your data names, dimensions, and/or units do not conform to these
        restrictions, please change beforehand. Dimensions and names can
        easily be changed with the `rename` method of xarray Datasets/DataArrays.

        Note that ds need not contain all of these variables, and this
        function will still provide as many diagnostics as possible.

    verbose : bool, optional
        Whether to print out progress information as the function proceeds.
        Defaults to False.
    include_waves : bool, optional
        Whether to include diagnostics as a function of zonal wavenumber such
        as eddy covariances and fourier coefficients. Defaults to False.
    waves : array-like, optional
        The specific zonal wavenumbers to maintain in the output. This
        kwarg is only considered if include_waves is True.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        'scipy' or 'xrft'. Specifying scipy uses some operations that are memory-eager
        and leverages scipy.fft.rfft. Specifying xrft should leverage the benefits
        of xarray/dask for large datasets by using xrft.fft. Defaults to scipy.
        This kwarg is only considered if include_waves is True.

    Returns
    -------
    `xarray.Dataset`
        An xarray Dataset containing the possible zonal mean diagnostics.

    Notes
    -----
    Please see https://essd.copernicus.org/articles/10/1925/2018/ for
    a description of a different zonal mean dataset compiled for the
    SPARC Reanalysis Intercomparison Project. This function does *not*
    provide all the same diagnostics as listed in that publication.
    However, if this function is provided with all of u, v, w, and T,
    it will return all the terms necessary to compute further diagnostics
    for, e.g., zonal Eulerian and Transformed Eulerian Mean momentum budgets.

    """

    logger = logging.getLogger("create_zonal_mean_dataset")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    if lon_coord == "":
        coords = infer_xr_coord_names(ds, required=["lon"])
        lon_coord = coords["lon"]

    valid_vars = ("u", "v", "w", "T", "Z")
    valid_covs = (("u", "v"), ("v", "T"), ("u", "w"), ("w", "T"))
    wave_vars = ("T", "Z")

    vars_available = [v for v in ds.data_vars if v in valid_vars]
    if len(vars_available) == 0:
        raise ValueError("No valid fields found in provided dataset")
    cov_pairs = [
        (v1, v2)
        for v1, v2 in valid_covs
        if v1 in vars_available and v2 in vars_available
    ]

    inter = {}
    out_coords = None
    logger.info(" *** Computing zonal means and eddies")
    for var in vars_available:
        logger.info(f"   {var}")

        zm = zonal_mean(ds[var])
        ed = ds[var] - zm

        inter[f"{var}"] = zm
        inter[f"{var}ed"] = ed
        out_coords = inter[f"{var}"].coords

    # another for-loop is not strictly necessary, but it makes the
    # logging messages a bit cleaner
    if include_waves is True:
        logger.info(" *** Computing zonal fourier transforms")
        for var in vars_available:
            logger.info(f"   fft({var})")
            fc = zonal_wave_coeffs(ds[var], waves=waves, fftpkg=fftpkg)
            if var in wave_vars:
                inter[f"{var}_k_real"] = fc.real
                inter[f"{var}_k_imag"] = fc.imag
            inter[f"{var}fc"] = fc
            out_coords = inter[f"{var}fc"].coords

    if len(cov_pairs) > 0:
        logger.info(" *** Computing eddy covariances")
        for var1, var2 in cov_pairs:
            logger.info(f"   {var1}{var2}")

            cov = zonal_mean(inter[f"{var1}ed"] * inter[f"{var2}ed"])
            inter[f"{var1}{var2}"] = cov
            if include_waves is True:
                logger.info(f"   {var1}{var2}_k")
                cov = zonal_wave_covariance(inter[f"{var1}fc"], inter[f"{var2}fc"])
                inter[f"{var1}{var2}_k"] = cov

    # remove unneeded fields
    out_vars = list(inter.keys())
    for var in out_vars:
        if "ed" in var or "fc" in var:
            inter.pop(var)

    # add attributes
    out_vars = inter.keys()
    for var in out_vars:
        inter[var].name = var
        inter[var].attrs["long_name"] = LONG_NAMES[var]
        inter[var].attrs["units"] = UNITS[var]

    out_ds = xr.Dataset(inter, coords=out_coords)
    out_ds.attrs["nlons"] = ds[lon_coord].size
    out_ds.attrs["lon0"] = ds[lon_coord].values[0]

    return out_ds
