from typing import Union, List

import numpy as np
import xarray as xr
import scipy
import xrft

from .checks import (has_global_regular_lons, infer_xr_coord_names)


def zonal_mean(dat: Union[xr.DataArray, xr.Dataset],
               lon_coord: str = "",
               strict: bool = False) -> Union[xr.DataArray, xr.Dataset]:
    r"""Compute the zonal mean.

    This is primarily a convenience function that will make other
    code more explicit/readable.

    Parameters
    ----------
    dat : `xarray.DataArray` or `xarray.Dataset`
        data containing a dimension named longitude that spans all 360 degrees

    lon_coord : str, optional
        The coordinate name of the longitude dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the longitude

    strict : bool, optional
        If True, the function will check whether the longitudes span 360
        degrees with regular spacing. If False (the default), this check is
        skipped.

    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        The mean across the longitude dimension

    """

    if lon_coord == "":
        coords = infer_xr_coord_names(dat, required=["lon"])
        lon_coord = coords["lon"]

    if strict is True:
        has_global_regular_lons(dat[lon_coord], enforce=True)

    return dat.mean(lon_coord)


def meridional_mean(dat: Union[xr.DataArray, xr.Dataset],
                    lat1: float, lat2: float,
                    lat_coord: str = "") -> Union[xr.DataArray, xr.Dataset]:
    r"""Compute the cos(lat) weighted mean of data between two latitudes.

    Parameters
    ----------
    dat : `xarray.DataArray` or `xarray.Dataset`
        data containing a latitude dimension that spans
        lat1 and lat2. The cos(lat) weighting assumes that the
        latitudes are equally spaced.

    lat1 : float
        The beginning latitude limit of the band average

    lat2 : float
        The ending latitude limit of the band average

    lat_coord : str, optional
        The coordinate name of the latitude dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the latitude

    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        the weighted mean across the latitude dimension limited
        by lat1 and lat2

    """

    if lat_coord == "":
        coords = infer_xr_coord_names(dat, required=["lat"])
        lat_coord = coords["lat"]

    if (lat1 >= lat2):
        msg = "lat1 must be less than lat2"
        raise ValueError(msg)

    min_lat = float(dat[lat_coord].min())
    max_lat = float(dat[lat_coord].max())
    if not ((min_lat <= lat1 <= max_lat) and (min_lat <= lat2 <= max_lat)):
        msg = f"data only contains lats in range of {min_lat} to {max_lat} " + \
              f"(chose lat1={lat1}, lat2={lat2})"
        raise ValueError(msg)

    lats = dat[lat_coord]
    ixs = {lat_coord: np.logical_and(lats >= lat1, lats <= lat2)}
    wgts = np.cos(np.deg2rad(dat[lat_coord].isel(ixs)))

    return dat.isel(ixs).weighted(wgts).mean(lat_coord)


def zonal_wave_coeffs(dat: xr.DataArray, *,
                      waves: Union[None, List] = None,
                      fftpkg: str = "scipy",
                      lon_coord: str = "") -> xr.DataArray:
    r"""Calculate the Fourier coefficients of waves in the zonal direction.

    This is a primarily a driver function that shifts the data depending
    on the specified fftpkg.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None
        for all.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        scipy or xrft. Specifying scipy uses some operations that are
        memory-eager and leverages scipy.fft.rfft. Specifying xrft should
        leverage the benefits of xarray/dask for large datasets by using
        xrft.fft. Defaults to scipy.
    lon_coord : str, optional
        The coordinate name of the longitude dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the longitude

    Returns
    -------
    `xarray.DataArray`
        Output of the rFFT along the longitude dimension, for specified
        waves only.

    """

    if fftpkg not in ["scipy", "xrft"]:
        msg = "fftpkg keyword arg must be one of scipy or xarray"
        raise ValueError(msg)

    if lon_coord == "":
        coords = infer_xr_coord_names(dat, required=["lon"])
        lon_coord = coords["lon"]
    has_global_regular_lons(dat[lon_coord], enforce=True)

    funcs = {
        "scipy": _zonal_wave_coeffs_scipy,
        "xrft": _zonal_wave_coeffs_xrft
    }
    fc = funcs[fftpkg](dat, lon_coord)

    fc.attrs["nlons"] = dat[lon_coord].size
    fc.attrs["lon0"] = dat[lon_coord].values[0]
    if (waves is not None):
        fc = fc.sel(wavenum_lon=waves)

    return fc


def _zonal_wave_coeffs_scipy(dat: xr.DataArray,
                             lon_coord: str) -> xr.DataArray:
    r"""Calculate the Fourier coefficients of waves in the zonal direction.

    Uses scipy.fft.rfft to perform the calculation.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees
    lon_coord : string
        name of the dimension/coordinate corresponding to the longitudes

    Returns
    -------
    `xarray.DataArray`
        Output of the rFFT along the longitude dimension.

    """
    nlons = dat[lon_coord].size
    lon_ax = dat.get_axis_num(lon_coord)

    new_dims = list(dat.dims)
    new_dims[lon_ax] = "wavenum_lon"

    new_coords = dict(dat.coords)
    new_coords.pop(lon_coord)
    new_coords["wavenum_lon"] = np.arange(0, nlons//2 + 1)

    fc = scipy.fft.rfft(dat.values, axis=lon_ax)
    fc = xr.DataArray(fc, coords=new_coords, dims=new_dims)

    return fc


def _zonal_wave_coeffs_xrft(dat: xr.DataArray, lon_coord: str) -> xr.DataArray:
    r"""Calculate the Fourier coefficients of waves in the zonal direction.

    Uses xrft.fft to perform the calculation.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees
    lon_coord : string
        name of the dimension/coordinate corresponding to the longitudes

    Returns
    -------
    `xarray.DataArray`
        Output of the rFFT along the longitude dimension.

    """

    fc = xrft.fft(dat, dim=lon_coord, real_dim=lon_coord,
                  true_phase=False, true_amplitude=False)

    fc = fc.rename({f"freq_{lon_coord}": "wavenum_lon"})
    fc = fc.assign_coords({"wavenum_lon": np.arange(fc.wavenum_lon.size)})

    return fc


def zonal_wave_ampl_phase(dat: xr.DataArray,
                          waves: Union[None, List] = None,
                          phase_deg: bool = False,
                          fftpkg: str = "scipy",
                          lon_coord: str = "") -> xr.DataArray:
    r"""Calculates the amplitudes and relative phases of waves in the
    zonal direction.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None
        for all.
    phase_deg : boolean, optional
        Whether to return the relative phases in radians or degrees.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options
        are scipy or xrft. Specifying scipy uses some operations that are
        memory-eager and leverages scipy.fft.rfft. Specifying xrft should
        leverage the benefits of xarray/dask for large datasets by using
        xrft.fft. Defaults to scipy.
    lon_coord : str, optional
        The coordinate name of the longitude dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the longitude

    Returns
    -------
    Tuple of two `xarray.DataArray`
        Tuple contains (amplitudes, phases)

    See Also
    --------
    zonal_wave_coeffs

    """

    if lon_coord == "":
        coords = infer_xr_coord_names(dat, required=["lon"])
        lon_coord = coords["lon"]

    fc = zonal_wave_coeffs(dat, waves=waves, fftpkg=fftpkg,
                           lon_coord=lon_coord)

    # where the longitudinal wavenumber is 0, `where' will
    # mask to nan, so np.isfinite will return False in those
    # spots and true everywhere else. Thus, add 1 to get
    # the multiplying mask that keeps in mind the "zeroth"
    # mode (which would be the zonal mean, if kept)
    #
    # this is necessary because of the symmetric spectrum,
    # so all other wavenumbers except the 0th need to
    # be multipled by 2 to get the right amplitude
    mult_mask = np.isfinite(fc.where(fc.wavenum_lon != 0)) + 1

    ampl = mult_mask*np.abs(fc) / fc.nlons
    phas = np.angle(fc, deg=phase_deg)

    return (ampl.astype(dat.dtype), phas.astype(dat.dtype))


def zonal_wave_contributions(dat: xr.DataArray,
                             waves: Union[None, List] = None,
                             fftpkg: str = "scipy",
                             lon_coord: str = "") -> xr.DataArray:
    r"""Computes contributions of waves with zonal wavenumber k to the
    input field.

    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension named longitude that spans all 360 degrees
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None
        for all.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options
        are scipy or xrft. Specifying scipy uses some operations that are
        memory-eager and leverages scipy.fft.rfft. Specifying xrft should
        leverage the benefits of xarray/dask for large datasets by using
        xrft.fft. Defaults to scipy.
    lon_coord : str, optional
        The coordinate name of the longitude dimension. If given an empty
        string (the default), the function tries to infer which coordinate
        corresponds to the longitude

    Returns
    -------
    `xarray.DataArray`

    See Also
    --------
    zonal_wave_coeffs

    """

    if lon_coord == "":
        coords = infer_xr_coord_names(dat, required=["lon"])
        lon_coord = coords["lon"]

    fc = zonal_wave_coeffs(dat, fftpkg=fftpkg, lon_coord=lon_coord)

    if (waves is None):
        waves = fc.wavenum_lon.values

    recons = []
    if (fftpkg == "scipy"):
        new_dims = list(dat.dims)
        new_dims += ["wavenum_lon"]
        new_coords = dict(dat.coords)
        new_coords["wavenum_lon"] = waves

        for k in waves:
            mask = np.isnan(fc.where(fc.wavenum_lon != k))

            kcont = scipy.fft.irfft((fc*mask).values,
                                    axis=fc.get_axis_num("wavenum_lon"))
            recons.append(kcont[..., np.newaxis])

        recons = np.concatenate(recons, axis=-1)
        recons = xr.DataArray(recons, dims=new_dims, coords=new_coords)

    elif (fftpkg == "xrft"):
        for k in waves:
            mask = np.isnan(fc.where(fc.wavenum_lon != k))

            kcont = xrft.ifft(fc*mask, dim="wavenum_lon",
                              real_dim="wavenum_lon")
            recons.append(kcont)

        recons = xr.concat(recons, dim="wavenum_lon")
        recons = recons.assign_coords({"wavenum_lon": waves,
                                       lon_coord: dat[lon_coord]})

    return recons.astype(dat.dtype)


def zonal_wave_covariance(dat1: xr.DataArray,
                          dat2: xr.DataArray,
                          waves: Union[None, List] = None,
                          fftpkg: str = "scipy",
                          lon_coord: str = "") -> xr.DataArray:
    r"""Calculates the covariance of two fields partititioned into
    zonal wavenumbers.

    Parameters
    ----------
    dat1 : `xarray.DataArray`
        field containing a dimension named longitude that spans all 360
        degrees. Should have the same shape as dat2.
    dat2 : `xarray.DataArray`
        another field also containing a dimension named longitude that spans
        all 360 degrees. Should have the same shape as dat1.
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None
        for all.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options
        are scipy or xrft. Specifying scipy uses some operations that are
        memory-eager and leverages scipy.fft.rfft. Specifying xrft should
        leverage the benefits of xarray/dask for large datasets by using
        xrft.fft. Defaults to scipy.

    Returns
    -------
    `xarray.DataArray`

    See Also
    --------
    zonal_wave_coeffs

    """

    # Ensure that dat1 and dat2 are fully consistent
    xr.align(dat1, dat2, join="exact", copy=False)

    # If dat1 and dat2 are fully consistent, then this block will cover both
    if lon_coord == "":
        coords = infer_xr_coord_names(dat1, required=["lon"])
        lon_coord = coords["lon"]

    nlons = dat1[lon_coord].size

    fc1 = zonal_wave_coeffs(dat1, waves=waves, fftpkg=fftpkg)
    fc2 = zonal_wave_coeffs(dat2, waves=waves, fftpkg=fftpkg)

    mult_mask = np.isfinite(fc1.where(fc1.wavenum_lon != 0)) + 1
    cov = mult_mask*np.real(fc1 * fc2.conj())/(nlons**2)

    return cov
