from __future__ import annotations

import numpy as np
import xarray as xr

from .checks import has_global_regular_lons, infer_xr_coord_names


def zonal_mean(
    dat: xr.DataArray | xr.Dataset,
    lon_coord: str = "",
    strict: bool = False,
) -> xr.DataArray | xr.Dataset:
    r"""Compute the zonal mean.

    This is primarily a convenience function that will make other
    code more explicit/readable. This function is imported at the
    top level of the package by default.

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
    zonal average: `xarray.DataArray` or `xarray.Dataset`
        The mean across the longitude dimension

    """

    if lon_coord == "":
        coords = infer_xr_coord_names(dat, required=["lon"])
        lon_coord = coords["lon"]

    if strict is True:
        has_global_regular_lons(dat[lon_coord], enforce=True)

    return dat.mean(lon_coord)


def meridional_mean(
    dat: xr.DataArray | xr.Dataset,
    lat1: float,
    lat2: float,
    lat_coord: str = "",
) -> xr.DataArray | xr.Dataset:
    r"""Compute the cos(lat) weighted mean of data between two latitudes.

    This function is imported at the top level of the package by default.

    Parameters
    ----------
    dat : `xarray.DataArray` or `xarray.Dataset`
        data containing a latitude dimension that spans
        lat1 and lat2. The cos(lat) weighting assumes that the
        latitudes are equally spaced. If given a dataset, the
        function assumes all variables are on the same
        latitude grid.

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
    meridional average: `xarray.DataArray` or `xarray.Dataset`
        the weighted mean across the latitude dimension limited
        by lat1 and lat2

    """

    if lat_coord == "":
        coords = infer_xr_coord_names(dat, required=["lat"])
        lat_coord = coords["lat"]

    if lat1 >= lat2:
        msg = "lat1 must be less than lat2"
        raise ValueError(msg)

    min_lat = float(dat[lat_coord].min())
    max_lat = float(dat[lat_coord].max())
    if not ((min_lat <= lat1 <= max_lat) and (min_lat <= lat2 <= max_lat)):
        msg = (
            f"data only contains lats in range of {min_lat} to {max_lat} "
            + f"(chose lat1={lat1}, lat2={lat2})"
        )
        raise ValueError(msg)

    lats = dat[lat_coord]
    ixs = {lat_coord: np.logical_and(lats >= lat1, lats <= lat2)}
    wgts = np.cos(np.deg2rad(dat[lat_coord].isel(ixs)))

    return dat.isel(ixs).weighted(wgts).mean(lat_coord)  # type: ignore
