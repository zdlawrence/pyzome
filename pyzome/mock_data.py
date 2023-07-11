from __future__ import annotations

import numpy as np
import xarray as xr


def _make_regular_coord(
    left: float | int,
    right: float | int,
    step: float | int,
    name: str,
    inclusive: bool = False,
) -> xr.DataArray:
    """
    Generate an xr.DataArray representing a regularly spaced coordinate array

    Parameters
    ----------
    left : float | int
        The left limit of the coordinate
    right : float | int
        The right limit of the coordinate
    step : float | int
        The step size between each element of the coordinate
    name : str
        The name of the coordinate
    inclusive : bool, optional
        Whether to include the right limit in the coordinate

    Returns
    -------
    `xarray.DataArray`

    """
    if step == 0.0:
        msg = "step cannot be zero"
        raise ValueError(msg)

    if left == right:
        msg = "left- and right-side values cannot be equal"
        raise ValueError(msg)

    if ((left < right) and (step < 0)) or ((left > right) and (step > 0)):
        msg = f"Invalid step for the given limits: left={left}, right={right}, step={step}"
        raise ValueError(msg)

    coord = np.arange(left, right + (inclusive * step), step).astype("float64")
    return xr.DataArray(coord, dims=[name], name=name).assign_coords({name: coord})


def lat_coord(
    resolution: float | int,
    name: str = "lat",
    left_lim: float | int = -90.0,
    right_lim: float | int = 90.0,
    inclusive: bool = True,
) -> xr.DataArray:
    """
    Generate a regularly spaced latitude-like coordinate

    Parameters
    ----------
    resolution : float | int
        The spacing between latitudes in degrees
    name : str, optional
        The name of the coordinate, defaults to "lat"
    left_lim : float | int, optional
        The left latitude limit, which defaults to -90
    right_lim : float | int, optional
        The right latitude limit, which defaults to 90
    inclusive : bool, optional
        Whether to include the right latitude limit, which defaults to True

    Returns
    -------
    latitudes: `xarray.DataArray` 
        Regularly spaced latitudes with the given resolution 
        (and name/limits, if given)

    """
    if (abs(left_lim) > 90) or (abs(right_lim) > 90):
        msg = f"Latitudes cannot be below -90 or above 90: left_lim={left_lim}, upp_lim={right_lim}"
        raise ValueError(msg)

    lats = _make_regular_coord(
        left_lim, right_lim, resolution, name, inclusive=inclusive
    )
    lats.attrs["standard_name"] = "latitude"
    lats.attrs["units"] = "degrees_north"

    return lats


def lon_coord(
    resolution: float | int,
    name: str = "lon",
    left_lim: float | int = 0.0,
    right_lim: float | int = 360.0,
    inclusive: bool = False,
) -> xr.DataArray:
    """
    Generate a regularly spaced longitude-like coordinate

    Parameters
    ----------
    resolution : float | int
        The spacing between longitudes in degrees
    name : str, optional
        The name of the coordinate, defaults to "lon"
    left_lim : float | int, optional
        The left longitude limit, which defaults to 0
    right_lim : float | int, optional
        The right longitude limit, which defaults to 360
    inclusive : bool, optional
        Whether to include the right longitude limit, which defaults to False

    Returns
    -------
    longitudes: `xarray.DataArray` 
        Regularly spaced longitudes with the given resolution
        (and name/limits, if given)

    """

    if ((left_lim < 0) and (right_lim > 180)) or ((right_lim < 0) and (left_lim > 180)):
        msg = "Proper longitude range should be within [-180,180] or [0, 360]"
        raise ValueError(msg)
    elif (left_lim > 360) or (right_lim > 360):
        msg = "upper longitude limit cannot exceed 360"
        raise ValueError(msg)
    elif (left_lim < -180) or (right_lim < -180):
        msg = "lower longitude limit cannot be below -180"
        raise ValueError(msg)

    lons = _make_regular_coord(
        left_lim, right_lim, resolution, name, inclusive=inclusive
    )
    lons.attrs["standard_name"] = "longitude"
    lons.attrs["units"] = "degrees_east"

    return lons


def plev_coord(
    levels_per_decade: int,
    left_lim_exponent: int = 3,
    right_lim_exponent: int = 0,
    name: str = "lev",
    units: str = "hPa",
) -> xr.DataArray:
    """
    Generate a pressure-like vertical coordinate with logarithmic spacing

    Parameters
    ----------
    levels_per_decade : int
        The number of levels per decade (i.e., between 10^3 and 10^2, 10^2 
        and 10^1, etc.)
    left_lim_exponent : int, optional
        The base-10 exponent for the left-most pressure level, which defaults 
        to 3 (for 1000 hPa)
    right_lim_exponent : int, optional
        The base-10 exponent for the right-most pressure level, which defaults 
        to 0 (for 1 hPa)
    name : str, optional
        The name of the coordinate, which defaults to "lev"
    units : str, optional
        The units of the coordinate, which defaults to "hPa"

    Returns
    -------
    pressures: `xarray.DataArray`
        Pressure-like coordinate with logarithmic spacing having the given
        number of levels per decade (and name/limits/units, if given)

    """

    if levels_per_decade <= 0:
        raise ValueError(
            f"'levels_per_decade' must be greater than 0, got {levels_per_decade}"
        )
    if left_lim_exponent == right_lim_exponent:
        raise ValueError(
            "'left_lim_exponent' and 'right_lim_exponent' must not be the same"
        )

    num_elements = (
        levels_per_decade * np.abs(left_lim_exponent - right_lim_exponent) + 1
    )
    pres = np.logspace(left_lim_exponent, right_lim_exponent, num_elements)
    pres = xr.DataArray(pres, dims=[name], name=name).assign_coords({name: pres})
    pres.name = name
    pres.attrs["standard_name"] = "pressure"
    pres.attrs["units"] = units

    return pres


def time_coord(
    start: str = "2000-01-01",
    end: str = "2001-01-01",
    freq: str = "M",
) -> xr.DataArray:
    """
    Generate a time-like coordinate

    Parameters
    ----------
    start : str, optional
        The start date, which defaults to "2000-01-01"
    end : str, optional
        The end date, which defaults to "2001-01-01"
    freq : str, optional
        The frequency of the time coordinate, which defaults to "M" (monthly).
        See https://numpy.org/doc/stable/reference/arrays.datetime.html#basic-datetimes
        for more information.

    Returns
    -------
    times: `xarray.DataArray`
        Time-like coordinate with the given start/end dates and frequency

    """

    if start >= end:
        raise ValueError("start time must be before end time")

    times = np.arange(start, end, dtype=f"datetime64[{freq}]").astype("datetime64[ns]")
    times = xr.DataArray(times, dims=["time"], name="time").assign_coords(
        {"time": times}
    )
    times.attrs["standard_name"] = "time"
    times.attrs["units"] = f"days since {start}"

    return times


def create_dummy_geo_field(
    lons: None | xr.DataArray = None,
    lats: None | xr.DataArray = None,
    levs: None | xr.DataArray = None,
    times: None | xr.DataArray = None,
    name: str = "dummy",
    attrs: dict = {},
) -> xr.DataArray:
    """
    Mock a geophysical field with random data

    Parameters
    ----------
    lons : `xarray.DataArray`
        The longitudes of the data. Defaults to None (longitudes excluded)
    lats : `xarray.DataArray`
        The latitudes of the data. Defaults to None (latitudes excluded)
    levs : `xarray.DataArray`, optional
        The vertical levels of the data. Defaults to None (vertical levels excluded)
    times : `xarray.DataArray`, optional
        The times of the data. Defaults to None (times excluded)
    name : str, optional
        The name of the data. Defaults to "dummy"
    attrs : dict, optional
        The attributes of the data. Defaults to an empty dictionary.

    Returns
    -------
    dummy_data: `xarray.DataArray` 
        Random data with corresponding geophysical coordinates

    """

    dat_shape = []
    coords = []
    dims = []

    for coord in (lons, lats, levs, times):
        if coord is not None:
            dat_shape.insert(0, coord.size)
            coords.insert(0, coord)
            dims.insert(0, coord.name)

    if len(dat_shape) == 0:
        msg = "At least one of 'lons', 'lats', 'levs', or 'times' must be provided"
        raise ValueError(msg)

    dummy = np.random.normal(size=dat_shape).astype("float32")
    da = xr.DataArray(dummy, coords=coords, dims=dims, name=name)
    da.attrs = attrs
    return da


def create_dummy_geo_dataset(
    field_names: list[str],
    lons: None | xr.DataArray = None,
    lats: None | xr.DataArray = None,
    levs: None | xr.DataArray = None,
    times: None | xr.DataArray = None,
    field_attrs: None | dict = None,
) -> xr.Dataset:
    """
    Mock a dataset of geophysical fields with random data

    Parameters
    ----------
    field_names : list[str]
        The names of the fields
    lons : `xarray.DataArray`
        The longitudes of the data
    lats : `xarray.DataArray`
        The latitudes of the data
    levs : `xarray.DataArray`, optional
        The vertical levels of the data. Defaults to None (vertical levels excluded)
    times : `xarray.DataArray`, optional
        The times of the data. Defaults to None (times excluded)
    field_attrs : dict, optional
        The attributes of the data. Defaults to None. If provided, this should be
        a nested dictionary with the outer keys corresponding to the field names,
        and inner keys/dicts corresponding to the attributes for each field.
    
    Returns
    -------
    dummy_dataset: `xarray.Dataset` 
        Dataset with fields having random data matching the given geophysical coordinates

    """
    if field_attrs is not None and len(field_attrs) != len(field_names):
        raise ValueError(
            f"Number of field names ({len(field_names)}) must match number of keys in field attributes ({len(field_attrs)})"
        )

    ds = xr.Dataset()
    for name in field_names:
        da = create_dummy_geo_field(lons, lats, levs, times, name)
        ds[name] = da
        if field_attrs is not None:
            ds[name].attrs = field_attrs[name]
    return ds
