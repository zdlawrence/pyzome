from __future__ import annotations
import re

import numpy as np
import xarray as xr

from .exceptions import LongitudeError, CoordinateError, AttrError, UnitError


coord_regex = {
    "lon": re.compile("lon[a-z]*"),
    "lat": re.compile("lat[a-z]*"),
    "lev": re.compile("(p?lev|pre?s|isobaric)[a-z]*"),
    "zonal_wavenum": re.compile("(zonal_wavenum|wavenum_lon|lon_wavenum)[a-z]*"),
}

var_units = {
    "altitude": {"m", "meters"},
    "heatflux": {
        "K m s-1",
        "m K s-1",
        "K*m*s-1",
        "m*K*s-1",
        "K * m * s-1",
        "m * K * s-1",
        "K*m/s",
        "m*K/s",
        "K * m / s",
        "m * K / s",
        "Kelvin meters per second",
        "meters Kelvin per second",
    },
    "momentumflux": {
        "m+2 s-2",
        "m+2*s-2",
        "m+2 * s-2",
        "m+2/s+2",
        "m+2 / s+2",
        "meters squared per second squared",
    },
    "prs_vertmomflux": {
        "Pa m s-2",
        "m Pa s-2",
        "Pa*m*s-2",
        "m*Pa*s-2",
        "Pa * m * s-2",
        "m * Pa * s-2",
        "Pa*m/s",
        "m*Pa/s",
        "Pa * m / s",
        "m * Pa / s",
        "Pascal meters per second squared",
        "meter Pascals per second squared",
    },
    "pressure": {"Pa", "Pascals"},
    "temperature": {"K", "degK", "Kelvin"},
    "prs_vvel": {"Pa s-1", "Pa*s-1", "Pa * s-1", "Pa/s", "Pa / s", "Pascals per second"},
    "wind": {"m s-1", "m*s-1", "m * s-1", "m/s", "m / s", "meters per second"},
}


def has_global_regular_lons(
    lons: xr.DataArray | np.ndarray, enforce: bool = False
) -> bool:
    r"""Checks an array of longitudes to ensure it is regularly spaced
    and spans 360 degrees. This is primarily to allow for strict checks
    on data before trying to take zonal means, Fourier transforms, etc.

    Parameters
    ----------
    lons : `xarray.DataArray` or `np.array`
        data containing the longitudes, in degrees

    enforce : bool, optional
        If True, the function will throw an error if the provided
        longitudes are irregular and/or only span a fraction of the globe.
        Defaults to False.

    Returns
    -------
    bool
        True if the lons span 360 degrees and regularly spaced, False if not

    """

    # generally for regularly gridded data, the difference
    # between the max lon and min lon plus the lon delta
    # should equal 360 degrees
    span = float(lons.max() - lons.min())
    dlon = float(lons[1] - lons[0])

    # check that all the longitude differences are close to
    # the single dlon, to within 1.0e-3 degrees
    equal_spacing = np.allclose(np.diff(lons), dlon, rtol=0.0, atol=1.0e-3)

    # now check that the span+londelta cover 360 degrees
    full_span = (span + dlon) >= 360.0

    if (enforce is True) and (equal_spacing is False):
        msg = "longitudes are not equally spaced"
        raise LongitudeError(msg)
    elif (enforce is True) and (full_span is False):
        msg = "longitudes do not span all 360 degrees"
        raise LongitudeError(msg)

    return equal_spacing and full_span


def infer_xr_coord_names(
    dat: xr.DataArray | xr.Dataset, required: list[str] = []
) -> dict:
    r"""A convenience function that identifies commonly used coordinate names
    for gridded earth datasets. This function enables other functions in
    pyzome to perform operations across coordinates without the user having
    to specify  the coordinate names or change their data to use pre-defined
    coordinate names. Uses regex patterns to identify variations of "lon",
    "lat", and "lev".

    Parameters
    ----------
    dat : `xarray.DataArray` or `xr.Dataset`
        The data containing the coordinates of lat, lon, lev, etc.

    required: List of strings, optional
        If this kwarg is defined, this function will throw errors if it is
        unable to find a name that matches with one of the coordinate-name
        categories defined by coord_regex. Defaults to an empty list,
        meaning the function will not check for any required coordinates.

    Returns
    -------
    coord_names: dict 
        string keys and values, with the keys being the
        coordinate name category (e.g., lat, lon), and the values being the
        actual coordinate name in the given xarray data. E.g,
        {"lat":"latitude"} is a possible return value in which latitude was
        detected as the coordinate for the "lat" category.

    """

    coord_names = {}
    dat_coords = list(dat.coords)
    for dc in dat_coords:
        for coord in coord_regex:
            if coord_regex[coord].match(dc.lower()): # type: ignore
                # Check if more than one coord in dat matches the same pattern
                if coord in coord_names:
                    msg = (
                        "Found multiple coordinates in dat matching the "
                        + f"pattern for '{coord}: {coord_names[coord]} & {dc}"
                    )
                    raise CoordinateError(msg)
                else:
                    coord_names[coord] = dc

    for req in required:
        if req not in coord_names:
            msg = f"Unable to match any of the coordinates in dat for {req}"
            raise CoordinateError(msg)

    return coord_names


def check_var_SI_units(dat: xr.DataArray, var: str, enforce: bool = False) -> bool:
    r"""A function that checks whether the units attribute of a DataArray
    matches SI units for a specific variable category.

    In cases where units matter, pyzome assumes that variables are provided
    in SI units. While there are python libraries that could be used to
    automatically detect and do conversions, pyzome errs on the side of being
    small, simple, and strict (e.g., so that a user would have to explicitly
    do unit conversions themselves, and set units attributes before applying
    a relevant pyzome function).


    Parameters
    ----------
    dat : `xarray.DataArray`
        The data with attributes to check for SI units.

    var: string
        The variable category for checking SI units. These are defined by the
        var_units global dictionary.

    enforce : bool, optional
        If True, the function will throw an error if the provided
        units string in the DataArray does not match the SI units for the
        given variable category. Defaults to False.

    Returns
    -------
    bool
        True if an SI units match is found, False otherwise.

    """

    if "units" not in dat.attrs:
        msg = "units is not an attribute of the given DataArray"
        raise AttrError(msg)

    units_SI = dat.units in var_units[var]
    if (enforce is True) and (units_SI is False):
        msg = (
            f"The units '{dat.units}' do not match SI units for the {var}"
            + " category."
        )
        raise UnitError(msg)

    return units_SI


def check_for_logp_coord(dat: xr.DataArray, enforce: bool = False) -> bool:
    r"""A function that checks whether a log-pressure altitude coordinate
    (assumed to be created by pyzome) exists in the given DataArray. Uses
    a combination of units and long_name to check.

    Parameters
    ----------
    dat : `xarray.DataArray`
        The data to check.

    enforce : bool, optional
        If True, the function will throw an error if the provided
        DataArray does not contain a coordinate named "z" with correct
        units and long_name attributes.

    Returns
    -------
    bool
        True if log-pressure altitude coordinate is found. False otherwise.

    """

    if "z" not in dat.coords:
        if enforce is True:
            msg = "z is not a coordinate in the data"
            raise CoordinateError(msg)
        return False

    if not {"units", "long_name", "note"} <= dat.z.attrs.keys():
        if enforce is True:
            msg = "z is missing at least one of the required attributes 'units', 'long_name', and/or 'note'"
            raise AttrError(msg)
        return False

    if dat.z.attrs["long_name"] != "log-pressure altitude":
        if enforce is True:
            msg = "z must have a long_name = 'log-pressure altitude'"
            raise AttrError(msg)
        return False

    if dat.z.attrs["note"] != "added by pyzome":
        if enforce is True:
            msg = "z must have a note = 'added by pyzome'"
            raise AttrError(msg)
        return False

    return check_var_SI_units(dat.z, "altitude", enforce=enforce)
