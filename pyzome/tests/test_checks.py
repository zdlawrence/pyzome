import pytest

import numpy as np
import xarray as xr

from pyzome.checks import (
    has_global_regular_lons,
    infer_xr_coord_names,
    check_var_SI_units,
    var_units
)
from pyzome.exceptions import (
    AttrError,
    CoordinateError,
    LongitudeError,
    UnitError
)
from pyzome.testing import create_dummy_xr_data


def test_global_regular_lons():
    """ Test that longitudes span globe and are evenly spaced """

    # Full 360 span from 0 to 360
    lons_full_360 = np.arange(0, 360, 2.)
    assert has_global_regular_lons(lons_full_360, enforce=True)

    # Full 360 span from -180 to 180
    lons_full_180 = np.arange(-180, 180, 3.)
    assert has_global_regular_lons(lons_full_180, enforce=True)

def test_partial_lons_fail():
    """ Test that partial longitude coverage fails """

    # Partial span from 45 to 225
    lons_part_equal = np.arange(45, 225, 5.)
    assert has_global_regular_lons(lons_part_equal, enforce=False) is False
    with pytest.raises(LongitudeError) as e:
        has_global_regular_lons(lons_part_equal, enforce=True)
    assert "longitudes do not span all 360 degrees" in str(e.value)


def test_irregular_lons_fail():
    """ Test that irregularly spaced longitudes fail """

    # Full span with unequal spacing
    lons_full_unequal = np.array([0,20,40,60,90,120,150,180,225,270,315,359])
    assert has_global_regular_lons(lons_full_unequal, enforce=False) is False
    with pytest.raises(LongitudeError) as e:
        has_global_regular_lons(lons_full_unequal, enforce=True)
    assert "longitudes are not equally spaced" in str(e.value)


def test_infer_xr_coord_names():
    """ Test that standard coordinates can be inferred/detected """

    da = create_dummy_xr_data(["lev","lat","lon"])
    mapper = {"lev":"lev","lat":"lat","lon":"lon"}
    assert infer_xr_coord_names(da, required = ["lev","lat","lon"]) == mapper

    da = create_dummy_xr_data(["time","level","latitude","longitude"])
    mapper = {"lev":"level","lat":"latitude","lon":"longitude"}
    assert infer_xr_coord_names(da, required = ["lev","lat","lon"]) == mapper


def test_infer_multiple_coords_fail():
    """ Test detection of multiple ambiguous coordinate names """

    da = create_dummy_xr_data(["lon","LONGI"])
    with pytest.raises(CoordinateError) as e:
        infer_xr_coord_names(da, required = ["lon"])
    assert "Found multiple coordinates in dat matching the" in str(e.value)


def test_infer_coords_missing_fail():
    """ Test that a required coordinate missing is detected """

    da = create_dummy_xr_data(["lev","lat"])
    with pytest.raises(CoordinateError) as e:
        infer_xr_coord_names(da, required = ["lon"])
    assert "Unable to match any of the coordinates in dat for" in str(e.value)


def test_check_var_SI_units():
    """ Test that all relevant SI units can be detected """

    da = create_dummy_xr_data(["lat"])
    for var in var_units:
        for unit in var_units[var]:
            da.attrs["units"] = unit
        assert check_var_SI_units(da, var, enforce=True)


def test_check_var_SI_units_missing():
    """ Test that missing units attribute throws an error """

    da = create_dummy_xr_data(["lat"])
    with pytest.raises(AttrError) as e:
        check_var_SI_units(da, "wind", enforce=False)
    assert "units is not an attribute of the given DataArray" in str(e.value)


def test_check_SI_units_wrong():
    """ Test that wrong/non-SI units are detected """

    da = create_dummy_xr_data(["lat"])
    da.attrs["units"] = "N/A"

    assert check_var_SI_units(da, "temperature", enforce=False) is False
    with pytest.raises(UnitError) as e:
        check_var_SI_units(da, "pressure", enforce=True)
    assert "do not match SI units for the" in str(e.value)