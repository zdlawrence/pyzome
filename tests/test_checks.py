import pytest

import numpy as np

from pyzome.checks import (
    has_global_regular_lons,
    infer_xr_coord_names,
    check_var_SI_units,
    check_for_logp_coord,
    var_units,
)
from pyzome.exceptions import AttrError, CoordinateError, LongitudeError, UnitError
from pyzome.mock_data import create_dummy_geo_field, lon_coord, lat_coord, plev_coord


def test_global_regular_lons():
    """Test that longitudes span globe and are evenly spaced"""

    # Full 360 span from 0 to 360
    lons_full_360 = np.arange(0, 360, 2.0)
    assert has_global_regular_lons(lons_full_360, enforce=True)

    # Full 360 span from -180 to 180
    lons_full_180 = np.arange(-180, 180, 3.0)
    assert has_global_regular_lons(lons_full_180, enforce=True)


def test_partial_lons_fail():
    """Test that partial longitude coverage fails"""

    # Partial span from 45 to 225
    lons_part_equal = np.arange(45, 225, 5.0)
    assert has_global_regular_lons(lons_part_equal, enforce=False) is False
    with pytest.raises(LongitudeError) as e:
        has_global_regular_lons(lons_part_equal, enforce=True)
    assert "longitudes do not span all 360 degrees" in str(e.value)


def test_irregular_lons_fail():
    """Test that irregularly spaced longitudes fail"""

    # Full span with unequal spacing
    lons_full_unequal = np.array([0, 20, 40, 60, 90, 120, 150, 180, 225, 270, 315, 359])
    assert has_global_regular_lons(lons_full_unequal, enforce=False) is False
    with pytest.raises(LongitudeError) as e:
        has_global_regular_lons(lons_full_unequal, enforce=True)
    assert "longitudes are not equally spaced" in str(e.value)


def test_infer_xr_coord_names():
    """Test that standard coordinates can be inferred/detected"""

    da = create_dummy_geo_field(lon_coord(10), lat_coord(10), plev_coord(3))
    mapper = {"lev": "lev", "lat": "lat", "lon": "lon"}
    assert infer_xr_coord_names(da, required=["lev", "lat", "lon"]) == mapper

    da = create_dummy_geo_field(
        lon_coord(10, name="longitude"),
        lat_coord(10, name="latitude"),
        plev_coord(3, name="level"),
    )
    print(da)
    mapper = {"lev": "level", "lat": "latitude", "lon": "longitude"}
    assert infer_xr_coord_names(da, required=["lev", "lat", "lon"]) == mapper


def test_infer_multiple_coords_fail():
    """Test detection of multiple ambiguous coordinate names"""

    da = create_dummy_geo_field(lon_coord(10), lon_coord(10, name="LONGI"))
    with pytest.raises(CoordinateError) as e:
        infer_xr_coord_names(da, required=["lon"])
    assert "Found multiple coordinates in dat matching the" in str(e.value)


def test_infer_coords_missing_fail():
    """Test that a required coordinate missing is detected"""

    da = create_dummy_geo_field(
        lon_coord(10, name="notgitude"), lat_coord(10, name="latitude")
    )
    with pytest.raises(CoordinateError) as e:
        infer_xr_coord_names(da, required=["lon"])
    assert "Unable to match any of the coordinates in dat for" in str(e.value)


def test_check_var_SI_units():
    """Test that all relevant SI units can be detected"""

    da = create_dummy_geo_field(lon_coord(10), lat_coord(10))
    for var in var_units:
        for unit in var_units[var]:
            da.attrs["units"] = unit
        assert check_var_SI_units(da, var, enforce=True)


def test_check_var_SI_units_missing():
    """Test that missing units attribute throws an error"""

    da = create_dummy_geo_field(lon_coord(10), lat_coord(10))
    with pytest.raises(AttrError) as e:
        check_var_SI_units(da, "wind", enforce=False)
    assert "units is not an attribute of the given DataArray" in str(e.value)


def test_check_SI_units_wrong():
    """Test that wrong/non-SI units are detected"""

    da = create_dummy_geo_field(lon_coord(10), lat_coord(10), attrs={"units": "N/A"})

    assert check_var_SI_units(da, "temperature", enforce=False) is False
    with pytest.raises(UnitError) as e:
        check_var_SI_units(da, "pressure", enforce=True)
    assert "do not match SI units for the" in str(e.value)


def test_check_for_logp_coord():
    """Test that check_for_logp_coord properly detects a log-p altitude coordinate"""
    da = create_dummy_geo_field(
        lon_coord(10), 
        lat_coord(10), 
        plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
    )

    assert check_for_logp_coord(da) is False 
    with pytest.raises(CoordinateError) as e:
        check_for_logp_coord(da, enforce=True)
    assert "z is not a coordinate in the data" in str(e.value) 

    da = da.assign_coords({"z": np.log(da.lev)})
    assert check_for_logp_coord(da) is False
    with pytest.raises(AttrError) as e:
        check_for_logp_coord(da, enforce=True)
    assert "z is missing at least one of the required attributes" in str(e.value)

    da.z.attrs["long_name"] = "not the right name"
    da.z.attrs["units"] = "not the right units"
    da.z.attrs["note"] = "not the right note"
    assert check_for_logp_coord(da) is False
    with pytest.raises(AttrError) as e:
        check_for_logp_coord(da, enforce=True)
    assert "z must have a long_name = 'log-pressure altitude'" in str(e.value)

    da.z.attrs["long_name"] = "log-pressure altitude"
    assert check_for_logp_coord(da) is False
    with pytest.raises(AttrError) as e:
        check_for_logp_coord(da, enforce=True)
    assert "z must have a note = 'added by pyzome'" in str(e.value)

    da.z.attrs["note"] = "added by pyzome"
    assert check_for_logp_coord(da) is False
    with pytest.raises(UnitError) as e:
        check_for_logp_coord(da, enforce=True)
    assert "do not match SI units for the" in str(e.value)

    da.z.attrs["units"] = "m"
    assert check_for_logp_coord(da, enforce=True) is True


    