import pytest

import numpy as np


from pyzome.mock_data import (
    create_dummy_geo_field,
    create_dummy_geo_dataset,
    _make_regular_coord,
    lon_coord,
    lat_coord,
    plev_coord,
    time_coord,
)


def test_reg_coord_start_and_step():
    """Test that regular coordinate is created correctly"""
    da = _make_regular_coord(1, 10, 2, "coord", False)
    np.testing.assert_allclose(da.values, np.arange(1, 10, 2))

    da = _make_regular_coord(0, 10, 2, "coord", False)
    np.testing.assert_allclose(da.values, np.arange(0, 10, 2))


def test_reg_coord_inclusive():
    """Test that _make_regular_coord is inclusive of the right-side value"""
    da = _make_regular_coord(0, 10, 2, "coord", True)
    np.testing.assert_allclose(da.values, np.arange(0, 12, 2))

    da = _make_regular_coord(0, 10, 2, "coord", False)
    np.testing.assert_allclose(da.values, np.arange(0, 10, 2))


def test_reg_coord_custom_name():
    """Test that _make_regular_coord can set a custom name"""
    da = _make_regular_coord(1, 10, 2, "testname", False)
    assert da.name == "testname"


def test_reg_coord_invalid_resolution():
    """
    Test that _make_regular_coord raises ValueError for invalid step size
    for the given limits
    """
    with pytest.raises(ValueError) as e:
        _make_regular_coord(90, -90, 10, "coord", True)
    assert "Invalid step for the given limits" in str(e.value)

    with pytest.raises(ValueError):
        _make_regular_coord(-180, 180, -10, "coord", False)
    assert "Invalid step for the given limits" in str(e.value)


def test_reg_coord_zero_step():
    """Test that _make_regular_coord raises ValueError for zero step size"""
    with pytest.raises(ValueError) as e:
        _make_regular_coord(-180, 180, 0, "coord", False)
    assert "step cannot be zero" in str(e.value)


def test_reg_coord_equal_step():
    """Test that _make_regular_coord raises ValueError for equal left and right values"""
    with pytest.raises(ValueError) as e:
        _make_regular_coord(10, 10, 1, "coord", False)
    assert "left- and right-side values cannot be equal" in str(e.value)


def test_lats_default_values():
    """Test that lat_coord creates a latitude-like coordinate correctly for default values"""
    da = lat_coord(30)
    assert da.name == "lat"
    np.testing.assert_allclose(da.values, np.arange(-90, 91, 30))
    assert da.attrs["standard_name"] == "latitude"
    assert da.attrs["units"] == "degrees_north"


def test_lats_custom_values():
    """Test that lat_coord creates a latitude-like coordinate correctly for custom values"""
    da = lat_coord(20, "latitude", -80, 80, False)
    assert da.name == "latitude"
    np.testing.assert_allclose(da.values, np.arange(-80, 80, 20))

    da = lat_coord(-20, "latitude", 80, -80, False)
    np.testing.assert_allclose(da.values, np.arange(80, -80, -20))


def test_lats_invalid_latitudes():
    """Test that lat_coord raises ValueError for invalid latitude limits"""
    with pytest.raises(ValueError) as e:
        lat_coord(10, left_lim=-91, right_lim=90)
    assert "Latitudes cannot be below -90 or above 90" in str(e.value)

    with pytest.raises(ValueError) as e:
        lat_coord(10, left_lim=-90, right_lim=91)
    assert "Latitudes cannot be below -90 or above 90" in str(e.value)


def test_lons_default_values():
    """Test that lon_coord creates a longitude-like coordinate correctly for default values"""
    da = lon_coord(30)
    assert da.name == "lon"
    np.testing.assert_allclose(da.values, np.arange(0, 360, 30))
    assert da.attrs["standard_name"] == "longitude"
    assert da.attrs["units"] == "degrees_east"


def test_lons_custom_values():
    """Test that lon_coord creates a longitude-like coordinate correctly for custom values"""
    da = lon_coord(45, "longitude", 0, 180, True)
    assert da.name == "longitude"
    np.testing.assert_allclose(da.values, np.arange(0, 181, 45))


def test_invalid_longitude_limits():
    """Test that lon_coord raises ValueError for invalid longitude limits"""
    with pytest.raises(ValueError) as e:
        lon_coord(30, "longitude", -180, 400)
    assert "Proper longitude range should be within [-180,180] or [0, 360]" in str(
        e.value
    )

    with pytest.raises(ValueError) as e:
        lon_coord(30, "longitude", 200, -10)
    assert "Proper longitude range should be within [-180,180] or [0, 360]" in str(
        e.value
    )

    with pytest.raises(ValueError) as e:
        lon_coord(30, "longitude", -200, 180)
    assert "lower longitude limit cannot be below -180" in str(e.value)

    with pytest.raises(ValueError) as e:
        lon_coord(30, "longitude", 30, 390)
    assert "upper longitude limit cannot exceed 360" in str(e.value)


def test_plev_default_values():
    """Test that plev_coord creates a pressure-like coordinate correctly for default values"""
    da = plev_coord(3)
    assert da.name == "lev"
    np.testing.assert_allclose(da.values, np.logspace(3, 0, 3 * 3 + 1))
    assert da.attrs["standard_name"] == "pressure"
    assert da.attrs["units"] == "hPa"


def test_plev_custom_values():
    """Test that plev_coord creates a pressure-like coordinate correctly for custom values"""
    da = plev_coord(5, 2, 0, "pressure", "Pa")
    assert da.name == "pressure"
    np.testing.assert_allclose(da.values, np.logspace(2, 0, 5 * 2 + 1))
    assert da.attrs["standard_name"] == "pressure"
    assert da.attrs["units"] == "Pa"


def test_plev_negative_levels_per_decade():
    """Test that plev_coord raises ValueError for negative levels_per_decade"""
    with pytest.raises(ValueError) as e:
        plev_coord(-3)
    assert "'levels_per_decade' must be greater than 0" in str(e.value)


def test_plev_same_exponent_limits():
    """Test that plev_coord raises ValueError for same left_lim_exponent and right_lim_exponent"""
    with pytest.raises(ValueError) as e:
        plev_coord(3, 2, 2)
    assert "'left_lim_exponent' and 'right_lim_exponent' must not be the same" in str(
        e.value
    )


def test_time_default_values():
    """Test that time_coord creates a time-like coordinate correctly for default values"""
    da = time_coord()
    assert da.name == "time"
    np.testing.assert_array_equal(
        da.values,
        np.arange("2000-01-01", "2001-01-01", dtype="datetime64[M]").astype(
            "datetime64[ns]"
        ),
    )
    assert da.attrs["standard_name"] == "time"
    assert da.attrs["units"] == "days since 2000-01-01"


def test_time_custom_values():
    """Test that time_coord creates a time-like coordinate correctly for custom values"""
    da = time_coord("2000-01-01", "2002-01-01", "D")
    assert da.name == "time"
    np.testing.assert_array_equal(
        da.values,
        np.arange("2000-01-01", "2002-01-01", dtype="datetime64[D]").astype(
            "datetime64[ns]"
        ),
    )


def test_time_start_after_end():
    """Test that time_coord raises ValueError for start time after end time"""
    with pytest.raises(ValueError) as e:
        time_coord("2001-01-02", "2000-01-01")
    assert "start time must be before end time" in str(e.value)


def test_invalid_frequency():
    """Test that time_coord raises a TypeError for invalid frequency"""
    with pytest.raises(TypeError):
        time_coord("2000-01-01", "2002-01-01", "Z")


def test_create_dummy_geo_field():
    """Test that create_dummy_geo_field creates a dummy field correctly"""
    lons = lon_coord(10)
    lats = lat_coord(10)
    levs = plev_coord(3)
    times = time_coord()

    da = create_dummy_geo_field(
        lons, lats, levs, times, name="test_field", attrs={"unit": "K"}
    )

    # Check the dimensions
    assert da.dims == ("time", "lev", "lat", "lon")

    # Check the sizes
    assert da.sizes == {
        "time": times.size,
        "lev": levs.size,
        "lat": lats.size,
        "lon": lons.size,
    }

    # Check the name
    assert da.name == "test_field"

    # Check the attributes
    assert da.attrs == {"unit": "K"}

    # Check without levels and time
    da = create_dummy_geo_field(lons, lats)

    # Check the dimensions
    assert da.dims == ("lat", "lon")

    # Check the sizes
    assert da.sizes == {"lat": lats.size, "lon": lons.size}


def test_create_dummy_geo_field_no_dimensions():
    """Test that create_dummy_geo_field raises a ValueError for no dimensions"""
    with pytest.raises(ValueError) as e:
        create_dummy_geo_field(None, None, None, None)
    assert "At least one of 'lons', 'lats', 'levs', or 'times' must be provided" in str(
        e.value
    )


def test_create_dummy_geo_field_invalid_arguments():
    """Test that create_dummy_geo_field raises errors for invalid arguments"""
    invalid_dataarray = "not a DataArray"
    lons = lon_coord(10)
    lats = lat_coord(10)

    with pytest.raises(AttributeError):
        create_dummy_geo_field(invalid_dataarray, lats)

    with pytest.raises(AttributeError):
        create_dummy_geo_field(lons, invalid_dataarray)


def test_create_dummy_geo_dataset():
    """Test that create_dummy_geo_dataset creates a dummy Dataset correctly"""
    lons = lon_coord(10)
    lats = lat_coord(10)
    levs = plev_coord(5)
    times = time_coord()
    field_names = ["temp", "precip", "wind"]
    field_attrs = {
        "temp": {"units": "K"},
        "precip": {"units": "mm/hr"},
        "wind": {"units": "m/s"},
    }

    ds = create_dummy_geo_dataset(field_names, lons, lats, levs, times, field_attrs)

    # Check if the created Dataset has the expected DataArrays
    for name in field_names:
        assert name in ds

        # Check if the DataArray has the expected attributes
        assert ds[name].attrs == field_attrs[name]

        # Check the dimensions
        assert ds[name].dims == ("time", "lev", "lat", "lon")

        # Check the sizes
        assert ds[name].sizes == {
            "time": times.size,
            "lev": levs.size,
            "lat": lats.size,
            "lon": lons.size,
        }


def test_create_dummy_geo_dataset_invalid_arguments():
    """Test that create_dummy_geo_dataset raises errors for invalid arguments"""
    invalid_dataarray = "not a DataArray"
    lons = lon_coord(10)
    lats = lat_coord(10)
    field_names = ["temp"]

    with pytest.raises(AttributeError):
        create_dummy_geo_dataset(field_names, invalid_dataarray, lats)

    with pytest.raises(AttributeError):
        create_dummy_geo_dataset(field_names, lons, invalid_dataarray)

    mismatched_field_attrs = {"temp": {"units": "K"}, "missing_field": {}}
    with pytest.raises(ValueError) as e:
        create_dummy_geo_dataset(
            field_names, lons, lats, field_attrs=mismatched_field_attrs
        )
    assert (
        "Number of field names (1) must match number of keys in field attributes (2)"
        in str(e.value)
    )
