import pytest
import numpy as np

from pyzome.basic import zonal_mean, meridional_mean
from pyzome.exceptions import AttrError, CoordinateError, LongitudeError, UnitError
from pyzome.mock_data import create_dummy_geo_field, lon_coord, lat_coord, plev_coord


def test_zonal_mean():
    """Test that zonal mean  works correctly"""
    da = create_dummy_geo_field(lon_coord(10), lat_coord(10), plev_coord(3))
    da_zonal_mean = zonal_mean(da, strict=True)

    # Basic tests that the function works as expected
    assert np.allclose(da_zonal_mean.values, np.mean(da.values, axis=2))
    assert da_zonal_mean.shape == (10, 19)
    assert da_zonal_mean.dims == ("lev", "lat")


def test_meridional_mean():
    """Test that meridional mean works correctly"""
    da = create_dummy_geo_field(lon_coord(10), lat_coord(10), plev_coord(3))
    da_meridional_mean = meridional_mean(da, -90, 90)

    # Basic tests that the function works as expected
    global_mean = np.average(
        da.values, axis=1, weights=np.cos(np.deg2rad(da.lat.values))
    )
    assert np.allclose(da_meridional_mean.values, global_mean)
    assert da_meridional_mean.shape == (10, 36)
    assert da_meridional_mean.dims == ("lev", "lon")


def test_meridional_mean_wrong_lat_order_fail():
    """Test that meridional mean fails with lats in wrong order"""
    da = create_dummy_geo_field(lon_coord(10), lat_coord(10), plev_coord(3))
    with pytest.raises(ValueError) as e:
        da_meridional_mean = meridional_mean(da, 90, -90)
    assert "lat1 must be less than lat2" in str(e.value)


def test_meridional_mean_too_few_lats_fail():
    """Test that meridional mean fails with too few lats"""
    da = create_dummy_geo_field(lon_coord(10), lat_coord(10), plev_coord(3))
    with pytest.raises(ValueError) as e:
        da_meridional_mean = meridional_mean(da.sel(lat=slice(30, 60)), 0, 90)
    assert "data only contains lats in range of" in str(e.value)
