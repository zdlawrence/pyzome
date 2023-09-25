import pytest
import numpy as np

from pyzome.accessor import PyzomeDataArrayAccessor, PyzomeDatasetAccessor
from pyzome.checks import check_for_logp_coord
from pyzome.mock_data import (
    create_dummy_geo_field,
    create_dummy_geo_dataset,
    lon_coord,
    lat_coord,
    plev_coord,
)


def test_pzm_accessor_coord_map():
    """Test that the internal coordinate map works correctly"""
    da = create_dummy_geo_field(
        lon_coord(10, name="longitude"), lat_coord(10), plev_coord(3, name="pressures")
    )

    assert da.pzm.coord_map("lon") == "longitude"
    assert da.pzm.coord_map("lat") == "lat"
    assert da.pzm.coord_map("plev") == "pressures"

    with pytest.raises(KeyError) as e:
        da.pzm.coord_map("foo")
    assert "pyzome coordinate 'foo' not identified in original data" in str(e.value)


def test_pzm_accessor_coord_props():
    """Test that the coordinate properties lead to the right references"""
    da = create_dummy_geo_field(
        lon_coord(10, name="lons"),
        lat_coord(10, name="lats"),
        plev_coord(3, name="pres"),
    )
    da = da.assign_coords({"zonal_wavenum": 1}).expand_dims("zonal_wavenum")

    assert da["lons"].equals(da.pzm.lon)
    assert da["lats"].equals(da.pzm.lat)
    assert da["pres"].equals(da.pzm.plev)
    assert da["zonal_wavenum"].equals(da.pzm.zonal_wavenum)


def test_pzm_dataset_accessor():
    ds = create_dummy_geo_dataset(
        field_names=["u", "v", "T"],
        lons=lon_coord(2.5),
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
    )
    assert isinstance(ds.pzm, PyzomeDatasetAccessor)

    # zonal mean should remove lon coord
    ds_zm = ds.pzm.zonal_mean(strict=True)
    with pytest.raises(KeyError):
        ds_zm.pzm.coord_map("lon")

    # meridional mean should remove lat coord
    ds_zm_mm = ds_zm.pzm.meridional_mean(-90, 90, strict=True)
    with pytest.raises(KeyError):
        ds_zm_mm.pzm.coord_map("lat")

    # check that logp altitude coord gets added
    assert check_for_logp_coord(ds.pzm.add_logp_altitude())


def test_pzm_dataarray_accessor():
    da = create_dummy_geo_field(
        lon_coord(2.5),
        lat_coord(2.5),
        plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
    )
    assert isinstance(da.pzm, PyzomeDataArrayAccessor)

    # zonal mean should remove lon coord
    da_zm = da.pzm.zonal_mean(strict=True)
    with pytest.raises(KeyError):
        da_zm.pzm.coord_map("lon")

    # meridional mean should remove lat coord
    da_zm_mm = da_zm.pzm.meridional_mean(-90, 90, strict=True)
    with pytest.raises(KeyError):
        da_zm_mm.pzm.coord_map("lat")

    # check that logp altitude coord gets added
    assert check_for_logp_coord(da_zm_mm.pzm.add_logp_altitude())

    # zonal wave coeffs adds zonal_wavenum and removes lon
    fc = da.pzm.zonal_wave_coeffs(waves=[1, 2, 3])
    assert fc.pzm.coord_map("zonal_wavenum") == "zonal_wavenum"
    assert np.allclose(fc.pzm.zonal_wavenum.values, [1, 2, 3])
    with pytest.raises(KeyError):
        fc.pzm.coord_map("lon")

    # inflate zonal wave coeffs and reconstruct; should add lons back
    # and remove zonal_wavenums
    fc_inflated = fc.pzm.inflate_zonal_wave_coeffs()
    recons = fc_inflated.pzm.filter_by_zonal_wave_truncation([1, 2, 3])
    assert recons.pzm.lon.equals(da.pzm.lon)
    with pytest.raises(KeyError):
        recons.pzm.coord_map("zonal_wavenum")

    # zonal wave contributions should add back lon but keep wavenums
    contribs = fc_inflated.pzm.zonal_wave_contributions([1, 2, 3])
    assert contribs.pzm.lon.equals(da.pzm.lon)
    assert contribs.pzm.zonal_wavenum.equals(fc.pzm.zonal_wavenum)
