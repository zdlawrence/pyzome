import pytest

from pyzome.mock_data import create_dummy_geo_dataset, lon_coord, lat_coord, plev_coord
from pyzome.recipes.zmd import create_zonal_mean_dataset


def test_create_zonal_mean_dataset_no_waves():
    """Test that create_zonal_mean_dataset works when no wave-output is requested"""
    ds = create_dummy_geo_dataset(
        field_names=["u", "v", "w", "T", "Z"],
        lons=lon_coord(2.5),
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
    )

    zmds = create_zonal_mean_dataset(ds, include_waves=False)
    assert "zonal_wavenum" not in zmds.coords
    for var in ["u", "v", "w", "T", "Z", "uv", "vT", "uw", "wT"]:
        assert var in zmds.data_vars
        assert "units" in zmds[var].attrs
        assert "long_name" in zmds[var].attrs

    ds = ds.drop_vars(["u", "Z"])
    zmds = create_zonal_mean_dataset(ds, include_waves=False)
    for var in ["v", "w", "T", "vT", "wT"]:
        assert var in zmds.data_vars
    for var in ["u", "uv", "uw"]:
        assert var not in zmds.data_vars


def test_create_zonal_mean_dataset_with_waves():
    """Test that create_zonal_mean_dataset works when wave-output is requested"""
    ds = create_dummy_geo_dataset(
        field_names=["u", "v", "w", "T", "Z"],
        lons=lon_coord(2.5),
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
    )

    zmds = create_zonal_mean_dataset(ds, include_waves=True, waves=[1, 2, 3])
    assert "zonal_wavenum" in zmds.coords

    expected_vars = [
        "u",
        "v",
        "w",
        "T",
        "Z",
        "uv",
        "vT",
        "uw",
        "wT",
        "uv_k",
        "vT_k",
        "uw_k",
        "wT_k",
        "Z_k_real",
        "Z_k_imag",
        "T_k_real",
        "T_k_imag",
    ]
    for var in expected_vars:
        assert var in zmds.data_vars
        assert "units" in zmds[var].attrs
        assert "long_name" in zmds[var].attrs

    ds = ds.drop_vars(["u", "Z"])
    zmds = create_zonal_mean_dataset(ds, include_waves=True, waves=[1, 2, 3])

    expected_vars = ["v", "w", "T", "vT", "wT", "vT_k", "wT_k", "T_k_real", "T_k_imag"]
    for var in expected_vars:
        assert var in zmds.data_vars

    expected_missing = ["u", "uv", "uw", "uv_k", "uw_k", "Z_k_real", "Z_k_imag"]
    for var in expected_missing:
        assert var not in zmds.data_vars


def test_create_zonal_mean_dataset_fails_no_valid_fields():
    """Test that create_zonal_mean_dataset fails when no valid fields are provided"""
    ds = create_dummy_geo_dataset(
        field_names=["foo", "bar", "baz"],
        lons=lon_coord(2.5),
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
    )

    with pytest.raises(ValueError) as e:
        zmds = create_zonal_mean_dataset(ds)
    assert "No valid fields found in provided dataset" in str(e.value)
