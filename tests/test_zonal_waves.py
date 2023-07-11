import pytest

import xarray as xr
import numpy as np
import scipy

from pyzome.zonal_waves import (
    zonal_wave_coeffs,
    zonal_wave_ampl_phase,
    inflate_zonal_wave_coeffs,
    filter_by_zonal_wave_truncation,
    zonal_wave_contributions,
    zonal_wave_covariance,
)
from pyzome.mock_data import create_dummy_geo_field, lon_coord, lat_coord


def test_zonal_wave_coeffs_scipy():
    """Test that zonal wave coeffs works correctly with scipy FFT"""
    test_lons = lon_coord(2.0)
    da = create_dummy_geo_field(test_lons, lat_coord(3.0))
    da_zonal_wave_coeffs = zonal_wave_coeffs(da, fftpkg="scipy")
    scipy_fft = scipy.fft.rfft(da.values, axis=1)

    # Basic tests that the function works as expected
    assert da_zonal_wave_coeffs.shape == (61, 91)
    assert da_zonal_wave_coeffs.dims == ("lat", "zonal_wavenum")
    assert da_zonal_wave_coeffs.lon0 == test_lons[0]
    assert da_zonal_wave_coeffs.nlons == test_lons.size
    np.testing.assert_allclose(da_zonal_wave_coeffs.values, scipy_fft)

    # Selecting specific wave numbers
    da_zonal_wave_coeffs = zonal_wave_coeffs(da, waves=[1, 2, 3, 4], fftpkg="scipy")
    assert da_zonal_wave_coeffs.shape == (61, 4)
    np.testing.assert_allclose(da_zonal_wave_coeffs.values, scipy_fft[:, 1:5])


def test_zonal_wave_coeffs_xrft():
    """Test that zonal wave coeffs works correctly with xrft FFT"""
    test_lons = lon_coord(2.0)
    da = create_dummy_geo_field(test_lons, lat_coord(3.0))
    da_zonal_wave_coeffs = zonal_wave_coeffs(da, fftpkg="xrft")
    numpy_fft = np.fft.rfft(da.values, axis=1)

    # Basic tests that the function works as expected
    assert da_zonal_wave_coeffs.shape == (61, 91)
    assert da_zonal_wave_coeffs.dims == ("lat", "zonal_wavenum")
    assert da_zonal_wave_coeffs.lon0 == test_lons[0]
    assert da_zonal_wave_coeffs.nlons == test_lons.size
    np.testing.assert_allclose(da_zonal_wave_coeffs.values, numpy_fft)

    # Selecting specific wave numbers
    da_zonal_wave_coeffs = zonal_wave_coeffs(da, waves=[1, 2, 3, 4], fftpkg="xrft")
    assert da_zonal_wave_coeffs.shape == (61, 4)
    np.testing.assert_allclose(da_zonal_wave_coeffs.values, numpy_fft[:, 1:5])


def test_zonal_wave_coeffs_scipy_xrft_equal():
    """Test that zonal wave coeffs give same results with scipy and xrft"""
    test_lons = lon_coord(2.0)
    da = create_dummy_geo_field(test_lons, lat_coord(3.0))
    da_fc_xrft = zonal_wave_coeffs(da, fftpkg="xrft").astype("complex64")
    da_fc_scipy = zonal_wave_coeffs(da, fftpkg="scipy")

    assert da_fc_xrft.shape == da_fc_scipy.shape
    assert da_fc_xrft.dims == da_fc_scipy.dims
    assert da_fc_xrft.lon0 == da_fc_scipy.lon0
    assert da_fc_xrft.nlons == da_fc_scipy.nlons
    np.testing.assert_allclose(da_fc_xrft.values, da_fc_scipy.values, atol=1.0e-5)


def test_zonal_wave_coeffs_unknown_fftpkg():
    """Test that zonal wave coeffs fails with unknown fftpkg"""
    da = create_dummy_geo_field(lon_coord(10), lat_coord(10))
    with pytest.raises(ValueError) as e:
        da_zonal_wave_coeffs = zonal_wave_coeffs(da, fftpkg="unknown")
    assert "fftpkg must be 'scipy' or 'xrft'" in str(e.value)


def test_inflate_zonal_wave_coeffs():
    """Test that inflate_zonal_wave_coeffs works correctly"""
    da = create_dummy_geo_field(lon_coord(2.0), lat_coord(3.0))

    da_fc = zonal_wave_coeffs(da, fftpkg="scipy", waves=[1, 2, 3, 4])
    da_fc_inflated = inflate_zonal_wave_coeffs(da_fc)
    assert da_fc_inflated.shape == (61, 91)
    assert np.allclose(da_fc_inflated.sel(zonal_wavenum=[1, 2, 3, 4]), da_fc)

    # Test the case when non-consecutive wavenums are initially kept
    da_fc = zonal_wave_coeffs(da, fftpkg="scipy", waves=[1, 5, 10, 20])
    da_fc_inflated = inflate_zonal_wave_coeffs(da_fc)
    assert np.allclose(da_fc_inflated.sel(zonal_wavenum=[1, 5, 10, 20]), da_fc)


def test_inflate_zonal_wave_coeffs_missing_nlons():
    """Test that inflate_zonal_wave_coeffs fails without nlons attribute"""
    da = create_dummy_geo_field(lon_coord(2.0), lat_coord(3.0))

    da_fc = zonal_wave_coeffs(da, fftpkg="scipy", waves=[1, 2, 3, 4])
    _ = da_fc.attrs.pop("nlons")

    with pytest.raises(KeyError) as e:
        da_fc_inflated = inflate_zonal_wave_coeffs(da_fc)
    assert "input DataArray must have an 'nlons' attribute" in str(e.value)


def test_inflate_zonal_wave_coeffs_return_input():
    """Test that inflate_zonal_wave_coeffs returns input if it matches desired wavenums"""
    da = create_dummy_geo_field(lon_coord(2.0), lat_coord(3.0))

    da_fc = zonal_wave_coeffs(da, fftpkg="scipy")

    da_fc_inflated = inflate_zonal_wave_coeffs(da_fc)
    assert da_fc_inflated is da_fc


def test_inflate_zonal_wave_coeffs_unexpected_wavenums():
    """Test that inflate_zonal_wave_coeffs fails with unexpected wavenums"""
    da = create_dummy_geo_field(lon_coord(2.0), lat_coord(3.0))

    da_fc = zonal_wave_coeffs(da, fftpkg="scipy", waves=[1, 2, 3, 4])
    da_fc = da_fc.assign_coords({"zonal_wavenum": [1, 2, 103, 104]})

    with pytest.raises(ValueError) as e:
        da_fc_inflated = inflate_zonal_wave_coeffs(da_fc)
    assert (
        "input DataArray wavenumbers are not a subset of the expected wavenumbers based on the 'nlons' attribute"
        in str(e.value)
    )


def test_zonal_wave_ampl_phase():
    """Test that zonal wave ampl phase works correctly with scipy FFT"""
    test_lons = lon_coord(1.0)
    test_lats = lat_coord(1.0)

    wave_num = 3
    wave_ampl = 4
    wave_phase = -np.pi / 3

    # Test with a cosine wave
    wave_field = np.broadcast_to(
        4 * np.cos(wave_num * np.deg2rad(test_lons) + wave_phase),
        (test_lats.size, test_lons.size),
    )
    wave_field = xr.DataArray(
        wave_field, coords=[test_lats, test_lons], dims=["lat", "lon"]
    )
    wave_field_fc = zonal_wave_coeffs(wave_field, fftpkg="scipy")

    ampl, phase = zonal_wave_ampl_phase(wave_field_fc)

    assert np.allclose(ampl.sel(zonal_wavenum=wave_num, lat=0), wave_ampl)
    assert np.allclose(phase.sel(zonal_wavenum=wave_num, lat=0), wave_phase)


def test_zonal_wave_ampl_phase_missing_nlons():
    da = create_dummy_geo_field(lon_coord(2.0), lat_coord(3.0))
    da_fc = zonal_wave_coeffs(da, fftpkg="scipy", waves=[1, 2, 3])

    with pytest.raises(KeyError) as e:
        _ = da_fc.attrs.pop("nlons")
        ampl, phase = zonal_wave_ampl_phase(da_fc)
    assert "input DataArray must have an 'nlons' attribute" in str(e.value)


def test_filter_by_zonal_wave_truncation_scipy():
    """Test that zonal wave truncation filtering works correctly with scipy IFFT"""
    test_lons = lon_coord(1.0)
    test_lats = lat_coord(1.0)

    wave_fields = [
        np.broadcast_to(
            k * np.cos(k * np.deg2rad(test_lons)), (test_lats.size, test_lons.size)
        )
        for k in range(0, 5)
    ]
    wave_field_13 = wave_fields[1] + wave_fields[3]
    wave_field_24 = wave_fields[2] + wave_fields[4]
    wave_field_134 = wave_fields[1] + wave_fields[3] + wave_fields[4]

    full_wave_field = xr.DataArray(
        wave_fields[1] + wave_fields[2] + wave_fields[3] + wave_fields[4],
        coords=[test_lats, test_lons],
        dims=["lat", "lon"],
    )
    wave_field_fc = zonal_wave_coeffs(full_wave_field, fftpkg="scipy")

    wave_field_filtered = filter_by_zonal_wave_truncation(
        wave_field_fc, [1, 3], fftpkg="scipy"
    )
    np.testing.assert_allclose(wave_field_filtered.values, wave_field_13, atol=1.0e-12)

    wave_field_filtered = filter_by_zonal_wave_truncation(
        wave_field_fc, [2, 4], fftpkg="scipy"
    )
    np.testing.assert_allclose(wave_field_filtered.values, wave_field_24, atol=1.0e-12)

    wave_field_filtered = filter_by_zonal_wave_truncation(
        wave_field_fc, [1, 3, 4], fftpkg="scipy"
    )
    np.testing.assert_allclose(wave_field_filtered.values, wave_field_134, atol=1.0e-12)


def test_filter_by_zonal_wave_truncation_xrft():
    """Test that zonal wave truncation filtering works correctly with xrft IFFT"""
    test_lons = lon_coord(1.0)
    test_lats = lat_coord(1.0)

    wave_fields = [
        np.broadcast_to(
            k * np.cos(k * np.deg2rad(test_lons)), (test_lats.size, test_lons.size)
        )
        for k in range(0, 5)
    ]
    wave_field_13 = wave_fields[1] + wave_fields[3]
    wave_field_24 = wave_fields[2] + wave_fields[4]
    wave_field_134 = wave_fields[1] + wave_fields[3] + wave_fields[4]

    full_wave_field = xr.DataArray(
        wave_fields[1] + wave_fields[2] + wave_fields[3] + wave_fields[4],
        coords=[test_lats, test_lons],
        dims=["lat", "lon"],
    )
    wave_field_fc = zonal_wave_coeffs(full_wave_field, fftpkg="xrft")

    wave_field_filtered = filter_by_zonal_wave_truncation(
        wave_field_fc, [1, 3], fftpkg="xrft"
    )
    np.testing.assert_allclose(wave_field_filtered.values, wave_field_13, atol=1.0e-12)

    wave_field_filtered = filter_by_zonal_wave_truncation(
        wave_field_fc, [2, 4], fftpkg="xrft"
    )
    np.testing.assert_allclose(wave_field_filtered.values, wave_field_24, atol=1.0e-12)

    wave_field_filtered = filter_by_zonal_wave_truncation(
        wave_field_fc, [1, 3, 4], fftpkg="xrft"
    )
    np.testing.assert_allclose(wave_field_filtered.values, wave_field_134, atol=1.0e-12)


def test_filter_by_zonal_wave_truncation_no_attrs():
    da = create_dummy_geo_field(lon_coord(2.0), lat_coord(3.0))
    da_fc = zonal_wave_coeffs(da, fftpkg="scipy")

    with pytest.raises(KeyError) as e:
        _ = da_fc.attrs.pop("lon0")
        da_filt = filter_by_zonal_wave_truncation(da_fc, [1, 2, 3])
    assert "input DataArray must have a 'lon0' attribute" in str(e.value)

    with pytest.raises(KeyError) as e:
        _ = da_fc.attrs.pop("nlons")
        da_filt = filter_by_zonal_wave_truncation(da_fc, [1, 2, 3])
    assert "input DataArray must have an 'nlons' attribute" in str(e.value)


def test_filter_by_zonal_wave_truncation_infer_lons():
    """Test that filter_by_zonal_wave_truncation can infer longitudes correctly"""
    test_lons = lon_coord(2.0)
    da = create_dummy_geo_field(test_lons, lat_coord(2.0))
    da_fc = zonal_wave_coeffs(da, fftpkg="scipy")

    with pytest.warns(UserWarning) as w:
        da_filtered = filter_by_zonal_wave_truncation(
            da_fc, [1, 6], fftpkg="scipy", lons=None
        )
    assert "attempting to infer the input longitude coordinate" in str(w[0].message)
    np.testing.assert_allclose(da_filtered.lon.values, test_lons)


def test_filter_by_zonal_wave_truncation_fftpkg_fails():
    """Test that filter_by_zonal_wave_truncation fails with invalid fftpkg"""
    test_lons = lon_coord(2.0)
    da = create_dummy_geo_field(test_lons, lat_coord(2.0))
    da_fc = zonal_wave_coeffs(da, fftpkg="scipy")

    with pytest.raises(ValueError) as e:
        da_filt = filter_by_zonal_wave_truncation(da_fc, [1, 2, 3], fftpkg="foo")
    assert "Invalid fftpkg: must be" in str(e.value)


def test_zonal_wave_contributions():
    """Test that zonal wave truncation filtering works correctly with scipy IFFT"""
    test_lons = lon_coord(1.0)
    test_lats = lat_coord(1.0)

    wave_fields = [
        np.broadcast_to(
            k * np.cos(k * np.deg2rad(test_lons)), (test_lats.size, test_lons.size)
        )
        for k in range(0, 5)
    ]

    full_wave_field = xr.DataArray(
        wave_fields[1] + wave_fields[2] + wave_fields[3] + wave_fields[4],
        coords=[test_lats, test_lons],
        dims=["lat", "lon"],
    )
    wave_field_fc = zonal_wave_coeffs(full_wave_field, fftpkg="scipy")
    contributions = zonal_wave_contributions(
        wave_field_fc,
        [1, 2, 3, 4],
        lons=test_lons,
    )

    for k in range(1, 5):
        np.testing.assert_allclose(
            contributions.sel(zonal_wavenum=k).values, wave_fields[k], atol=1.0e-12
        )


def test_zonal_wave_covariance():
    """Test that zonal wave covariance works correctly"""
    test_lons = lon_coord(1.0)
    test_lats = lat_coord(2.0)

    wave1 = xr.DataArray(
        np.broadcast_to(
            np.cos(np.deg2rad(test_lons)), (test_lats.size, test_lons.size)
        ),
        coords=[test_lats, test_lons],
        dims=["lat", "lon"],
    )
    wave1_phased = xr.DataArray(
        np.broadcast_to(
            np.cos(np.deg2rad(test_lons) + np.pi / 3), (test_lats.size, test_lons.size)
        ),
        coords=[test_lats, test_lons],
        dims=["lat", "lon"],
    )

    fc_k1 = zonal_wave_coeffs(wave1, fftpkg="scipy")
    fc_k1_phased = zonal_wave_coeffs(wave1_phased, fftpkg="scipy")

    zon_cov = zonal_wave_covariance(fc_k1, fc_k1_phased)

    assert zon_cov.dims == ("lat", "zonal_wavenum")
    np.testing.assert_allclose(zon_cov.sel(zonal_wavenum=1).values, 0.25)
    np.testing.assert_allclose(
        zon_cov.isel(zonal_wavenum=slice(2, None)).values, 0.0, rtol=0, atol=1.0e-30
    )


def test_zonal_wave_covariance_no_nlons():
    """Test that zonal_wave_covariance fails without any nlons given"""

    da1 = create_dummy_geo_field(lon_coord(2.5), lat_coord(2.5))
    da2 = create_dummy_geo_field(lon_coord(2.5), lat_coord(2.5))
    da_fc1 = zonal_wave_coeffs(da1, fftpkg="scipy")
    da_fc2 = zonal_wave_coeffs(da2, fftpkg="scipy")

    with pytest.raises(ValueError) as e:
        _ = da_fc1.attrs.pop("nlons")
        _ = da_fc2.attrs.pop("nlons")

        zon_cov = zonal_wave_covariance(da_fc1, da_fc2)
    assert (
        "nlons must either be provided as a kwarg or be in the attrs of fc1 or fc2"
        in str(e.value)
    )
