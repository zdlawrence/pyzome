import pytest

import numpy as np

from pyzome.qglogp import (
    add_logp_altitude,
    buoyancy_frequency_squared,
    merid_grad_qgpv,
    refractive_index,
    plumb_wave_activity_flux,
)
from pyzome.mock_data import (
    create_dummy_geo_dataset, 
    create_dummy_geo_field, 
    lon_coord, 
    lat_coord, 
    plev_coord
)
from pyzome.constants import SCALE_HEIGHT, PREF


def test_add_logp_altitude():
    """Test that add_logp_altitude works correctly"""
    da = create_dummy_geo_field(
        lons=lon_coord(2.5),
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
        name = "T",
        attrs={"units": "K"},
    )

    da = add_logp_altitude(da)
    assert da.z.attrs["units"] == "m"
    assert da.z.attrs["long_name"] == "log-pressure altitude"
    assert da.z.attrs["note"] == "added by pyzome"
    np.testing.assert_allclose(da.z.values, -SCALE_HEIGHT*np.log(da.lev.values/PREF))


def test_buoyancy_frequency_squared():
    """Test that buoyancy_frequency_squared computation works from start to finish"""
    T = create_dummy_geo_field(
        lons=None,
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
        name = "T",
        attrs={"units": "K"},
    )
    T = add_logp_altitude(T)

    N2 = buoyancy_frequency_squared(T)
    assert N2.attrs["units"] == "s-2"
    assert N2.attrs["long_name"] == "buoyancy frequency squared"


def test_merid_grad_qgpv():
    """Test that merid_grad_qgpv computation works from start to finish"""
    ds = create_dummy_geo_dataset(
        field_names = ["u", "N2"],
        lons=None,
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
        field_attrs={"u": {"units": "m s-1"}, "N2": {"units": "s-2"}},
    )
    ds = add_logp_altitude(ds)

    dqdy = merid_grad_qgpv(ds.u, ds.N2)
    term1, term2, term3 = merid_grad_qgpv(ds.u, ds.N2, terms=True)
    np.testing.assert_allclose(dqdy.values, (term1 + term2 + term3).values)


def test_refractive_index():
    """Test that refractive_index computation works from start to finish"""

    ds = create_dummy_geo_dataset(
        field_names = ["u", "q_phi", "N2"],
        lons=None,
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
        field_attrs={"u": {"units": "m s-1"}, "q_phi": {"units": "s-1"}, "N2": {"units": "s-2"}},
    )
    ds = add_logp_altitude(ds)

    ri2 = refractive_index(ds.u, ds.q_phi, np.abs(ds.N2), 1)
    term1, term2, term3  = refractive_index(ds.u, ds.q_phi, np.abs(ds.N2), 1, terms=True)
    np.testing.assert_allclose(ri2.values, (term1 + term2 + term3).values)

    # Test with alternate buoyancy frequency term
    ri2 = refractive_index(ds.u, ds.q_phi, np.abs(ds.N2), 1, cd_approx_term=True)
    term1, term2, term3 = refractive_index(ds.u, ds.q_phi, np.abs(ds.N2), 1, terms=True, cd_approx_term=True)
    np.testing.assert_allclose(ri2.values, (term1 + term2 + term3).values)


def test_plumb_wave_activity_flux():
    """Test that plumb_wave_activity_flux computation works from start to finish"""
    
    ds = create_dummy_geo_dataset(
        field_names = ["psi", "N2"],
        lons=lon_coord(2.5),
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
        field_attrs={"psi": {"units": "m+2 s-1"}, "N2": {"units": "s-2"}},
    )
    ds = add_logp_altitude(ds)

    combos = [["x","y","z"], ["x","z"], ["y","z"], ["x"], ["y"], ["z"]]
    for components in combos:
        F = plumb_wave_activity_flux(ds.psi, ds.N2, components=components)
        assert len(F) == len(components)


def test_plumb_wave_activity_flux_no_lats():
    ds = create_dummy_geo_dataset(
        field_names = ["psi", "N2"],
        lons=lon_coord(2.5),
        lats=None,
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
        field_attrs={"psi": {"units": "m+2 s-1"}, "N2": {"units": "s-2"}},
    )
    ds = add_logp_altitude(ds)

    with pytest.raises(ValueError) as e:
        F = plumb_wave_activity_flux(ds.psi, ds.N2, components=["y"])
    assert "A latitude coordinate must be available" in str(e.value)


def test_plumb_wave_activity_flux_no_lons():
    ds = create_dummy_geo_dataset(
        field_names = ["psi", "N2"],
        lons=None,
        lats=lat_coord(2.5),
        levs=plev_coord(3, left_lim_exponent=5, right_lim_exponent=2, units="Pa"),
        field_attrs={"psi": {"units": "m+2 s-1"}, "N2": {"units": "s-2"}},
    )
    ds = add_logp_altitude(ds)

    with pytest.raises(ValueError) as e:
        F = plumb_wave_activity_flux(ds.psi, ds.N2, components=["x", "z"])
    assert "A longitude coordinate must be available" in str(e.value)
