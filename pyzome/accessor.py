import xarray as xr

from .constants import SCALE_HEIGHT, PREF
from .checks import infer_xr_coord_names
from .basic import zonal_mean, meridional_mean
from .qglogp import add_logp_altitude
from .zonal_waves import (
    zonal_wave_coeffs,
    inflate_zonal_wave_coeffs,
    filter_by_zonal_wave_truncation,
    zonal_wave_contributions,
)


class PyzomeAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._coord_map = infer_xr_coord_names(xarray_obj)

    def coord_map(self, coord: str):
        if coord not in self._coord_map:
            raise KeyError(
                f"pyzome coordinate '{coord}' not identified in original data"
            )
        return self._coord_map[coord]

    @property
    def lon(self):
        return self._obj[self.coord_map("lon")]

    @property
    def lat(self):
        return self._obj[self.coord_map("lat")]

    @property
    def plev(self):
        return self._obj[self.coord_map("plev")]

    @property
    def zonal_wavenum(self):
        return self._obj[self.coord_map("zonal_wavenum")]

    def add_logp_altitude(self, H=SCALE_HEIGHT, p0=PREF):
        return add_logp_altitude(self._obj, plev_coord=self.plev.name, H=H, p0=p0)

    def zonal_mean(self, strict=False):
        return zonal_mean(self._obj, lon_coord=self.lon.name, strict=strict)

    def meridional_mean(self, lat1, lat2, strict=False):
        return meridional_mean(
            self._obj, lat1, lat2, lat_coord=self.lat.name, strict=strict
        )


@xr.register_dataarray_accessor("pzm")
class PyzomeDataArrayAccessor(PyzomeAccessor):
    def zonal_wave_coeffs(self, waves, fftpkg="scipy"):
        return zonal_wave_coeffs(
            self._obj, waves=waves, fftpkg=fftpkg, lon_coord=self.lon.name
        )

    def inflate_zonal_wave_coeffs(self):
        return inflate_zonal_wave_coeffs(self._obj, wave_coord=self.zonal_wavenum.name)

    def filter_by_zonal_wave_truncation(self, waves, fftpkg="scipy", lons=None):
        return filter_by_zonal_wave_truncation(
            self._obj,
            waves=waves,
            fftpkg=fftpkg,
            wave_coord=self.zonal_wavenum.name,
            lons=lons,
        )

    def zonal_wave_contributions(self, waves, fftpkg="scipy", lons=None):
        return zonal_wave_contributions(
            self._obj,
            waves=waves,
            fftpkg=fftpkg,
            wave_coord=self.zonal_wavenum.name,
            lons=lons,
        )


@xr.register_dataset_accessor("pzm")
class PyzomeDatasetAccessor(PyzomeAccessor):
    pass
