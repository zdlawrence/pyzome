from typing import Sequence, Optional

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
    r"""Base class for pyzome xarray accessors. Provides the common coordinate
    mapping functionality for selecting correct xarray coordinate names for
    use in pyzome functions. The properties and methods of this base class
    apply to both xarray DataArray and Dataset objects.

    Examples
    --------
    >>> import xarray as xr
    >>> import pyzome as pzm
    >>> ds = xr.open_dataset("...")
    >>> ds.pzm.lon
    >>> ds.pzm.zonal_mean()
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._coord_map = infer_xr_coord_names(xarray_obj)

    def coord_map(self, coord: str):
        r"""Provides the mapping between the pyzome standard coordinate names
        (lon, lat, plev, zonal_wavenum) and the actual coordinate names in the
        xarray object.

        Parameters
        ----------
        coord : str
            The pyzome standard coordinate name (lon, lat, plev, zonal_wavenum)

        Returns
        -------
        str
            The actual coordinate name in the xarray object

        Raises
        ------
        KeyError
            If the underlying coordinate name is not found in the xarray object
        """
        if coord not in self._coord_map:
            raise KeyError(
                f"pyzome coordinate '{coord}' not identified in original data"
            )
        return self._coord_map[coord]

    @property
    def lon(self):
        r"""The longitude coordinate of the xarray object, if it exists/was identified"""
        return self._obj[self.coord_map("lon")]

    @property
    def lat(self):
        r"""The latitude coordinate of the xarray object, if it exists/was identified"""
        return self._obj[self.coord_map("lat")]

    @property
    def plev(self):
        r"""The pressure level coordinate of the xarray object, if it exists/was identified"""
        return self._obj[self.coord_map("plev")]

    @property
    def zonal_wavenum(self):
        r"""The zonal wavenumber coordinate of the xarray object, if it exists/was identified"""
        return self._obj[self.coord_map("zonal_wavenum")]

    def add_logp_altitude(self, H: float = SCALE_HEIGHT, p0: float = PREF):
        r"""Adds a log-pressure altitude coordinate to the xarray object. Requires
        that the pyzome `plev` coordinate be identified, and that it be in units of Pa.

        See Also
        --------
        pyzome.qglogp.add_logp_altitude
        """
        return add_logp_altitude(self._obj, plev_coord=self.plev.name, H=H, p0=p0)

    def zonal_mean(self, strict: bool = True):
        r"""Computes the zonal mean of the xarray object, if a longitude coordinate
        was identifed.

        See Also
        --------
        pyzome.basic.zonal_mean
        """
        return zonal_mean(self._obj, lon_coord=self.lon.name, strict=strict)

    def meridional_mean(self, lat1: float, lat2: float, strict: bool = True):
        r"""Computes the meridional mean of the xarray object, if a latitude coordinate
        was identifed.

        See Also
        --------
        pyzome.basic.meridional_mean
        """
        return meridional_mean(
            self._obj, lat1, lat2, lat_coord=self.lat.name, strict=strict
        )


@xr.register_dataarray_accessor("pzm")
class PyzomeDataArrayAccessor(PyzomeAccessor):
    r"""xarray DataArray accessor for pyzome functions. Inherits from
    PyzomeAccessor for maintaining the same coordinate mapping functionality.

    Examples
    --------
    >>> import xarray as xr
    >>> import pyzome as pzm
    >>> da = xr.open_dataarray("...") # must be a DataArray
    >>> da.pzm.lon
    >>> da.pzm.zonal_wave_coeffs().pzm.filter_by_zonal_wave_truncation(waves=[1,2,3])
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        self._coord_map = infer_xr_coord_names(xarray_obj)

    def zonal_wave_coeffs(
        self, waves: Optional[Sequence[int]] = None, fftpkg: str = "scipy"
    ):
        r"""Computes the zonal wave coefficients of a DataArray, if a longitude
        coordinate was identifed.

        See Also
        --------
        pyzome.zonal_waves.zonal_wave_coeffs
        """
        return zonal_wave_coeffs(
            self._obj, waves=waves, fftpkg=fftpkg, lon_coord=self.lon.name
        )

    def inflate_zonal_wave_coeffs(self):
        r"""Inflates the zonal wave coefficients of a DataArray to the full
        spectrum expected for inverse transforms.

        See Also
        --------
        pyzome.zonal_waves.inflate_zonal_wave_coeffs
        """
        return inflate_zonal_wave_coeffs(self._obj, wave_coord=self.zonal_wavenum.name)

    def filter_by_zonal_wave_truncation(
        self,
        waves: Sequence[int],
        fftpkg: str = "scipy",
        lons: Optional[xr.DataArray] = None,
    ):
        r"""Filters the input DataArray by truncating to include only the specified
        zonal wavenumbers.

        See Also
        --------
        pyzome.zonal_waves.filter_by_zonal_wave_truncation
        """
        return filter_by_zonal_wave_truncation(
            self._obj,
            waves=waves,
            fftpkg=fftpkg,
            wave_coord=self.zonal_wavenum.name,
            lons=lons,
        )

    def zonal_wave_contributions(
        self,
        waves: Sequence[int],
        fftpkg: str = "scipy",
        lons: Optional[xr.DataArray] = None,
    ):
        r"""Computes the individual contributions of each given zonal wavenumber
        to the input DataArray.

        See Also
        --------
        pyzome.zonal_waves.zonal_wave_contributions
        """
        return zonal_wave_contributions(
            self._obj,
            waves=waves,
            fftpkg=fftpkg,
            wave_coord=self.zonal_wavenum.name,
            lons=lons,
        )


@xr.register_dataset_accessor("pzm")
class PyzomeDatasetAccessor(PyzomeAccessor):
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self._coord_map = infer_xr_coord_names(xarray_obj)
