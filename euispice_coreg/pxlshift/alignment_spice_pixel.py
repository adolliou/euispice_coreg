from .alignment_pixels import AlignmentPixels
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from ..utils import Util


class AlignmentSpicePixel(AlignmentPixels):
    def __init__(self, fsi_path: str, fsi_window: int, spice_path: str, spice_window: int, index_amplitude=None,):
        super().__init__(fsi_path, fsi_window, spice_path, spice_window)
        self.fsi_path = fsi_path
        self.spice_path = spice_path
        self.fsi_window = fsi_window
        self.spice_window = spice_window

        level = None
        if "L2" in self.spice_path:
            level = 2
        elif "L3" in self.spice_path:
            level = 3
        self._extract_spice_data_header(level=level, index_amplitude=index_amplitude)

    def find_best_parameters(self, lag_dx: np.array, lag_dy: np.array, lag_drot: np.array, unit_rot="degree",
                             shift_solar_rotation_dx_large=False):

        return super().find_best_parameters(lag_dx, lag_dy, lag_drot, unit_rot,shift_solar_rotation_dx_large)

    def _extract_spice_data_header(self, level: int, index_amplitude=None):
        with fits.open(self.spice_path) as hdul_small:
            dt = hdul_small[self.spice_window].header.copy()["PC4_1"]
            if level == 2:
                self._prepare_spice_from_l2(hdul_small)
            elif level == 3:
                self._prepare_spice_from_l3(hdul_small, index_amplitude)

            self.hdr_small['SOLAR_B0'] = hdul_small[self.spice_window].header["SOLAR_B0"]
            self.hdr_small['RSUN_REF'] = hdul_small[self.spice_window].header["RSUN_REF"]
            self.hdr_small['DSUN_OBS'] = hdul_small[self.spice_window].header["DSUN_OBS"]

            # Correct solar rotation in SPICE header
            # rotation rate (on solar sphere)
            self._correct_solar_rotation(dt)
            # shift large fov if no dynamic pointing$
            hdul_small.close()

    def _correct_solar_rotation(self, dt):
        B0 = np.deg2rad(self.hdr_small['SOLAR_B0'])
        band = self.hdr_large['WAVELNTH']
        omega_car = np.deg2rad(360 / 25.38 / 86400)  # rad s-1
        if band == 174:
            band = 171
        omega = omega_car + Util.EUIUtil.diff_rot(B0, f'EIT {band}')  # rad s-1
        # helioprojective rotation rate for s/c
        Rsun = self.hdr_small['RSUN_REF']  # m
        Dsun = self.hdr_small['DSUN_OBS']  # m
        phi = omega * Rsun / (Dsun - Rsun)  # rad s-1
        phi = np.rad2deg(phi) * 3600  # arcsec s-1
        DTx_old = u.Quantity(self.hdr_small['CDELT1'], self.hdr_small['CUNIT1'])
        DTx_new = DTx_old - dt * phi * u.arcsec
        self.hdr_small['CDELT1'] = DTx_new.to(self.hdr_small['CUNIT1']).value
        print(f'Corrected solar rotation : changed SPICE CDELT1 from {DTx_old} to {DTx_new}')

    def _prepare_spice_from_l2(self, hdul_small):

        data_small = np.array(hdul_small[self.spice_window].data.copy(), dtype=np.float64)
        header_spice = hdul_small[self.spice_window].header.copy()
        Util.SpiceUtil.recenter_crpix_in_header_L2(header_spice)
        ymin, ymax = Util.SpiceUtil.vertical_edges_limits(header_spice)
        w_spice = WCS(header_spice)
        w_xyt = w_spice.dropaxis(2)
        w_xyt.wcs.pc[2, 0] = 0
        w_xy = w_xyt.dropaxis(2)
        self.hdr_small = w_xy.to_header().copy()
        self.hdr_small["NAXIS1"] = data_small.shape[3]
        self.hdr_small["NAXIS2"] = data_small.shape[2]
        Util.EUIUtil.recenter_crpix_in_header(self.hdr_small)
        ylen = data_small.shape[2]

        ylim = np.array([ymin, ylen - ymax - 1]).max()
        self.data_small = np.nansum(data_small[0, :, ylim:(ylen - ylim), :], axis=0)
        self.hdr_small["CRPIX1"] = (self.data_small.shape[1] + 1) / 2
        self.hdr_small["CRPIX2"] = (self.data_small.shape[0] + 1) / 2

        self.hdr_small["NAXIS1"] = self.data_small.shape[1]
        self.hdr_small["NAXIS2"] = self.data_small.shape[0]

    def _prepare_spice_from_l3(self, hdul_small, index_amplitude):
        w = WCS(hdul_small[self.spice_window].header.copy())
        w2 = w.deepcopy()
        w2.wcs.pc[3, 0] = 0
        w2.wcs.pc[3, 1] = 0
        w_xyt = w2.dropaxis(0)
        w_xy = w_xyt.dropaxis(2)
        data_small = np.array(hdul_small[self.spice_window].data.copy(), dtype=np.float64)
        self.data_small = data_small[:, :, index_amplitude]
        self.data_small[self.data_small == hdul_small[self.spice_window].header["ANA_MISS"]] = np.nan
        self.hdr_small = w_xy.to_header().copy()

        self.hdr_small["NAXIS1"] = self.data_small.shape[1]
        self.hdr_small["NAXIS2"] = self.data_small.shape[0]
