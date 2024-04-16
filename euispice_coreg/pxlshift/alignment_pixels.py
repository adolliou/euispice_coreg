import warnings

import numpy as np
from astropy.io import fits
from ..utils import rectify
from ..utils import c_correlate
from tqdm import tqdm
from ..utils import matrix_transform
import astropy.units as u
from astropy.time import Time
from ..utils import Util


class AlignmentPixels:

    def __init__(self, large_fov_known_pointing: str, window_large: int, small_fov_to_correct: str,
                 window_small: int, ):
        with fits.open(large_fov_known_pointing) as hdul_large:
            hdu_large = hdul_large[window_large]
            self.hdr_large = hdu_large.header.copy()
            self.data_large = np.array(hdu_large.data.copy(), dtype=np.float64)
            hdul_large.close()
        with fits.open(small_fov_to_correct) as hdul_small:
            hdu_small = hdul_small[window_small]
            self.hdr_small = hdu_small.header.copy()
            self.data_small = np.array(hdu_small.data.copy(), dtype=np.float64)
            hdul_small.close()
        self.slc_small_ref = None
        self.x_large = None
        self.y_large = None

    def _iteration_along_dy(self, dx, ):
        results = np.zeros((len(self.lag_dy)), dtype=np.float64)
        for jj, dy in enumerate(tqdm(self.lag_dy, desc="dx = %i" % dx)):
            results[jj] = self._step(dx=dx, dy=dy)
        return results

    def _step(self, dx, dy):
        slc = (slice(self.slc_small_ref[0].start + dy,
                     self.slc_small_ref[0].stop + dy),
               slice(self.slc_small_ref[1].start + dx,
                     self.slc_small_ref[1].stop + dx)
               )
        self._check_boundaries(slc, self.data_large.shape)

        if (self.data_large[slc[0], slc[1]].shape != self.data_small_rotated.shape):
            raise ValueError("shapes not similar")

        lag = [0]
        is_nan = np.array(
            (np.isnan(self.data_large[slc[0], slc[1]].ravel())) | (np.isnan(self.data_small_rotated.ravel())),
            dtype="bool")
        return c_correlate.c_correlate(s_2=self.data_large[slc[0], slc[1]].ravel()[~is_nan],
                                       s_1=self.data_small_rotated.ravel()[~is_nan],
                                       lags=lag)

    def find_best_parameters(self, lag_dx: np.array, lag_dy: np.array, lag_drot: np.array, unit_rot="degree",
                             shift_solar_rotation_dx_large=False):

        if shift_solar_rotation_dx_large:
            self._shift_large_fov()

        self._sub_resolution_large_fov()
        self._initialise_slice_corresponding_to_small()
        corr = np.zeros((len(lag_dx), len(lag_dy), len(lag_drot)), dtype=np.float64)
        self.lag_dx = lag_dx
        self.lag_dy = lag_dy
        self.lag_drot = lag_drot
        self.unit_rot = unit_rot
        self.data_small_rotated = self.data_small.copy()

        for kk, drot in enumerate(lag_drot):
            if drot != 0:
                xx, yy = np.meshgrid(np.arange(self.data_small.shape[1]),
                                     np.arange(self.data_small.shape[0]),
                                     )
                nx, ny = matrix_transform.MatrixTransform.polar_transform(xx, yy, theta=drot, units=self.unit_rot)
                self.data_small_rotated = rectify.interpol2d(self.data_small.copy(), x=nx, y=ny, fill=-32762, order=1)
                self.data_small_rotated[self.data_small_rotated == -32762] = np.nan
            else:
                self.data_small_rotated = self.data_small.copy()
            for ii, dx in enumerate(lag_dx):
                corr[ii, :, kk] = self._iteration_along_dy(dx=dx, )
        return corr

    def _shift_large_fov(self):
        xx, yy = np.meshgrid(np.arange(self.data_large.shape[1]),
                             np.arange(self.data_large.shape[0]), )
        data_large = Util.CommonUtil.interpol2d(self.data_large, xx, yy, fill=-32762, order=1)
        data_large = np.where(data_large == -32762, np.nan, data_large)
        dcrval = self._return_shift_large_fov_solar_rotation()
        if "CROTA" in self.hdr_large:
            warnings.warn("CROTA must be in degree", Warning)
            theta = np.deg2rad(self.hdr_large["CROTA"])
            dx = (dcrval.to(self.hdr_large["CUNIT1"]).value / self.hdr_large["CDELT1"]) * np.cos(-theta)
            dy = (dcrval.to(self.hdr_large["CUNIT2"]).value / self.hdr_large["CDELT2"]) * np.sin(-theta)
        else:
            dx = dcrval.to(self.hdr_large["CUNIT1"]).value / self.hdr_large["CDELT1"]
            dy = 0
        mat = matrix_transform.MatrixTransform.displacement_matrix(dx=dx, dy=dy)
        nx, ny = matrix_transform.MatrixTransform.linear_transform(xx, yy, matrix=mat)
        data_large = Util.CommonUtil.interpol2d(data_large, nx, ny, fill=-32762, order=1)
        # norm = ImageNormalize(stretch=LogStretch(20), vmin=1, vmax=1000)
        # PlotFunctions.plot_fov(self.data_large, norm=norm)
        self.data_large = np.where(data_large == -32762, np.nan, data_large)
        # PlotFunctions.plot_fov(self.data_large, norm=norm)
        print(f"corrected solar rotation on FSI on CRVAL1: {dx=}, {dy=}")

    def _return_shift_large_fov_solar_rotation(self):
        band = self.hdr_large['WAVELNTH']
        B0 = np.deg2rad(self.hdr_large['SOLAR_B0'])
        omega_car = np.deg2rad(360 / 25.38 / 86400)  # rad s-1
        if band == 174:
            band = 171
        omega = omega_car + Util.EUIUtil.diff_rot(B0, f'EIT {band}')  # rad s-1
        # helioprojective rotation rate for s/c
        Rsun = self.hdr_large['RSUN_REF']  # m
        Dsun = self.hdr_large['DSUN_OBS']  # m
        phi = omega * Rsun / (Dsun - Rsun)  # rad s-1
        phi = np.rad2deg(phi) * 3600  # arcsec s-1
        time_spice = Time(self.hdr_small["DATE-AVG"])
        time_fsi = Time(self.hdr_large["DATE-AVG"])
        dt = (time_spice - time_fsi).to("s").value
        return u.Quantity(dt * phi, "arcsec")

    def _sub_resolution_large_fov(self):
        cdelt1_conv = u.Quantity(self.hdr_small["CDELT1"],
                                 self.hdr_small["CUNIT1"]).to(self.hdr_large["CUNIT1"]).value
        cdelt2_conv = u.Quantity(self.hdr_small["CDELT2"],
                                 self.hdr_small["CUNIT2"]).to(self.hdr_large["CUNIT2"]).value
        self.ratio_res_1 = cdelt1_conv / self.hdr_large["CDELT1"]
        self.ratio_res_2 = cdelt2_conv / self.hdr_large["CDELT2"]

        x, y = np.meshgrid(np.arange(0, self.data_large.shape[1], self.ratio_res_1),
                           np.arange(0, self.data_large.shape[0], self.ratio_res_2), )

        self.data_large = rectify.interpol2d(self.data_large, x=x, y=y, order=1, fill=-32768)
        self.data_large[self.data_large == -32768] = np.nan

        y_new, x_new = np.meshgrid(np.arange(self.data_large.shape[0]),
                                   np.arange(self.data_large.shape[1]))
        self.x_large = x_new
        self.y_large = y_new

    def _initialise_slice_corresponding_to_small(self, ):
        l = [int((self.data_large.shape[n] - self.data_small.shape[n] - 1) / 2) for n in range(2)]
        self.slc_small_ref = (slice(l[0], l[0] + self.data_small.shape[0]),
                              slice(l[1], l[1] + self.data_small.shape[1]))

    @staticmethod
    def _check_boundaries(slc, shape, ):
        for n in range(2):
            if (slc[n].start < 0):
                raise ValueError("too large shift : outside FSI")
            if (slc[n].stop > shape[n]):
                raise ValueError("too large shift : outside FSI")
