from .alignement import Alignment
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from ..synras import map_builder
from . import c_correlate
import copy
from ..utils import Util

class AlignmentSpice(Alignment):
    def __init__(self, large_fov_known_pointing: str, small_fov_to_correct: str, lag_crval1: np.array,
                 lag_crval2: np.array, lag_cdelta1, lag_cdelta2, lag_crota, small_fov_value_min=None,
                 parallelism=False, small_fov_value_max=None, counts_cpu_max=40, large_fov_window=-1,
                 small_fov_window=-1, use_tqdm=False, lag_solar_r=None,
                 path_save_figure=None):

        super().__init__(large_fov_known_pointing=large_fov_known_pointing, small_fov_to_correct=small_fov_to_correct,
                         lag_crval1=lag_crval1, lag_crval2=lag_crval2, lag_cdelta1=lag_cdelta1, lag_cdelta2=lag_cdelta2,
                         lag_crota=lag_crota, use_tqdm=use_tqdm,
                         lag_solar_r=lag_solar_r, small_fov_value_min=small_fov_value_min, parallelism=parallelism,
                         small_fov_value_max=small_fov_value_max, counts_cpu_max=counts_cpu_max,
                         large_fov_window=large_fov_window, small_fov_window=small_fov_window,
                         path_save_figure=path_save_figure, )

    def align_using_helioprojective(self, method='correlation', index_amplitude=None, extend_pixel_size=False,
                                    cut_from_center=None):
        self.lonlims = None
        self.latlims = None
        self.shape = None
        self.reference_date = None
        self.function_to_apply = self._interpolate_on_large_data_grid
        self.method = method
        self.coordinate_frame = "helioprojective"
        self.extend_pixel_size = extend_pixel_size
        self.cut_from_center = cut_from_center
        self._extract_imager_data_header()

        level = None
        if "L2" in self.small_fov_to_correct:
            level = 2
        elif "L3" in self.small_fov_to_correct:
            level = 3
        self._extract_spice_data_header(level=level, index_amplitude=index_amplitude, )

        results = super()._find_best_header_parameters()
        # A
        return results

    def align_using_carrington(self, lonlims: list, latlims: list, shape: tuple, reference_date,
                               method='correlation', index_amplitude=None):
        self.lonlims = lonlims
        self.latlims = latlims
        self.shape = shape
        self.reference_date = reference_date
        self.function_to_apply = self._carrington_transform
        self.extend_pixel_size=False
        self.method = method
        self.coordinate_frame = "carrington"
        self.shift_hdr_solar_rotation = False
        self._extract_imager_data_header()

        level = None
        if "L2" in self.small_fov_to_correct:
            level = 2
        elif "L3" in self.small_fov_to_correct:
            level = 3
        self._extract_spice_data_header(level=level, index_amplitude=index_amplitude)
        self.hdr_small["CRVAL1"] = Util.CommonUtil.ang2pipi(u.Quantity(self.hdr_small["CRVAL1"],
                                                                  self.hdr_small["CUNIT1"])).to("arcsec").value
        self.hdr_small["CRVAL2"] = Util.CommonUtil.ang2pipi(u.Quantity(self.hdr_small["CRVAL2"],
                                                                  self.hdr_small["CUNIT2"])).to("arcsec").value
        self.hdr_small["CDELT1"] = u.Quantity(self.hdr_small["CDELT1"], self.hdr_small["CUNIT1"]).to("arcsec").value
        self.hdr_small["CDELT2"] = u.Quantity(self.hdr_small["CDELT2"], self.hdr_small["CUNIT2"]).to("arcsec").value
        self.hdr_small["CUNIT1"] = "arcsec"
        self.hdr_small["CUNIT2"] = "arcsec"

        results = super()._find_best_header_parameters()

        return results

    def _extract_imager_data_header(self, ):
        with fits.open(self.large_fov_known_pointing) as hdul_large:
            self.data_large = np.array(hdul_large[self.large_fov_window].data.copy(), dtype=np.float64)
            self.hdr_large = hdul_large[self.large_fov_window].header.copy()
            # super()._recenter_crpix_in_header(self.hdr_large)
            hdul_large.close()

    def _extract_spice_data_header(self, level: int, index_amplitude=None):
        with fits.open(self.small_fov_to_correct) as hdul_small:
            dt = hdul_small[self.small_fov_window].header.copy()["PC4_1"]
            if level == 2:
                self._prepare_spice_from_l2(hdul_small)
            elif level == 3:
                self._prepare_spice_from_l3(hdul_small, index_amplitude)

            self.hdr_small['SOLAR_B0'] = hdul_small[self.small_fov_window].header["SOLAR_B0"]
            self.hdr_small['RSUN_REF'] = hdul_small[self.small_fov_window].header["RSUN_REF"]
            self.hdr_small['DSUN_OBS'] = hdul_small[self.small_fov_window].header["DSUN_OBS"]
            self.hdr_small['CROTA'] = hdul_small[self.small_fov_window].header["CROTA"]

            # self.hdr_small['RSUN_OBS'] = hdul_small[self.small_fov_window].header["RSUN_OBS"]

            # Correct solar rotation in SPICE header
            # rotation rate (on solar sphere)
            if self.extend_pixel_size:
                self._correct_solar_rotation(dt)
            # shift large fov if no dynamic pointing$

            hdul_small.close()

    def _correct_solar_rotation(self, dt):
        B0 = np.deg2rad(self.hdr_small['SOLAR_B0'])
        band = self.hdr_large['WAVELNTH']
        omega_car = np.deg2rad(360 / 25.38 / 86400)  # rad s-1
        if band == 174:
            band = 171
        omega = omega_car + Util.AlignEUIUtil.diff_rot(B0, f'EIT {band}')  # rad s-1
        # helioprojective rotation rate for s/c
        Rsun = self.hdr_small['RSUN_REF']  # m
        Dsun = self.hdr_small['DSUN_OBS']  # m
        phi_rot = 1.004 * omega * Rsun / (Dsun - 1.004 * Rsun)  # rad s-1
        phi_rot = np.rad2deg(phi_rot) * 3600  # arcsec s-1

        # rake into account angle to the limb
        alpha = u.Quantity(self.hdr_small["CRVAL1"], self.hdr_small["CUNIT1"]).to("rad").value

        phi = np.arcsin(((Dsun - 1.004 * Rsun) / (1.004 * Rsun)) * np.sin(
            alpha))  # heliocentric longitude with respect to spacecraft pointing
        if (-np.pi / 2 > phi > np.pi / 2):
            raise ValueError("Error in estimating heliocentric latitude")

        DTx_old = u.Quantity(self.hdr_small['CDELT1'], self.hdr_small['CUNIT1']).to("arcsec")
        DTx_new = DTx_old - dt * phi_rot * u.arcsec * np.cos(phi)  # last term to take into account that structure
        # are less shrinked on the limbs
        self.hdr_small['CDELT1'] = DTx_new.to(self.hdr_small['CUNIT1']).value
        print(f'Corrected solar rotation : changed SPICE CDELT1 from {DTx_old} to {DTx_new}')

    def _prepare_spice_from_l2(self, hdul_small):

        data_small = np.array(hdul_small[self.small_fov_window].data.copy(), dtype=np.float64)
        header_spice = hdul_small[self.small_fov_window].header
        ymin, ymax = Util.AlignSpiceUtil.vertical_edges_limits(header_spice)
        w_spice = WCS(hdul_small[self.small_fov_window].header.copy())
        w_xyt = w_spice.dropaxis(2)
        w_xyt.wcs.pc[2, 0] = 0

        w_xy = w_xyt.dropaxis(2)
        self.hdr_small = w_xy.to_header().copy()
        # self.hdr_small["NAXIS1"] = data_small.shape[3]
        # self.hdr_small["NAXIS2"] = data_small.shape[2]
        # super()._recenter_crpix_in_header(self.hdr_small)
        # ylen = data_small.shape[2]

        # ylim = np.array([ymin, ylen - ymax - 1]).max()
        data_small[:, :, :ymin, :] = np.nan
        data_small[:, :, ymax:, :] = np.nan

        self.data_small = np.nansum(data_small[0, :, :, :], axis=0)
        self.data_small[:ymin, :] = np.nan
        self.data_small[ymax:, :] = np.nan

        if self.cut_from_center is not None:
            xlen = self.cut_from_center
            xmid = self.data_small.shape[1]//2
            self.data_small[:, :(xmid - xlen//2 - 1)] = np.nan
            self.data_small[:, (xmid + xlen // 2):] = np.nan



        #
        # self.hdr_small["CRPIX1"] = (self.data_small.shape[1] + 1) / 2
        # self.hdr_small["CRPIX2"] = (self.data_small.shape[0] + 1) / 2
        #
        self.hdr_small["NAXIS1"] = self.data_small.shape[1]
        self.hdr_small["NAXIS2"] = self.data_small.shape[0]

    def _prepare_spice_from_l3(self, hdul_small, index_amplitude):
        w = WCS(hdul_small[self.small_fov_window].header.copy())
        w2 = w.deepcopy()
        w2.wcs.pc[3, 0] = 0
        w2.wcs.pc[3, 1] = 0
        w_xyt = w2.dropaxis(0)
        w_xy = w_xyt.dropaxis(2)
        data_small = np.array(hdul_small[self.small_fov_window].data.copy(), dtype=np.float64)
        self.data_small = data_small[:, :, index_amplitude]
        self.data_small[self.data_small == hdul_small[self.small_fov_window].header["ANA_MISS"]] = np.nan
        self.hdr_small = w_xy.to_header().copy()

        self.hdr_small["NAXIS1"] = self.data_small.shape[1]
        self.hdr_small["NAXIS2"] = self.data_small.shape[0]


class AlignementSpice_iterative_context_raster(AlignmentSpice):
    def __init__(self, large_fov_list_paths: list, small_fov_to_correct: str, threshold_time: u.Quantity,
                 lag_crval1: np.array,
                 lag_crval2: np.array, lag_cdelta1, lag_cdelta2, lag_crota, small_fov_value_min=None,
                 parallelism=False, small_fov_value_max=None, counts_cpu_max=40, large_fov_window=-1,
                 small_fov_window=-1, use_tqdm=False, path_save_figure=None,):

        super().__init__(large_fov_known_pointing="No_specific_path", small_fov_to_correct=small_fov_to_correct,
                         lag_crval1=lag_crval1, lag_crval2=lag_crval2, lag_cdelta1=lag_cdelta1, lag_cdelta2=lag_cdelta2,
                         lag_crota=lag_crota, use_tqdm=use_tqdm,
                         lag_solar_r=None, small_fov_value_min=small_fov_value_min, parallelism=parallelism,
                         small_fov_value_max=small_fov_value_max, counts_cpu_max=counts_cpu_max,
                         large_fov_window=large_fov_window, small_fov_window=small_fov_window,
                         path_save_figure=path_save_figure, )
        self.step_figure = False
        self.large_fov_list_paths = large_fov_list_paths
        self.small_fov_to_correct = small_fov_to_correct
        self.threshold_time = threshold_time

    def _step(self, d_crval2, d_crval1, d_cdelta1, d_cdelta2, d_crota, d_solar_r, method: str, ):
        hdr_small_shft = self.hdr_small.copy()

        self._shift_header(hdr_small_shft, d_crval1=d_crval1, d_crval2=d_crval2,
                           d_cdelta1=d_cdelta1, d_cdelta2=d_cdelta2,
                           d_crota=d_crota)
        hdr_small_shft_unflattened = self.header_spice_unflattened.copy()
        Util.AlignSpiceUtil.recenter_crpix_in_header_L2(hdr_small_shft_unflattened)

        self._shift_header(hdr_small_shft_unflattened, d_crval1=d_crval1, d_crval2=d_crval2,
                           d_cdelta1=d_cdelta1, d_cdelta2=d_cdelta2,
                           d_crota=d_crota)
        C = map_builder.SPICEComposedMapBuilder(path_to_spectro=self.small_fov_to_correct,
                               list_imager_paths=self.large_fov_list_paths,
                               threshold_time=self.threshold_time,
                               window_imager=self.large_fov_window, window_spectro=self.small_fov_window)
        C.process_from_header(hdr_spice=hdr_small_shft_unflattened)

        self.data_large = copy.deepcopy(C.data_composed)
        self.hdr_large = copy.deepcopy(C.hdr_composed)
        del C

        data_small = self.function_to_apply(d_solar_r=d_solar_r, data=self.data_small,
                                            hdr=hdr_small_shft)
        condition_1 = np.ones(len(data_small.ravel()), dtype='bool')
        condition_2 = np.ones(len(data_small.ravel()), dtype='bool')

        if self.small_fov_value_min is not None:
            condition_1 = np.array(data_small.ravel() > self.small_fov_value_min, dtype='bool')
        if self.small_fov_value_max is not None:
            condition_2 = np.array(data_small.ravel() < self.small_fov_value_max, dtype='bool')

        if method == 'correlation':

            lag = [0]
            is_nan = np.array((np.isnan(self.data_large.ravel(), dtype='bool')
                               | (np.isnan(data_small.ravel(), dtype='bool'))),
                              dtype='bool')
            corr = c_correlate.c_correlate(self.data_large.ravel()[(~is_nan) & (condition_1) & (condition_2)],
                                           data_small.ravel()[(~is_nan) & (condition_1) & (condition_2)],
                                           lags=lag)

            return corr

        elif method == 'residus':
            norm = np.sqrt(self.data_large.ravel())
            diff = (self.data_large.ravel() - data_small.ravel()) / norm
            return np.std(diff[(condition_1) & (condition_2)])
        else:
            raise NotImplementedError

    def _extract_imager_data_header(self, ):
        C = map_builder.SPICEComposedMapBuilder(path_to_spectro=self.small_fov_to_correct,
                               list_imager_paths=self.large_fov_list_paths,
                               threshold_time=self.threshold_time,
                               window_imager=self.large_fov_window, window_spectro=self.small_fov_window)
        C.process_from_header(hdr_spice=self.header_spice_unflattened)

        self.data_large = copy.deepcopy(C.data_composed)
        self.hdr_large = copy.deepcopy(C.hdr_composed)
        del C


    def _create_submap_of_large_data(self, data_large):
        return data_large

    def align_using_helioprojective(self, method='correlation', index_amplitude=None, shift_hdr_solar_rotation=False,
                                    extend_pixel_size=False):
        self.lonlims = None
        self.latlims = None
        self.shape = None
        self.reference_date = None
        self.function_to_apply = self._interpolate_on_large_data_grid
        self.method = method
        self.coordinate_frame = "helioprojective"
        self.shift_hdr_solar_rotation = shift_hdr_solar_rotation
        self.extend_pixel_size = extend_pixel_size

        level = None
        if "L2" in self.small_fov_to_correct:
            level = 2
        elif "L3" in self.small_fov_to_correct:
            level = 3
        self._extract_spice_data_header(level=level, index_amplitude=index_amplitude)
        self._extract_imager_data_header()
        results = super()._find_best_header_parameters()

        return results

    def _prepare_spice_from_l2(self, hdul_small):
        self.header_spice_unflattened = hdul_small[self.small_fov_window].header.copy()
        super()._prepare_spice_from_l2(hdul_small=hdul_small)



