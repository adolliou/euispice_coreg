from .alignment import Alignment
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from ..synras import map_builder
from . import c_correlate
import copy
from ..utils import Util
from.AlignmentResults import AlignmentResults

#slimane imports ----------------------------
from ..slimane_funcs.utils import *
from sunpy.map import Map
from collections.abc import Iterable
import warnings
from astropy.wcs.utils import WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS
from sunpy.time import parse_time
#--------------------------------------------
class AlignmentSpice(Alignment):
    def __init__(self, large_fov_known_pointing: str, small_fov_to_correct: str,
                 lag_crval1: np.array = None, lag_crval2: np.array = None,
                 lag_cdelt1: np.array = None, lag_cdelt2: np.array = None,
                 lag_crota: np.array = None, lag_solar_r: np.array = None,
                 large_fov_window: int | str = -1, small_fov_window: int | str = -1,
                 parallelism: bool = False, counts_cpu_max: int = 40,
                 display_progress_bar: bool = False, path_save_figure: str | None = None,
                 wavelength_interval_to_sum: list[u.Quantity] | str = "all",
                 sub_fov_window: list[u.Quantity] | str = "all",
                 opencv: bool = False,
                 ):
        """

        :param large_fov_known_pointing: path to the reference image or synthetic raster FITS file with knwown pointing
        :param small_fov_to_correct: path to the L2 SPICE FITS file, where the pointing must be corrected.
        :param lag_crval1: array of floats  (in arcsec) of shift applied to the CRVAL1 attribute
        in the SPICE Fits file header. Set to None if no shift.
        :param lag_crval2: array of floats  (in arcsec), same for CRVAL2
        :param lag_cdelt1: array of floats (in arcsec), same for CDELT1
        :param lag_cdelt2: array of floats (in arcsec), same for CDELT2
        :param lag_crota: array values (in degrees), same for CROTA
        :param lag_solar_r: used for carrington reprojections only. set an array if you want to change the
        radius of the sphere where the reprojection is done. by default equal to 1.004.
        :param large_fov_window: window (int or str) of the reference imager/synthetic raster HDULIST you want to use.
        by default is -1.
        :param small_fov_window: window (int or str) of the SPICE HDULIST you want to use (spectral window in general).
        by default is -1.
        :param wavelength_interval_to_sum: has the form [wave_min * u.angstrom, wave_max * u.angstrom].
        for the given SPICE window, set the wavelength interval over which
        the sum is performed, to obtain image (X, Y) from the SPICE L2 data (X, Y, lambda).
        Default is "all" for the entire window.
        :param parallelism: choose to use parallelism or not.
        :param counts_cpu_max: choose the maximum number of CPU used.
        :param display_progress_bar: choose to display the progress bar or not.
        :param path_save_figure: path where to save the figures.
        :param sub_fov_window: for SPICE only. if "all", select the entire SPICE window. Else enter a list of the form
        [lon_min * u.arcsec, lon_max * u.arcsec, lat_min * u.arcsec, lat_max * u.arcsec].
        :param opencv, set to True to use OpenCV library.
        """
        super().__init__(large_fov_known_pointing=large_fov_known_pointing, small_fov_to_correct=small_fov_to_correct,
                         lag_crval1=lag_crval1, lag_crval2=lag_crval2, lag_cdelt1=lag_cdelt1, lag_cdelt2=lag_cdelt2,
                         lag_crota=lag_crota, display_progress_bar=display_progress_bar,
                         lag_solar_r=lag_solar_r, parallelism=parallelism,
                         counts_cpu_max=counts_cpu_max,
                         large_fov_window=large_fov_window, small_fov_window=small_fov_window,
                         path_save_figure=path_save_figure,opencv=opencv )

        self.sub_fov_window = sub_fov_window
        self.function_to_apply = None
        self.coordinate_frame = None
        self.extend_pixel_size = None
        self.cut_from_center = None
        self.wavelength_interval_to_sum = wavelength_interval_to_sum

    def align_using_helioprojective(self, method='correlation', extend_pixel_size=False,
                                    cut_from_center=None, return_type="AlignmentResults", 
                                    coefficient_l3: int = None):
        """
        Returns the results for the correlation algorithm in helioprojective frame

        Args:
            method (str, optional): Method to co align the data. Defaults to 'correlation'.
            return_type (str, optional): Determinates the output object of the method 
            either 'corr' or "AlignmentResults". Defaults to 'AlignmentResults'.
            coefficient_l3 (int, optional). Only if level = 3. Which coefficient to use
            in the L3 file for the coalignment. 

        Returns:
            corr matrix or AlignmentResults depending on return_type
        """
        self.lonlims = None
        self.latlims = None
        self.shape = None
        self.reference_date = None
        self.function_to_apply = self._interpolate_on_large_data_grid
        self.method = method
        self.coordinate_frame = "final_helioprojective"
        self.extend_pixel_size = extend_pixel_size
        self.cut_from_center = cut_from_center
        self._extract_imager_data_header()
        self.lon_ctype="HPLN-TAN"
        self.lat_ctype="HPLT-TAN"
        level = None
        if "L2" in self.small_fov_to_correct:
            level = 2
        elif "L3" in self.small_fov_to_correct:
            level = 3
        self._extract_spice_data_header(level=level, coeff=coefficient_l3)

        results = super()._find_best_header_parameters()
        # A
        if return_type == "corr":
            return results
        elif return_type == "AlignmentResults":
            return AlignmentResults(corr=results, unit_lag=self.unit_lag,
                                    lag_crval1=self.lag_crval1, lag_crval2=self.lag_crval2, 
                                    lag_cdelt1=self.lag_cdelt1, lag_cdelt2=self.lag_cdelt2, 
                                    lag_crota=self.lag_crota, 
                                    image_to_align_path=self.small_fov_to_correct, image_to_align_window=self.small_fov_window,  
                                    reference_image_path=self.large_fov_known_pointing, reference_image_window=self.large_fov_window)

    def align_using_carrington(self, lonlims: tuple[int, int], latlims: tuple[int, int],
                               size_deg_carrington=None, shape=None,
                               reference_date=None, method='correlation', 
                               return_type="AlignmentResults", 
                               coefficient_l3: int = None):
            
        if (lonlims is None) and (latlims is None) & (size_deg_carrington is not None):

            CRLN_OBS = self.hdr_small["CRLN_OBS"]
            CRLT_OBS = self.hdr_small["CRLT_OBS"]

            self.lonlims = [CRLN_OBS - 0.5 * size_deg_carrington[0], CRLN_OBS + 0.5 * size_deg_carrington[0]]
            self.latlims = [CRLT_OBS - 0.5 * size_deg_carrington[1], CRLT_OBS + 0.5 * size_deg_carrington[1]]
            self.shape = [self.hdr_small["NAXIS1"], self.hdr_small["NAXIS2"]]
            print(f"{self.lonlims=}")

        elif (lonlims is not None) and (latlims is not None) & (shape is not None):

            self.lonlims = lonlims
            self.latlims = latlims
            self.shape = shape
        else:
            raise ValueError("either set lonlims as None, or not. no in between.")
        self.reference_date = reference_date
        self.function_to_apply = self._carrington_transform
        self.extend_pixel_size = False
        self.method = method
        self.coordinate_frame = "final_carrington"
        self._extract_imager_data_header()
        self.lon_ctype="HPLN-TAN"
        self.lat_ctype="HPLT-TAN"
        level = None
        if "L2" in self.small_fov_to_correct:
            level = 2
        elif "L3" in self.small_fov_to_correct:
            level = 3
        self._extract_spice_data_header(level=level, coeff=coefficient_l3)
        self.hdr_small["CRVAL1"] = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.hdr_small["CRVAL1"],
                                                                            self.hdr_small["CUNIT1"])).to(
            "arcsec").value
        self.hdr_small["CRVAL2"] = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.hdr_small["CRVAL2"],
                                                                            self.hdr_small["CUNIT2"])).to(
            "arcsec").value
        self.hdr_small["CDELT1"] = u.Quantity(self.hdr_small["CDELT1"], self.hdr_small["CUNIT1"]).to("arcsec").value
        self.hdr_small["CDELT2"] = u.Quantity(self.hdr_small["CDELT2"], self.hdr_small["CUNIT2"]).to("arcsec").value
        self.hdr_small["CUNIT1"] = "arcsec"
        self.hdr_small["CUNIT2"] = "arcsec"

        results = super()._find_best_header_parameters()

        if return_type == "corr":
            return results
        elif return_type == "AlignmentResults":
            return AlignmentResults(corr=results, unit_lag=self.unit_lag,
                                    lag_crval1=self.lag_crval1, lag_crval2=self.lag_crval2, 
                                    lag_cdelt1=self.lag_cdelt1, lag_cdelt2=self.lag_cdelt2, 
                                    lag_crota=self.lag_crota, 
                                    image_to_align_path=self.small_fov_to_correct, image_to_align_window=self.small_fov_window,  
                                    reference_image_path=self.large_fov_known_pointing, reference_image_window=self.large_fov_window)

    def _extract_imager_data_header(self, ):
        with fits.open(self.large_fov_known_pointing) as hdul_large:
            self.data_large = np.array(hdul_large[self.large_fov_window].data.copy(), dtype=np.float64)
            self.hdr_large = hdul_large[self.large_fov_window].header.copy()
            # super()._recenter_crpix_in_header(self.hdr_large)
            hdul_large.close()

    def _extract_spice_data_header(self, level: int, coeff: int = None,):
        """prepare the SPICE data for the colalignement. Accepts L2 or L3 files.

        Args:
            level (int): Level of the input SPICE data. Must be 2 or 3.
            coeff (int, optional): only if L3. Coefficient that will be use for the co-alignment. Defaults to None.

        Raises:
            ValueError: raise Error if incorrect input level.
        """        
        with fits.open(self.small_fov_to_correct) as hdul_small:
            dt = hdul_small[self.small_fov_window].header.copy()["PC4_1"]
            if level == 2:
                self._prepare_spice_from_l2(hdul_small)
            elif level == 3:
                self._prepare_spice_from_l3(hdul_small, coeff)
            else:
                raise ValueError("level must be 2 or 3")

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
        w_wave = w_spice.sub(['spectral'])

        w_xy = w_xyt.dropaxis(2)
        self.hdr_small = w_xy.to_header().copy()
        # self.hdr_small["NAXIS1"] = data_small.shape[3]
        # self.hdr_small["NAXIS2"] = data_small.shape[2]
        # super()._recenter_crpix_in_header(self.hdr_small)
        # ylen = data_small.shape[2]

        # ylim = np.array([ymin, ylen - ymax - 1]).max()
        data_small[:, :, :ymin, :] = np.nan
        data_small[:, :, ymax:, :] = np.nan
        print(self.wavelength_interval_to_sum)
        if self.wavelength_interval_to_sum is "all":
            self.data_small = np.nansum(data_small[0, :, :, :], axis=0)
        elif type(self.wavelength_interval_to_sum).__name__ == "list":
            z = np.arange(data_small.shape[1])
            wave = w_wave.pixel_to_world(z)
            selection_wave = np.logical_and(wave >= self.wavelength_interval_to_sum[0],
                                            wave <= self.wavelength_interval_to_sum[1])
            self.data_small = np.nansum(data_small[0, selection_wave, :, :], axis=0)
        else:
            raise ValueError("wavelength_interval_to_sum must be a [wave_min * u.angstrom, wave_max * u.angstrom] "
                             "or 'all' str ")
        self.data_small[:ymin, :] = np.nan
        self.data_small[ymax:, :] = np.nan

        if self.cut_from_center is not None:
            xlen = self.cut_from_center
            xmid = self.data_small.shape[1] // 2
            self.data_small[:, :(xmid - xlen // 2 - 1)] = np.nan
            self.data_small[:, (xmid + xlen // 2):] = np.nan

        idx_lon = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
        idx_lat = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
        if self.sub_fov_window == "all":
            pass
        elif type(self.sub_fov_window).__name__ == "list":

            x, y = np.meshgrid(np.arange(w_xy.pixel_shape[idx_lon]),
                               np.arange(w_xy.pixel_shape[idx_lat]))

            if self.use_sunpy:
                coords_spice = w_xy.pixel_to_world(x, y)
                lon_spice = coords_spice.Tx
                lat_spice = coords_spice.Ty
            else:
                lon_spice, lat_spice = w_xy.pixel_to_world(x, y)

            selection_subfov_lon = np.logical_and(lon_spice >= self.sub_fov_window[0],
                                                  lon_spice <= self.sub_fov_window[1], )
            selection_subfov_lat = np.logical_and(lat_spice >= self.sub_fov_window[2],
                                                  lat_spice <= self.sub_fov_window[3], )

            selection_subfov = np.logical_and(selection_subfov_lon, selection_subfov_lat)
            self.data_small[~selection_subfov] = np.nan
        else:
            raise ValueError("sub_fov_window must be a [lon_min * u.arcsec, lon_max * u.arcsec,"
                             " lat_min * u.arcsec, lat_max * u.arcsec] "
                             "or 'all' str ")
        #
        # self.hdr_small["CRPIX1"] = (self.data_small.shape[1] + 1) / 2
        # self.hdr_small["CRPIX2"] = (self.data_small.shape[0] + 1) / 2
        #
        self.hdr_small["NAXIS1"] = self.data_small.shape[1]
        self.hdr_small["NAXIS2"] = self.data_small.shape[0]

    def _prepare_spice_from_l3(self, hdul_small, coeff: int):

        data_small = np.array(hdul_small[self.small_fov_window].data.copy(), dtype=np.float64)
        header_spice = hdul_small[self.small_fov_window].header

        self.data_small = data_small[coeff, ...]
        ymin, ymax = Util.AlignSpiceUtil.vertical_edges_limits(header_spice)
        self.data_small[:ymin, :] = np.nan
        self.data_small[ymax:, :] = np.nan       
        
        w_spice = WCS(header_spice)

        w_xyt = w_spice.dropaxis(0)
        w_xyt.wcs.pc[2, 0] = 0
        w_xy = w_xyt.dropaxis(2)
        self.hdr_small = w_xy.to_header().copy()

class AlignementSpiceIterativeContextRaster(AlignmentSpice):
    def __init__(self, large_fov_list_paths: list, small_fov_to_correct: str, threshold_time: u.Quantity,
                 lag_crval1: np.array,
                 lag_crval2: np.array, lag_cdelt1, lag_cdelt2, lag_crota, small_fov_value_min=None,
                 parallelism=False, small_fov_value_max=None, counts_cpu_max=40, large_fov_window=-1,
                 small_fov_window=-1, use_tqdm=False, path_save_figure=None, ):

        super().__init__(large_fov_known_pointing="No_specific_path", small_fov_to_correct=small_fov_to_correct,
                         lag_crval1=lag_crval1, lag_crval2=lag_crval2, lag_cdelt1=lag_cdelt1, lag_cdelt2=lag_cdelt2,
                         lag_crota=lag_crota, use_tqdm=use_tqdm,
                         lag_solar_r=None, small_fov_value_min=small_fov_value_min, parallelism=parallelism,
                         small_fov_value_max=small_fov_value_max, counts_cpu_max=counts_cpu_max,
                         large_fov_window=large_fov_window, small_fov_window=small_fov_window,
                         path_save_figure=path_save_figure, )
        self.step_figure = False
        self.large_fov_list_paths = large_fov_list_paths
        self.small_fov_to_correct = small_fov_to_correct
        self.threshold_time = threshold_time

    def _step(self, d_crval2, d_crval1, d_cdelt1, d_cdelt2, d_crota, d_solar_r, method: str, ):
        hdr_small_shft = self.hdr_small.copy()

        self._shift_header(hdr_small_shft, d_crval1=d_crval1, d_crval2=d_crval2,
                           d_cdelt1=d_cdelt1, d_cdelt2=d_cdelt2,
                           d_crota=d_crota)
        hdr_small_shft_unflattened = self.header_spice_unflattened.copy()
        Util.AlignSpiceUtil.recenter_crpix_in_header_L2(hdr_small_shft_unflattened)

        self._shift_header(hdr_small_shft_unflattened, d_crval1=d_crval1, d_crval2=d_crval2,
                           d_cdelt1=d_cdelt1, d_cdelt2=d_cdelt2,
                           d_crota=d_crota)
        C = map_builder.SPICEComposedMapBuilder(path_to_spectro=self.small_fov_to_correct,
                                                list_imager_paths=self.large_fov_list_paths,
                                                threshold_time=self.threshold_time,
                                                window_imager=self.large_fov_window,
                                                window_spectro=self.small_fov_window)
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
            A = self.data_large.ravel()[(~is_nan) & (condition_1) & (condition_2)]
            B = data_small.ravel()[(~is_nan) & (condition_1) & (condition_2)]
            
            corr = self.correlation_function(A, B, lags=lag)
            # corr = np.corrcoef(A, B)
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
                                                window_imager=self.large_fov_window,
                                                window_spectro=self.small_fov_window)
        C.process_from_header(hdr_spice=self.header_spice_unflattened)

        self.data_large = copy.deepcopy(C.data_composed)
        self.hdr_large = copy.deepcopy(C.hdr_composed)
        del C

    def _create_submap_of_large_data(self, data_large):
        return data_large

    def align_using_helioprojective(self, method='correlation', index_amplitude=None,
                                    extend_pixel_size=False):
        self.lonlims = None
        self.latlims = None
        self.shape = None
        self.reference_date = None
        self.function_to_apply = self._interpolate_on_large_data_grid
        self.method = method
        self.coordinate_frame = "final_helioprojective"
        self.extend_pixel_size = extend_pixel_size
        self.lon_ctype="HPLN-TAN"
        self.lat_ctype="HPLT-TAN"
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


class AlignmentSpiceSaffronL3(Alignment):
    def __init__(self, 
                 large_fov_known_pointing: str, 
                 small_fov_to_correct: str,
                 param_steps: dict,
                 starting_point: dict,
                #  lag_crval1: np.array = None, #NOTE: No Need for these Anymore
                #  lag_crval2: np.array = None,
                #  lag_cdelt1: np.array = None, 
                #  lag_cdelt2: np.array = None,
                #  lag_crota: np.array = None, 
                 lag_solar_r: np.array = None,
                 
                 parallelism: bool = False, 
                 counts_cpu_max: int = 40,
                 display_progress_bar: bool = False, 
                 path_save_figure: str | None = None,
                 opencv: bool = False,
                 ):
        """
            #NOTE: Slimane Added these
                - All paths are now pathlib objects better control of the paths
                - No need for self.wavelength_interval_to_sum because L3 has no wavelength axis
                - self._extract_imager_data_header now gets the meta of the Map object
                - self._extract_spice_data_header now uses sunpy map however sunpy do not show the 
                - no need for small_fov_window because the user can manipulate the data externally and then give the resulting map object 
                - I descovered in my L3 fits files I forgot to put the CROTA so the users (for now) have to add them manually unfortunatly 
                - Since there is not much hdus to pass there is no need for coefficients (the user would have already chosen which coeffiecient and sent it as an input)
                - no need for self._prepare_spice_from_l2 this class is SAFFRON focused only
                - Removing data_small,header_spice,data_large,hdr_large All of these are easiely accessible by map objects
                - no need to remove dumbells user can do this with their map object
                - from previous I concluded that _prepare_spice_from_l3 is not needed 
                - since the map object is our play grounds I made sure to copy the maps that I have got as an input
                - I don't know what is _correct_solar_rotation doing but I have only change to map meta objects
                -  No Need for lag lists (for now) gradient ascent does not need boundaries
            :param large_fov_known_pointing: path to the reference image or synthetic raster FITS file with knwown pointing
            :param small_fov_to_correct: path to the L2 SPICE FITS file, where the pointing must be corrected.
            :param lag_crval1: array of floats  (in arcsec) of shift applied to the CRVAL1 attribute
            in the SPICE Fits file header. Set to None if no shift.
            :param lag_crval2: array of floats  (in arcsec), same for CRVAL2
            :param lag_cdelt1: array of floats (in arcsec), same for CDELT1
            :param lag_cdelt2: array of floats (in arcsec), same for CDELT2
            :param lag_crota: array values (in degrees), same for CROTA
            :param lag_solar_r: used for carrington reprojections only. set an array if you want to change the
            radius of the sphere where the reprojection is done. by default equal to 1.004.
            :param large_fov_window: window (int or str) of the reference imager/synthetic raster HDULIST you want to use.
            by default is -1.
            :param small_fov_window: window (int or str) of the SPICE HDULIST you want to use (spectral window in general).
            by default is -1.
            :param wavelength_interval_to_sum: has the form [wave_min * u.angstrom, wave_max * u.angstrom].
            for the given SPICE window, set the wavelength interval over which
            the sum is performed, to obtain image (X, Y) from the SPICE L2 data (X, Y, lambda).
            Default is "all" for the entire window.
            :param parallelism: choose to use parallelism or not.
            :param counts_cpu_max: choose the maximum number of CPU used.
            :param display_progress_bar: choose to display the progress bar or not.
            :param path_save_figure: path where to save the figures.
            :param sub_fov_window: for SPICE only. if "all", select the entire SPICE window. Else enter a list of the form
            [lon_min * u.arcsec, lon_max * u.arcsec, lat_min * u.arcsec, lat_max * u.arcsec].
            :param opencv, set to True to use OpenCV library.
        """
        super().__init__(large_fov_known_pointing=large_fov_known_pointing, 
                         small_fov_to_correct=small_fov_to_correct,
                         lag_crval1=np.array([0]),#Just to initialize the parent class than not used 
                         lag_crval2=np.array([0]), 
                         lag_cdelt1=np.array([0]), 
                         lag_cdelt2=np.array([0]),
                         lag_crota= np.array([0]), 
                         display_progress_bar=display_progress_bar,
                         lag_solar_r=lag_solar_r, 
                         parallelism=parallelism,
                         counts_cpu_max=counts_cpu_max,
                         large_fov_window=-1, 
                         small_fov_window=-1,
                         path_save_figure=path_save_figure,opencv=opencv )
        if True:#Preparing small and large map
            # 1- turning all given formats to map objects
            self.large_fov_known_pointing = get_data_object(large_fov_known_pointing,as_object='map')
            self.small_fov_to_correct = get_data_object(small_fov_to_correct,as_object='map')
            if isinstance(self.large_fov_known_pointing, Iterable):
                self.large_fov_known_pointing = self.large_fov_known_pointing[0]
            if isinstance(self.small_fov_to_correct, Iterable):
                self.small_fov_to_correct = self.small_fov_to_correct[0]
            # 2- make a copy of the map objects
            self.large_fov_known_pointing = Map(self.large_fov_known_pointing.data.copy(),
                                                self.large_fov_known_pointing.meta.copy())
            self.small_fov_to_correct = Map(self.small_fov_to_correct.data.copy(),
                                            self.small_fov_to_correct.meta.copy())
            
            
        self.function_to_apply = None
        self.coordinate_frame = None
        self.extend_pixel_size = None
        self.cut_from_center = None
        self.param_steps = param_steps
        self.starting_point = starting_point
        if True:#Preparing the parameters
            for key in self.param_steps.keys():
                if not isinstance(self.param_steps[key] ,Iterable):
                    self.param_steps[key] = [self.param_steps[key]]
            
            nsteps = np.unique(([len(self.param_steps[key]) for key in self.param_steps.keys()]))
            if len(nsteps)!=1:
                raise ValueError("All the parameters must have the same number of steps")
            for key in ['crval1', 'crval2','crota', 'cdelt1', 'cdelt2'] :
                if key not in (self.param_steps.keys()):
                    self.param_steps[key] = [0]*nsteps[0]
            for key in ['crval1', 'crval2','crota', 'cdelt1', 'cdelt2']:
                if key not in (self.starting_point.keys()):
                    self.starting_point[key] = 0
            
        
        # self.wavelength_interval_to_sum = wavelength_interval_to_sum

    def align_using_helioprojective(self, method='correlation', extend_pixel_size=False,
                                    cut_from_center=None, return_type="AlignmentResults", 
                                    coefficient_l3: int = None):
        """
        TODO: Apparently this needs to be updates
        Returns the results for the correlation algorithm in helioprojective frame

        Args:
            method (str, optional): Method to co align the data. Defaults to 'correlation'.
            return_type (str, optional): Determinates the output object of the method 
            either 'corr' or "AlignmentResults". Defaults to 'AlignmentResults'.
            coefficient_l3 (int, optional). Only if level = 3. Which coefficient to use
            in the L3 file for the coalignment. 

        Returns:
            corr matrix or AlignmentResults depending on return_type
        """
        self.lonlims = None
        self.latlims = None
        self.shape = None
        self.reference_date = None
        self.function_to_apply = self._interpolate_on_large_data_grid
        self.method = method
        self.coordinate_frame = "final_helioprojective"
        self.extend_pixel_size = extend_pixel_size
        self.cut_from_center = cut_from_center
        # self._extract_imager_data_header() No need anymore
        self.lon_ctype="HPLN-TAN"
        self.lat_ctype="HPLT-TAN"
        self._extract_spice_data_header()
        results = self._find_best_header_parameters()
        return results

        if return_type == "corr":
            return results
        elif return_type == "AlignmentResults":
            return AlignmentResults(corr=results, unit_lag=self.unit_lag,
                                    # lag_crval1=self.lag_crval1, lag_crval2=self.lag_crval2, No need
                                    # lag_cdelt1=self.lag_cdelt1, lag_cdelt2=self.lag_cdelt2, No need
                                    lag_crota=self.lag_crota, 
                                    image_to_align_path=self.small_fov_to_correct, image_to_align_window=self.small_fov_window,  
                                    reference_image_path=self.large_fov_known_pointing, reference_image_window=self.large_fov_window)
    def _find_best_header_parameters(self, ang2pipi=True):

        self.crval1_ref = self.small_fov_to_correct.meta['CRVAL1']
        self.crval2_ref = self.small_fov_to_correct.meta['CRVAL2']
        self.use_crota = True

        if 'CROTA' in self.small_fov_to_correct.meta:
            self.crota_ref = self.small_fov_to_correct.meta['CROTA']
        elif 'CROTA2' in self.small_fov_to_correct.meta:
            self.crota_ref = self.small_fov_to_correct.meta['CROTA2']
        else:
            s = - np.sign(self.small_fov_to_correct.meta['PC1_2']) + (self.small_fov_to_correct.meta['PC1_2'] == 0)
            self.crota_ref = np.rad2deg(np.arccos(self.small_fov_to_correct.meta['PC1_1'])) * s
            self.small_fov_to_correct.meta["CROTA"] = np.rad2deg(np.arccos(self.small_fov_to_correct.meta['PC1_1']))
            # self.use_crota = False
        self.cdelt1_ref = self.small_fov_to_correct.meta['CDELT1']
        self.cdelt2_ref = self.small_fov_to_correct.meta['CDELT2']

        self.unit1 = self.small_fov_to_correct.meta["CUNIT1"]
        self.unit2 = self.small_fov_to_correct.meta["CUNIT2"]


        
        for key in self.param_steps.keys():
            for i in range(len(self.param_steps[key])):
                self.param_steps[key][i] = u.Quantity(
                    self.param_steps[key][i],self.unit_lag).to(self.unit1).value
        for key in self.starting_point.keys():
            self.starting_point[key] = u.Quantity(
                self.starting_point[key],self.unit_lag).to(self.unit1).value
        self.unit_lag = self.unit1
        if self.unit1 != self.unit2:
            raise ValueError("CUNIT1 and CUNIT2 must be equal")
        if self.lag_solar_r is None:
            self.lag_solar_r = np.array([1.004])
            
        self.data_large = self.large_fov_known_pointing.data.copy()   
        
        shmm_large, data_large = Util.MpUtils.gen_shmm(create=True, ndarray=self.data_large.copy())
        self._large = {"name": shmm_large.name, "shape": data_large.shape,
                        "dtype": data_large.dtype, "size": data_large.size, }
        del self.data_large # After all it is deleted XD

        shmm_small, data_small = Util.MpUtils.gen_shmm(create=True, ndarray=self.small_fov_to_correct.data.copy())
        self._small = {"name": shmm_small.name, "size": data_small.size, "shape": data_small.shape,
                        "dtype": data_small.dtype}
        # del self.data_small
        if self.counts is None:
                    self.counts = mp.cpu_count()
        
        #NOTE: Slimane method gradient ascend 
        iter_num= 1000 
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # no need for lock
        self.lock = None
        

        for kk, d_solar_r in enumerate(self.lag_solar_r):
            Processes = []

            current_index = np.array(
                [
                    self.starting_point.get("crval1", 0),
                    self.starting_point.get("crval2", 0),
                    self.starting_point.get("crota", 0),
                    self.starting_point.get("cdelt1", 0),
                    self.starting_point.get("cdelt2", 0),
                    ], dtype="float")
            
            self.correlation_results = {}
            # current_value = TODOcalc_function(tuple(current_index)) #TODO replace calc_function with the cross-correlation function
            current_value = self._step(
                d_crval1=current_index[0], d_crval2=current_index[1],
                d_crota=current_index[2], d_cdelt1=current_index[3],
                d_cdelt2=current_index[4], d_solar_r=d_solar_r,
                method=self.method)
            
            ndim = len(current_index)
            
            for ind in range(len(self.param_steps["crval1"])):
                visited = {}
                step = np.array(
                    [
                        self.param_steps.get("crval1", 0)[ind],
                        self.param_steps.get("crval2", 0)[ind],
                        self.param_steps.get("crota", 0)[ind],
                        self.param_steps.get("cdelt1", 0)[ind],
                        self.param_steps.get("cdelt2", 0)[ind],
                    ], dtype="float")
                
                print(f'working on step {step}')
                last_index = None
            
                for _ in range(iter_num):#the number of iteration is chosen to be as lage as posisble but not the exact number
                    #searching for first neighbors 
                    neighbors = get_first_n_neighbors(
                        point=current_index, 
                        step=step,
                        n = max(
                            count_touching_hypercubes(np.sum(step>0)),
                            self.counts
                            )
                        )
                    neighbors
                    # Check for already visited neighbors and walk if needed
                    if last_index is not None:
                        gradient_vector = current_index - last_index 
                        #draw a arrow starts from last_index to current_index
                        updated_neighbors = []
                        for neighbor in neighbors:
                            if neighbor in visited:
                                test_point = np.array(neighbor, dtype=float)
                                # while tuple(test_point) in visited or tuple(test_point) in neighbors or tuple(test_point) in updated_neighbors:
                                while tuple(test_point) in visited or tuple(test_point) in neighbors or tuple(test_point) in updated_neighbors:
                                    test_point += gradient_vector
                                new_neighbor = tuple(test_point)
                                updated_neighbors.append(new_neighbor)
                            else:
                                updated_neighbors.append(neighbor)
                        neighbors = updated_neighbors
                
                    # Evaluate neighbors (compute if not visited)
                    #DONE This should be a multiprocessing function that runs over all the jobs available
                    #TODO Check if I well understood the self._step and it is doing as it should be
                    # Step 1: Create the kwargs list for all new neighbors
                    tasks = []
                    new_neighbors = []
                    for n in neighbors:
                        kwargs = {
                            "d_crval1": n[0],
                            "d_crval2": n[1],
                            "d_crota": n[2],
                            "d_cdelt1": n[3],
                            "d_cdelt2": n[4],
                            "d_solar_r": d_solar_r,
                            "method": self.method,
                        }
                        tasks.append(kwargs)
                        new_neighbors.append(n)
                    if self.parallelism: 
                        # Step 2: Run in parallel
                        results = {}
                        if tasks:
                            with ProcessPoolExecutor(max_workers=self.counts) as executor:
                                futures = {executor.submit(self._step, **task): i for i, task in enumerate(tasks)}
                                for future in as_completed(futures):
                                    i = futures[future]
                                    n = new_neighbors[i]
                                    result = future.result()
                                    visited[n] = result  # Store in visited
                                    results[n] = result  # Store to later populate neighbor_values
                                    if  ind not  in self.correlation_results:
                                        self.correlation_results[ind] = {}
                                    self.correlation_results[ind][n]= result
                    else:# do the same thing without parallelism
                        results = {}
                        for i, task in enumerate(tasks):
                            task2 =  {key:task[key]*3600 for key in ["d_crval1","d_crval2","d_crota","d_cdelt1","d_cdelt2"]}
                            print(task2)
                            for key in task2:
                                if np.abs(task2[key]) > 100:
                                    raise ValueError("The values of the parameters are too large")
                            n = new_neighbors[i]
                            result = self._step(**task)
                            visited[n] = result
                            results[n] = result  # Store to later populate neighbor_values
                            if  ind not  in self.correlation_results:
                                self.correlation_results[ind] = {}
                            self.correlation_results[ind][n]= result
                            
                                
                    neighbor_values = []
                    for n in neighbors:
                        val = visited[n] if n in visited else results[n]
                        neighbor_values.append((n, val))
                            
                    #TODO===============================================================================
                    
                    # Find the best neighbor
                    sorted_neighbors = sorted(neighbor_values, key=lambda x: x[1], reverse=True)
                    best_neighbor, best_value = sorted_neighbors[0]
                    if best_value > current_value:
                        last_index = tuple(current_index.copy())
                        current_index = np.array(best_neighbor)
                        current_value = best_value
                    else:
                        break

            
            shmm_large.close()
            shmm_large.unlink()
            shmm_small.close()
            shmm_small.unlink()
        

    def align_using_carrington(self, lonlims: tuple[int, int], latlims: tuple[int, int],
                               size_deg_carrington=None, shape=None,
                               reference_date=None, method='correlation', 
                               return_type="AlignmentResults", 
                               coefficient_l3: int = None):
            
        if (lonlims is None) and (latlims is None) & (size_deg_carrington is not None):

            CRLN_OBS = self.small_fov_to_correct.meta["CRLN_OBS"]
            CRLT_OBS = self.small_fov_to_correct.meta["CRLT_OBS"]

            self.lonlims = [CRLN_OBS - 0.5 * size_deg_carrington[0], CRLN_OBS + 0.5 * size_deg_carrington[0]]
            self.latlims = [CRLT_OBS - 0.5 * size_deg_carrington[1], CRLT_OBS + 0.5 * size_deg_carrington[1]]
            self.shape = [self.small_fov_to_correct.meta["NAXIS1"], self.small_fov_to_correct.meta["NAXIS2"]]
            print(f"{self.lonlims=}")

        elif (lonlims is not None) and (latlims is not None) & (shape is not None):

            self.lonlims = lonlims
            self.latlims = latlims
            self.shape = shape
        else:
            raise ValueError("either set lonlims as None, or not. no in between.")
        self.reference_date = reference_date
        self.function_to_apply = self._carrington_transform
        self.extend_pixel_size = False
        self.method = method
        self.coordinate_frame = "final_carrington"
        self._extract_imager_data_header()
        self.lon_ctype="HPLN-TAN"
        self.lat_ctype="HPLT-TAN"
        level = None
        if "L2" in self.small_fov_to_correct:
            level = 2
        elif "L3" in self.small_fov_to_correct:
            level = 3
        self._extract_spice_data_header(level=level, coeff=coefficient_l3)
        self.small_fov_to_correct.meta["CRVAL1"] = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.small_fov_to_correct.meta["CRVAL1"],
                                                                            self.small_fov_to_correct.meta["CUNIT1"])).to(
            "arcsec").value
        self.small_fov_to_correct.meta["CRVAL2"] = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.small_fov_to_correct.meta["CRVAL2"],
                                                                            self.small_fov_to_correct.meta["CUNIT2"])).to(
            "arcsec").value
        self.small_fov_to_correct.meta["CDELT1"] = u.Quantity(self.small_fov_to_correct.meta["CDELT1"], self.small_fov_to_correct.meta["CUNIT1"]).to("arcsec").value
        self.small_fov_to_correct.meta["CDELT2"] = u.Quantity(self.small_fov_to_correct.meta["CDELT2"], self.small_fov_to_correct.meta["CUNIT2"]).to("arcsec").value
        self.small_fov_to_correct.meta["CUNIT1"] = "arcsec"
        self.small_fov_to_correct.meta["CUNIT2"] = "arcsec"

        results = super()._find_best_header_parameters()

        if return_type == "corr":
            return results
        elif return_type == "AlignmentResults":
            return AlignmentResults(corr=results, unit_lag=self.unit_lag,
                                    lag_crval1=self.lag_crval1, lag_crval2=self.lag_crval2, 
                                    lag_cdelt1=self.lag_cdelt1, lag_cdelt2=self.lag_cdelt2, 
                                    lag_crota=self.lag_crota, 
                                    image_to_align_path=self.small_fov_to_correct, image_to_align_window=self.small_fov_window,  
                                    reference_image_path=self.large_fov_known_pointing, reference_image_window=self.large_fov_window)

    def _extract_spice_data_header(self):
        """prepare the SPICE data for the colalignement. Accepts L2 or L3 files.

        Args:
            level (int): Level of the input SPICE data. Must be 2 or 3.
            coeff (int, optional): only if L3. Coefficient that will be use for the co-alignment. Defaults to None.

        Raises:
            ValueError: raise Error if incorrect input level.
        """   
        
        dt = self.small_fov_to_correct.meta.copy()["PC3_1"]
        
        if self.extend_pixel_size:
            self._correct_solar_rotation(dt)

    def _correct_solar_rotation(self, dt):
        B0 = np.deg2rad(self.small_fov_to_correct.meta['SOLAR_B0'])
        band = self.large_fov_known_pointing.meta['WAVELNTH']
        omega_car = np.deg2rad(360 / 25.38 / 86400)  # rad s-1
        if band == 174:
            band = 171
        omega = omega_car + Util.AlignEUIUtil.diff_rot(B0, f'EIT {band}')  # rad s-1
        # helioprojective rotation rate for s/c
        Rsun = self.small_fov_to_correct.meta['RSUN_REF']  # m
        Dsun = self.small_fov_to_correct.meta['DSUN_OBS']  # m
        phi_rot = 1.004 * omega * Rsun / (Dsun - 1.004 * Rsun)  # rad s-1
        phi_rot = np.rad2deg(phi_rot) * 3600  # arcsec s-1

        # rake into account angle to the limb
        alpha = u.Quantity(self.small_fov_to_correct.meta["CRVAL1"], self.small_fov_to_correct.meta["CUNIT1"]).to("rad").value

        phi = np.arcsin(((Dsun - 1.004 * Rsun) / (1.004 * Rsun)) * np.sin(
            alpha))  # heliocentric longitude with respect to spacecraft pointing
        if (-np.pi / 2 > phi > np.pi / 2):
            raise ValueError("Error in estimating heliocentric latitude")

        DTx_old = u.Quantity(self.small_fov_to_correct.meta['CDELT1'], self.small_fov_to_correct.meta['CUNIT1']).to("arcsec")
        DTx_new = DTx_old - dt * phi_rot * u.arcsec * np.cos(phi)  # last term to take into account that structure
        # are less shrinked on the limbs
        self.small_fov_to_correct.meta['CDELT1'] = DTx_new.to(self.small_fov_to_correct.meta['CUNIT1']).value
        print(f'Corrected solar rotation : changed SPICE CDELT1 from {DTx_old} to {DTx_new}')

    def _iteration_step(self, list_d_crval1, list_d_crval2, list_d_cdelt1, list_d_cdelt2, list_d_crota, d_solar_r, method: str,
                                     position: tuple, lock=None):
        # NOTE: Again what the hell is this
        # TODO search whether important
        # A = np.array([1, 2], dtype="float")
        # B = np.array([1, 2], dtype="float")
        # lag = [0]
        # c = self.correlation_function(A, B, lag)
        results = np.zeros(len(list_d_crval1), dtype=np.float64)
        if self.display_progress_bar:
            for ii, d_crval1 in enumerate(tqdm(list_d_crval1)):
                d_crval2 = list_d_crval2[ii]
                d_cdelt1 = list_d_cdelt1[ii]
                d_cdelt2 = list_d_cdelt2[ii]
                d_crota = list_d_crota[ii]

                results[ii] = self._step(d_crval2=d_crval2, d_crval1=d_crval1,
                                         d_cdelt1=d_cdelt1, d_cdelt2=d_cdelt2, d_crota=d_crota,
                                         method=method, d_solar_r=d_solar_r,
                                         )

        else:

            for ii, d_crval1 in enumerate(list_d_crval1):
                d_crval2 = list_d_crval2[ii]
                d_cdelt1 = list_d_cdelt1[ii]
                d_cdelt2 = list_d_cdelt2[ii]
                d_crota = list_d_crota[ii]

                results[ii] = self._step(d_crval2=d_crval2, d_crval1=d_crval1,
                                         d_cdelt1=d_cdelt1, d_cdelt2=d_cdelt2, d_crota=d_crota,
                                         method=method, d_solar_r=d_solar_r,
                                         )

        lock.acquire()
        shmm_correlation, data_correlation = Util.MpUtils.gen_shmm(create=False, **self._correlation)
        # for uu in range(len(position[0])):
        data_correlation[position[0], position[1], position[2], position[3], position[4], position[5],] = results
        lock.release()
        shmm_correlation.close()
    
    def _step(self, d_crval1,d_crval2, d_crota, d_cdelt1, d_cdelt2, d_solar_r, method: str):
        shmm_small, data_small = Util.MpUtils.gen_shmm(create=False, **self._small)
        shmm_large, data_large = Util.MpUtils.gen_shmm(create=False, **self._large)

        hdr_small_shft = self.small_fov_to_correct.fits_header.copy()
        # hdr_small_shft['history'] = None
        # hdr_small_shft['keycomments'] = None
        
        # print(d_crval1, d_crval2, d_crota, d_cdelt1, d_cdelt2)
        self._shift_header(hdr_small_shft, d_crval1=d_crval1, d_crval2=d_crval2,
                           d_cdelt1=d_cdelt1, d_cdelt2=d_cdelt2,
                           d_crota=d_crota)
        # if np.abs(hdr_small_shft['CRVAL1'])>=1:
        #     raise ValueError("CRVAL1 is so large in arcsec")
        # else:print(f"CRVAL1 is {hdr_small_shft['CRVAL1']} in arcsec CRVAL2 is {hdr_small_shft['CRVAL2']} in arcsec")
        
        data_small_interp = self.function_to_apply(d_solar_r=d_solar_r, data=data_small, hdr=hdr_small_shft)
        data_small_interp = copy.deepcopy(data_small_interp)

        if method == 'correlation':   
            lag = [0]
            is_nan = np.array((np.isnan(data_large.ravel(), dtype='bool')
                               | (np.isnan(data_small_interp.ravel(), dtype='bool'))),
                              dtype='bool')
            # if data_large.ravel()[(~is_nan)].shape == data_small_interp.ravel()[(~is_nan)].shape:
            # c = np.corrcoef(data_large.ravel()[(~is_nan)], data_small_interp.ravel()[(~is_nan)])[1, 0]
            A = np.array(data_large.ravel()[(~is_nan)], dtype="float")
            B = np.array(data_small_interp.ravel()[(~is_nan)], dtype="float")
            c = self.correlation_function(A, B, lags=lag)
            

            # print(f'{data_large=}')
            # l = data_small_interp.shape
            # print(f'{data_small_interp[l[0]//2, l[1]//2]=}')
            # print(f'{c=}')

            # c = copy.deepcopy(c)
            shmm_large.close()
            shmm_small.close()

            return c

        elif method == 'residus':
            norm = np.sqrt(data_large.ravel())
            diff = (data_large.ravel() - data_small_interp.ravel()) / norm
            return np.std(diff)
        else:
            raise NotImplementedError
        
    def _interpolate_on_large_data_grid(self, d_solar_r, data, hdr, ):
        w_xy_small = WCS(hdr)
        
        use_sunpy = False
        for mapping in [WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS]:
            if mapping[-1][0].__module__ == 'sunpy.coordinates.wcs_utils':
                use_sunpy = True
        if use_sunpy:

            w_large = WCS(self.large_fov_known_pointing.fits_header)
            idx_lon = np.where(np.array(w_large.wcs.ctype, dtype="str") == self.lon_ctype)[0][0]
            idx_lat = np.where(np.array(w_large.wcs.ctype, dtype="str") == self.lat_ctype)[0][0]
            x, y = np.meshgrid(np.arange(self.large_fov_known_pointing.data.shape[::-1][idx_lon]),
                               np.arange(self.large_fov_known_pointing.data.shape[::-1][idx_lat]), )  # t dépend de x,
            coords = w_large.pixel_to_world(x, y)
            # x_large, y_large = w_xy_small.world_to_pixel(coords,) Saffron has 3 axis (XYand time )
            x_large,y_large,time = w_xy_small.world_to_pixel(
                coords,
                parse_time(WCS(self.small_fov_to_correct.fits_header).wcs.dateobs)
                )
            
        else:
            longitude_large, latitude_large = Util.AlignEUIUtil.extract_EUI_coordinates(self.hdr_large,
                                                                                        lon_ctype=self.lon_ctype,
                                                                                        lat_ctype=self.lat_ctype,
                                                                                        dsun=False)
            x_large, y_large = w_xy_small.world_to_pixel(longitude_large, latitude_large)

        image_small_shft = np.zeros_like(x_large, dtype="float32")
        Util.AlignCommonUtil.interpol2d(data.copy(), x=x_large, y=y_large, order=self.order,
                                        fill=np.nan, dst=image_small_shft, opencv=self.opencv)
        # if np.all(np.isnan(image_small_shft)):
        #     raise ValueError("No data in the small FOV")
        # image_small_shft = np.where(image_small_shft == -32768, np.nan, image_small_shft)

        return image_small_shft
    
    def _shift_header(self, hdr, **kwargs):
        if 'd_crval1' in kwargs.keys():
            if self.unit_lag == hdr["CUNIT1"]:
                hdr['CRVAL1'] = self.crval1_ref + kwargs["d_crval1"]
            else:
                raise ValueError("lag.unit and cUNIT are not the same")
                hdr['CRVAL1'] = u.Quantity(self.crval1_ref, self.unit_lag).to(hdr["CUNIT1"]).value \
                                + u.Quantity(kwargs["d_crval1"], self.unit_lag).to(hdr["CUNIT1"]).value

        if 'd_crval2' in kwargs.keys():
            if self.unit_lag == hdr["CUNIT2"]:
                hdr['CRVAL2'] = self.crval2_ref + kwargs["d_crval2"]
            else:
                raise ValueError("lag.unit and cUNIT are not the same")

                hdr['CRVAL2'] = u.Quantity(self.crval2_ref, self.unit_lag).to(hdr["CUNIT2"]).value \
                                + u.Quantity(kwargs["d_crval2"], self.unit_lag).to(hdr["CUNIT2"]).value
        change_pcij = False

        if ('d_cdelt1' in kwargs.keys()):
            if kwargs["d_cdelt1"] != 0.0:
                change_pcij = True
                if self.unit_lag == hdr["CUNIT1"]:
                    cdelt1 = self.cdelt1_ref + kwargs["d_cdelt1"]
                else:
                    raise ValueError("lag.unit and cUNIT are not the same")

                    cdelt1 = (u.Quantity(self.cdelt1_ref, self.unit_lag)
                              + u.Quantity(kwargs["d_cdelt1"], self.unit_lag))
                    hdr['CDELT1'] = cdelt1.to(hdr["CUNIT1"]).value
        if 'd_cdelt2' in kwargs.keys():
            if kwargs["d_cdelt2"] != 0.0:
                change_pcij = True
                if self.unit_lag == hdr["CUNIT2"]:
                    cdelt2 = self.cdelt2_ref + kwargs["d_cdelt2"]
                else:
                    raise ValueError("lag.unit and CUNIT are not the same")
                    cdelt2 = (u.Quantity(self.cdelt2_ref, self.unit_lag)
                              + u.Quantity(kwargs["d_cdelt2"], self.unit_lag))
                hdr['CDELT2'] = cdelt2.to(hdr["CUNIT2"]).value
        if 'd_crota' in kwargs.keys():
            if kwargs["d_crota"] != 0.0:
                change_pcij = True

                if 'CROTA' in hdr:
                    hdr['CROTA'] = self.crota_ref + kwargs["d_crota"]
                    # crot = hdr['CROTA']
                elif 'CROTA2' in hdr:
                    hdr['CROTA2'] = self.crota_ref + kwargs["d_crota"]
                    # crot = hdr['CROTA2']
                else:
                    if kwargs["d_crota"] != 0.0:
                        crot = np.rad2deg(np.arccos(hdr["PC1_1"]))
                        s = - np.sign(hdr["PC1_2"]) + (hdr["PC1_2"] == 0.0)
                        crot = crot * s
                        hdr["CROTA"] = crot
            if kwargs["d_crota"] != 0.0:
                crot = self.crota_ref + kwargs["d_crota"]
            else:
                crot = self.crota_ref
            # raise NotImplementedError
        if change_pcij:
            rho = np.deg2rad(crot)
            lam = hdr["CDELT2"] / hdr["CDELT1"]
            hdr["PC1_1"] = np.cos(rho)
            hdr["PC2_2"] = np.cos(rho)
            hdr["PC1_2"] = - lam * np.sin(rho)
            hdr["PC2_1"] = (1 / lam) * np.sin(rho)

