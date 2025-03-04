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
        self.coordinate_frame = "helioprojective"
        self.extend_pixel_size = extend_pixel_size
        self.cut_from_center = cut_from_center
        self._extract_imager_data_header()

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
        self.coordinate_frame = "carrington"
        self._extract_imager_data_header()

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

    # def _prepare_spice_from_l3(self, hdul_small, index_amplitude):
    #     w = WCS(hdul_small[self.small_fov_window].header.copy())
    #     w2 = w.deepcopy()
    #     w2.wcs.pc[3, 0] = 0
    #     w2.wcs.pc[3, 1] = 0
    #     w_xyt = w2.dropaxis(0)
    #     w_xy = w_xyt.dropaxis(2)
    #     data_small = np.array(hdul_small[self.small_fov_window].data.copy(), dtype=np.float64)
    #     self.data_small = data_small[:, :, index_amplitude]
    #     self.data_small[self.data_small == hdul_small[self.small_fov_window].header["ANA_MISS"]] = np.nan
    #     self.hdr_small = w_xy.to_header().copy()
    #
    #     self.hdr_small["NAXIS1"] = self.data_small.shape[1]
    #     self.hdr_small["NAXIS2"] = self.data_small.shape[0]

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
        self.coordinate_frame = "helioprojective"
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
