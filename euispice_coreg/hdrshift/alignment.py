import copy

import numpy as np
import multiprocessing as mp
# from functools import partial
from astropy.coordinates import SkyCoord
from tqdm import tqdm
from multiprocessing import Process, Lock
import astropy.io.fits as Fits
from . import c_correlate
from ..utils import rectify
from astropy.wcs import WCS, FITSFixedWarning
import astropy.units as u
from ..plot import plot
import warnings
from ..utils import Util
import os
from astropy.wcs.utils import WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS
# from sunpy.map import Map
import astropy.constants
# from matplotlib import pyplot as plt

warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)
import sys
from .AlignmentResults import AlignmentResults



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Alignment:

    def __init__(self, large_fov_known_pointing: str, small_fov_to_correct: str, lag_crval1: np.array,
                 lag_crval2: np.array, lag_cdelt1: object, lag_cdelt2: object, lag_crota: object,
                 lag_solar_r: object = None,
                 small_fov_value_min: object = None,
                 parallelism: object = False, display_progress_bar: bool = False,
                 small_fov_value_max: object = None, counts_cpu_max: int = 40, large_fov_window: object = -1,
                 small_fov_window: object = -1,
                 path_save_figure: str = None, reprojection_order=2, force_crota_0=False,
                 unit_lag="arcsec"):
        """

        @param large_fov_known_pointing: (str) path to the reference file fits (most of the time an imager or a synthetic raster)
        @param small_fov_to_correct: (str)  path to the fits file to align. Only the header values will be changed.
        @param lag_crval1: (unit_lag) array of header CRVAL1 lags.
        @param lag_crval2: (unit_lag) array of header CRVAL2 lags.
        @param lag_cdelt1: (unit_lag) array of header CDELT1 lags.
        @param lag_cdelt2: (unit_lag) array of header CDELT2 lags.
        @param lag_crota: (deg) array of header CROTA lags. the PC1_1/2 matrixes will be updated accordingly.
        @param lag_solar_r: ([1/Rsun]) set to 1.004 by default. Only needed if apply carrington transformation.
        Important: If you align PHI data, you should set it at 1.000 .
        @param small_fov_value_min: min value to the absolute values. Applies to the image_to_align (optional)
        @param small_fov_value_max: max value to the absolute values. Applies to the image_to_alig (optional)
        @param parallelism: set true to allow parallelism.
        @param display_progress_bar: show progress bar in terminal.
        @param counts_cpu_max: allow max number of cpu for the parallelism.
        @param large_fov_window: (str or int) HDULIST window for the reference file
        @param small_fov_window: (str or int) HDULIST window for the fits to align
        @param path_save_figure (str): folder where to save figs following the alignement (optional, will increase computational time)
        @param reprojection_order: (int) order of the spline interpolation. Default is 2.
        @param force_crota_0: if no CROTA, CROTA2 or Pci_j matrix, force the CROTA parameter to 0.
        """
        self.large_fov_known_pointing = large_fov_known_pointing
        self.small_fov_to_correct = small_fov_to_correct
        self.lag_crval1 = lag_crval1
        self.lag_crval2 = lag_crval2
        self.lag_cdelt1 = lag_cdelt1
        self.lag_cdelt2 = lag_cdelt2
        self.lag_crota = lag_crota
        self.lag_solar_r = lag_solar_r
        self.unit_lag = unit_lag
        self.unit_lag_input = copy.deepcopy(unit_lag)

        self.lonlims = None
        self.latlims = None
        self.shape = None
        self.reference_date = None
        self.parallelism = parallelism
        self.small_fov_window = small_fov_window
        self.large_fov_window = large_fov_window

        self.crval1_ref = None
        self.crval2_ref = None
        self.crota_ref = None
        self.cdelt_ref = None
        self.data_large = None
        self.counts = counts_cpu_max
        self.data_small = None
        self.hdr_small = None
        self.hdr_large = None
        self.method = None
        self.rat_wave = {'171': '171', '193': '195', '211': '195', '131': '171', '304': '304', '335': '304',
                         '94': '171', '174': '171'}
        self.small_fov_value_min = small_fov_value_min
        self.small_fov_value_max = small_fov_value_max
        self.path_save_figure = path_save_figure
        self.display_progress_bar = display_progress_bar
        self.marker = False
        self.force_crota_0 = force_crota_0
        self._large = None
        self._small = None
        self.method_carrington_reprojection = None
        self.use_pcij = True
        self.correlation_function = c_correlate.c_correlate
        if (lag_crota is None) and (lag_cdelt1 is None) and (lag_cdelt2 is None):
            self.use_pcij = False
        self._correlation = None

        self.order = reprojection_order

        self.lock = Lock()
        self.lon_ctype = None
        self.lat_ctype = None

        # check whether the Helioprojective frame is imported through an sunpy.map import for instance.
        use_sunpy = False
        for mapping in [WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS]:
            if mapping[-1][0].__module__ == 'sunpy.coordinates.wcs_utils':
                use_sunpy = True
        self.use_sunpy = use_sunpy
        # set None values to np.array([0]) lags.
        for lag_name, lag_value in zip(["lag_crval1", "lag_crval2", "lag_crota", "lag_cdelt1", "lag_cdelt2"],
                                       [lag_crval1, lag_crval2, lag_crota, lag_cdelt1, lag_cdelt2]):
            if lag_value is None:
                self.__setattr__(lag_name, np.array([0.0]))

    # def __del__(self):

    def align_using_carrington(self, lonlims: tuple[int, int] = None, latlims: tuple[int, int] = None,
                               size_deg_carrington=None, shape=None,
                               reference_date=None, method='correlation',
                               method_carrington_reprojection="fa",
                               return_type='AlignmentResults'):
        """Align the two images. 

        Args:
            lonlims (tuple[int, int]): tuple of length 2, containing the the limits on the carrington longitude grid where 
            all data will be reprojected before alignment (in degrees)
            latlims (tuple[int, int]): tuple of length 2, containing the the limits on the carrington latitude grid (in degrees)
            size_deg_carrington (_type_, optional): Tuple of length 2 The size of a carrington pixel for the grid. If set, then
            shape is ignored. Defaults to None.
            shape (_type_, optional): Typle of length 2. Shape of the carrington grid. Defaults to None.
            reference_date (_type_, optional): Reference date when the carrington reprojection is performed, 
            to take into account for the solar rotation (and differential rotation). Defaults to None.
            method (str, optional): method to co-align to imgages. either "correlation" or "residues". Defaults to 'correlation'.
            method_carrington_reprojection (str, optional): Method to use for the carrington reprojection. Either "fa" or "sunpy". 
            If set to "sunpy", then no lonlims, latlims, size_deg or shape is required. 
            Defaults to "fa".
            return_type (str, optional): Determinates the output object of the method 
            either 'corr' or "AlignmentResults". Defaults to 'AlignmentResults'.
        Raises:
            ValueError: If some input value is incorrect or if the reference date is not manually implemented,
              while the  

        Returns:
            _type_: correlation matrix 
        """

        self.method = method
        self.coordinate_frame = "final_carrington"
        self.lon_ctype = "HPLN-TAN"
        self.lat_ctype = "HPLT-TAN"
        self.ang2pipi = True


        self.method_carrington_reprojection = method_carrington_reprojection
        f_large = Fits.open(self.large_fov_known_pointing)
        f_small = Fits.open(self.small_fov_to_correct)
        if self.method_carrington_reprojection == "fa":
            self.function_to_apply = self._carrington_transform_fa
        elif self.method_carrington_reprojection == "sunpy":
            self.function_to_apply = self._carrington_transform_sunpy
        else:
            raise ValueError("method_carrington_reprojection must be either 'fa' or 'sunpy")

        self.data_large = np.array(f_large[self.large_fov_window].data.copy(), dtype=np.float64)
        self.hdr_large = f_large[self.large_fov_window].header.copy()
        # self._recenter_crpix_in_header(self.hdr_large)

        self.hdr_small = f_small[self.small_fov_window].header.copy()
        # self._recenter_crpix_in_header(self.hdr_small)

        self.data_small = np.array(f_small[self.small_fov_window].data.copy(), dtype=np.float64)

        if method_carrington_reprojection == "fa":

            if reference_date is None:
                if "DATE-AVG":
                    raise ValueError(
                        "Either provide a reference date manualy or the reference file header must have a DATE-AVG keyword.")
                self.reference_date = self.hdr_large["DATE-AVG"]
            else:
                self.reference_date = reference_date

            if (lonlims is None) and (latlims is None) & (size_deg_carrington is not None):

                CRLN_OBS = self.hdr_small["CRLN_OBS"]
                CRLT_OBS = self.hdr_small["CRLT_OBS"]

                self.lonlims = [CRLN_OBS - 0.5 * size_deg_carrington[0], CRLN_OBS + 0.5 * size_deg_carrington[0]]
                self.latlims = [CRLT_OBS - 0.5 * size_deg_carrington[1], CRLT_OBS + 0.5 * size_deg_carrington[1]]
                self.shape = [self.hdr_small["NAXIS1"], self.hdr_small["NAXIS2"]]

            elif (lonlims is not None) and (latlims is not None) & (shape is not None):

                self.lonlims = lonlims
                self.latlims = latlims
                self.shape = shape
            else:
                raise ValueError("either set lonlims as None, or not. no in between.")
            
            if self.shape.size > 25000000:
                warnings.warn(f"shape parameter is [{shape.shape[0]}, {shape.shape[1]}], which is very large."
                               "Computational time might significantly increase")

        # if self.use_pcij:
        self._check_ant_create_pcij_matrix(self.hdr_small)
        self._check_ant_create_pcij_matrix(self.hdr_large)

        f_large.close()
        f_small.close()
        results = self._find_best_header_parameters()
        if return_type == "corr":
            return results
        elif return_type == "AlignmentResults":

            self.lag_crval1 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_crval1,
                                                                        self.unit_lag)).to(self.unit_lag_input).value
            self.lag_crval2 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_crval2,
                                                                        self.unit_lag)).to(self.unit_lag_input).value
            self.lag_cdelt1 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_cdelt1,
                                                                        self.unit_lag)).to(self.unit_lag_input).value
            self.lag_cdelt2 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_cdelt2,
                                                                        self.unit_lag)).to(self.unit_lag_input).value
            self.unit_lag = self.unit_lag_input


            return AlignmentResults(corr=results,
                                    lag_crval1=self.lag_crval1, lag_crval2=self.lag_crval2,
                                    lag_cdelt1=self.lag_cdelt1, lag_cdelt2=self.lag_cdelt2,
                                    lag_crota=self.lag_crota, unit_lag=self.unit_lag_input,
                                    image_to_align_path=self.small_fov_to_correct,
                                    image_to_align_window=self.small_fov_window,
                                    reference_image_path=self.large_fov_known_pointing,
                                    reference_image_window=self.large_fov_window)
        return results

    def align_using_helioprojective(self, method='correlation',
                                    return_type='AlignmentResults', 
                                    fov_limits=None):
        """
        Returns the results for the correlation algorithm in helioprojective frame

        Args:
            method (str, optional): Method to co align the data. Defaults to 'correlation'.
            return_type (str, optional): Determinates the output object of the method 
            either 'corr' or "AlignmentResults". Defaults to 'AlignmentResults'.
            fov_limits: list of the longitude and latitude limits to set to the small image for the correlation. 
            [[lonmin , lonmax], [latmin, latmax]] * u.arcsec

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
        self.lon_ctype = "HPLN-TAN"
        self.lat_ctype = "HPLT-TAN"
        self.ang2pipi = True

        f_large = Fits.open(self.large_fov_known_pointing)
        f_small = Fits.open(self.small_fov_to_correct)
        dat_large_var = np.array(f_large[self.large_fov_window].data.copy(), dtype=np.float64)
        self.data_large = dat_large_var

        self.hdr_large = f_large[self.large_fov_window].header.copy()
        # self._recenter_crpix_in_header(self.hdr_large)

        self.hdr_small = f_small[self.small_fov_window].header.copy()

        # if self.use_pcij:
        self._check_ant_create_pcij_matrix(self.hdr_small)
        self._check_ant_create_pcij_matrix(self.hdr_large)

        # self._recenter_crpix_in_header(self.hdr_small)
        self.data_small = np.array(f_small[self.small_fov_window].data.copy(), dtype=np.float64)
        f_large.close()
        f_small.close()


        results = self._find_best_header_parameters(fov_limits=fov_limits)
        if return_type == "corr":
            return results
        elif return_type == "AlignmentResults":
            

            self.lag_crval1 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_crval1,
                                                                        self.unit_lag)).to(self.unit_lag_input).value
            self.lag_crval2 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_crval2,
                                                                        self.unit_lag)).to(self.unit_lag_input).value
            self.lag_cdelt1 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_cdelt1,
                                                                        self.unit_lag)).to(self.unit_lag_input).value
            self.lag_cdelt2 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_cdelt2,
                                                                        self.unit_lag)).to(self.unit_lag_input).value
            self.unit_lag = self.unit_lag_input

            return AlignmentResults(corr=results,
                                    lag_crval1=self.lag_crval1, lag_crval2=self.lag_crval2,
                                    lag_cdelt1=self.lag_cdelt1, lag_cdelt2=self.lag_cdelt2,
                                    lag_crota=self.lag_crota, unit_lag=self.unit_lag_input,
                                    image_to_align_path=self.small_fov_to_correct,
                                    image_to_align_window=self.small_fov_window,
                                    reference_image_path=self.large_fov_known_pointing,
                                    reference_image_window=self.large_fov_window)

    def align_using_initial_carrington(self, method='correlation',
                                       return_type='AlignmentResults'):
        """
        Returns the results for the correlation algorithm in carrington frame, starting from images in carrington coordinates

        Args:
            method (str, optional): Method to co align the data. Defaults to 'correlation'.
            return_type (str, optional): Determinates the output object of the method 
            either 'corr' or "AlignmentResults". Defaults to 'AlignmentResults'.

        Returns:
            corr matrix or AlignmentResults depending on return_type
        """
        self.lonlims = None
        self.latlims = None
        self.shape = None
        self.reference_date = None
        self.function_to_apply = self._interpolate_on_large_data_grid

        self.method = method
        self.coordinate_frame = "initial_carrington"
        self.lon_ctype = "CRLN-CAR"
        self.lat_ctype = "CRLT-CAR"
        self.ang2pipi = False

        f_large = Fits.open(self.large_fov_known_pointing)
        f_small = Fits.open(self.small_fov_to_correct)
        dat_large_var = np.array(f_large[self.large_fov_window].data.copy(), dtype="float32")
        self.data_large = dat_large_var

        self.hdr_large = f_large[self.large_fov_window].header.copy()
        # self._recenter_crpix_in_header(self.hdr_large)

        self.hdr_small = f_small[self.small_fov_window].header.copy()

        # if self.use_pcij:
        self._check_ant_create_pcij_matrix(self.hdr_small)
        self._check_ant_create_pcij_matrix(self.hdr_large)

        # self._recenter_crpix_in_header(self.hdr_small)
        self.data_small = np.array(f_small[self.small_fov_window].data.copy(), dtype="float32")
        f_large.close()
        f_small.close()

        results = self._find_best_header_parameters(ang2pipi=False)
        if return_type == "corr":
            return results
        elif return_type == "AlignmentResults":
            return AlignmentResults(corr=results,
                                    lag_crval1=self.lag_crval1, lag_crval2=self.lag_crval2,
                                    lag_cdelt1=self.lag_cdelt1, lag_cdelt2=self.lag_cdelt2,
                                    lag_crota=self.lag_crota, unit_lag=self.unit_lag,
                                    image_to_align_path=self.small_fov_to_correct,
                                    image_to_align_window=self.small_fov_window,
                                    reference_image_path=self.large_fov_known_pointing,
                                    reference_image_window=self.large_fov_window)

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

    def _iteration_step(self, list_d_crval1, list_d_crval2, list_d_cdelt1, list_d_cdelt2, list_d_crota, d_solar_r, method: str,
                                     position: tuple, lock=None):
        A = np.array([1, 2], dtype="float")
        B = np.array([1, 2], dtype="float")
        lag = [0]
        c = self.correlation_function(A, B, lag)
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

    def _step(self, d_crval2, d_crval1, d_cdelt1, d_cdelt2, d_crota, d_solar_r, method: str):

        shmm_small, data_small = Util.MpUtils.gen_shmm(create=False, **self._small)
        shmm_large, data_large = Util.MpUtils.gen_shmm(create=False, **self._large)

        hdr_small_shft = self.hdr_small.copy()
        self._shift_header(hdr_small_shft, d_crval1=d_crval1, d_crval2=d_crval2,
                           d_cdelt1=d_cdelt1, d_cdelt2=d_cdelt2,
                           d_crota=d_crota)

        data_small_interp = self.function_to_apply(d_solar_r=d_solar_r, data=data_small, hdr=hdr_small_shft)
        data_small_interp = copy.deepcopy(data_small_interp)

        if method == 'correlation':

            lag = [0]
            is_nan = np.logical_or(np.logical_not(np.isfinite(data_large.ravel())),np.logical_not(np.isfinite(data_small_interp.ravel())))

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

    def _step_no_shmm(self, d_crval2, d_crval1, d_cdelt1, d_cdelt2, d_crota, d_solar_r, method: str):

        data_small = self.data_small.copy()
        data_large = self.data_large
        hdr_small_shft = self.hdr_small.copy()
        self._shift_header(hdr_small_shft, d_crval1=d_crval1, d_crval2=d_crval2,
                           d_cdelt1=d_cdelt1, d_cdelt2=d_cdelt2,
                           d_crota=d_crota)

        data_small_interp = self.function_to_apply(d_solar_r=d_solar_r, data=data_small, hdr=hdr_small_shft)

        if method == 'correlation':

            lag = [0]
            is_nan = np.logical_or(np.logical_not(np.isfinite(data_large.ravel())),np.logical_not(np.isfinite(data_small_interp.ravel())))
            A = np.array(data_large.ravel()[(~is_nan)], dtype="float")
            B = np.array(data_small_interp.ravel()[(~is_nan)], dtype="float")
            c = self.correlation_function(A, B, lags=lag)
            # c = np.corrcoef(data_large.ravel()[(~is_nan)], data_small_interp.ravel()[(~is_nan)])[1, 0]

            return c

        elif method == 'residus':
            norm = np.sqrt(data_large.ravel())
            diff = (data_large.ravel() - data_small_interp.ravel()) / norm
            return np.std(diff)
        else:
            raise NotImplementedError

    def _check_ant_create_pcij_matrix(self, hdr):
        if ("PC1_1" not in hdr):
            warnings.warn("PCi_j matrix not found in header of the FITS file to align. Adding it to the header.")
            if "CROTA" in hdr:
                crot = hdr["CROTA"]
            elif "CROTA2" in hdr:
                crot = hdr["CROTA2"]
            else:
                if self.force_crota_0:
                    crot = 0.0
                    hdr["CROTA"] = 0.0
                else:
                    raise ValueError("No, CROTA, CROTA2 or PCi_j matrix in your FITS file. If want to force a CROTA=0, "
                                     "please set the force_crota_0 to True when initializing Alignment ")

            rho = np.deg2rad(crot)
            lam = hdr["CDELT2"] / hdr["CDELT1"]
            hdr["PC1_1"] = np.cos(rho)
            hdr["PC2_2"] = np.cos(rho)
            hdr["PC1_2"] = - lam * np.sin(rho)
            hdr["PC2_1"] = (1 / lam) * np.sin(rho)
        if hdr["PC1_1"] >= 1.0:
            warnings.warn(f'{hdr["PC1_1"]=}, setting to  1.0.')
            hdr["PC1_1"] = 1.0
            hdr["PC2_2"] = 1.0
            hdr["PC1_2"] = 0.0
            hdr["PC2_1"] = 0.0
            hdr["CROTA"] = 0.0

        if 'CROTA' not in hdr:
            s = - np.sign(hdr["PC1_2"]) + (hdr["PC1_2"] == 0)
            hdr["CROTA"] = s * np.rad2deg(np.arccos(hdr["PC1_1"]))

    def _find_best_header_parameters(self, ang2pipi=True, fov_limits=None):

        self._set_removed_values_to_nan_in_datasmall(fov_limits)
        
        self._set_initial_header_values(ang2pipi)

        # for lag in [self.lag_crval1, self.lag_crval2, self.lag_cdelt1, self.lag_cdelt2, self.lag_crota]:
        #     if lag is None:
        #         lag = np.array([0])
        
        
        # A = np.array([1, 2], dtype="float")
        # B = np.array([1, 2], dtype="float")
        # lag = [0]
        # c = self.correlation_function(A, B, lag)

        if self.parallelism:
            results = np.zeros(
                (len(self.lag_crval1), len(self.lag_crval2), len(self.lag_cdelt1), len(self.lag_cdelt2),
                 len(self.lag_crota), len(self.lag_solar_r)), dtype="float")

            shmm_correlation, data_correlation = Util.MpUtils.gen_shmm(create=True, ndarray=results)
            self._correlation = {"name": shmm_correlation.name, "size": data_correlation.size,
                                 "shape": data_correlation.shape, "dtype": data_correlation.dtype}
            del results
            for kk, d_solar_r in enumerate(self.lag_solar_r):
                Processes = []

                if self.coordinate_frame == "final_carrington":
                    self.data_large = self.function_to_apply(d_solar_r=d_solar_r, data=self.data_large,
                                                             hdr=self.hdr_large)
                elif (self.coordinate_frame == "final_helioprojective") or (
                        self.coordinate_frame == "initial_carrington"):
                    self.data_large = self._create_submap_of_large_data(data_large=self.data_large, fov_limits=fov_limits)


                isnan = np.isnan(self.data_small)
                if isnan.all():
                    raise ValueError("minimum or maximum value have set all small FOV to nan")
                shmm_large, data_large = Util.MpUtils.gen_shmm(create=True, ndarray=copy.deepcopy(self.data_large))
                self._large = {"name": shmm_large.name, "shape": data_large.shape,
                               "dtype": data_large.dtype, "size": data_large.size, }
                del self.data_large

                shmm_small, data_small = Util.MpUtils.gen_shmm(create=True, ndarray=copy.deepcopy(self.data_small))
                self._small = {"name": shmm_small.name, "size": data_small.size, "shape": data_small.shape,
                               "dtype": data_small.dtype}
                del self.data_small

                list_d_crval1_, list_d_crval2_, list_d_cdelt1_, list_d_cdelt2_, list_d_crota_ = \
                    np.meshgrid(self.lag_crval1, self.lag_crval2, self.lag_cdelt1, self.lag_cdelt2, self.lag_crota, indexing='ij')
                xx_, yy_, zz_, nn_, pp_ = np.meshgrid(np.arange(len(self.lag_crval1)),
                                                np.arange(len(self.lag_crval2)),
                                                np.arange(len(self.lag_cdelt1)), 
                                                np.arange(len(self.lag_cdelt2)), 
                                                np.arange(len(self.lag_crota)), 
                                                indexing='ij')
                if self.counts is None:
                    self.counts = mp.cpu_count()
                list_d_crval1 = np.array_split(list_d_crval1_.ravel(), self.counts)
                list_d_crval2 = np.array_split(list_d_crval2_.ravel(), self.counts)
                list_d_cdelt1 = np.array_split(list_d_cdelt1_.ravel(), self.counts)
                list_d_cdelt2 = np.array_split(list_d_cdelt2_.ravel(), self.counts)
                list_d_crota = np.array_split(list_d_crota_.ravel(), self.counts)
                
                xx = np.array_split(xx_.ravel(), self.counts)
                yy = np.array_split(yy_.ravel(), self.counts)
                zz = np.array_split(zz_.ravel(), self.counts)
                nn = np.array_split(nn_.ravel(), self.counts)
                pp = np.array_split(pp_.ravel(), self.counts)




                for ii, sublist_d_crval1 in enumerate(list_d_crval1):
                    sublist_d_crval2 = list_d_crval2[ii]
                    sublist_d_cdelt1 = list_d_cdelt1[ii]
                    sublist_d_cdelt2 = list_d_cdelt2[ii]
                    sublist_d_crota = list_d_crota[ii]

                    position =  (
                        xx[ii], 
                        yy[ii], 
                        zz[ii], 
                        nn[ii], 
                        pp[ii], 
                        [kk] * len(xx[ii])
                    )
                    
                    
                    
                    kwargs = {
                        "list_d_crval1": sublist_d_crval1,
                        "list_d_crval2": sublist_d_crval2,
                        "list_d_cdelt1": sublist_d_cdelt1,
                        "list_d_cdelt2": sublist_d_cdelt2,
                        "list_d_crota": sublist_d_crota,
                        "d_solar_r": d_solar_r,
                        "method": self.method,
                        "lock": self.lock,
                        "position": position,
                    }

                    Processes.append(Process(target=self._iteration_step, kwargs=kwargs))

                lenp = len(Processes)
                ii = -1
                is_close = []
                while (ii < lenp - 1):
                    ii += 1
                    Processes[ii].start()
                    while (np.sum([p.is_alive() for mm, p in zip(range(lenp), Processes) if
                                   (mm not in is_close)]) > self.counts):
                        pass
                    for kk, P in zip(range(lenp), Processes):
                        if kk not in is_close:
                            if (not (P.is_alive())) and (kk <= ii):
                                P.close()
                                is_close.append(kk)

                while (np.sum([p.is_alive() for mm, p in zip(range(lenp), Processes) if (mm not in is_close)]) != 0):
                    pass
                for kk, P in zip(range(lenp), Processes):
                    if kk not in is_close:
                        if (not (P.is_alive())) and (kk <= ii):
                            P.close()
                            is_close.append(kk)

            shmm_correlation, data_correlation = Util.MpUtils.gen_shmm(create=False, **self._correlation)
            shmm_large, data_large = Util.MpUtils.gen_shmm(create=False, **self._large)
            shmm_small, data_small = Util.MpUtils.gen_shmm(create=False, **self._small)

            data_correlation_cp = copy.deepcopy(data_correlation)
            shmm_correlation.close()
            shmm_large.close()
            shmm_large.unlink()
            shmm_small.close()
            shmm_small.unlink()
            shmm_correlation.unlink()
        else:
            data_correlation_cp = np.zeros(
                (len(self.lag_crval1), len(self.lag_crval2), len(self.lag_cdelt1), len(self.lag_cdelt2),
                 len(self.lag_crota), len(self.lag_solar_r)), dtype="float")
            for hh, d_solar_r in enumerate(self.lag_solar_r):
                if self.coordinate_frame == "final_carrington":
                    self.data_large = self.function_to_apply(d_solar_r=d_solar_r, data=self.data_large,
                                                             hdr=self.hdr_large)
                elif (self.coordinate_frame == "initial_helioprojective") or (
                        self.coordinate_frame == "initial_carrington"):
                    self.data_large = self._create_submap_of_large_data(data_large=self.data_large, fov_limits=fov_limits)


                isnan = np.isnan(self.data_small)
                # shmm_large, data_large = Util.MpUtils.gen_shmm(create=True, ndarray=self.data_large)
                # self._large = {"name": shmm_large.name, "dtype": data_large.dtype, "shape": data_large.shape}
                # self.data_large = None
                #
                # shmm_small, data_small = Util.MpUtils.gen_shmm(create=True, ndarray=self.data_small)
                # self._small = {"name": shmm_small.name, "dtype": data_small.dtype, "shape": data_small.shape}
                # self.data_small = None
                #
                # shmm_large.close()
                # shmm_small.close()

                for ii, d_crval1 in enumerate(self.lag_crval1):
                    for jj, d_crval2 in enumerate(tqdm(self.lag_crval2)):
                        for kk, d_cdelt1 in enumerate(self.lag_cdelt1):
                            for mm, d_cdelt2 in enumerate(self.lag_cdelt2):
                                for ll, d_crota in enumerate(self.lag_crota):
                                    data_correlation_cp[ii, jj, kk, mm, ll, hh] = self._step_no_shmm(d_crval2=d_crval2,
                                                                                                     d_crval1=d_crval1,
                                                                                                     d_cdelt1=d_cdelt1,
                                                                                                     d_cdelt2=d_cdelt2,
                                                                                                     d_crota=d_crota,
                                                                                                     method=self.method,
                                                                                                     d_solar_r=d_solar_r,

                                                                                                     )

        return data_correlation_cp

    def _set_initial_header_values(self, ang2pipi):
        self.crval1_ref = self.hdr_small['CRVAL1']
        self.crval2_ref = self.hdr_small['CRVAL2']
        self.use_crota = True

        if 'CROTA' in self.hdr_small:
            self.crota_ref = self.hdr_small['CROTA']
        elif 'CROTA2' in self.hdr_small:
            self.crota_ref = self.hdr_small['CROTA2']
        else:
            s = - np.sign(self.hdr_small['PC1_2']) + (self.hdr_small['PC1_2'] == 0)
            self.crota_ref = np.rad2deg(np.arccos(self.hdr_small['PC1_1'])) * s
            self.hdr_small["CROTA"] = np.rad2deg(np.arccos(self.hdr_small['PC1_1']))
            # self.use_crota = False
        self.cdelt1_ref = self.hdr_small['CDELT1']
        self.cdelt2_ref = self.hdr_small['CDELT2']

        self.unit1 = self.hdr_small["CUNIT1"]
        self.unit2 = self.hdr_small["CUNIT2"]

        if self.unit_lag in self.unit1:
            pass
        else:
            warnings.warn("Units of headers in deg: Modyfying inputs units to deg.")
            if ang2pipi:
                self.lag_crval1 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_crval1,
                                                                           self.unit_lag)).to(self.unit1).value
                self.lag_crval2 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_crval2,
                                                                           self.unit_lag)).to(self.unit2).value
                self.lag_cdelt1 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_cdelt1,
                                                                           self.unit_lag)).to(self.unit1).value
                self.lag_cdelt2 = Util.AlignCommonUtil.ang2pipi(u.Quantity(self.lag_cdelt2,
                                                                           self.unit_lag)).to(self.unit2).value
            else:
                self.lag_crval1 = u.Quantity(self.lag_crval1, self.unit_lag).to(self.unit1).value
                self.lag_crval2 = u.Quantity(self.lag_crval2, self.unit_lag).to(self.unit2).value
                self.lag_cdelt1 = u.Quantity(self.lag_cdelt1, self.unit_lag).to(self.unit1).value
                self.lag_cdelt2 = u.Quantity(self.lag_cdelt2, self.unit_lag).to(self.unit2).value
            self.unit_lag = self.unit1

        if self.unit1 != self.unit2:
            raise ValueError("CUNIT1 and CUNIT2 must be equal")
        if self.lag_solar_r is None:
            self.lag_solar_r = np.array([1.004])

    def _set_removed_values_to_nan_in_datasmall(self, fov_limits=None):
        condition_1 = np.ones(self.data_small.shape, dtype='bool')
        condition_2 = np.ones(self.data_small.shape, dtype='bool')

  
        if self.small_fov_value_min is not None:
            condition_1[np.abs(self.data_small) < self.small_fov_value_min] = False
        if self.small_fov_value_max is not None:
            condition_2[np.abs(self.data_small) > self.small_fov_value_max] = False
        set_to_nan = np.logical_not(np.logical_and(condition_1, condition_2))

        self.data_small[set_to_nan] = np.nan
        if fov_limits is not None:
            self._select_fov_in_small_data(fov_limits)

    def _carrington_transform_fa(self, d_solar_r, data, hdr):
        rate_wave_ = None
        if self.hdr_large['WAVELNTH'] not in self.rat_wave.keys():
            rate_wave_ = None
        else:
            rate_wave_ = self.rat_wave['%i' % (self.hdr_large['WAVELNTH'])]

        spherical = rectify.CarringtonTransform(hdr, radius_correction=d_solar_r,
                                                reference_date=self.reference_date,
                                                rate_wave=rate_wave_)
        spherizer = rectify.Rectifier(spherical)
        image = spherizer(data, self.shape, self.lonlims, self.latlims, order=self.order, fill=-32762)
        image = np.where(image == -32762, np.nan, image)
        if Fits.HeaderDiff(hdr, self.hdr_large).identical:
            if self.path_save_figure is not None:
                date_obs = hdr["DATE-OBS"]
                dlon = (self.lonlims[1] - self.lonlims[0]) / self.shape[0]
                dlat = (self.latlims[1] - self.lonlims[0]) / self.shape[1]

                plot.PlotFunctions.plot_fov(data=image, show=False,
                                            path_save=os.path.join(self.path_save_figure,
                                                                   f'image_large_{date_obs[:19]}.pdf'),
                                            extent=(
                                                self.lonlims[0] - 0.5 * dlon, self.lonlims[1] + 0.5 * dlon,
                                                self.latlims[0] - 0.5 * dlat, self.latlims[1] + 0.5 * dlat,),
                                            xlabel="carrington longitude [°]", ylabel="carrington latitude [°]"
                                            )
                spherical = rectify.CarringtonTransform(self.hdr_small, radius_correction=d_solar_r,
                                                        reference_date=self.reference_date,
                                                        rate_wave=rate_wave_)
                spherizer = rectify.Rectifier(spherical)

                image_small = spherizer(self.data_small, self.shape, self.lonlims, self.latlims, 
                                        order=self.order, fill=-32762, )
                image_small = np.where(image_small == -32762, np.nan, image_small)
                date_obs = self.hdr_small["DATE-OBS"]

                plot.PlotFunctions.plot_fov(data=image_small, show=False,
                                            path_save=os.path.join(self.path_save_figure,
                                                                   f'image_small_{date_obs[:19]}.pdf'),
                                            extent=(
                                                self.lonlims[0] - 0.5 * dlon, self.lonlims[1] + 0.5 * dlon,
                                                self.latlims[0] - 0.5 * dlat, self.latlims[1] + 0.5 * dlat,
                                            ),
                                            xlabel="carrington longitude [°]", ylabel="carrington latitude [°]"

                                            )

        return image

    def _carrington_transform_sunpy(self, d_solar_r, data, hdr, data_large=None):
        rsun = (d_solar_r * astropy.constants.R_sun).to("m").value
        from sunpy.map import Map
        from sunpy.coordinates import propagate_with_solar_surface

        if Fits.HeaderDiff(hdr, self.hdr_large).identical:

            map_ref = Map(data, hdr)
            map_to_align = Map(self.data_small, self.hdr_small)
            map_to_align.meta["rsun_ref"] = rsun
            map_ref.meta["rsun_ref"] = rsun
            with propagate_with_solar_surface():
                map_ref_rep = map_ref.reproject_to(map_to_align.wcs)
            image = copy.deepcopy(map_ref_rep.data)
            self.hdr_large = copy.deepcopy(self.hdr_small)

            if self.path_save_figure is not None:
                date_obs = hdr["DATE-OBS"]

                plot.PlotFunctions.simple_plot_sunpy(map_to_align, show=False,
                                                     path_save=os.path.join(self.path_save_figure,
                                                                            f"image_small_{date_obs[:19]}.pdf"))
                date_obs = self.hdr_small["DATE-OBS"]
                plot.PlotFunctions.simple_plot_sunpy(map_ref, show=False,
                                                     path_save=os.path.join(self.path_save_figure,
                                                                            f"image_large_{date_obs[:19]}.pdf"))

                map_to_align = Map(self.data_small, self.hdr_small)
                map_to_align.meta["rsun_ref"] = rsun
                with propagate_with_solar_surface():
                    with HiddenPrints():
                        map_to_align_rep = map_to_align.reproject_to(map_ref.wcs)
                plot.PlotFunctions.simple_plot_sunpy(map_ref_rep, show=False,
                                                     path_save=os.path.join(self.path_save_figure,
                                                                            f"image_large_rep_{date_obs[:19]}.pdf"))

        else:
            map_to_align = Map(data, hdr)
            map_to_align.meta["rsun_ref"] = rsun
            hdr_large = copy.deepcopy(self.hdr_large)
            hdr_large["RSUN_REF"] = rsun
            w_large = WCS(hdr_large)
            with propagate_with_solar_surface():
                map_to_align_rep = map_to_align.reproject_to(w_large)
            image = copy.deepcopy(map_to_align_rep.data)

        return image

    def _create_submap_of_large_data(self, data_large,fov_limits=None ):
        if self.path_save_figure is not None:
            plot.PlotFunctions.simple_plot(self.hdr_large, data_large, show=False,
                                           path_save='%s/large_fov_before_cut.pdf' % (self.path_save_figure))

        hdr_cut = self.hdr_small.copy()
        x_cut, y_cut = self._extract_coordinates_pixels(hdr_cut, self.hdr_large)

        image_large_cut = np.zeros_like(x_cut, dtype="float32")
        Util.AlignCommonUtil.interpol2d(data_large.copy(), x=x_cut, y=y_cut,
                                        dst=image_large_cut,
                                        order=self.order, fill=np.nan)

        self.hdr_large = hdr_cut.copy()

        if self.path_save_figure is not None:
            levels = [0.15 * np.nanmax(self.data_small)]

            date_small = self.hdr_small["DATE-AVG"]
            date_small = date_small.replace(":", "_")
            plot.PlotFunctions.simple_plot(self.hdr_large, image_large_cut, show=False,
                                           path_save='%s/large_fov_%s.pdf' % (self.path_save_figure, date_small))
            plot.PlotFunctions.simple_plot(self.hdr_small, self.data_small, show=False,
                                           path_save='%s/small_fov_%s.pdf' % (self.path_save_figure, date_small))
            plot.PlotFunctions.contour_plot(self.hdr_large, image_large_cut, self.hdr_small, self.data_small,
                                            show=False, path_save='%s/compare_plot_%s.pdf' % (self.path_save_figure,
                                                                                              date_small),
                                            levels=levels)
        self.step_figure = False
        return np.array(image_large_cut)

    def _interpolate_on_large_data_grid(self, d_solar_r, data, hdr, ):



        x_large, y_large = self._extract_coordinates_pixels(self.hdr_large, hdr)

        image_small_shft = np.zeros_like(x_large, dtype="float32")
        Util.AlignCommonUtil.interpol2d(data.copy(), x=x_large, y=y_large, order=self.order,
                                        fill=np.nan, dst=image_small_shft,)
        # image_small_shft = np.where(image_small_shft == -32768, np.nan, image_small_shft)

        return image_small_shft

    def _check_sunpy(self):
        use_sunpy = False
        for mapping in [WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS]:
            if mapping[-1][0].__module__ == 'sunpy.coordinates.wcs_utils':
                use_sunpy = True
        return use_sunpy

    def _extract_coordinates_pixels(self, header_initial_to_project, header_target_projection=None, return_type="xy", ang2pipi=True):
        use_sunpy = self._check_sunpy()
        if return_type == "xy":
            w_to_project = WCS(header_target_projection)

        if use_sunpy:
            w_initial_to_project = WCS(header_initial_to_project)
            idx_lon = np.where(np.array(w_initial_to_project.wcs.ctype, dtype="str") == self.lon_ctype)[0][0]
            idx_lat = np.where(np.array(w_initial_to_project.wcs.ctype, dtype="str") == self.lat_ctype)[0][0]
            x, y = np.meshgrid(np.arange(w_initial_to_project.pixel_shape[idx_lon]),
                               np.arange(w_initial_to_project.pixel_shape[idx_lat]), )  # t dépend de x,
            if return_type=="xy":
                coords = w_initial_to_project.pixel_to_world(x, y)
                x_large, y_large = w_to_project.world_to_pixel(coords)
            else:
                if self.lon_ctype == "HPLN-TAN":
                    longitude_large = Util.AlignCommonUtil.ang2pipi(coords.Tx)
                    latitude_large = Util.AlignCommonUtil.ang2pipi(coords.Ty)
                elif  self.lon_ctype == "CRLN-CAR":
                    longitude_large = coords.lon
                    latitude_large = coords.lat

        else:
            longitude_large, latitude_large = Util.AlignEUIUtil.extract_EUI_coordinates(header_initial_to_project,
                                                                                        lon_ctype=self.lon_ctype,
                                                                                        lat_ctype=self.lat_ctype,
                                                                                        dsun=False)
            x_large, y_large = w_to_project.world_to_pixel(longitude_large, latitude_large)
        if return_type=="xy":
            return x_large,y_large
        elif return_type=="lonlat":
            return longitude_large, latitude_large

    @staticmethod
    def _get_naxis(hdr):
        if "ZNAXIS1" in hdr:
            naxis1 = hdr["ZNAXIS1"]
            naxis2 = hdr["ZNAXIS2"]
        else:
            naxis1 = hdr["NAXIS1"]
            naxis2 = hdr["NAXIS2"]
        return naxis1, naxis2


    def _select_fov_in_small_data(self, fov_limits):
        lonlims = fov_limits[0]
        latlims = fov_limits[1]
        longitude, latitude = Util.AlignEUIUtil.extract_EUI_coordinates(self.hdr_small, 
                                                                        lon_ctype=self.lon_ctype,
                                                                        lat_ctype=self.lat_ctype,
                                                                        dsun=False)
        # set_to_nan = np.logical_or(
        #     np.logical_or(longitude < lonlims[0], longitude > lonlims[1]), 
        #     np.logical_or(latitude  < latlims[0], latitude  > latlims[1]), 
        # )

        
        # self.data_small[set_to_nan] = np.nan


        long, latg, dlon, dlat = Util.PlotFits.build_regular_grid(longitude, latitude, lonlims=lonlims, latlims=latlims)


        mid_point = [long.shape[0]//2, long.shape[1]//2]
        hdrg_small = self.hdr_small.copy()
        hdrg_small["CRVAL1"] = long[mid_point[0], mid_point[1]].to(hdrg_small["CUNIT1"]).value
        hdrg_small["CRVAL2"] = latg[mid_point[0], mid_point[1]].to(hdrg_small["CUNIT2"]).value
        hdrg_small["CRPIX1"] = mid_point[0] + 1
        hdrg_small["CRPIX2"] = mid_point[1] + 1
        hdrg_small["CDELT1"] = dlon.to(hdrg_small["CUNIT1"]).value
        hdrg_small["CDELT2"] = dlat.to(hdrg_small["CUNIT2"]).value

        hdrg_small["PC1_1"] = 1.0
        hdrg_small["PC2_2"] = 1.0
        hdrg_small["PC1_2"] = 0.0
        hdrg_small["PC2_1"] = 0.0
        hdrg_small["CROTA"] = 0.0
        hdrg_small["CROTA2"] = 0.0
        
        hdrg_small["NAXIS1"] = long.shape[0]
        hdrg_small["NAXIS2"] = long.shape[1]

        xg, yg = self._extract_coordinates_pixels(header_initial_to_project=hdrg_small,
                                                  header_target_projection=self.hdr_small, 
                                                  ang2pipi = self.ang2pipi,
                                                  )
        data_small_interp = np.zeros_like(xg)
        Util.AlignCommonUtil.interpol2d(self.data_small, x=xg, y=yg, order=self.order, fill=np.nan, dst=data_small_interp)
        self.data_small = data_small_interp
        self.hdr_small = hdrg_small
        #A


