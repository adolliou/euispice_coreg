import warnings

import astropy.io.fits as fits
from astropy.time import Time, TimeDelta
import astropy.constants
from tqdm import tqdm
from scipy.ndimage import map_coordinates
import astropy.constants
from matplotlib import pyplot as plt
import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from astropy.visualization import ImageNormalize, AsymmetricPercentileInterval, SqrtStretch, LinearStretch, LogStretch
from multiprocess.shared_memory import SharedMemory
from astropy.coordinates import SkyCoord
import cv2


class AlignCommonUtil:

    @staticmethod
    def find_closest_dict_index(utc_to_find, dict_file_reference, threshold_time, time_delay=False,
                                dsun_obs_to_find=None):
        if time_delay:
            if dsun_obs_to_find is None:
                raise ValueError("please enter dsun_obs_ref if time delay is not negligeable.")
            time = np.array([n - TimeDelta(((d * u.m - dsun_obs_to_find) / astropy.constants.c).to("s"))
                             for n, d in zip(dict_file_reference["date-avg"], dict_file_reference["dsun-obs"])],
                            dtype="object")
        else:
            time = dict_file_reference["date-avg"]

        delta_time = np.array([np.abs((utc_to_find - n).to(u.s).value) for n in time])
        closest_index = delta_time.argmin()
        delta_time_min = delta_time[closest_index]

        if delta_time_min > threshold_time:
            raise ValueError("Delta time between EUI and SPICE file "
                             "equal to %2f s > %.2f" % (delta_time_min, threshold_time))
        return closest_index, delta_time_min

    @staticmethod
    def find_closest_time(list_to_find: list, list_ref: list, window_to_find=-1, window_ref=-1, time_delay=True,
                          maximal_threshold=15 * u.s):
        """
        Returns a list (of size length(list_to_find) with the index of the closest Fits files.
        Returns
        """
        list_index = np.arr
        for ii, fits_path_to_find in enumerate(tqdm(list_to_find, desc="find_closest_time")):
            with fits.open(fits_path_to_find) as hdul:
                hdu = hdul[window_to_find]
                date_obs_to_find = Time(hdu.header["DATE-AVG"])
                if time_delay:
                    dsun_obs_to_find = hdu.header["DSUN_OBS"]
                time_diff = []
                for jj, fits_path_ref in enumerate(list_ref):
                    with fits.open(fits_path_ref) as hdul_tmp:
                        hdu_tmp = hdul_tmp[window_ref]
                        date_obs_ref = Time(hdu_tmp.header["DATE-AVG"])

                        if time_delay:
                            dsun_obs_ref = hdu_tmp.header["DSUN_OBS"]
                            dt = (np.array(dsun_obs_to_find) - np.array(dsun_obs_ref)) / astropy.constants.c.value
                            date_obs_ref = date_obs_ref + dt * u.s
                        time_diff.append(np.abs((date_obs_to_find - date_obs_ref).to(u.s).value))
                        hdul_tmp.close()
                list_index.append(np.array(time_diff).argmin())
                hdul.close()
            list_index = np.array(list_index, dtype='int')
            selection_error = (list_index > maximal_threshold.to(u.s).value)
            if selection_error.sum() > 0:
                raise ValueError("Threshold delta time of %i s attained" % (maximal_threshold.to(u.s).value))

    @staticmethod
    def ang2pipi(ang):
        """ put angle between ]-180, +180] deg """
        pi = u.Quantity(180, 'deg')
        return - ((- ang + pi) % (2 * pi) - pi)

    @staticmethod
    def interpol2d(image, x, y, fill, order, dst=None, opencv=False):
        """"
        taken from Frederic interpol2d function
        """
        bad = np.logical_or(x == np.nan, y == np.nan)
        x = np.where(bad, -1, x)
        y = np.where(bad, -1, y)

        coords = np.stack((y.ravel(), x.ravel()), axis=0)
        return_ = False
        if opencv:
            image = np.array(image, dtype="float32")
        if dst is None:
            return_ = True
            dst = np.empty(x.shape, dtype=image.dtype)
        if opencv:
            if order == 0:
                inter = cv2.INTER_NEAREST
            elif order == 1:
                inter = cv2.INTER_LINEAR
            elif order == 2:
                inter = cv2.INTER_CUBIC
            else:
                raise ValueError("order must be 0, 1 or 2 for openCV")
            cv2.remap(image,
                      x.astype(np.float32),  # converts to float 32 for opencv
                      y.astype(np.float32),  # does nothing with default dtype
                      inter,  # interpolation method
                      dst,  # destination array
                      cv2.BORDER_CONSTANT,  # fills in with constant value
                      fill)
        else:
            map_coordinates(image,
                            coords,
                            order=order,
                            mode='constant',
                            cval=fill, output=dst.ravel(), prefilter=False)
        if return_:
            return dst

    @staticmethod
    def write_corrected_fits(path_l2_input: str, window_list, path_l3_output: str, corr: np.array,
                             lag_crval1=None, lag_crval2=None, lag_crota=None,
                             lag_cdelt1=None, lag_cdelt2=None,
                             ):
        max_index = np.unravel_index(np.nanargmax(corr), corr.shape)
        with fits.open(path_l2_input) as hdul:
            for window in window_list:
                header = hdul[window].header.copy()
                data = hdul[window].data.copy()

                AlignCommonUtil.correct_pointing_header(header,
                                                        lag_crval1=lag_crval1[max_index[0]],
                                                        lag_crval2=lag_crval2[max_index[1]],
                                                        lag_cdelt1=lag_cdelt1[max_index[2]],
                                                        lag_cdelt2=lag_cdelt2[max_index[3]],
                                                        lag_crota=lag_crota[max_index[4]]
                                                        )

                if isinstance(hdul[window], astropy.io.fits.hdu.compressed.compressed.CompImageHDU):
                    hdu = fits.CompImageHDU(data=data, header=header)
                elif isinstance(hdul[window], astropy.io.fits.hdu.image.ImageHDU):
                    hdu = fits.ImageHDU(data=data, header=header)
                elif isinstance(hdul[window], astropy.io.fits.hdu.image.PrimaryHDU):
                    hdu = fits.PrimaryHDU(data=data, header=header)
                hdul[window] = hdu
            hdul.writeto(path_l3_output, overwrite=True)
            hdul.close()

    @staticmethod
    def correct_pointing_header(header, lag_cdelt1, lag_cdelt2, lag_crota, lag_crval1, lag_crval2):
        AlignCommonUtil._check_and_create_pcij_crota_hdr(header)
        if header["PC1_1"] > 1.0:
            warnings.warn(f'{header["PC1_1"]=}, set it to 1.0')
            header["PC1_1"] = 1.0
            header["PC2_2"] = 1.0
            header["PC1_2"] = 0.0
            header["PC2_1"] = 0.0
            header["CROTA"] = 0.0
        change_pcij = False
        if lag_crval1 is not None:
            header['CRVAL1'] = header['CRVAL1'
                               ] + u.Quantity(lag_crval1, "arcsec").to(
                header['CUNIT1']).value
        if lag_crval2 is not None:
            header['CRVAL2'] = header['CRVAL2'
                               ] + u.Quantity(lag_crval2, "arcsec").to(
                header['CUNIT2']).value
        key_rota = None
        if "CROTA" in header:
            key_rota = "CROTA"
            crota = header[key_rota]

        elif "CROTA2" in header:
            key_rota = "CROTA2"
            crota = header[key_rota]
        else:
            crota = u.Quantity(np.arccos(header["PC1_1"]), "rad").to("deg").value
            s = - np.sign(header["PC1_2"]) + (header["PC1_2"] == 0.0)
            crota = crota * s
        if lag_crota is not None:
            crota += lag_crota
            if key_rota is not None:
                header[key_rota] = crota
            change_pcij = True
        if lag_cdelt1 is not None:
            header['CDELT1'] = header['CDELT1'] + u.Quantity(
                lag_cdelt1, "arcsec").to(
                header['CUNIT1']).value
            change_pcij = True
        if lag_cdelt2 is not None:
            header['CDELT2'] = header['CDELT2'] + u.Quantity(
                lag_cdelt2, "arcsec").to(
                header['CUNIT2']).value
            change_pcij = True
        if change_pcij:
            theta = u.Quantity(crota, "deg").to("rad").value
            lam = header["CDELT2"] / header["CDELT1"]
            header["PC1_1"] = np.cos(theta)
            header["PC2_2"] = np.cos(theta)
            header["PC1_2"] = - lam * np.sin(theta)
            header["PC2_1"] = (1 / lam) * np.sin(theta)

    # @staticmethod
    # def write_corrected_fits(path_l2_input: str, window_list, path_l3_output: str, corr: np.array,
    #                          lag_crval1=None, lag_crval2=None, lag_crota=None,
    #                          lag_cdelt1=None, lag_cdelt2=None,
    #                          ):
    #     max_index = np.unravel_index(np.nanargmax(corr), corr.shape)
    #     with fits.open(path_l2_input) as hdul:
    #         for window in window_list:
    #
    #             AlignCommonUtil._check_and_create_pcij_crota_hdr(hdul[window].header)
    #
    #             if hdul[window].header["PC1_1"] > 1.0:
    #                 warnings.warn(f'{hdul[window].header["PC1_1"]=}, set it to 1.0')
    #                 hdul[window].header["PC1_1"] = 1.0
    #                 hdul[window].header["PC2_2"] = 1.0
    #                 hdul[window].header["PC1_2"] = 0.0
    #                 hdul[window].header["PC2_1"] = 0.0
    #                 hdul[window].header["CROTA"] = 0.0
    #             change_pcij = False
    #             if lag_crval1 is not None:
    #                 hdul[window].header['CRVAL1'] = hdul[window].header['CRVAL1'
    #                                                 ] + u.Quantity(lag_crval1[max_index[0]], "arcsec").to(
    #                     hdul[window].header['CUNIT1']).value
    #             if lag_crval2 is not None:
    #                 hdul[window].header['CRVAL2'] = hdul[window].header['CRVAL2'
    #                                                 ] + u.Quantity(lag_crval2[max_index[1]],
    #                                                                "arcsec").to(
    #                     hdul[window].header['CUNIT2']).value
    #             key_rota = None
    #
    #             if "CROTA" in hdul[window].header:
    #                 key_rota = "CROTA"
    #                 crota = hdul[window].header[key_rota]
    #
    #             elif "CROTA2" in hdul[window].header:
    #                 key_rota = "CROTA2"
    #                 crota = hdul[window].header[key_rota]
    #             else:
    #                 crota = u.Quantity(np.arccos(hdul[window].header["PC1_1"]), "rad").to("deg").value
    #                 s = - np.sign(hdul[window].header["PC1_2"]) + (hdul[window].header["PC1_2"] == 0.0)
    #                 crota = crota * s
    #
    #             if lag_crota is not None:
    #                 crota += lag_crota[max_index[4]]
    #                 if key_rota is not None:
    #                     hdul[window].header[key_rota] = crota
    #                 change_pcij = True
    #
    #             if lag_cdelt1 is not None:
    #                 hdul[window].header['CDELT1'] = hdul[window].header['CDELT1'] + u.Quantity(
    #                     lag_cdelt1[max_index[2]],
    #                     "arcsec").to(
    #                     hdul[window].header['CUNIT1']).value
    #                 change_pcij = True
    #
    #             if lag_cdelt2 is not None:
    #                 hdul[window].header['CDELT2'] = hdul[window].header['CDELT2'] + u.Quantity(
    #                     lag_cdelt2[max_index[3]],
    #                     "arcsec").to(
    #                     hdul[window].header['CUNIT2']).value
    #                 change_pcij = True
    #             if change_pcij:
    #                 theta = u.Quantity(crota, "deg").to("rad").value
    #                 lam = hdul[window].header["CDELT2"] / hdul[window].header["CDELT1"]
    #                 hdul[window].header["PC1_1"] = np.cos(theta)
    #                 hdul[window].header["PC2_2"] = np.cos(theta)
    #                 hdul[window].header["PC1_2"] = - lam * np.sin(theta)
    #                 hdul[window].header["PC2_1"] = (1 / lam) * np.sin(theta)
    #
    #         hdul.writeto(path_l3_output, overwrite=True)
    #         hdul.close()

    @staticmethod
    def _check_and_create_pcij_crota_hdr(hdr):
        if ("PC1_1" not in hdr):
            warnings.warn(
                "PCi_j matrix not found in header of the FITS file to align. Adding it to the header.")
            if "CROTA" in hdr:
                crot = hdr["CROTA"]
            elif "CROTA2" in hdr:
                crot = hdr["CROTA2"]
            else:
                hdr["CROTA"] = 0.0
                crot=0.0

            rho = np.deg2rad(crot)
            lam = hdr["CDELT2"] / hdr["CDELT1"]
            hdr["PC1_1"] = np.cos(rho)
            hdr["PC2_2"] = np.cos(rho)
            hdr["PC1_2"] = - lam * np.sin(rho)
            hdr["PC2_1"] = (1 / lam) * np.sin(rho)
        if hdr["PC1_1"] > 1.0:
            warnings.warn(f'{hdr["PC1_1"]=}, setting to  1.0.')
            hdr["PC1_1"] = 1.0
            hdr["PC2_2"] = 1.0
            hdr["PC1_2"] = 0.0
            hdr["PC2_1"] = 0.0
            hdr["CROTA"] = 0.0
        if 'CROTA' not in hdr:
            s = - np.sign(hdr["PC1_2"]) + (hdr["PC1_2"] == 0)
            hdr["CROTA"] = s * np.rad2deg(np.arccos(hdr["PC1_1"]))

    @staticmethod
    def align_pixels_shift(delta_pix1, delta_pix2, windows, large_fov_fits_path,
                           large_fov_window, small_fov_path, ):
        with fits.open(small_fov_path) as hdul_spice_CR_ref:
            for win in windows:
                header_spice_CR_ref = hdul_spice_CR_ref[win].header.copy()
                with fits.open(large_fov_fits_path) as hdul_fsi_ref:
                    header_fsi = hdul_fsi_ref[large_fov_window].header.copy()
                    w_fsi = WCS(header_fsi)
                    if "ZNAXIS1" in header_fsi:
                        naxis1 = header_fsi["ZNAXIS1"]
                        naxis2 = header_fsi["ZNAXIS2"]
                    else:
                        naxis1 = header_fsi["NAXIS1"]
                        naxis2 = header_fsi["NAXIS2"]
                    x_mid = (naxis1 - 1) / 2
                    y_mid = (naxis2 - 1) / 2
                    lon_mid, lat_mid = w_fsi.pixel_to_world(np.array([x_mid]), np.array([y_mid]))
                    lon_mid_fsi = lon_mid[0].to(header_spice_CR_ref["CUNIT1"]).value
                    lat_mid_fsi = lat_mid[0].to(header_spice_CR_ref["CUNIT2"]).value
                    if "ZNAXIS1" in hdul_spice_CR_ref[win].header:
                        naxis1 = hdul_spice_CR_ref[win].header["ZNAXIS1"]
                        naxis2 = hdul_spice_CR_ref[win].header["ZNAXIS2"]
                    else:
                        naxis1 = hdul_spice_CR_ref[win].header["NAXIS1"]
                        naxis2 = hdul_spice_CR_ref[win].header["NAXIS2"]

                    hdul_spice_CR_ref[win].header["CRVAL1"] = lon_mid_fsi + delta_pix1 * header_spice_CR_ref["CDELT1"]
                    hdul_spice_CR_ref[win].header["CRVAL2"] = lat_mid_fsi + delta_pix2 * header_spice_CR_ref["CDELT2"]
                    hdul_spice_CR_ref[win].header["CRPIX1"] = (naxis1 + 1) / 2
                    hdul_spice_CR_ref[win].header["CRPIX2"] = (naxis2 + 1) / 2
        return hdul_spice_CR_ref[win].header


class AlignEUIUtil:
    @staticmethod
    def extract_EUI_coordinates(hdr, dsun=True, lon_ctype="HPLN-TAN", lat_ctype="HPLT-TAN"):
        w = WCS(hdr)
        idx_lon = np.where(np.array(w.wcs.ctype, dtype="str") == lon_ctype)[0][0]
        idx_lat = np.where(np.array(w.wcs.ctype, dtype="str") == lat_ctype)[0][0]
        x, y = np.meshgrid(np.arange(w.pixel_shape[idx_lon]),
                           np.arange(w.pixel_shape[idx_lat]), )  # t dépend de x,
        # should reproject on a new coordinate grid first : suppose slits at the same time :
        coords = w.pixel_to_world(x, y)
        if isinstance(coords, SkyCoord):
            if lon_ctype == "HPLN-TAN":
                longitude = AlignCommonUtil.ang2pipi(coords.Tx)
                latitude = AlignCommonUtil.ang2pipi(coords.Ty)
            elif lon_ctype == "CRLN-CAR":
                longitude = coords.lon
                latitude = coords.lat
        elif isinstance(coords, list):
            longitude = AlignCommonUtil.ang2pipi(coords[0])
            latitude = AlignCommonUtil.ang2pipi(coords[1])
        else:
            raise ValueError("output from w.pixel_to_world() must be SkyCoord or list")
        if dsun:
            dsun_obs_large = hdr["DSUN_OBS"]
            return AlignCommonUtil.ang2pipi(longitude), \
                AlignCommonUtil.ang2pipi(latitude), dsun_obs_large
        else:
            return longitude, latitude

    @staticmethod
    def diff_rot(lat, wvl='default'):
        """ Return the angular velocity difference between differential and
        Carrington rotation.
        Parameters
        ==========
        lat : float
            The latitude, in radians
        wvl : str (default: 'default'
            The wavelength, or the band to return the rotation from.
        Returns
        =======
        corr : float
            The difference in angular velocities between the differential and
            Carrington rotations, in radians per second:
                Δω(θ) = ω_Car - ω_diff(θ)
                with ω_Car = 360° / (25.38 days)
                and  ω_diff(θ) = A + B sin² θ + C sin⁴ θ
        """
        p = {
            # ° day⁻¹; Hortin (2003):
            'EIT 171': (14.56, -2.65, 0.96),
            'EIT 195': (14.50, -2.14, 0.66),
            'EIT 284': (14.60, -0.71, -1.18),
            'EIT 304': (14.51, -3.12, 0.34),
        }
        p['default'] = p['EIT 195']
        A, B, C = p[wvl]
        A_car = 360 / 25.38  # ° day⁻¹
        corr = A - A_car + B * np.sin(lat) ** 2 + C * np.sin(lat) ** 4  # ° day⁻¹
        corr = np.deg2rad(corr / 86400)  # rad s⁻¹
        return corr

    @staticmethod
    def recenter_crpix_in_header(hdr):
        pass
        # w = WCS(hdr)
        # if "ZNAXIS1" in hdr:
        #     naxis1 = hdr["ZNAXIS1"]
        #     naxis2 = hdr["ZNAXIS2"]
        # else:
        #     naxis1 = hdr["NAXIS1"]
        #     naxis2 = hdr["NAXIS2"]
        # x_mid = (naxis1 - 1) / 2
        # y_mid = (naxis2 - 1) / 2
        # lon_mid, lat_mid = w.pixel_to_world(np.array([x_mid]), np.array([y_mid]))
        # lon_mid = lon_mid[0].to(hdr["CUNIT1"]).value
        # lat_mid = lat_mid[0].to(hdr["CUNIT2"]).value
        # hdr["CRVAL1"] = lon_mid
        # hdr["CRVAL2"] = lat_mid
        # hdr["CRPIX1"] = (naxis1 + 1) / 2
        # hdr["CRPIX2"] = (naxis2 + 1) / 2

    # @staticmethod
    # def write_corrected_fits(path_eui_l2_input: str, window_eui, path_eui_l2_output: str, corr: np.array,
    #                          lag_crval1=None, lag_crval2=None, lag_crota=None,
    #                          lag_cdelt1=None, lag_cdelt2=None,
    #                          ):
    #     raise DeprecationWarning
    #
    #     max_index = np.unravel_index(np.nanargmax(corr), corr.shape)
    #
    #     with fits.open(path_eui_l2_input) as hdul:
    #         # hdu = hdul[window_spice]
    #         # hdr_shifted = hdu.header
    #         # AlignSpiceUtil.recenter_crpix_in_header_L2(hdul[window_eui].header)
    #         change_pcij = False
    #         if lag_crval1 is not None:
    #             hdul[window_eui].header['CRVAL1'] = hdul[window_eui].header['CRVAL1'
    #                                                 ] + u.Quantity(lag_crval1[max_index[0]], "arcsec").to(
    #                 hdul[window_eui].header['CUNIT1']).value
    #         if lag_crval2 is not None:
    #             hdul[window_eui].header['CRVAL2'] = hdul[window_eui].header['CRVAL2'
    #                                                 ] + u.Quantity(lag_crval2[max_index[1]],
    #                                                                "arcsec").to(
    #                 hdul[window_eui].header['CUNIT2']).value
    #         key_rota = None
    #         crota = np.rad2deg(np.arccos(hdul[window_eui].header["PC1_1"]))
    #         if "CROTA" in hdul[window_eui].header:
    #             key_rota = "CROTA"
    #         elif "CROTA2" in hdul[window_eui].header:
    #             key_rota = "CROTA2"
    #
    #         if lag_crota is not None:
    #             crota += lag_crota[max_index[4]]
    #             if key_rota is not None:
    #                 hdul[window_eui].header[key_rota] = crota
    #             change_pcij = True
    #
    #         if lag_cdelt1 is not None:
    #             hdul[window_eui].header['CDELT1'] = hdul[window_eui].header['CDELT1'] + u.Quantity(
    #                 lag_cdelt1[max_index[2]],
    #                 "arcsec").to(
    #                 hdul[window_eui].header['CUNIT1']).value
    #             change_pcij = True
    #
    #         if lag_cdelt2 is not None:
    #             hdul[window_eui].header['CDELT2'] = hdul[window_eui].header['CDELT2'] + u.Quantity(
    #                 lag_cdelt2[max_index[3]],
    #                 "arcsec").to(
    #                 hdul[window_eui].header['CUNIT2']).value
    #             change_pcij = True
    #         if change_pcij:
    #             theta = np.deg2rad(crota)
    #             lam = hdul[window_eui].header["CDELT2"] / hdul[window_eui].header["CDELT1"]
    #             hdul[window_eui].header["PC1_1"] = np.cos(theta)
    #             hdul[window_eui].header["PC2_2"] = np.cos(theta)
    #             hdul[window_eui].header["PC1_2"] = -lam * np.sin(theta)
    #             hdul[window_eui].header["PC2_1"] = (1 / lam) * np.sin(theta)
    #
    #     hdul.writeto(path_eui_l2_output, overwrite=True)
    #     hdul.close()


class AlignSpiceUtil:

    @staticmethod
    def slit_pxl(header):

        """ Compute the first and last pixel of the slit from a FITS header """
        ybin = header['NBIN2']
        h_detector = 1024 / ybin
        if header['DETECTOR'] == 'SW':
            h_slit = 600 / ybin
        elif header['DETECTOR'] == 'LW':
            h_slit = 626 / ybin
        else:
            raise ValueError(f"unknown detector: {header['DETECTOR']}")
        slit_beg = (h_detector - h_slit) / 2
        slit_end = h_detector - slit_beg
        slit_beg = slit_beg - header['PXBEG2'] / ybin + 1
        slit_end = slit_end - header['PXBEG2'] / ybin + 1
        slit_beg = int(np.ceil(slit_beg))
        slit_end = int(np.floor(slit_end))
        return slit_beg, slit_end

    @staticmethod
    def vertical_edges_limits(header):
        iymin, iymax = AlignSpiceUtil.slit_pxl(header)
        iymin += int(20 / header['NBIN2'])
        iymax -= int(20 / header['NBIN2'])
        return iymin, iymax

    # @staticmethod
    # def create_intensity_map(path_to_l3, index_amplitude=0, index_width=2):
    #     hdul = fits.open(path_to_l3)
    #     hdu_results = hdul[0]
    #     data = hdu_results.data.copy()
    #     hdr = hdu_results.header.copy()
    #     hdul.close()
    #
    #     missing = hdr["ANA_MISS"]
    #     condition_missing = np.array((data[:, :, index_amplitude] == missing) | (data[:, :, index_width] == missing),
    #                                  dtype="bool")
    #
    #     intensity = data[:, :, index_amplitude] * data[:, :, index_width] * np.sqrt(2 * np.pi)
    #     intensity[condition_missing] = np.nan
    #
    #     return hdr, intensity

    @staticmethod
    def _data_treatment_spice(self, hdr, data):
        w_small = WCS(hdr)
        w2 = w_small.deepcopy()
        w2.wcs.pc[3, 0] = 0
        w2.wcs.pc[3, 1] = 0
        w_xyt = w2.dropaxis(2)
        w_xy_small = w_xyt.dropaxis(2)
        return w_xy_small

    @staticmethod
    def extract_spice_coordinates_l3(hdr, return_type='xy'):
        w_small = WCS(hdr)
        w2 = w_small.deepcopy()

        w2.wcs.pc[3, 0] = 0

        if return_type == 'xy':
            w2.wcs.pc[3, 1] = 0
            w_xyt = w2.dropaxis(0)
            w_xy = w_xyt.dropaxis(2)

            idx_lon = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
            idx_lat = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
            x_small, y_small = np.meshgrid(np.arange(w_xy.pixel_shape[idx_lon]),
                                           np.arange(w_xy.pixel_shape[idx_lat]), )
            longitude_small, latitude_small = w_xy.pixel_to_world(x_small, y_small)
            return longitude_small, latitude_small
        elif return_type == 'xyt':
            w_xyt = w2.dropaxis(0)

            idx_lon = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
            idx_lat = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
            idx_utc = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "UTC")[0][0]
            x_small, y_small, z_small = np.meshgrid(np.arange(w_xyt.pixel_shape[idx_lon]),
                                                    np.arange(w_xyt.pixel_shape[idx_lat]),
                                                    np.arange(w_xyt.pixel_shape[idx_utc]), )
            longitude_small, latitude_small, utc_small = w_xyt.pixel_to_world(x_small, y_small, z_small)
            return longitude_small, latitude_small, utc_small

    @staticmethod
    def extract_spice_coordinates_l2(hdr, return_type='xy'):
        w = WCS(hdr)
        w_xyt = w.dropaxis(2)

        if return_type == 'xy':
            w_xyt.wcs.pc[2, 0] = 0
            w_xy = w_xyt.dropaxis(2)
            idx_lon = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
            idx_lat = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
            x_small, y_small = np.meshgrid(np.arange(w_xy.pixel_shape[idx_lon]),
                                           np.arange(w_xy.pixel_shape[idx_lat]), )

            # in case imported sunpy, and helioprojective coordinate frame has been imported, need to consider that a
            # Skycoord object will be in output of pixel_to_world.
            coords = w_xy.pixel_to_world(x_small, y_small)
            if isinstance(coords, SkyCoord):
                longitude_small = AlignCommonUtil.ang2pipi(coords.Tx)
                latitude_small = AlignCommonUtil.ang2pipi(coords.Ty)

            elif isinstance(coords, list):
                longitude_small = coords[0]
                latitude_small = coords[1]
            else:
                raise ValueError("outputs from wcs.pixel_to_world() must be SkyCoord or list")

            return longitude_small, latitude_small
        elif return_type == 'xyt':
            idx_lon = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
            idx_lat = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
            idx_utc = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "UTC")[0][0]
            x_small, y_small, z_small = np.meshgrid(np.arange(w_xyt.pixel_shape[idx_lon]),
                                                    np.arange(w_xyt.pixel_shape[idx_lat]),
                                                    np.arange(w_xyt.pixel_shape[idx_utc]), )
            coords = w_xyt.pixel_to_world(x_small, y_small, z_small)

            if len(coords) == 2:
                longitude_small = AlignCommonUtil.ang2pipi(coords[0].Tx)
                latitude_small = AlignCommonUtil.ang2pipi(coords[0].Ty)
                utc_small = coords[1]

            elif len(coords) == 3:
                longitude_small = coords[0]
                latitude_small = coords[1]
                utc_small = coords[2]

            else:
                raise ValueError("outputs from wcs.pixel_to_world() must be SkyCoord or list")
            return longitude_small, latitude_small, utc_small

    @staticmethod
    def recenter_crpix_in_header_L2(hdr):
        pass
        # w = WCS(hdr)
        # w_xyt = w.dropaxis(2)
        #
        # if "ZNAXIS1" in hdr:
        #     naxis1 = hdr["ZNAXIS1"]
        #     naxis2 = hdr["ZNAXIS2"]
        #     naxis3 = hdr["ZNAXIS3"]
        # else:
        #     naxis1 = hdr["NAXIS1"]
        #     naxis2 = hdr["NAXIS2"]
        #     naxis3 = hdr["NAXIS3"]
        #
        # x_mid = (naxis1 - 1) / 2
        # y_mid = (naxis2 - 1) / 2
        # t_mid = (naxis3 - 1) / 2
        #
        # lon_mid, lat_mid, utc_mid = w_xyt.pixel_to_world(np.array([x_mid]), np.array([y_mid]), np.array(t_mid))
        # lon_mid = lon_mid[0].to(hdr["CUNIT1"]).value
        # lat_mid = lat_mid[0].to(hdr["CUNIT2"]).value
        #
        # hdr["CRVAL1"] = lon_mid
        # hdr["CRVAL2"] = lat_mid
        # hdr["CRPIX1"] = (naxis1 + 1) / 2
        # hdr["CRPIX2"] = (naxis2 + 1) / 2

        # must also shift the rotation

    @staticmethod
    def extract_l3_data(path_spice: str, line: dict, index_line: int, window=0):

        with fits.open(path_spice) as hdul_spice:
            hdu = hdul_spice[window]
            data = hdu.data
            data_l3 = {"amplitude": data[:, :, line["amplitude"][index_line]],
                       "width": data[:, :, line["width"][index_line]],
                       "chi2": data[:, :, line["chi2"][index_line]],
                       "background": data[:, :, line["background"][index_line]],
                       "lambda": data[:, :, line["lambda"][index_line]]}
            data_l3["chi2"] = np.where(data_l3["amplitude"] == hdu.header["ANA_MISS"], np.nan, data_l3["chi2"])

            for key in ["amplitude", "width", "background", "lambda"]:
                data_l3[key] = np.where(data_l3["chi2"] == 0, np.nan, data_l3[key])
                data_l3[key] = np.where(data_l3[key] == hdu.header["ANA_MISS"], np.nan,
                                        data_l3[key])

            data_l3["radiance"] = data_l3["amplitude"] * data_l3["width"] * np.sqrt(2 * np.pi) * 0.424660900

            return data_l3

    # @staticmethod
    # def write_corrected_fits(path_spice_l2_input: str, window_spice_list, path_spice_l2_output: str, corr: np.array,
    #                          lag_crval1=None, lag_crval2=None, lag_crota=None,
    #                          lag_cdelt1=None, lag_cdelt2=None,
    #                          ):
    #
    #     max_index = np.unravel_index(np.nanargmax(corr), corr.shape)
    #
    #     with fits.open(path_spice_l2_input) as hdul:
    #         for window_spice in window_spice_list:
    #             # hdu = hdul[window_spice]
    #             # hdr_shifted = hdu.header
    #             # AlignSpiceUtil.recenter_crpix_in_header_L2(hdul[window_spice].header)
    #             change_pcij = False
    #             if lag_crval1 is not None:
    #                 hdul[window_spice].header['CRVAL1'] = hdul[window_spice].header['CRVAL1'
    #                                                       ] + u.Quantity(lag_crval1[max_index[0]], "arcsec").to(
    #                     hdul[window_spice].header['CUNIT1']).value
    #             if lag_crval2 is not None:
    #                 hdul[window_spice].header['CRVAL2'] = hdul[window_spice].header['CRVAL2'
    #                                                       ] + u.Quantity(lag_crval2[max_index[1]],
    #                                                                      "arcsec").to(
    #                     hdul[window_spice].header['CUNIT2']).value
    #             key_rota = None
    #             crota = np.rad2deg(np.arccos(hdul[window_spice].header["PC1_1"]))
    #             if "CROTA" in hdul[window_spice].header:
    #                 key_rota = "CROTA"
    #             elif "CROTA2" in hdul[window_spice].header:
    #                 key_rota = "CROTA2"
    #
    #             if lag_crota is not None:
    #                 crota += lag_crota[max_index[4]]
    #                 if key_rota is not None:
    #                     hdul[window_spice].header[key_rota] = crota
    #                 change_pcij = True
    #
    #             if lag_cdelt1 is not None:
    #                 hdul[window_spice].header['CDELT1'] = hdul[window_spice].header['CDELT1'] + u.Quantity(
    #                     lag_cdelt1[max_index[2]],
    #                     "arcsec").to(
    #                     hdul[window_spice].header['CUNIT1']).value
    #                 change_pcij = True
    #
    #             if lag_cdelt2 is not None:
    #                 hdul[window_spice].header['CDELT2'] = hdul[window_spice].header['CDELT2'] + u.Quantity(
    #                     lag_cdelt2[max_index[3]],
    #                     "arcsec").to(
    #                     hdul[window_spice].header['CUNIT2']).value
    #                 change_pcij = True
    #             if change_pcij:
    #                 theta = np.deg2rad(crota)
    #                 lam = hdul[window_spice].header["CDELT2"] / hdul[window_spice].header["CDELT1"]
    #                 hdul[window_spice].header["PC1_1"] = np.cos(theta)
    #                 hdul[window_spice].header["PC2_2"] = np.cos(theta)
    #                 hdul[window_spice].header["PC1_2"] = -lam * np.sin(theta)
    #                 hdul[window_spice].header["PC2_1"] = (1 / lam) * np.sin(theta)
    #
    #         hdul.writeto(path_spice_l2_output, overwrite=True)
    #         hdul.close()
    #


class PlotFits:
    @staticmethod
    def get_range(data, stre='log', imax=99.5, imin=2):
        """
        :param data:
        :param stretch: 'sqrt', 'log', or 'linear' (default)
        :return: norm
        """
        isnan = np.isnan(data)
        data = data[~isnan]

        if data.size == 0:
            return None
        do = False
        if imax > 100:
            vmin, vmax = AsymmetricPercentileInterval(imin, 100).get_limits(data)
            vmax = vmax * imax / 100
        else:
            vmin, vmax = AsymmetricPercentileInterval(imin, imax).get_limits(data)

        #    print('Vmin:', vmin, 'Vmax', vmax)
        if stre is None:
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
        elif stre == 'sqrt':
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        elif stre == 'log':
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
        else:
            raise ValueError('Bad stre value: either None or sqrt')
        return norm

    @staticmethod
    def plot_fov_rectangle(data, slc=None, path_save=None, show=True, plot_colorbar=True, norm=None, angle=0):
        fig = plt.figure()
        ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        PlotFits.plot_fov(data=data, show=False, fig=fig, ax=ax, norm=norm)
        rect = patches.Rectangle((slc[1].start, slc[0].start),
                                 slc[1].stop - slc[1].start, slc[0].stop - slc[0].start, linewidth=1,
                                 edgecolor='r', facecolor='none', angle=angle)
        ax.add_patch(rect)
        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)

    @staticmethod
    def plot_fov(data, slc=None, path_save=None, show=True, plot_colorbar=True, fig=None, ax=None, norm=None):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        if slc is not None:
            im = ax.imshow(data[slc[0], slc[1]], origin="lower", interpolation="none", norm=norm)
        else:
            im = ax.imshow(data, origin="lower", interpolation="None", norm=norm)
        if plot_colorbar:
            fig.colorbar(im, label="DN/s")
        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)

    @staticmethod
    def simple_plot(hdr_main, data_main, path_save=None, show=True, ax=None, fig=None, norm=None,
                    show_xlabel=True, show_ylabel=True, plot_colorbar=True):

        longitude, latitude, dsun = AlignEUIUtil.extract_EUI_coordinates(hdr_main)
        longitude_grid, latitude_grid, dlon, dlat = PlotFits.build_regular_grid(longitude=longitude, latitude=latitude)

        dlon = dlon.to("arcsec").value
        dlat = dlat.to("arcsec").value
        w = WCS(hdr_main)
        x, y = w.world_to_pixel(longitude_grid, latitude_grid)
        image_on_regular_grid = AlignCommonUtil.interpol2d(data_main, x=x, y=y, fill=-32762, order=1)
        image_on_regular_grid[image_on_regular_grid == -32762] = np.nan
        # dlon = (longitude_grid[1, 1] - longitude_grid[0, 0]).to("arcsec").value
        # dlat = (latitude_grid[1, 1] - latitude_grid[0, 0]).to("arcsec").value

        return_im = False
        if fig is None:
            fig = plt.figure()
            return_im = True
        if ax is None:
            ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        im = ax.imshow(image_on_regular_grid, origin="lower", interpolation="none", norm=norm,
                       extent=[longitude_grid[0, 0].to(u.arcsec).value - 0.5 * dlon,
                               longitude_grid[-1, -1].to(u.arcsec).value + 0.5 * dlon,
                               latitude_grid[0, 0].to(u.arcsec).value - 0.5 * dlat,
                               latitude_grid[-1, -1].to(u.arcsec).value + 0.5 * dlat])
        # im = ax.imshow(data_main, origin="lower", interpolation="none", norm=norm,)

        if show_xlabel:
            ax.set_xlabel("Solar-X [arcsec]")
        if show_ylabel:
            ax.set_ylabel("Solar-Y [arcsec]")
        if plot_colorbar:
            fig.colorbar(im, label=hdr_main["BUNIT"])
        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)
        if return_im:
            return im

    @staticmethod
    def contour_plot(hdr_main, data_main, hdr_contour, data_contour, path_save=None, show=True, levels=None,
                     ax=None, fig=None, norm=None, show_xlabel=True, show_ylabel=True, plot_colorbar=True):
        longitude_main, latitude_main, dsun = AlignEUIUtil.extract_EUI_coordinates(hdr_contour)
        longitude_grid, latitude_grid = PlotFits._build_regular_grid(longitude=longitude_main,
                                                                     latitude=latitude_main)

        w_xy_main = WCS(hdr_main)
        x_small, y_small = w_xy_main.world_to_pixel(longitude_grid, latitude_grid)
        image_main_cut = AlignCommonUtil.interpol2d(np.array(data_main,
                                                             dtype=np.float64), x=x_small, y=y_small,
                                                    order=1, fill=-32768)
        image_main_cut[image_main_cut == -32768] = np.nan

        w_xy_contour = WCS(hdr_contour)
        x_contour, y_contour = w_xy_contour.world_to_pixel(longitude_grid, latitude_grid)
        image_contour_cut = AlignCommonUtil.interpol2d(np.array(data_contour, dtype=np.float64),
                                                       x=x_contour, y=y_contour,
                                                       order=1, fill=-32768)
        image_contour_cut[image_contour_cut == -32768] = np.nan
        dlon = (longitude_grid[1, 1] - longitude_grid[0, 0]).to("arcsec").value
        dlat = (latitude_grid[1, 1] - latitude_grid[0, 0]).to("arcsec").value

        return_im = True
        if fig is None:
            fig = plt.figure()
            return_im = False
        if ax is None:
            ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        im = ax.imshow(image_main_cut, origin="lower", interpolation="none", norm=norm,
                       extent=[longitude_grid[0, 0].to(u.arcsec).value - 0.5 * dlon,
                               longitude_grid[-1, -1].to(u.arcsec).value + 0.5 * dlon,
                               latitude_grid[0, 0].to(u.arcsec).value - 0.5 * dlat,
                               latitude_grid[-1, -1].to(u.arcsec).value + 0.5 * dlat])
        if levels is None:
            max_small = np.nanmax(image_contour_cut)
            levels = [0.5 * max_small]
        ax.contour(image_contour_cut, levels=levels, origin='lower', linewidths=0.5, colors='w',
                   extent=[longitude_grid[0, 0].to(u.arcsec).value - 0.5 * dlon,
                           longitude_grid[-1, -1].to(u.arcsec).value + 0.5 * dlon,
                           latitude_grid[0, 0].to(u.arcsec).value - 0.5 * dlat,
                           latitude_grid[-1, -1].to(u.arcsec).value + 0.5 * dlat])
        if show_xlabel:
            ax.set_xlabel("Solar-X [arcsec]")
        if show_ylabel:
            ax.set_ylabel("Solar-Y [arcsec]")
        if plot_colorbar:
            fig.colorbar(im, label=hdr_main["BUNIT"])
        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)
        if return_im:
            return im

    @staticmethod
    def compare_plot(hdr_main, data_main, hdr_contour_1, data_contour_1,
                     hdr_contour_2, data_contour_2, norm, path_save=None, show=True, levels=None, ):

        if (norm.vmin is None) or (norm.vmax is None):
            raise ValueError("Must explicit vmin and vmax in norm, so that the cbar is the same for both figures.")

        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax_cbar = fig.add_subplot(gs[2])

        im = PlotFits.contour_plot(hdr_main=hdr_main, data_main=data_main, plot_colorbar=False,
                                   hdr_contour=hdr_contour_1, data_contour=data_contour_1,
                                   path_save=None, show=False, levels=levels, fig=fig, ax=ax1, norm=norm)

        im = PlotFits.contour_plot(hdr_main=hdr_main, data_main=data_main, show_ylabel=False, plot_colorbar=False,
                                   hdr_contour=hdr_contour_2, data_contour=data_contour_2,
                                   path_save=None, show=False, levels=levels, fig=fig, ax=ax2, norm=norm)

        fig.colorbar(im, cax=ax_cbar, label=hdr_main["BUNIT"])

        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)

    @staticmethod
    def build_regular_grid(longitude, latitude, lonlims=None, latlims=None):
        # breakpoint()
        x = np.abs((longitude[0, 1] - longitude[0, 0]).to("deg").value)
        y = np.abs((latitude[0, 1] - latitude[0, 0]).to("deg").value)
        dlon = np.sqrt(x ** 2 + y ** 2)

        x = np.abs((longitude[1, 0] - longitude[0, 0]).to("deg").value)
        y = np.abs((latitude[1, 0] - latitude[0, 0]).to("deg").value)
        dlat = np.sqrt(x ** 2 + y ** 2)
        if longitude.unit == "deg":
            longitude1D = np.arange(np.min(longitude.to(u.deg).value),
                                    np.max(longitude.to(u.deg).value), dlon)
            latitude1D = np.arange(np.min(latitude.to(u.deg).value),
                                np.max(latitude.to(u.deg).value), dlat)
        elif longitude.unit == "arcsec":
            longitude1D = np.arange(np.min(AlignCommonUtil.ang2pipi(longitude).to(u.deg).value),
                                    np.max(AlignCommonUtil.ang2pipi(longitude).to(u.deg).value), dlon)
            latitude1D = np.arange(np.min(AlignCommonUtil.ang2pipi(latitude).to(u.deg).value),
                                np.max(AlignCommonUtil.ang2pipi(latitude).to(u.deg).value), dlat)
        if (lonlims is not None) or (latlims is not None):
            longitude1D = longitude1D[(longitude1D >lonlims[0].to("deg").value) &
                                      (longitude1D <lonlims[1].to("deg").value)]
            latitude1D = latitude1D[(latitude1D > latlims[0].to("deg").value) &
                                    (latitude1D < latlims[1].to("deg").value)]
        longitude_grid, latitude_grid = np.meshgrid(longitude1D, latitude1D)

        longitude_grid = longitude_grid * u.deg
        latitude_grid = latitude_grid * u.deg
        dlon = dlon * u.deg
        dlat = dlat * u.deg
        return longitude_grid, latitude_grid, dlon, dlat

    @staticmethod
    def extend_regular_grid(longitude_grid, latitude_grid, delta_longitude, delta_latitude):
        x = np.abs((longitude_grid[0, 1] - longitude_grid[0, 0]).to("deg").value)
        y = np.abs((latitude_grid[0, 1] - latitude_grid[0, 0]).to("deg").value)
        dlon = np.sqrt(x ** 2 + y ** 2)

        x = np.abs((longitude_grid[1, 0] - longitude_grid[0, 0]).to("deg").value)
        y = np.abs((latitude_grid[1, 0] - latitude_grid[0, 0]).to("deg").value)
        dlat = np.sqrt(x ** 2 + y ** 2)
        if longitude_grid.unit == 'deg':
            delta_longitude_deg = delta_longitude.to("deg").value
            delta_latitude_deg = delta_latitude.to("deg").value

            longitude1D = np.arange(
                np.min(longitude_grid.to(u.deg).value - 0.5 * delta_longitude_deg),
                np.max(longitude_grid.to(u.deg).value) + 0.5 * delta_longitude_deg,
                dlon)
            latitude1D = np.arange(
                np.min(latitude_grid.to(u.deg).value - 0.5 * delta_latitude_deg),
                np.max(latitude_grid.to(u.deg).value) + 0.5 * delta_latitude_deg,
                dlat)

        elif longitude_grid.unit == "arcsec":
            delta_longitude_deg = AlignCommonUtil.ang2pipi(delta_longitude).to("deg").value
            delta_latitude_deg = AlignCommonUtil.ang2pipi(delta_latitude).to("deg").value

            longitude1D = np.arange(
                np.min(AlignCommonUtil.ang2pipi(longitude_grid).to(u.deg).value - 0.5 * delta_longitude_deg),
                np.max(AlignCommonUtil.ang2pipi(longitude_grid).to(u.deg).value) + 0.5 * delta_longitude_deg,
                dlon)
            latitude1D = np.arange(
                np.min(AlignCommonUtil.ang2pipi(latitude_grid).to(u.deg).value - 0.5 * delta_latitude_deg),
                np.max(AlignCommonUtil.ang2pipi(latitude_grid).to(u.deg).value) + 0.5 * delta_latitude_deg,
                dlat)

        longitude_grid_ext, latitude_grid_ext = np.meshgrid(longitude1D, latitude1D)
        longitude_grid_ext = longitude_grid_ext * u.deg
        latitude_grid_ext = latitude_grid_ext * u.deg

        return longitude_grid_ext, latitude_grid_ext


class MpUtils:
    @staticmethod
    def gen_shmm(create=False, name=None, ndarray=None, size=0, shape=None, dtype=None):

        assert (type(ndarray) != type(None) or size != 0) or type(name) != type(None)
        assert type(ndarray) != type(None) or type(shape) != type(None)
        if (dtype is None) & create:
            dtype = ndarray.dtype
        elif (dtype is None) & (create == False):
            raise ValueError("dtype must be set")

        size = size if type(ndarray) == type(None) else ndarray.nbytes
        shmm = SharedMemory(create=create, size=size, name=name)
        shmm_data = np.ndarray(shape=shape if type(ndarray) == type(None) else ndarray.shape, buffer=shmm.buf,
                               dtype=dtype)
        if create and type(ndarray) != type(None):
            shmm_data[:] = ndarray[:]
        elif create:
            shmm_data[:] = np.nan
        return shmm, shmm_data
