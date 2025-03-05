import os
from abc import ABC
from astropy.wcs import WCS
import numpy as np
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from ..utils import Util
import warnings
from astropy.wcs.utils import WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS
from astropy.coordinates import SkyCoord


class MapBuilder(ABC):

    def __init__(self):
        pass

    def process(self, path_output: str):
        pass


class ComposedMapBuilder(MapBuilder):

    def __init__(self, path_to_spectro: str, list_imager_paths: list, threshold_time: u.Quantity,
                 window_imager=-1, window_spectro=0,):
        """

        :param path_to_spectro: path to the reference spectrometer L2 fits
        :param list_imager_paths: list of L2 imager paths used to create the synthetic raster
        :param threshold_time: if can not find imager fits file with date-avg time lower than threshold_time
        for a given raster step, raise an error.
        :param window_imager: chosen window for the imager HDULIST
        :param window_spectro: chosen window for the spectro HDULIST
        # :param divide_exposure: divide the imager data by the exposure time
        """
        warnings.warn("Sorry, there is an critical error with composedmaps right now. Currently under review.\n /For now, please do not use the function, and coalign SPICE with an image close to DATE-AVG")
        super().__init__()
        self.path_to_spectro = path_to_spectro
        self.list_imager_paths = np.array(list_imager_paths, dtype="str")
        self.window_imager = window_imager
        self.window_spectro = window_spectro
        self.threshold_time = threshold_time
        self.path_composed_map = None
        self._extract_imager_metadata()
        self.path_output = None

        use_sunpy = False
        for mapping in [WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS]:
            if mapping[-1][0].__module__ == 'sunpy.coordinates.wcs_utils':
                use_sunpy = True
                # import sunpy.map
        self.use_sunpy = use_sunpy
        self.skycoord_spice = None
        # self.divide_exposure = divide_exposure

    def process(self, folder_path_output=None, basename_output=None, print_filename=True, level=2,
                keep_original_imager_pixel_size=False, return_synras_name=False):
        """

        :param folder_path_output:  path to the output folder.
        :param basename_output: basename of the output FITS file. mist end with .fits
        :param print_filename: print names of the files used from the folder
        :param level: number of dimensions for the input file. Keep to 2 except for L3 SPICE FITS files
        :param keep_original_imager_pixel_size: keep the original pixel size of the L2 imagers lists
        :param return_synras_name: (bool) if True, return the path to the output synthetic raster.
        """
        self.path_output = folder_path_output
        with fits.open(self.path_to_spectro) as hdul_spice:
            hdu_spice = hdul_spice[self.window_spectro]
            hdr_spice = hdu_spice.header.copy()
            output_synras_name = self._create_map_from_hdu(hdr_spice, basename_output, folder_path_output,
                                                           print_filename=print_filename,
                                                           level=level,
                                                    keep_original_imager_pixel_size=keep_original_imager_pixel_size)
            hdul_spice.close()

        if return_synras_name:
            return output_synras_name

    def process_from_header(self, hdr_spice, path_output=None, basename_output=None, print_filename=False, level=2,
                            keep_original_imager_pixel_size=False):
        self.path_output = path_output
        self._create_map_from_hdu(hdr_spice, basename_output, path_output, print_filename=print_filename,
                                  level=level, keep_original_imager_pixel_size=keep_original_imager_pixel_size)

    def _create_map_from_hdu(self, hdr_spice, basename_output=None, path_output=None, print_filename=True, level=2,
                             keep_original_imager_pixel_size=False):

        hdr_im, latitude_spice, longitude_spice, naxis1, \
            naxis2, naxis_long, utc_spice, w_xy = self._prepare_spectro_data(hdr_spice,
                                                                             keep_original_imager_pixel_size, level)
        # EUIUtil
        
        for ii in range(naxis_long):
            utc_slit, deltat = self._return_mean_time(utc_spice[:, ii, 0])
            index_closest, dt = self._find_closest_imager_time(utc_slit)

            self.dates_selected[ii] = self.dates[index_closest]
            dt = u.Quantity(dt, "s")
            if dt > self.threshold_time:
                print(f"{self.dates=}")
                print(f"{utc_slit=}")

                raise ValueError(f"{dt=}: Could not find imager sufficiently close in time")
            with fits.open(self.list_imager_paths[index_closest]) as hdul_imager:
                if print_filename:
                    print(f"\nUse imager {os.path.basename(self.list_imager_paths[index_closest])}")
                hdu_imager = hdul_imager[self.window_imager]
                hdr_imager = hdu_imager.header
                data_imager = hdu_imager.data

                # if self.divide_exposure:
                #     if ('DN/s' in hdr_imager["BUNIT"]) or
                #     pass
                # else:
                #     pass

                w_im = WCS(hdr_imager)
                if self.use_sunpy:

                    coords_tmp = SkyCoord(longitude_spice[:, ii, 0], latitude_spice[:, ii, 0],
                                          frame=self.skycoord_spice.frame)
                    x_fsi, y_fsi = w_im.world_to_pixel(coords_tmp)

                else:
                    x_fsi, y_fsi = w_im.world_to_pixel(longitude_spice[:, ii, 0], latitude_spice[:, ii, 0])
                data_imager_on_slit = Util.AlignCommonUtil.interpol2d(data_imager, x=x_fsi, y=y_fsi, order=2,
                                                                      fill=np.nan)
                self.data_composed[:, ii] = data_imager_on_slit
                hdul_imager.close()
        # keys = ["DATE-AVG", "DSUN_OBS", "RSUN_REF", "SOLAR_B0", "WAVELNTH", "DETECTOR", "BUNIT", "XPOSURE"]
        keys = ["CRPIX1", "CRPIX2", "CRPIX3", "CRPIX4",
                "CRVAL1", "CRVAL2", "CRVAL3", "CRVAL4",
                "CDELT1", "CDELT2", "CDELT3", "CDELT4",
                "CUNIT1", "CUNIT2", "CUNIT3", "CUNIT4",
                "CROTA2", "CROTA"]
        for ii, jj in zip([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                          [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]):
            keys.append(f"PC{ii}_{jj}")
        hdr_fsi_mid = self.headers[len(self.list_imager_paths) // 2]
        self.hdr_composed = hdr_fsi_mid
        for k in keys:
            if k in self.hdr_spice_:
                self.hdr_composed[k] = self.hdr_spice_[k]
            else:
                warnings.warn(f"{k} no in original header. It is not added to the synthetic raster header")
        self.hdr_composed["DATE-AVG"] = hdr_spice["DATE-AVG"]
        utc_composed, deltat = self._return_mean_time(self.dates_selected)
        wave = self.hdr_composed["WAVELNTH"]
        if "DETECTOR" in self.hdr_composed:
            detector = self.hdr_composed["DETECTOR"]
        elif "INSTRUME" in self.hdr_composed:
            detector = self.hdr_composed["INSTRUME"]
        else:
            raise ValueError("No info on reference instrument")

        if keep_original_imager_pixel_size:
            # self.hdr_composed["NAXIS1"] = self.data_composed.shape[1]
            # self.hdr_composed["NAXIS2"] = self.data_composed.shape[0]
            # EUIUtil.recenter_crpix_in_header(hdr_im)
            # EUIUtil.recenter_crpix_in_header(self.hdr_composed)
            # naxis1 = self.data_composed.shape[1]
            # naxis2 = self.data_composed.shape[0]
            #
            x_mid = (naxis1 - 1) / 2
            y_mid = (naxis2 - 1) / 2
            lon_mid, lat_mid = w_xy.pixel_to_world(np.array([x_mid]), np.array([y_mid]))
            self.hdr_composed["CDELT1"] = u.Quantity(hdr_im["CDELT1"],
                                                     hdr_im["CUNIT1"]).to(self.hdr_composed["CUNIT1"]).value
            self.hdr_composed["CDELT2"] = u.Quantity(hdr_im["CDELT2"],
                                                     hdr_im["CUNIT2"]).to(self.hdr_composed["CUNIT2"]).value
            lam = self.hdr_composed["CDELT2"] / self.hdr_composed["CDELT1"]
            rho = np.arccos(self.hdr_composed["PC1_1"])
            s = - np.sign(self.hdr_composed["PC1_2"])
            rho = rho * s

            self.hdr_composed["PC1_2"] = - lam * np.sin(rho)
            self.hdr_composed["PC2_1"] = (1 / lam) * np.sin(rho)
            self.hdr_composed["CRPIX1"] = (self.data_composed.shape[1] + 1) / 2
            self.hdr_composed["CRPIX2"] = (self.data_composed.shape[0] + 1) / 2
            #
            self.hdr_composed["CRVAL1"] = lon_mid[0].to(self.hdr_composed["CUNIT1"]).value
            self.hdr_composed["CRVAL2"] = lat_mid[0].to(self.hdr_composed["CUNIT2"]).value

        if basename_output is None:
            date = utc_composed.fits[:19]
            date = date.replace(":", "_")
            basename_new = f"solo_L3_{detector}{wave}-image-composed-{date}.fits"
        else:
            basename_new = basename_output
        if path_output is not None:
            hdu_composed = fits.PrimaryHDU(self.data_composed, header=self.hdr_composed)
            hdul_composed = fits.HDUList([hdu_composed])
            hdul_composed.writeto(os.path.join(self.path_output, basename_new), overwrite=True)
            self.path_composed_map = os.path.join(self.path_output, basename_new)
        else:
            if level == 2:
                self.hdr_composed["NAXIS1"] = self.data_composed.shape[1]
                self.hdr_composed["NAXIS2"] = self.data_composed.shape[0]
            else:
                raise NotImplementedError
        return os.path.join(self.path_output, basename_new)

    def _prepare_spectro_data(self, hdr_spice, keep_original_imager_pixel_size, level):
        pass

    def get_path_to_composed_map(self):
        return self.path_composed_map

    def _extract_imager_metadata(self):
        self.dates = np.empty(len(self.list_imager_paths), dtype="object")
        self.headers = np.empty(len(self.list_imager_paths), dtype="object")
        for ii, path in enumerate(self.list_imager_paths):
            with fits.open(path) as hdul:
                self.dates[ii] = Time(hdul[self.window_imager].header["DATE-AVG"])
                self.headers[ii] = hdul[self.window_imager].header.copy()
                hdul.close()

    def _find_closest_imager_time(self, utc_ref):
        Delta_t = np.array([np.abs((utc_ref - n).to("s").value) for n in self.dates], dtype=np.float64)
        return Delta_t.argmin(), Delta_t.min()

    @staticmethod
    def _return_mean_time(utc_list):
        utc_ref = utc_list[0]
        Delta_t = np.array([(utc_ref - n).to("s").value for n in utc_list], dtype=np.float64)
        mean_dt = u.Quantity(Delta_t.mean(), "s")
        utc_mean = utc_ref - mean_dt
        return utc_mean, Delta_t


class SPICEComposedMapBuilder(ComposedMapBuilder):

    def __init__(self, path_to_spectro: str, list_imager_paths: list, threshold_time: u.Quantity,
                 window_imager=-1, window_spectro=0, ):

        super().__init__(path_to_spectro=path_to_spectro, list_imager_paths=list_imager_paths,
                         threshold_time=threshold_time,
                         window_imager=window_imager, window_spectro=window_spectro, )

    def _prepare_spectro_data(self, hdr_spice, keep_original_imager_pixel_size, level):
        if level == 2:
            w_spice = WCS(hdr_spice)
            self.dates_selected = np.empty(hdr_spice["NAXIS1"], dtype="object")
            self.data_composed = np.empty((hdr_spice["NAXIS2"], hdr_spice["NAXIS1"]), dtype=np.float64)
            naxis1 = hdr_spice["NAXIS1"]
            naxis2 = hdr_spice["NAXIS2"]
            w_xyt = w_spice.dropaxis(2)
            with fits.open(self.list_imager_paths[0]) as hdul_im:
                hdu = hdul_im[self.window_imager]
                hdr_im = hdu.header.copy()
            if keep_original_imager_pixel_size:


                x, y, t = np.meshgrid(np.arange(0, hdr_spice["NAXIS1"], hdr_im["CDELT1"] / hdr_spice["CDELT1"]),
                        np.arange(0, hdr_spice["NAXIS2"], hdr_im["CDELT2"] / hdr_spice["CDELT2"]),
                        np.arange(hdr_spice["NAXIS4"]))
                self.data_composed = np.empty((len(np.arange(0, hdr_spice["NAXIS2"],
                                                             hdr_im["CDELT2"] / hdr_spice["CDELT2"])),
                                               len(np.arange(0, hdr_spice["NAXIS1"],
                                                             hdr_im["CDELT1"] / hdr_spice["CDELT1"]))),
                                              dtype=np.float64)
                self.dates_selected = np.empty(len(np.arange(0, hdr_spice["NAXIS1"],
                                                             hdr_im["CDELT1"] / hdr_spice["CDELT1"])), dtype="object")



                naxis_long = len(np.arange(0, hdr_spice["NAXIS1"], hdr_im["CDELT1"] / hdr_spice["CDELT1"]))
            else:
                x, y, t = np.meshgrid(np.arange(hdr_spice["NAXIS1"]),
                                      np.arange(hdr_spice["NAXIS2"]),
                                      np.arange(hdr_spice["NAXIS4"]))
                naxis_long = hdr_spice["NAXIS1"]

            if self.use_sunpy:
                coords_spice = w_xyt.pixel_to_world(x, y, t)
                self.skycoord_spice = coords_spice[0]

                longitude_spice = coords_spice[0].Tx
                latitude_spice = coords_spice[0].Ty
                utc_spice = coords_spice[1]
            else:
                longitude_spice, latitude_spice, utc_spice = w_xyt.pixel_to_world(x, y, t)
            w_xy = w_xyt.deepcopy()
            w_xy.wcs.pc[2, 0] = 0
            w_xy = w_xy.dropaxis(2)
        elif level == 3:
            w_spice = WCS(hdr_spice)
            self.dates_selected = np.empty(hdr_spice["NAXIS2"], dtype="object")
            self.data_composed = np.empty((hdr_spice["NAXIS3"], hdr_spice["NAXIS2"]), dtype=np.float64)
            naxis1 = hdr_spice["NAXIS2"]
            naxis2 = hdr_spice["NAXIS3"]

            w2 = w_spice.deepcopy()
            w2.wcs.pc[3, 0] = 0
            w_xyt = w2.dropaxis(0)

            idx_lon = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
            idx_lat = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
            idx_utc = np.where(np.array(w_xyt.wcs.ctype, dtype="str") == "UTC")[0][0]
            with fits.open(self.list_imager_paths[0]) as hdul_im:
                hdu = hdul_im[self.window_imager]
                hdr_im = hdu.header.copy()
            if keep_original_imager_pixel_size:

                x, y, z = np.meshgrid(np.arange(0, w_xyt.pixel_shape[idx_lon], hdr_im["CDELT1"] / hdr_spice["CDELT2"]),
                                      np.arange(0, w_xyt.pixel_shape[idx_lat], hdr_im["CDELT2"] / hdr_spice["CDELT3"]),
                                      np.arange(w_xyt.pixel_shape[idx_utc]), )
                self.data_composed = np.empty((len(np.arange(0, hdr_spice["NAXIS3"],
                                                             hdr_im["CDELT2"] / hdr_spice["CDELT3"])),
                                               len(np.arange(0, hdr_spice["NAXIS2"],
                                                             hdr_im["CDELT1"] / hdr_spice["CDELT2"]))),
                                              dtype=np.float64)
                self.dates_selected = np.empty(len(np.arange(0, hdr_spice["NAXIS2"],
                                                             hdr_im["CDELT1"] / hdr_spice["CDELT2"])), dtype="object")

                # x2, y2, z2 = np.meshgrid(np.arange(w_xyt.pixel_shape[idx_lon]), np.arange(w_xyt.pixel_shape[idx_lat]),
                #                          np.arange(w_xyt.pixel_shape[idx_utc]), )

                naxis_long = len(np.arange(0, hdr_spice["NAXIS2"], hdr_im["CDELT1"] / hdr_spice["CDELT2"]))

            else:

                x, y, z = np.meshgrid(np.arange(w_xyt.pixel_shape[idx_lon]), np.arange(w_xyt.pixel_shape[idx_lat]),
                                      np.arange(w_xyt.pixel_shape[idx_utc]), )
                naxis_long = hdr_spice["NAXIS2"]

            if self.use_sunpy:
                coords_spice = w_xyt.pixel_to_world(x, y, z)
                self.skycoord_spice = coords_spice[0]

                longitude_spice = coords_spice[0].Tx
                latitude_spice = coords_spice[0].Ty
                utc_spice = coords_spice[1]
            else:
                longitude_spice, latitude_spice, utc_spice = w_xyt.pixel_to_world(x, y, z)

            w_xyt.wcs.pc[2, 0] = 0
            w_xy = w_xyt.dropaxis(2)
        self.hdr_spice_ = w_xy.to_header()
        return hdr_im, latitude_spice, longitude_spice, naxis1, naxis2, naxis_long, utc_spice, w_xy
