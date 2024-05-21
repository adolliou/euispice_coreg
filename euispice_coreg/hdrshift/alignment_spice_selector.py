from .alignment_spice import AlignmentSpice
import numpy as np
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import copy
from ..synras.map_builder import SPICEComposedMapBuilder
from ..selector.selector_eui import SelectorEui


class AlignmentSpiceSelector(AlignmentSpice):
    def __init__(self, path_to_spice_fits: str, lag_crval1: np.array, lag_crval2: np.array,
                 window_spice="Ly-gamma-CIII group (Merged)",
                 lag_cdelta1=None, lag_cdelta2=None, lag_crota=None, small_fov_value_min=None,
                 parallelism=False, counts_cpu_max=40,
                 small_fov_window=-1, use_tqdm=False, lag_solar_r=None, small_fov_value_max=None,
                 path_save_figure=None, threshold_time=1000 * u.s):
        """

        :param path_to_spice_fits:
        :param lag_crval1:
        :param lag_crval2:
        :param window_spice:
        :param lag_cdelta1:
        :param lag_cdelta2:
        :param lag_crota:
        :param small_fov_value_min:
        :param parallelism:
        :param counts_cpu_max:
        :param small_fov_window:
        :param use_tqdm:
        :param lag_solar_r:
        :param small_fov_value_max:
        :param path_save_figure:
        :param threshold_time:
        """
        with fits.open(path_to_spice_fits) as hdulist:
            hdu = hdulist[window_spice]
            hdr = hdu.header
            date_start = Time(hdr["DATE-BEG"])
            date_end = Time(hdr["DATE-END"])
            s = SelectorEui(release=6.0, level=2)
            l_url, l_time = s.get_url_from_time_interval(time1=date_start, time2=date_end,
                                                         file_name_str="eui-fsi304-image")
        self.list_url_fsi304 = l_url
        self.list_time_fsi304 = l_time
        self.threshold_time = threshold_time
        self.header_spice_unflattened = None

        super().__init__(large_fov_known_pointing="selector", small_fov_to_correct=path_to_spice_fits,
                         lag_crval1=lag_crval1, lag_crval2=lag_crval2, lag_cdelta1=lag_cdelta1, lag_cdelta2=lag_cdelta2,
                         lag_crota=lag_crota, use_tqdm=use_tqdm,
                         lag_solar_r=lag_solar_r, small_fov_value_min=small_fov_value_min, parallelism=parallelism,
                         small_fov_value_max=small_fov_value_max, counts_cpu_max=counts_cpu_max,
                         large_fov_window=-1, small_fov_window=small_fov_window,
                         path_save_figure=path_save_figure, )

    def _extract_imager_data_header(self, ):
        C = SPICEComposedMapBuilder(path_to_spectro=self.small_fov_to_correct,
                                    list_imager_paths=self.list_url_fsi304,
                                    threshold_time=self.threshold_time,
                                    window_imager=self.large_fov_window,
                                    window_spectro=self.small_fov_window)
        C.process_from_header(hdr_spice=self.header_spice_unflattened)

        self.data_large = copy.deepcopy(C.data_composed)
        self.hdr_large = copy.deepcopy(C.hdr_composed)
        with fits.open(self.large_fov_known_pointing) as hdul_large:
            self.data_large = np.array(hdul_large[self.large_fov_window].data.copy(), dtype=np.float64)
            self.hdr_large = hdul_large[self.large_fov_window].header.copy()
            # super()._recenter_crpix_in_header(self.hdr_large)
            hdul_large.close()

    def _prepare_spice_from_l2(self, hdul_small):
        self.header_spice_unflattened = hdul_small[self.small_fov_window].header.copy()
        super()._prepare_spice_from_l2(hdul_small=hdul_small)