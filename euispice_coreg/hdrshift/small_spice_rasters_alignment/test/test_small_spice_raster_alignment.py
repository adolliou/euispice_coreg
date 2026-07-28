import os
from ....utils.Selector.selector_eui import SelectorEui
from ....utils.Selector.selector_spice import SelectorSpice
from ....hdrshift.small_spice_rasters_alignment.small_spice_raster_alignment import small_spice_raster_alignment
from astropy.time import Time
from pathlib import Path
from astropy.io import fits
import numpy as np


def test_small_spice_raster_alignment():


    data_input_folder   = os.path.join(Path(__file__).parents[0], "data_input")

    path_cr             = os.path.join(data_input_folder, "release-6.0\\level2\\2022\\03\\17\\solo_L2_spice-n-ras_20220317T000032_V24_100663831-000-DR6.fits")
    window_cr           = 3

    selector_eui        = SelectorEui(base_url=data_input_folder)
    list_imagers_cr, time_all_cr     = selector_eui.get_url_from_time_interval(
        time1           = Time("2022-03-17T00:00:00"),
        time2           = Time("2022-03-17T00:30:00"),
        file_name_str   = "eui-fsi304",
        )

    window_imagers_cr   = -1
    selector_eui        = SelectorEui(base_url=data_input_folder)
    list_imagers_sr, time_all_sr     = selector_eui.get_url_from_time_interval(
        time1 = Time("2022-03-17T00:00:00"),
        time2 = Time("2022-03-17T01:00:00"), 
        file_name_str="eui-hrieuvopn",
    )
    window_imagers_sr   = -1

    selector_spice      = SelectorSpice(base_url=data_input_folder, release=6.0)
    list_small_rasters, time_all_spice  = selector_spice.get_url_from_time_interval(
        time1 = Time("2022-03-17T00:18:00"),
        time2 = Time("2022-03-17T01:00:00"), 
        file_name_str="spice-n-ras",
    )
    window_small_raster = 3

    windows_to_correct  = [
        0, 1, 2, 3, 4, 5
    ]

    folder_save         = Path(__file__).parents[0]

    # name                = list_imagers_sr[0]
    # ff                                              = fits.open(name)

    param_alignment_cr = {
        "lag_crval1": np.arange(-40, 40, 4), # lag crvals in the headers, in arcsec
        "lag_crval2": np.arange(-40, 40, 4),  # in arcsec
        "lag_crota": np.array([0]), # in degrees
        "lag_cdelt1": np.array([0]), # in arcsec
        "lag_cdelt2": np.array([0]), # in arcsec
        }

    small_spice_raster_alignment(
        small_raster_list_path                  = list_small_rasters,
        small_raster_window                     = window_small_raster, 
        context_raster_path                     = path_cr, 
        context_raster_window                   = window_cr, 
        list_reference_imagers_context_raster   = list_imagers_cr, 
        window_reference_imagers_context_raster = window_imagers_cr, 
        list_reference_imagers_small_raster     = list_imagers_sr, 
        window_reference_imagers_small_raster   = window_imagers_sr,
        windows_to_correct                      = windows_to_correct,
        folder_save                             = folder_save,  
        param_alignment_cr                      = param_alignment_cr, 
        cpu_count                               = 6, 

    )







    # imagers_cr_names    = [
    #     "solo_L2_eui-fsi304-image_20220317T000300228_V02.fits", 
    #     "solo_L2_eui-fsi304-image_20220317T000400208_V02.fits",
    #     "solo_L2_eui-fsi304-image_20220317T000500208_V02.fits", 
    #     "solo_L2_eui-fsi304-image_20220317T000600208_V02.fits", 
    #     "solo_L2_eui-fsi304-image_20220317T000700208_V02.fits", 
    #     "solo_L2_eui-fsi304-image_20220317T000800208_V02.fits", 
    #     "solo_L2_eui-fsi304-image_20220317T000900208_V02.fits", 
    #     "solo_L2_eui-fsi304-image_20220317T001000208_V02.fits",
    #     "solo_L2_eui-fsi304-image_20220317T001100208_V02.fits", 

    # ]
