import os
from pathlib import Path
import astropy.units as u
from ..map_builder import SPICEComposedMapBuilder
from astropy.io import fits


def test_map_builder_spice():
    list_path_fsi_2 = [
        os.path.join(Path().absolute(), "SPICE_alignment", "synras", "test",
                     "fitsfiles", 'solo_L2_eui-fsi304-image_20220317T000300228_V02.fits'),
        os.path.join(Path().absolute(), "SPICE_alignment", "hdrshift", "test",
                     "fitsfiles", 'solo_L2_eui-fsi304-image_20220317T000800208_V02.fits'),
        os.path.join(Path().absolute(), "SPICE_alignment", "synras", "test",
                     "fitsfiles", 'solo_L2_eui-fsi304-image_20220317T001600209_V02.fits'),
    ]
    path_spice = os.path.join(Path().absolute(), "SPICE_alignment", "hdrshift", "test",
                              "fitsfiles", 'solo_L2_spice-n-ras_20220317T000032_V02_100663831-000.fits')
    window_spice = 3
    window_imager = -1  # same for imagers in imager_list
    threshold_time = u.Quantity(1000, "s")  # maximum threshold time you want
    output_L3_fits_folder = os.path.join(Path().absolute(), "SPICE_alignment", "synras", "test",
                                         "fitsfiles")

    C = SPICEComposedMapBuilder(path_to_spectro=path_spice, list_imager_paths=list_path_fsi_2,
                                window_imager=window_imager, window_spectro=window_spice,
                                threshold_time=threshold_time)
    C.process(path_output=output_L3_fits_folder, basename_output="test_map_builder.fits")

    path_map_build = os.path.join(Path().absolute(), "SPICE_alignment", "synras", "test",
                                  "fitsfiles", "test_map_builder.fits")

    path_map_build_ref = os.path.join(Path().absolute(), "SPICE_alignment", "synras", "test",
                                      "fitsfiles", "test_map_builder_ref.fits")

    with fits.open(path_map_build) as hdul:
        with fits.open(path_map_build_ref) as hdul_ref:
            f = fits.FITSDiff(hdul, hdul_ref)
            assert f.identical()
