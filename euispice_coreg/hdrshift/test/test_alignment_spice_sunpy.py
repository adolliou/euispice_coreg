import os.path

import numpy as np
from ..alignment_spice import AlignmentSpice
from pathlib import Path
import astropy.units as u
import sunpy.map


def test_alignement_helioprojective_spice():
    folder = 'C:/Users/adolliou/PycharmProjects/Alignement/test'
    # path_spice = os.path.join(Path().absolute(), "euispice_coreg", "hdrshift", "test",
    #                         "fitsfiles_old", "solo_L2_spice-n-ras_20220317T000032_V02_100663831-000.fits")

    path_fsi = ("https://www.sidc.be/EUI/data/releases/202301_release_6.0/L2/2022/03/17/solo_L2_eui-fsi304"
                "-image_20220317T000800208_V02.fits")
    path_spice = ("https://spice.osups.universite-paris-saclay.fr/spice-data/release-3.0/level2/2022/03/17"
                  "/solo_L2_spice-n-ras_20220317T000032_V02_100663831-000.fits")
    small_fov_window = 3
    lag_crval1 = np.arange(-30, -15, 1)
    lag_crval2 = np.arange(30, 51, 1)
    lag_crota = np.array([0])
    lag_cdelt1 = np.array([0])
    lag_cdelt2 = np.array([0])
    parallelism = True

    wave_interval = [973 * u.angstrom, 979 * u.angstrom]
    # sub_fov_window = [400 * u.arcsec, 600 * u.arcsec, -300 * u.arcsec, 0 * u.arcsec]

    A = AlignmentSpice(large_fov_known_pointing=path_fsi, small_fov_to_correct=path_spice,
                       lag_crval1=lag_crval1, lag_crval2=lag_crval2, lag_crota=lag_crota, display_progress_bar=False,
                       lag_cdelt1=lag_cdelt1, lag_cdelt2=lag_cdelt2, parallelism=parallelism,
                       large_fov_window=-1, small_fov_window=small_fov_window,
                       path_save_figure=None, wavelength_interval_to_sum=wave_interval,
                       )

    results = A.align_using_helioprojective(method='correlation', )

    assert np.abs(results.shift_arcsec[0] + 22.736789342637366) < 1.0E-3
    assert np.abs(results.shift_arcsec[1] - 36.198098608759494) < 1.0E-3
    windows_spice = ["Mg IX 706 - Peak", # The windows where the pointing will be corrected. It is adviced to correct the shift in all of the spectral windows. 
            "Ne VIII 770 - Peak",
            "S V 786 / O IV 787 - Peak",
            "Ly-gamma-CIII group (Merged)",
            "LyB- FeX group (Merged)",
            "O VI 1032 - Peak"]
    path_save_fits = "./euispice_coreg/hdrshift/test/test_SPICE.fits"
    path_save_folder = "./euispice_coreg/hdrshift/test"
    results.write_corrected_fits(windows_spice, path_to_l3_output=path_save_fits)
    results.plot_correlation(path_save_figure=os.path.join(path_save_folder, "correlation_results.png"))
    results.plot_co_alignment(path_save_figure=os.path.join(path_save_folder, "co_alignment_results.png"))