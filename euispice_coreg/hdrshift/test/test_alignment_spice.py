import os.path

import numpy as np
from ..alignment_spice import AlignmentSpice
from pathlib import Path
import astropy.units as u


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
                       lag_crval1=lag_crval1, lag_crval2=lag_crval2, lag_crota=lag_crota, display_progress_bar=True,
                       lag_cdelta1=lag_cdelt1, lag_cdelta2=lag_cdelt2, parallelism=parallelism,
                       large_fov_window=-1, small_fov_window=small_fov_window,
                       path_save_figure=None, wavelength_interval_to_sum=wave_interval,
                       )

    corr = A.align_using_helioprojective(method='correlation')
    max_index = np.unravel_index(corr.argmax(), corr.shape)

    assert lag_crval1[max_index[0]] == -23
    assert lag_crval2[max_index[1]] == 36
