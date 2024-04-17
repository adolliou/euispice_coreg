import os.path
import numpy as np
from ..alignment_spice import AlignmentSpice
from pathlib import Path
import sunpy.map


def test_alignement_helioprojective_spice():
    folder = 'C:/Users/adolliou/PycharmProjects/Alignement/test'


    path_fsi = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-fsi304"
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

    A = AlignmentSpice(large_fov_known_pointing=path_fsi, small_fov_to_correct=path_spice,
                       lag_crval1=lag_crval1, lag_crval2=lag_crval2, lag_crota=lag_crota, use_tqdm=True,
                       lag_cdelta1=lag_cdelt1, lag_cdelta2=lag_cdelt2, parallelism=parallelism,
                       large_fov_window=-1, small_fov_window=small_fov_window,
                       path_save_figure=folder, )

    corr = A.align_using_helioprojective(method='correlation')
    max_index = np.unravel_index(corr.argmax(), corr.shape)

    assert lag_crval1[max_index[0]] == -23
    assert lag_crval2[max_index[1]] == 36
