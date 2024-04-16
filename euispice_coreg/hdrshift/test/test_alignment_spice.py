import os.path

from astropy.io import fits
import numpy as np
import pytest
from ..alignement_spice import AlignmentSpice
from pathlib import Path


def test_alignement_helioprojective_spice():
    folder = 'C:/Users/adolliou/PycharmProjects/Alignement/test'

    path_fsi = os.path.join(Path().absolute(), "SPICE_alignment", "hdrshift", "test",
                            "fitsfiles", 'solo_L2_eui-fsi304-image_20220317T000800208_V02.fits')
    path_spice = os.path.join(Path().absolute(), "SPICE_alignment", "hdrshift", "test",
                              "fitsfiles", 'solo_L2_spice-n-ras_20220317T000032_V02_100663831-000.fits')

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
