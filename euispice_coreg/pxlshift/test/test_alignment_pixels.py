import os.path

import numpy as np
from ..alignment_pixels import AlignmentPixels
from pathlib import Path


# def test_alignement_pixel_shift():
#     path_fsi = os.path.join(Path().absolute(), "SPICE_alignment", "hdrshift", "test",
#                             "fitsfiles", "solo_L2_eui-fsi174-image_20220317T095045281_V01.fits")
#     path_hri = os.path.join(Path().absolute(), "SPICE_alignment", "hdrshift", "test",
#                             "fitsfiles", "solo_L2_eui-hrieuv174-image_20220317T095045277_V01.fits")
#
#     lag_dx = np.arange(-235, -220, 5)
#     lag_dy = np.arange(55, 75, 5)
#     lag_drot = np.array([0.75])
#     unit = "degree"
#
#     A = AlignmentPixels(large_fov_known_pointing=path_fsi, small_fov_to_correct=path_hri, window_large=-1,
#                         window_small=-1)
#     corr = A.find_best_parameters(lag_dx=lag_dx, lag_dy=lag_dy, lag_drot=lag_drot, unit_rot=unit,
#                                   shift_solar_rotation_dx_large=True)
#
#     max_index = np.unravel_index(corr.argmax(), corr.shape)
#
#     assert lag_dx[max_index[0]] == -230
#     assert lag_dy[max_index[1]] == 65
#
