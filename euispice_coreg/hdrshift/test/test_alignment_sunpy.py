import os.path

import numpy as np
from ..alignment import Alignment
from pathlib import Path

import sunpy.map

def test_alignement_helioprojective_shift():
    path_fsi = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-fsi174"
                "-image_20220317T095045281_V01.fits")
    path_hri = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-hrieuv174"
                "-image_20220317T095045277_V01.fits")

    lag_crval1 = np.arange(15, 26, 1)
    lag_crval2 = np.arange(5, 11, 1)

    lag_cdelt1 = [0]
    lag_cdelt2 = [0]

    lag_crota = [0.75]

    min_value = 0
    max_value = 1310


    A = Alignment(large_fov_known_pointing=path_fsi, small_fov_to_correct=path_hri, lag_crval1=lag_crval1,
                  lag_crval2=lag_crval2, lag_cdelt1=lag_cdelt1, lag_cdelt2=lag_cdelt2, lag_crota=lag_crota,
                  parallelism=True, display_progress_bar=True, counts_cpu_max=20, small_fov_value_min=min_value,
                  small_fov_value_max=max_value,)

    corr = A.align_using_helioprojective(method='correlation', return_type="corr")
    max_index = np.unravel_index(corr.argmax(), corr.shape)

    assert lag_crval1[max_index[0]] == 24
    assert lag_crval2[max_index[1]] == 6


def test_alignement_carrington():
    path_fsi = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-fsi174"
                "-image_20220317T095045281_V01.fits")
    path_hri = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-hrieuv174"
                "-image_20220317T095045277_V01.fits")


    lonlims = (230, 260)
    latlims = (-65, -35)  # longitude min and max (degrees)

    parallelism = True
    shape = (2400, 2400)

    lag_crval1 = np.arange(20, 30, 2)
    lag_crval2 = np.arange(5, 15, 2)
    reference_date = "2022-03-17T09:50:45"
    lag_solar_r = [1.004]

    lag_cdelt1 = [0]
    lag_cdelt2 = [0]

    # lag_crota = [0.75]

    lag_crota = [0.75]

    min_value = 0
    max_value = 1310

    A = Alignment(large_fov_known_pointing=path_fsi, small_fov_to_correct=path_hri, lag_crval1=lag_crval1,
                  lag_crval2=lag_crval2, lag_cdelt1=lag_cdelt1, lag_cdelt2=lag_cdelt2, lag_crota=lag_crota,
                  parallelism=parallelism, display_progress_bar=True,
                  small_fov_value_min=min_value,
                  small_fov_value_max=max_value, lag_solar_r=lag_solar_r, )
    corr = A.align_using_carrington(method='correlation', shape=shape, lonlims=lonlims, latlims=latlims,
                                    reference_date=reference_date, method_carrington_reprojection="fa", return_type="corr")
    max_index = np.unravel_index(corr.argmax(), corr.shape)

    assert lag_crval1[max_index[0]] == 22
    assert lag_crval2[max_index[1]] == 5


if __name__ == "__main__":
    pass