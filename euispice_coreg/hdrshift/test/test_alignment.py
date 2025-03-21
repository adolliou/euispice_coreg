import os.path

import numpy as np
from ..alignment import Alignment



def test_alignement_helioprojective_shift():

    path_fsi = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-fsi174"
                "-image_20220317T095045281_V01.fits")
    path_hri = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-hrieuv174"
                "-image_20220317T095045277_V01.fits")

    lag_crval1 = np.arange(15, 26, 1)
    lag_crval2 = np.arange(5, 11, 1)

    # lag_crval1 = np.arange(15, 26, 1)
    # lag_crval2 = np.arange(10, 50, 1)

    lag_cdelt1 = None  #
    lag_cdelt2 = [0]

    lag_crota = [0.75]


    A = Alignment(large_fov_known_pointing=path_fsi, small_fov_to_correct=path_hri, lag_crval1=lag_crval1,
                  lag_crval2=lag_crval2, lag_cdelt1=lag_cdelt1, lag_cdelt2=lag_cdelt2, lag_crota=lag_crota,
                  parallelism=True, display_progress_bar=True, counts_cpu_max=None,  )

    corr = A.align_using_helioprojective(method='correlation', return_type="corr")
    max_index = np.unravel_index(corr.argmax(), corr.shape)

    assert lag_crval1[max_index[0]] == 24
    assert lag_crval2[max_index[1]] == 6

def test_alignement_helioprojective_noparallelism_shift():


    path_fsi = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-fsi174"
                "-image_20220317T095045281_V01.fits")
    path_hri = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-hrieuv174"
                "-image_20220317T095045277_V01.fits")

    lag_crval1 = np.arange(24, 25, 1)
    lag_crval2 = np.arange(6, 7, 1)

    lag_cdelt1 = [0]
    lag_cdelt2 = [0]

    lag_crota = [0.75]
    min_value = 0
    max_value = 1310

    A = Alignment(large_fov_known_pointing=path_fsi, small_fov_to_correct=path_hri, lag_crval1=lag_crval1,
                  lag_crval2=lag_crval2, lag_cdelt1=lag_cdelt1, lag_cdelt2=lag_cdelt2, lag_crota=lag_crota,
                  parallelism=False, display_progress_bar=True, counts_cpu_max=20, small_fov_value_min=min_value,
                  small_fov_value_max=max_value, )

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

    lag_cdelt1 = None
    lag_cdelt2 = [0]

    # lag_crota = [0.75]

    lag_crota = [0.75]

    min_value = 0
    max_value = 1310

    A = Alignment(large_fov_known_pointing=path_fsi, small_fov_to_correct=path_hri, lag_crval1=lag_crval1,
                  lag_crval2=lag_crval2, lag_cdelt1=lag_cdelt1, lag_cdelt2=lag_cdelt2, lag_crota=lag_crota,
                  parallelism=parallelism, display_progress_bar=True,
                    lag_solar_r=lag_solar_r, )
    corr = A.align_using_carrington(method='correlation', shape=shape, lonlims=lonlims, latlims=latlims,
                                    reference_date=reference_date,  method_carrington_reprojection="fa", return_type="corr")
    max_index = np.unravel_index(corr.argmax(), corr.shape)

    assert lag_crval1[max_index[0]] == 22
    assert lag_crval2[max_index[1]] == 5


# def test_alignement_minimal_header():
#     path_eis = "./euispice_coreg/euispice_coreg/hdrshift/test/fitsfiles/eis_20221024_024912_fe_12_195_119_calib_intensity.fits"
#     # path_hri = os.path.join(Path().absolute(), "euispice_coreg", "hdrshift", "test",
#                             # "fitsfiles", "solo_L2_eui-hrieuv174-image_20220317T095045277_V01.fits")

#     path_aia = "./fitsfiles/AIA_193_eis_20221024_024912_fe_12_195_119_calib_intensity.fits"

#     lag_crval1 = np.arange(-10, 10, 1)
#     lag_crval2 = np.arange(-10, 10, 1)

#     lag_cdelt1 = None
#     lag_cdelt2 = None

#     lag_crota = None

#     min_value = 0
#     max_value = 1310

#     A = Alignment(large_fov_known_pointing=path_eis, small_fov_to_correct=path_aia, lag_crval1=lag_crval1,
#                   lag_crval2=lag_crval2, lag_cdelt1=lag_cdelt1, lag_cdelt2=lag_cdelt2, lag_crota=lag_crota,
#                   parallelism=True, display_progress_bar=True, counts_cpu_max=20, small_fov_value_min=min_value,
#                   small_fov_value_max=max_value,force_crota_0=True )

#     corr = A.align_using_helioprojective(method='correlation')
#     max_index = np.unravel_index(corr.argmax(), corr.shape)

#     assert lag_crval1[max_index[0]] == 0
#     assert lag_crval2[max_index[1]] == 6




def test_alignement_carrington_sunpy():
    # path_fsi = os.path.join(Path().absolute(), "euispice_coreg", "hdrshift", "test",
    #                         "fitsfiles", "solo_L2_eui-fsi174-image_20220317T095045281_V01.fits")
    # path_hri = os.path.join(Path().absolute(), "euispice_coreg", "hdrshift", "test",
    #                         "fitsfiles", "solo_L2_eui-hrieuv174-image_20220317T095045277_V01.fits")
    path_fsi = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-fsi174"
                "-image_20220317T095045281_V01.fits")
    path_hri = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-hrieuv174"
                "-image_20220317T095045277_V01.fits")

    parallelism = True

    lag_crval1 = np.arange(20, 30, 2)
    lag_crval2 = np.arange(5, 15, 2)
    reference_date = "2022-03-17T09:50:45"
    lag_solar_r = [1.004]

    lag_cdelt1 = None
    lag_cdelt2 = [0]

    # lag_crota = [0.75]

    lag_crota = [0.75]

    min_value = 0
    max_value = 1310

    A = Alignment(large_fov_known_pointing=path_fsi, small_fov_to_correct=path_hri, lag_crval1=lag_crval1,
                  lag_crval2=lag_crval2, lag_cdelt1=lag_cdelt1, lag_cdelt2=lag_cdelt2, lag_crota=lag_crota,
                  parallelism=parallelism, display_progress_bar=True,
                     lag_solar_r=lag_solar_r, )
    corr = A.align_using_carrington(method='correlation', 
                                    reference_date=reference_date,   method_carrington_reprojection="sunpy", return_type="corr")
    max_index = np.unravel_index(corr.argmax(), corr.shape)
    print(f"{lag_crval1[max_index[0]]=}")
    print(f"{lag_crval2[max_index[1]]=}")
    assert lag_crval1[max_index[0]] == 24
    assert lag_crval2[max_index[1]] == 7


