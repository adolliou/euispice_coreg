import os.path
import numpy as np
from ..alignment import Alignment
from pathlib import Path
import pytest
from ..AlignmentResults import AlignmentResults
# This module is used to load images
from PIL import Image
# This module contains a number of arithmetical image operations
from PIL import ImageChops



def image_pixel_differences(base_image, compare_image):
  """
  Calculates the bounding box of the non-zero regions in the image.
  :param base_image: target image to find
  :param compare_image:  set of images containing the target image
  :return: The bounding box is returned as a 4-tuple defining the
           left, upper, right, and lower pixel coordinate. If the image
           is completely empty, this method returns None.
  """
  # Returns the absolute value of the pixel-by-pixel
  # difference between two images.

  diff = ImageChops.difference(base_image, compare_image)
  if diff.getbbox():
    return False
  else:
    return True
  

@pytest.fixture
def corr():
    return np.array(
        [
            [
                [[[[0.94431532]]]],
                [[[[0.94491356]]]],
                [[[[0.94490277]]]],
                [[[[0.94429364]]]],
                [[[[0.94309195]]]],
                [[[[0.94131598]]]],
            ],
            [
                [[[[0.9487374]]]],
                [[[[0.94936037]]]],
                [[[[0.94934872]]]],
                [[[[0.94870775]]]],
                [[[[0.94744547]]]],
                [[[[0.94558114]]]],
            ],
            [
                [[[[0.95292]]]],
                [[[[0.95356913]]]],
                [[[[0.95355487]]]],
                [[[[0.95288052]]]],
                [[[[0.95155507]]]],
                [[[[0.94959962]]]],
            ],
            [
                [[[[0.95678181]]]],
                [[[[0.95745709]]]],
                [[[[0.95743886]]]],
                [[[[0.95673169]]]],
                [[[[0.95534362]]]],
                [[[[0.95329829]]]],
            ],
            [
                [[[[0.96025253]]]],
                [[[[0.96095169]]]],
                [[[[0.96093119]]]],
                [[[[0.96019453]]]],
                [[[[0.95874962]]]],
                [[[[0.95662224]]]],
            ],
            [
                [[[[0.963255]]]],
                [[[[0.96397323]]]],
                [[[[0.96395091]]]],
                [[[[0.96318901]]]],
                [[[[0.96169552]]]],
                [[[[0.95949712]]]],
            ],
            [
                [[[[0.96570708]]]],
                [[[[0.9664386]]]],
                [[[[0.96641366]]]],
                [[[[0.96563084]]]],
                [[[[0.9640988]]]],
                [[[[0.96184383]]]],
            ],
            [
                [[[[0.9675529]]]],
                [[[[0.96828706]]]],
                [[[[0.96825363]]]],
                [[[[0.96745105]]]],
                [[[[0.96588888]]]],
                [[[[0.96359088]]]],
            ],
            [
                [[[[0.9687609]]]],
                [[[[0.9694829]]]],
                [[[[0.96943329]]]],
                [[[[0.96861061]]]],
                [[[[0.96702333]]]],
                [[[[0.96469464]]]],
            ],
            [
                [[[[0.96932341]]]],
                [[[[0.9700199]]]],
                [[[[0.9699457]]]],
                [[[[0.96910128]]]],
                [[[[0.96749419]]]],
                [[[[0.96514772]]]],
            ],
            [
                [[[[0.96927416]]]],
                [[[[0.96994215]]]],
                [[[[0.96984541]]]],
                [[[[0.96898563]]]],
                [[[[0.96737077]]]],
                [[[[0.96502305]]]],
            ],
        ]
    )


@pytest.fixture
def params_alignment():

    return {
        "lag_crval1": np.arange(15, 26, 1),
        "lag_crval2": np.arange(5, 11, 1),
        "lag_cdelt1": None,
        "lag_cdelt2": [0],
        "lag_crota": [0.75],
    }


@pytest.fixture
def path_hri():

    return (
        "https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-hrieuv174"
        "-image_20220317T095045277_V01.fits"
    )


@pytest.fixture
def path_fsi():
    return (
        "https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-fsi174"
        "-image_20220317T095045281_V01.fits"
    )



class TestAlignmentResults:

    def test_compute_shift(self, corr, params_alignment, path_hri, path_fsi):

        R = AlignmentResults(
            corr=corr,
            **params_alignment,
            unit_lag="arcsec",
            image_to_align_path=path_hri,
            image_to_align_window=-1,
            reference_image_path=path_fsi,
            reference_image_window=-1
        )
        assert np.abs(R.shift_pixels[0] - 9.33682107) < 1.0e-2
        assert np.abs(R.shift_pixels[1] - 1.42187891) < 1.0e-2
        save_fits = "./euispice_coreg/hdrshift/test/test.fits"
        windows = [-1]
        R.write_corrected_fits(windows, path_to_l3_output=save_fits)

    def test_plot_correlation(self, corr, params_alignment, path_hri, path_fsi):
        
        R = AlignmentResults(
            corr=corr,
            **params_alignment,
            unit_lag="arcsec",
            image_to_align_path=path_hri,
            image_to_align_window=-1,
            reference_image_path=path_fsi,
            reference_image_window=-1
        )

        save_plot = "./euispice_coreg/hdrshift/test/plot_correlation1.jpeg"
        save_plot_ref = "./euispice_coreg/hdrshift/test/plot_correlation2.jpeg"

        R.plot_correlation(path_save_figure=save_plot, show=False)
        base_image = Image.open(save_plot)
        ref_image = Image.open(save_plot_ref)
        assert(image_pixel_differences(base_image, ref_image))

    def test_plot_co_alignment(self, corr, params_alignment, path_hri, path_fsi):
        
        R = AlignmentResults(
            corr=corr,
            **params_alignment,
            unit_lag="arcsec",
            image_to_align_path=path_hri,
            image_to_align_window=-1,
            reference_image_path=path_fsi,
            reference_image_window=-1
        )

        save_plot = "./euispice_coreg/hdrshift/test/plot_co_alignment1_results.jpeg"
        save_plot_ref = "./euispice_coreg/hdrshift/test/plot_co_alignment2_results.jpeg"
        R.plot_co_alignment(path_save_figure=save_plot, show=False)
        base_image = Image.open(save_plot)
        ref_image = Image.open(save_plot_ref)
        assert(image_pixel_differences(base_image, ref_image))

