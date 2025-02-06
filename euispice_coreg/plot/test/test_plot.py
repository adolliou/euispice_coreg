import pytest
import numpy as np
from ..plot import PlotFits, PlotFunctions

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
    return np.array([[[[[[0.95776455]]]],
        [[[[0.95860984]]]],
        [[[[0.95631261]]]],
        [[[[0.95112539]]]],
        [[[[0.94335218]]]]],
       [[[[[0.96222236]]]],
        [[[[0.96295417]]]],
        [[[[0.96058389]]]],
        [[[[0.95510041]]]],
        [[[[0.94699545]]]]],
       [[[[[0.96400085]]]],
        [[[[0.96460102]]]],
        [[[[0.96218054]]]],
        [[[[0.95653924]]]],
        [[[[0.94835215]]]]],
       [[[[[0.96318198]]]],
        [[[[0.96368646]]]],
        [[[[0.96109933]]]],
        [[[[0.95569653]]]],
        [[[[0.94760919]]]]],
        [[[[[0.95984876]]]],
        [[[[0.96015055]]]],
        [[[[0.95749983]]]],
        [[[[0.9522271 ]]]],
        [[[[0.94451592]]]]]])

@pytest.fixture
def param_alignment():
   
   return {
    "lag_crval1" : np.arange(20, 30, 2),
    "lag_crval2" : np.arange(5, 15, 2),
    "lag_cdelt1" : None,
    "lag_cdelt2" : [0],
    "lag_crota" : [0.75],
    }


def test_plot_correlation(corr, ):
    params_alignment = {
    "lag_crval1" : np.arange(20, 30, 2),
    "lag_crval2" : np.arange(5, 15, 2),
    "lag_cdelt1" : None,
    "lag_cdelt2" : [0],
    "lag_crota" : [0.75],
    }
    reference_date = "2022-03-17T09:50:45"
    min_value = 0
    max_value = 1310
    lag_solar_r = [1.004]
    save_fig = "./euispice_coreg/plot/test/correlation2.jpeg"
    ref_fig = "./euispice_coreg/plot/test/correlation.jpeg"
    max_index = np.unravel_index(corr.argmax(), corr.shape)
    PlotFunctions.plot_correlation(corr=corr, path_save_figure=save_fig, **params_alignment)

    base_image = Image.open(save_fig)
    ref_image = Image.open(ref_fig)
    assert(image_pixel_differences(base_image, ref_image))


def test_plot_co_alignment(corr, param_alignment):
    path_fsi = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-fsi174"
                "-image_20220317T095045281_V01.fits")
    path_hri = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-hrieuv174"
                "-image_20220317T095045277_V01.fits")
    save_fig = "./euispice_coreg/plot/test/co_alignment2.jpeg"
    ref_fig = "./euispice_coreg/plot/test/co_alignment.jpeg"
    PlotFunctions.plot_co_alignment(reference_image_path=path_fsi, reference_image_window=-1, 
                                    image_to_align_path=path_hri, image_to_align_window=-1, 
                                    corr=corr, path_save_figure=save_fig, **param_alignment)
    
    base_image = Image.open(save_fig)
    ref_image = Image.open(ref_fig)
    assert(image_pixel_differences(base_image, ref_image))


def test_plot_co_alignment_sunpy(corr, param_alignment):
    path_fsi = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-fsi174"
                "-image_20220317T095045281_V01.fits")
    path_hri = ("https://www.sidc.be/EUI/data/releases/202204_release_5.0/L2/2022/03/17/solo_L2_eui-hrieuv174"
                "-image_20220317T095045277_V01.fits")
    save_fig = "./euispice_coreg/plot/test/co_alignment_supy2.pdf"
    ref_fig = "./euispice_coreg/plot/test/co_alignment_sunpy.pdf"
    PlotFunctions.plot_co_alignment(reference_image_path=path_fsi, reference_image_window=-1, 
                                    image_to_align_path=path_hri, image_to_align_window=-1, 
                                    corr=corr, path_save_figure=ref_fig, **param_alignment, type_plot="sunpy")
    
    # base_image = Image.open(save_fig)
    # ref_image = Image.open(ref_fig)
    # assert(image_pixel_differences(base_image, ref_image))
