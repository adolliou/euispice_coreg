import os
import numpy as np
from tqdm import tqdm
import warnings

warnings.simplefilter("ignore")
from euispice_coreg.hdrshift.alignment import Alignment
from matplotlib import pyplot as plt
import shutil
import astropy.units as u
from astropy.io import fits
from astropy.time import Time

def jitter_correction_imagers(
        list_files_input: list, path_files_output: str, 
        lonlims = None, latlims = None, shape = None,     
        lag_crval1: np.array = np.array(-5, 5, 0.1),
        lag_crval2: np.array = np.array(-5, 5, 0.1),
        lag_cdelt1: np.array = np.array(0, 1, 1),
        lag_cdelt2: np.array = np.array(0, 1, 1),
        lag_crota: np.array = np.array(0, 1, 1),    
        sublist_length: int=10, overlap: int=1, 
        window_files_input: int = -1, 
        method_carrington_reprojection: str = "fa",
        unit_lag: str = "arcsec",
        path_figures: str = None, plot_all_figures: bool = False,
        parallelism: bool=True, cpu_count: int = None,
        small_fov_value_max: float = None, small_fov_value_min: float = None,
        alignement_method = "carrington"
                             ):
    """Correct the jitter of a list of FITS files, by dividing them into overlapping sublists 
    and cross-correlating them to the first image of their sublist (see Chitta et al., A&A, 2022, 
    DOI: 10.1051/0004-6361/202244170)
    By default, the cross-correlation is performed in a carrington frame, not to remove the Earth rotation. 
    This is necessary if one want to reproject these corrected images into a carrington frame.
    One has to define the Carrington frame where the reprojection is performed. 

    Args:
        list_files_input (list): List of paths to the FITS files where the jitter must be corrected.
        path_files_output (str): Path to the folder where 
        lonlims (tuple): limits in carrington longitude (degrees, degrees) for the carrington grid 
        where the alignment is performed.
        latlims (tuple): Limits in carrington latitudes (degrees, degrees)
        shape (tuple): shape of the carrington grids (pixels, pixels)
        lag_crval1 (np.array, optional): Array to define the lag on the CRVAL1 keywords in the header.
        See the Alignment class for more details. Defaults to np.array(-5, 5, 0.1).
        lag_crval2 (np.array, optional): Same for CRVAL2. Defaults to np.array(-5, 5, 0.1).
        lag_cdelt1 (np.array, optional): Same for CDELT1. Defaults to np.array(0, 1, 1).
        lag_cdelt2 (np.array, optional): Same for CDELT2. Defaults to np.array(0, 1, 1).
        lag_crota (np.array, optional): Same for CROTA. Defaults to np.array(0, 1, 1).
        sublist_length (int, optional): Number of images in each sublists, where each image is
        co-aligned with the first one of each sublist. Defaults to 10.
        overlap (int, optional): Number of Overlapping images between each sublits (> 1). Defaults to 1.
        window_files_input (int, optional): window of the HDULIST to use for the input FITS files.
        Defaults to -1.
        method_carrington_reprojection (str, optional): either "fa" or "sunpy". Defaults to "fa".
        unit_lag (str, optional): Units of the input images. Defaults to "arcsec".
        path_figures (str, optional): path where figures related to the cross-correlation are saved. Defaults to None.
        plot_all_figures (bool, optional): If False, only plot the correlation values. 
        If True, also plot comparison images for each co-alignment. Defaults to False.
        parallelism (bool, optional): If True, use parallelism to fasten the co-alignment procedure. Defaults to True.
        cpu_count (int, optional): Maximum CPU count to use for the co-alignement. Defaults to None.
        small_fov_value_max (float, optional): Maximum pixel value to use for the co-alignment. Defaults to None.
        small_fov_value_min (float, optional): Minimum pixel value to use for the co-alignment. Defaults to None.
        alignement_method (str, optional): either "carrington" or "initial_carrington" (if the input images are already in carrington coordinates).
        Defaults to "carrington".

    """    
    if overlap == 0:
        raise ValueError("number of overlapping images between sublists can not be equal to 0.")
    dates_files_input = []
    for ii, path in enumerate(list_files_input):
         with fits.open(path) as hdul:
            hdu = hdul[window_files_input]
            dates_files_input.append(Time(hdu.header["DATE-AVG"]))

    parameter_alignment = {
     "lag_crval1": lag_crval1, 
     "lag_crval2": lag_crval2, 
     "lag_cdelt1": lag_cdelt1, 
     "lag_cdelt2": lag_cdelt2, 
     "lag_crota": lag_crota, 
    }

    kwargs_carrington = {
        "lonlims": lonlims, 
        "latlims": latlims, 
        "shape": shape,
    }

    idx_list = np.arange(list_files_input)

    list_after_ref = idx_list[idx_list[0]:]
    sublists_after = [list_after_ref[n: n + sublist_length + overlap] for n
                      in range(0, len(list_after_ref), sublist_length)]
    liste_before_ref = idx_list[idx_list[0]::-1]
    sublists_before = [liste_before_ref[n: n + sublist_length + overlap] for n
                       in range(0, len(liste_before_ref), sublist_length)]

    if len(sublists_after) > 0:
        for ii, list_ in enumerate(tqdm(sublists_after)):
            index_ref = list_[0]
            basename = os.path.basename(list_files_input[index_ref])
            basename_new = basename

            path_reference = os.path.join(path_files_output, basename_new)

            if ii == 0:
                shutil.copyfile(list_files_input[index_ref], path_reference)
            

            


            for index_to_align in list_[1:]:
                print(f'{index_to_align=}')
                date_to_align = dates_files_input[index_to_align].fits[11:19]
                date_to_align = date_to_align.replace(":", "_")
                reference_date = dates_files_input[index_ref]
                results = _align_hrieuv_with_hrieuv(path_output_figures=path_figures,
                                                   large_fov_fits_path=path_reference,
                                                   large_fov_window=window_files_input,
                                                   small_fov_path=list_files_input[index_to_align],
                                                   window_to_align=window_files_input,
                                                   date_to_align=date_to_align,
                                                   parameter_alignment=parameter_alignment,
                                                   cpu_count=cpu_count, do_plot_figure=plot_all_figures,
                                                   method_carrington_reprojection=method_carrington_reprojection,
                                                   reference_date=reference_date, parallelism=parallelism,
                                                   alignement_method=alignement_method,
                                                   small_fov_value_max=small_fov_value_max, small_fov_value_min=small_fov_value_min,
                                                   unit_lag=unit_lag,
                                                   **kwargs_carrington)

                basename = os.path.basename(list_files_input[index_to_align])
                basename_new = basename
                results.write_corrected_fits(window_list_to_apply_shift=[window_files_input],
                                             path_to_l3_output=os.path.join(path_files_output,basename_new), )
    if len(sublists_before) > 0:

        for ii, list_ in enumerate(tqdm(sublists_before)):
            index_ref = list_[0]
            basename = os.path.basename(list_files_input[index_ref])

            basename_new = basename
            path_reference = os.path.join(path_files_output,basename_new)

            if (not os.path.isfile(path_reference)) and (ii == 0):
                shutil.copyfile(list_files_input[index_ref], path_reference)
            
  

            for index_to_align in list_[1:]:
                date_to_align = date_to_align[index_to_align].fits[11:19]
                date_to_align = date_to_align.replace(":", "_")
                reference_date = date_to_align[index_ref]
                results = _align_hrieuv_with_hrieuv(path_output_figures=path_figures,
                                                   large_fov_fits_path=path_reference,
                                                   large_fov_window=window_files_input,
                                                   small_fov_path=list_files_input[index_to_align],
                                                   window_to_align=window_files_input,
                                                   date_to_align=date_to_align,
                                                   parameter_alignment=parameter_alignment,
                                                   cpu_count=cpu_count, do_plot_figure=plot_all_figures,
                                                   method_carrington_reprojection=method_carrington_reprojection,
                                                   reference_date=reference_date, parallelism=parallelism,
                                                   alignement_method=alignement_method,
                                                   small_fov_value_max=small_fov_value_max, small_fov_value_min=small_fov_value_min,
                                                   **kwargs_carrington)
                basename = os.path.basename(list_files_input[index_to_align])
                basename_new = basename
                results.write_corrected_fits(window_list_to_apply_shift=[window_files_input],
                                             path_to_l3_output=os.path.join(path_files_output,
                                                                            basename_new), )


def _align_hrieuv_with_hrieuv( large_fov_fits_path: str, large_fov_window: str,
                             small_fov_path: str, parameter_alignment: dict, date_to_align,
                             cpu_count=30, window_to_align=3, do_plot_figure=False,
                             parallelism=True,
                             lonlims=None, latlims=None, shape=None, unit_lag="arcsec",
                             reference_date=None,small_fov_value_max=None,small_fov_value_min=None,
                             method_carrington_reprojection="fa", alignement_method="carrington",
                            path_output_figures: str = None,
                             fov_limits=None ):
    """Sub-function of jittter_correction_imagers.

    Args:
        large_fov_fits_path (str): path to the reference FITS file
        large_fov_window (str): window of the reference FITS file
        small_fov_path (str): path to the FITS file to align
        parameter_alignment (dict): other kword parameters for the alignment algorithm
        date_to_align (_type_): date of the FITS file to align
        cpu_count (int, optional): number of CPU cores to use for parallelism. Defaults to 30.
        window_to_align (int, optional): Window of the FITS file to align . Defaults to 3.
        do_plot_figure (bool, optional): _description_. Defaults to False.
        parallelism (bool, optional): use parallelism or not. Defaults to True.
        lonlims (_type_, optional): parameter defining the carrington grid. Defaults to None.
        latlims (_type_, optional): parameter defining the carrington grid. Defaults to None.
        shape (_type_, optional): parameter defining the carrington grid. Defaults to None.
        unit_lag (str, optional): unit over which the lag arrays are given. Defaults to "arcsec".
        reference_date (_type_, optional): date over which to correct for differential rotation. Defaults to None.
        small_fov_value_max (_type_, optional): maximum threshold value in the FITS file to align to consider for the alignment.. Defaults to None.
        small_fov_value_min (_type_, optional): minimum threshold value in the FITS file to align to consider for the alignment. Defaults to None.
        method_carrington_reprojection (str, optional): method for the carrington alignement (fa or sunpy). Defaults to "fa".
        alignement_method (str, optional): method to do the alignment. Defaults to "carrington".
        path_output_figures (str, optional): path for output figures. Defaults to None.
        fov_limits (tuple, optional): 

    Returns:
        _type_: _description_
    """    

    A = Alignment(large_fov_known_pointing=large_fov_fits_path, large_fov_window=large_fov_window,
                  small_fov_to_correct=small_fov_path, small_fov_window=window_to_align,
                  display_progress_bar=False,small_fov_value_max=small_fov_value_max,
                  small_fov_value_min=small_fov_value_min,
                  parallelism=parallelism, counts_cpu_max=cpu_count,unit_lag=unit_lag,
                  **parameter_alignment)

    date_ref = (reference_date.fits[11:19]).replace(":", "_")
    unit = None
    if alignement_method == "carrington":
        results = A.align_using_carrington(method="correlation",
                                           lonlims=lonlims, latlims=latlims, shape=shape, reference_date=reference_date,
                                           method_carrington_reprojection=method_carrington_reprojection)
        results.plot_correlation(path_save_figure=os.path.join(path_output_figures,
                                                               f"correlation_{date_to_align}.pdf"))
        unit = "deg"
        lonlims_ = None
        latlims_ = None            
    elif alignement_method == "initial_carrington":
        results = A.align_using_initial_carrington(method="correlation", )
        results.plot_correlation(
            path_save_figure=os.path.join(path_output_figures, f"correlation_{date_to_align}_{date_ref}.pdf"))
        unit = "deg"
        lonlims_=u.Quantity(lonlims, unit)
        latlims_=u.Quantity(latlims, unit)
    elif alignement_method == "helioprojective":
    
        results = A.align_using_helioprojective(method="correlation", fov_limits=fov_limits)
        results.plot_correlation(
            path_save_figure=os.path.join(path_output_figures, f"correlation_{date_to_align}_{date_ref}.pdf"))
        lonlims_ = None
        latlims_ = None 

    if do_plot_figure:
        results.plot_co_alignment(type_plot="successive_plot",
                                  path_save_figure=os.path.join(path_output_figures,
                                                                f"plot_co_alignment_{date_to_align}_{date_ref}.pdf"),
                                                                lonlims=lonlims_, latlims=latlims_,
                                  )

    plt.close("all")


    return results
