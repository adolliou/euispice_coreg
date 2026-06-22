from euispice_coreg.utils.create_dict_file import create_dict_file
from euispice_coreg.synras.map_builder import SPICEComposedMapBuilder
from euispice_coreg.hdrshift.alignment_spice import AlignmentSpice
import os
import numpy as np
from pathlib import Path
from astropy.io import fits 


def small_spice_raster_alignment(small_raster_list_path: list, small_raster_window: int | str, 
                                context_raster_path: str, context_raster_window: int | str, 
                                list_reference_imagers_small_raster: list, 
                                window_reference_imagers_small_raster: int | str, 
                                  list_reference_imagers_context_raster: list, 
                                window_reference_imagers_context_raster: int | str, 
                                windows_to_correct: str,
                                folder_save: str,
                                threshold_time_synras: int=100, 
                                threshold_time_small_raster: int=100, 
                                parallelism: bool = True,
                                cpu_count: int = 10,
                                verbose: int = 1
                                ):
    """_summary_

    Args:
        small_raster_list_path (list): _description_
        small_raster_window (int | str): _description_
        context_raster_path (str): _description_
        context_raster_window (int | str): _description_
        list_reference_imagers_small_raster (list): _description_
        window_reference_imagers_small_raster (int | str): _description_
        list_reference_imagers_context_raster (list): _description_
        window_reference_imagers_context_raster (int | str): _description_
        windows_to_correct (str): _description_
        folder_save (str): _description_
    """
    datfolder   = os.path.join(folder_save, "data")
    figfolder   = os.path.join(folder_save, "figures")

    Path(datfolder).mkdir(exist_ok=True, parents=False)
    Path(figfolder).mkdir(exist_ok=True, parents=False)


    # First, ensure all list are temporally ordered

    filename, suffix_small  = os.path.splitext(list_reference_imagers_small_raster[0]) 
    dict_ref_imagers_small  = create_dict_file(
          path_instrument   = list_reference_imagers_small_raster, 
          window            = window_reference_imagers_small_raster, 
          suffix            = suffix_small,
     )

    filename, suffix_context    = os.path.splitext(list_reference_imagers_context_raster[0]) 
    dict_ref_imagers_context    = create_dict_file(
          path_instrument       = list_reference_imagers_context_raster, 
          window                = window_reference_imagers_context_raster, 
          suffix                = suffix_context,
     )

    if verbose > 0:
        print(f"Create synthetic raster for the context raster")


    path_to_synras = _create_sr_context_raster(
        context_raster_path     = context_raster_path, 
        context_raster_window   = context_raster_window, 
        list_reference_imagers_context_raster=dict_ref_imagers_context["path"], 
        window_reference_imagers_context_raster=window_reference_imagers_context_raster,
        folder_save=datfolder, 
        threshold_time_synras   = threshold_time_synras,
    )

    if verbose > 0:
        print(f"Co-align the context raster with the synthetic raster")


    param_alignment = {
        "lag_crval1": np.arange(-80, 80, 1), # lag crvals in the headers, in arcsec
        "lag_crval2": np.arange(-80, 80, 1),  # in arcsec
        "lag_crota": np.array([-0.5, 0, 0.5]), # in degrees
        "lag_cdelt1": np.array([0]), # in arcsec
        "lag_cdelt2": np.array([0]), # in arcsec
    }

    result  = _co_align_synras(
        path_to_synras          = path_to_synras, 
        window_synras           = 0, 
        context_raster_path     = context_raster_path, 
        context_raster_window   = context_raster_window,
        param_alignment         = param_alignment,
        figfolder               = figfolder, 
    )

    name_cr         = os.path.basename(context_raster_path)
    path_save_fits  = os.path.join(folder_save, name_cr)
    result.write_corrected_fits(
        windows_to_correct,
        path_to_l3_output=path_save_fits
        )
    

    delta_pc = _compute_shift_fov_context_small_rasters(
        context_raster_path                     = path_save_fits,
        context_raster_window                   = context_raster_window,
        list_reference_imagers_small_raster     = dict_ref_imagers_context["path"], 
        window_reference_imagers_small_raster   = window_reference_imagers_small_raster,
    )

    

def _create_sr_context_raster(
        context_raster_path: str, 
        context_raster_window: int | str,
        list_reference_imagers_context_raster: list, 
        window_reference_imagers_context_raster: int | str,
        folder_save: str, 
        threshold_time_syntras: int,
):
    

    C = SPICEComposedMapBuilder(path_to_spectro     = context_raster_path,
                                window_spectro      = context_raster_window,                                
                                list_imager_paths   = list_reference_imagers_context_raster,
                                window_imager       = window_reference_imagers_context_raster,
                                threshold_time      = threshold_time_syntras,
                                )
    path_to_synras = C.process(
        folder_path_output  = folder_save,
        return_synras_name  = True)
    return path_to_synras

def _co_align_synras(
    path_to_synras: str, 
    window_synras: int | str, 
    context_raster_path: str,
    context_raster_window: int | str,
    param_alignment: dict, 
    parallelism: bool, 
    counts_cpu_max: int,
    figfolder: str,
    ):

    A = AlignmentSpice(
        large_fov_known_pointing    = path_to_synras,
        large_fov_window            = window_synras,        
        small_fov_to_correct        = context_raster_path,
        small_fov_window            = context_raster_window,
        display_progress_bar        = True,
        parallelism                 = parallelism,
        counts_cpu_max              = counts_cpu_max,
                    **param_alignment)

    results = A.align_using_helioprojective(method='correlation')
    results.plot_correlation(path_save_figure=os.path.join(figfolder, "correlation_context_raster.pdf"), show=True)

    return results


def _compute_shift_fov_context_small_rasters(
        context_raster_path                    : str, 
        context_raster_window                  : int | str, 
        list_reference_imagers_small_raster    : list, 
        window_reference_imagers_small_raster  : int | str, 
    ):

    index_small     = 0
    with fits.open(context_raster_path) as hdul_context:
        with fits.open(list_reference_imagers_small_raster[index_small]) as hdul_small:
            
            hdu_context     = hdul_context[context_raster_window]   
            data_context    = hdu_context.data
            header_context  = hdu_context.header

            hdu_small       = hdul_small[window_reference_imagers_small_raster]
            data_small      = hdu_small.data
            header_small    = hdu_small.header

            p1_context      = header_context["XSTART"]
            p2_context      = (p1_context - header_context["NAXIS1"] * header_context["CDELT1"])
            pc_context      = (p1_context + p2_context) * 0.5

            p1_small        = header_small["XSTART"]
            p2_small        = (p1_context - header_small["NAXIS1"] * header_small["CDELT1"])
            pc_small        = (p1_small + p2_small) * 0.5

            delta_pc             = p2_context - p1_context

            return delta_pc


