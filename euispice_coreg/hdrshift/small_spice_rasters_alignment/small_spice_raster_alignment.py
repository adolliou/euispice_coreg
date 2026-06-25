from euispice_coreg.utils.create_dict_file import create_dict_file
from euispice_coreg.synras.map_builder import SPICEComposedMapBuilder
from euispice_coreg.hdrshift.alignment_spice import AlignmentSpice
import os
import numpy as np
from pathlib import Path
from astropy.io import fits 
from astropy.time import Time
import astropy.units as u
import sunpy.map
from sunpy.coordinates import HeliocentricInertial, propagate_with_solar_surface
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord


def small_spice_raster_alignment(small_raster_list_path: list, small_raster_window: int | str, 
                                context_raster_path: str, context_raster_window: int | str, 
                                list_reference_imagers_small_raster: list, 
                                window_reference_imagers_small_raster: int | str, 
                                list_reference_imagers_context_raster: list, 
                                window_reference_imagers_context_raster: int | str, 
                                windows_to_correct: str,
                                folder_save: str,
                                threshold_time_synras: u.Quantity = 200 * u.s, 
                                threshold_time_small_raster: u.Quantity = 200 * u.s, 
                                parallelism: bool = True,
                                cpu_count: int = 10,
                                verbose: int = 1,
                                param_alignment_cr=None,
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

    if param_alignment_cr is None:   
    
        param_alignment_cr = {
        "lag_crval1": np.arange(-80, 80, 1), # lag crvals in the headers, in arcsec
        "lag_crval2": np.arange(-80, 80, 1),  # in arcsec
        "lag_crota": np.array([0]), # in degrees
        "lag_cdelt1": np.array([0]), # in arcsec
        "lag_cdelt2": np.array([0]), # in arcsec
        }

    datfolder   = os.path.join(folder_save, "data")
    figfolder   = os.path.join(folder_save, "figures")

    Path(datfolder).mkdir(exist_ok=True, parents=False)
    Path(figfolder).mkdir(exist_ok=True, parents=False)


    # First, ensure all list are temporally ordered

    filename, suffix_small  = os.path.splitext(list_reference_imagers_small_raster[0]) 
    date_raster_small       = []
    date_im_small           = []
    date_im_context         = []
    for list_path, date_list, window in zip(
            (
            small_raster_list_path, 
            list_reference_imagers_small_raster, 
            list_reference_imagers_context_raster, 
            ), 
            ( date_raster_small, date_im_small, date_im_context),
            (small_raster_window,
             window_reference_imagers_small_raster, 
            window_reference_imagers_context_raster),
    ): 
        for path in list_path:
            with fits.open(str(path)) as hdul:
                hdu             = hdul[window]
                
                date_list.append(Time(hdu.header.copy()["DATE-AVG"]))


        dt_array    = np.array([(n - date_list[0]).to("s").value for n in date_list])
        sort        = np.argsort(dt_array)

        date_list   = np.array(date_list)[sort]
        list_path   = np.array(list_path)[sort]


    filename, suffix_context    = os.path.splitext(list_reference_imagers_context_raster[0]) 


    if verbose > 0:
        print(f"Create synthetic raster for the context raster")


    path_to_synras = _create_sr_context_raster(
        context_raster_path                     = context_raster_path, 
        context_raster_window                   = context_raster_window, 
        list_reference_imagers_context_raster   = list_reference_imagers_context_raster, 
        window_reference_imagers_context_raster = window_reference_imagers_context_raster,
        folder_save                             = datfolder, 
        threshold_time_synras                   = threshold_time_synras,
    )

    if verbose > 0:
        print(f"Co-align the context raster with the synthetic raster")




    result  = _co_align_synras(
        path_to_synras          = path_to_synras, 
        window_synras           = 0, 
        context_raster_path     = context_raster_path, 
        context_raster_window   = context_raster_window,
        param_alignment         = param_alignment_cr,
        figfolder               = figfolder,
        cpu_count               = cpu_count, 
        parallelism             = parallelism,  
    )

    name_cr         = os.path.basename(context_raster_path)
    path_save_fits  = os.path.join(folder_save, name_cr)
    result.write_corrected_fits(
        windows_to_correct,
        path_to_l3_output=path_save_fits
        )

    delta_pc_arcsec = _compute_shift_fov_context_small_rasters(
        context_raster_path         = context_raster_path,
        context_raster_window       = context_raster_window,
        small_raster_list_path      = small_raster_list_path, 
        small_raster_list_window    = small_raster_window,
    )

    coords_center_cr                = _get_coords_center_spice_cr(
        context_raster_path         = context_raster_path,
        context_raster_window       = context_raster_window,
    )

    co_align_small_rasters(
        small_raster_list_path      = small_raster_list_path, 
        small_raster_list_window    = small_raster_window, 
        list_reference_imagers_cr   = list_reference_imagers_context_raster,
        window_reference_imagers_cr = window_reference_imagers_context_raster, 

        delta_pc_arcsec             = delta_pc_arcsec, 
        coords_center_cr            = coords_center_cr,
        datfolder                   = datfolder, 
    )


    

def _create_sr_context_raster(
        context_raster_path: str, 
        context_raster_window: int | str,
        list_reference_imagers_context_raster: list, 
        window_reference_imagers_context_raster: int | str,
        folder_save: str, 
        threshold_time_synras: int,
):
    

    C = SPICEComposedMapBuilder(path_to_spectro     = context_raster_path,
                                window_spectro      = context_raster_window,                                
                                list_imager_paths   = list_reference_imagers_context_raster,
                                window_imager       = window_reference_imagers_context_raster,
                                threshold_time      = threshold_time_synras,
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
    cpu_count: int,
    figfolder: str,
    ):

    A = AlignmentSpice(
        large_fov_known_pointing    = path_to_synras,
        large_fov_window            = window_synras,        
        small_fov_to_correct        = context_raster_path,
        small_fov_window            = context_raster_window,
        display_progress_bar        = True,
        parallelism                 = parallelism,
        counts_cpu_max              = cpu_count,
                    **param_alignment)

    results = A.align_using_helioprojective(method='correlation')
    results.plot_correlation(path_save_figure=os.path.join(figfolder, "correlation_context_raster.pdf"), show=True)

    return results


def _compute_shift_fov_context_small_rasters(
        context_raster_path                    : str, 
        context_raster_window                  : int | str, 
        small_raster_list_path    : list, 
        small_raster_list_window  : int | str, 
    ):

    index_small     = 0
    with fits.open(context_raster_path) as hdul_context:
        with fits.open(small_raster_list_path[index_small]) as hdul_small:
            
            hdu_context     = hdul_context[context_raster_window]   
            data_context    = hdu_context.data
            header_context  = hdu_context.header

            hdu_small       = hdul_small[small_raster_list_window]
            data_small      = hdu_small.data
            header_small    = hdu_small.header

            p1_context      = header_context["XSTART"]
            p2_context      = (p1_context - header_context["NAXIS1"] * header_context["CDELT1"])
            pc_context      = (p1_context + p2_context) * 0.5

            p1_small        = header_small["XSTART"]
            p2_small        = (p1_small - header_small["NAXIS1"] * header_small["CDELT1"])
            pc_small        = (p1_small + p2_small) * 0.5

            delta_pc_arcsec             = pc_context - pc_small

            return delta_pc_arcsec


def _get_coords_center_spice_cr(
        context_raster_path         :list,
        context_raster_window       :str|int,
    ):

    date_avg_cr = None

    with fits.open(context_raster_path) as hdul_cr:
        hdu_cr              = hdul_cr[context_raster_window]
        header_cr           = hdu_cr.header.copy()

        naxis1              = header_cr["NAXIS1"]
        naxis2              = header_cr["NAXIS2"]

        crpix1              = (naxis1 + 1)/2
        crpix2              = (naxis1 + 1)/2

        w_spice             = WCS(header_cr)
        w_xyt               = w_spice.dropaxis(2)
        w_xyt.wcs.pc[2, 0]  = 0
        w_xy = w_xyt.dropaxis(2)

        with propagate_with_solar_surface():
            coords_center       = w_xy.pixel_to_world(crpix1 - 1, crpix2 - 1)

    return coords_center 

    
    

def  co_align_small_rasters(
        small_raster_list_path      : str, 
        small_raster_list_window    : int|str, 
        list_reference_imagers_cr   : list,
        window_reference_imagers_cr : int|str, 
        delta_pc_arcsec             : float, 
        coords_center_cr            : SkyCoord,
        datfolder                   : str, 
    ):
    pass