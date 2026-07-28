from euispice_coreg.utils.create_dict_file import create_dict_file
from euispice_coreg.synras.map_builder import SPICEComposedMapBuilder
from euispice_coreg.hdrshift.alignment_spice import AlignmentSpice
import os
import numpy as np
from pathlib import Path
from astropy.io import fits 
from astropy.time import Time
import astropy.units as u
from sunpy.map import Map
from sunpy.coordinates import HeliocentricInertial, propagate_with_solar_surface
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from sunpy.coordinates.screens import SphericalScreen
import warnings
import astropy

def small_spice_raster_alignment(
        small_raster_list_path                          : list,
        small_raster_window                             : int|str, 
        context_raster_path                             : str,
        context_raster_window                           : int|str, 
        list_reference_imagers_small_raster             : list, 
        window_reference_imagers_small_raster           : int|str, 
        list_reference_imagers_context_raster           : list, 
        window_reference_imagers_context_raster         : int|str,                         
        windows_to_correct                              : str,
        path_imager_sr_closeto_imager_cr                : str,
        window_imager_sr_closeto_imager_cr              : str|int,
        path_imager_cr_closeto_imager_sr                : str,
        window_imager_cr_closeto_imager_sr              : str|int,
        folder_save                                     : str,
        threshold_time_synras                           : u.Quantity = 200 * u.s, 
        threshold_time_small_raster                     : u.Quantity = 60 * u.s, 
        threshold_time_imager_sr_cr                     : u.Quantity = 60 * u.s, 
        parallelism                                     : bool = True,
        cpu_count                                       : int = 10,
        verbose                                         : int = 1,
        param_alignment_cr                              : dict = None,
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

    delta_pc_arcsec         = _compute_shift_fov_context_small_rasters(
        context_raster_path         = context_raster_path,
        context_raster_window       = context_raster_window,
        small_raster_list_path      = small_raster_list_path, 
        small_raster_list_window    = small_raster_window,
    )

    # Center of the sr rasters, in the imager cr pixels
    x_cr, y_cr              =  _get_sr_center_in_imager_cr(
        context_raster_path         = context_raster_path,
        context_raster_window       = context_raster_window,
        list_imager_cr              = list_reference_imagers_context_raster, 
        window_imager_cr            = window_reference_imagers_context_raster,
        date_imager_cr              = date_im_context, 
        delta_pc_arcsec             = delta_pc_arcsec,
    )

    x_sr, y_sr              = _cr_imager_to_sr_imager_pixels(
        x_cr                                = x_cr, 
        y_cr                                = y_cr, 
        path_imager_sr_closeto_imager_cr    = path_imager_sr_closeto_imager_cr  , 
        window_imager_sr_closeto_imager_cr  = window_imager_sr_closeto_imager_cr, 
        path_imager_cr_closeto_imager_sr    = path_imager_cr_closeto_imager_sr  , 
        window_imager_cr_closeto_imager_sr  = window_imager_cr_closeto_imager_sr, 
        threshold_time_imager_sr_cr         = threshold_time_imager_sr_cr, 
    )


    # Get sr center in FSI pixels

    # get HRIEUV pixels in FSI pixels

    # get sr center in HRIEUV pixels
    # 
    # Co align sr with HRIEUV files 

    # coords_center_sr                = _get_coords_center_spice_sr(
    #     context_raster_path         = context_raster_path,
    #     context_raster_window       = context_raster_window,
    #     delta_pc_arcsec             = delta_pc_arcsec,
    # )


    co_align_small_rasters(
        small_raster_list_path      = small_raster_list_path, 
        small_raster_list_window    = small_raster_window, 
        list_reference_imagers_sr   = list_reference_imagers_small_raster,
        window_reference_imagers_sr = window_reference_imagers_small_raster, 
        date_im_small               = date_im_small,
        x_sr                        = x_sr,
        y_sr                        = y_sr,
        datfolder                   = datfolder,
        windows_to_correct          = windows_to_correct,
        threshold_time_small_raster = threshold_time_small_raster, 
    )


    

def _create_sr_context_raster(
        context_raster_path                     :str, 
        context_raster_window                   :int|str,
        list_reference_imagers_context_raster   :list, 
        window_reference_imagers_context_raster :int|str,
        folder_save                             :str, 
        threshold_time_synras                   :int,
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
    path_to_synras              : str, 
    window_synras               : int|str, 
    context_raster_path         : str,
    context_raster_window       : int|str,
    param_alignment             : dict, 
    parallelism                 : bool, 
    cpu_count                   : int,
    figfolder                   : str,
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
        context_raster_window                  : int|str, 
        small_raster_list_path                 : list, 
        small_raster_list_window               : int|str, 
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


def     _get_sr_center_in_imager_cr(
        context_raster_path         : list,
        context_raster_window       : str|int,
        list_imager_cr              : list, 
        window_imager_cr            : str|int, 
        date_imager_cr              : list, 
        delta_pc_arcsec             : u.Quantity,
        ):

    with fits.open(context_raster_path) as hdul_cr:
        hdu_cr              = hdul_cr[context_raster_window]
        header_cr           = hdu_cr.header.copy()

        naxis1              = header_cr["NAXIS1"]
        naxis2              = header_cr["NAXIS2"]

        crpix1              = (naxis1 + 1)/2
        crpix2              = (naxis2 + 1)/2

        w_spice             = WCS(header_cr)
        w_xyt               = w_spice.dropaxis(2)
        w_xyt.wcs.pc[2, 0]  = 0
        w_xy = w_xyt.dropaxis(2)


        delta_pc_crpixels   = delta_pc_arcsec/header_cr["CDELT1"]

        with propagate_with_solar_surface():
            coords_center       = w_xy.pixel_to_world(crpix1 - 1 - delta_pc_crpixels, crpix2 - 1)

        date_cr                 = Time(header_cr["DATE-AVG"])
        index_imager_cr_closest = np.abs([(n - date_cr).to("s").value  \
                                          for n in date_imager_cr]).argmin()
        path_imager_cr          = list_imager_cr[index_imager_cr_closest]

        with fits.open(path_imager_cr) as hdul_im_cr:
            hdu_im_cr           = hdul_im_cr[window_imager_cr]

            header_im_cr        = hdu_im_cr.header
            w_im_cr             = WCS(header_im_cr)
            x_sr_in_imagercr, y_sr_in_imagercr = w_im_cr.world_to_pixels(coords_center)

    return x_sr_in_imagercr, y_sr_in_imagercr 

# def _get_coords_center_spice_sr(
#         context_raster_path         :list,
#         context_raster_window       :str|int,
#         delta_pc_arcsec             :u.Quantity
#     ):


#     with fits.open(context_raster_path) as hdul_cr:
#         hdu_cr              = hdul_cr[context_raster_window]
#         header_cr           = hdu_cr.header.copy()

#         naxis1              = header_cr["NAXIS1"]
#         naxis2              = header_cr["NAXIS2"]

#         crpix1              = (naxis1 + 1)/2
#         crpix2              = (naxis2 + 1)/2

#         w_spice             = WCS(header_cr)
#         w_xyt               = w_spice.dropaxis(2)
#         w_xyt.wcs.pc[2, 0]  = 0
#         w_xy = w_xyt.dropaxis(2)


#         delta_pc_crpixels   = delta_pc_arcsec/header_cr["CDELT1"]

#         with propagate_with_solar_surface():
#             coords_center       = w_xy.pixel_to_world(crpix1 - 1 - delta_pc_crpixels, crpix2 - 1)

#     return coords_center 


def  _cr_imager_to_sr_imager_pixels(
    x_cr                                : int, 
    y_cr                                : int, 
    path_imager_sr_closeto_imager_cr    : str  , 
    window_imager_sr_closeto_imager_cr  : int|str, 
    path_imager_cr_closeto_imager_sr    : str  , 
    window_imager_cr_closeto_imager_sr  : int|str, 
    threshold_time_imager_sr_cr         : u.Quantity
    ):

    with fits.open(path_imager_cr_closeto_imager_sr) as hdul_im_cr:
        with fits.open(path_imager_sr_closeto_imager_cr) as hdul_im_sr:

            hdu_im_cr       = hdul_im_cr[window_imager_sr_closeto_imager_cr]
            hdu_im_sr       = hdul_im_sr[window_imager_cr_closeto_imager_sr]

            w_sr            = WCS(hdu_im_sr.header)
            w_cr            = WCS(hdu_im_cr.header)

            date_cr         = Time(hdu_im_cr.header["DATE-AVG"])
            date_sr         = Time(hdu_im_sr.header["DATE-AVG"])
            if np.abs(date_cr - date_sr) >= threshold_time_imager_sr_cr:
                ValueError("DeltaTime between imager cr and sr too large")
            coords_center   = w_cr.pixels_to_world(x_cr, y_cr)
            x_sr, y_sr      = w_sr.world_to_pixels(coords_center)

    return x_sr, y_sr


    

    

def  co_align_small_rasters(
        small_raster_list_path      : str, 
        small_raster_list_window    : int|str, 
        list_reference_imagers_sr   : list,
        window_reference_imagers_sr : int|str,
        date_im_small               : list,
        threshold_time_small_raster : u.Quantity,  
        x_sr                        : float,
        y_sr                        : float, 
        datfolder                   : str, 
        windows_to_correct          : list,
    ):
    
    crpix1              = None
    crpix2              = None
    lon_arcsec          = None
    lat_arcsec          = None


    coords_center       = None
    for path_sr in small_raster_list_path:
        with fits.open(path_sr) as hdul_sr:
            hdu_sr                      = hdul_sr[small_raster_list_window]
            header_sr                   = hdu_sr.header.copy()     
            time_sr                     = Time(header_sr["DATE-AVG"])
            index_imager_sr_closest     = np.abs([(n - time_sr).to("s").value for n in date_im_small])
            if index_imager_sr_closest > threshold_time_small_raster:
                raise ValueError(f"could not find imager file close enough to {time_sr.fits[11:19]}")
            path_im_sr                  = list_reference_imagers_sr[index_imager_sr_closest]
            with fits.open(path_im_sr) as hdul_im_sr:
                hdu_im_sr       = hdul_im_sr[window_reference_imagers_sr] 
                map_im_sr       = Map(hdu_im_sr)
                header_im_sr    = hdu_im_sr.header.copy()
                w_im_sr         = WCS(header_im_sr)

                with (propagate_with_solar_surface(),
                    SphericalScreen(map_im_sr.observer_coordinate, only_off_disk=True)):
                    coords_center        = w_im_sr.pixel_to_world(x_sr, y_sr)

            lon_arcsec          = coords_center.lon.to("arcsec").value
            lat_arcsec          = coords_center.lat.to("arcsec").value

            crpix1              = (naxis1 + 1)/2
            crpix2              = (naxis2 + 1)/2

            hdul_out = fits.HDUList()

            for ii in range(len(hdul_sr)):
                hdu = hdul_sr[ii]
                if "EXTNAME" in hdu.header:
                    extname = hdu.header["EXTNAME"]
                else:
                    extname = "nothing98695"
                if (extname in windows_to_correct) or \
                      (ii in windows_to_correct) or \
                        ((ii - len(hdul_sr)) in windows_to_correct):
                    header_sr           = hdu.header.copy()
                    naxis1              = header_sr["NAXIS1"]
                    naxis2              = header_sr["NAXIS2"]

                    _check_ant_create_pcij_matrix(header_im_sr)

                    header_sr["CRPIX1"] = crpix1
                    header_sr["CRPIX2"] = crpix2
                    header_sr["CRVAL1"] = lon_arcsec
                    header_sr["CRVAL2"] = lat_arcsec


                    lam                 = header_sr["CDELT2"]/header_sr["CDELT1"]
                    rho                 = np.deg2rad(header_im_sr["CROTA"])

                    header_sr["PC1_1"]  = np.cos(rho)
                    header_sr["PC2_2"]  = np.cos(rho)
                    header_sr["PC1_2"]  = - lam * np.sin(rho)
                    header_sr["PC2_1"]  = (1 / lam) * np.sin(rho)
                    header_sr["CROTA"]  = header_im_sr["CROTA"]
                    header_sr["CROTA2"] = header_im_sr["CROTA"]


                    data = np.array(data, dtype="<f4")
                    if isinstance(hdu, astropy.io.fits.hdu.compressed.compressed.CompImageHDU):
                        hdu_out = fits.CompImageHDU(data=data, header=header_sr)
                    elif isinstance(hdu, astropy.io.fits.hdu.image.ImageHDU):
                        hdu_out = fits.ImageHDU(data=data, header=header_sr)
                    elif isinstance(hdu, astropy.io.fits.hdu.image.PrimaryHDU):
                        hdu_out = fits.PrimaryHDU(data=data, header=header_sr)
                    hdu_out.verify("silentfix")
            name            = os.path.basename(small_raster_list_path)
            hdul_out.savefig(os.path.join(datfolder, name))



                    
    def _check_ant_create_pcij_matrix(hdr):
        if ("PC1_1" not in hdr):
            warnings.warn("PCi_j matrix not found in header of the FITS file to align. Adding it to the header.")
            if "CROTA" in hdr:
                crot = hdr["CROTA"]
            elif "CROTA2" in hdr:
                crot = hdr["CROTA2"]
            else:

                raise ValueError("No, CROTA, CROTA2 or PCi_j matrix in your FITS file. If want to force a CROTA=0, "
                                    "please set the force_crota_0 to True when initializing Alignment ")

            rho = np.deg2rad(crot)
            lam = hdr["CDELT2"] / hdr["CDELT1"]
            hdr["PC1_1"] = np.cos(rho)
            hdr["PC2_2"] = np.cos(rho)
            hdr["PC1_2"] = - lam * np.sin(rho)
            hdr["PC2_1"] = (1 / lam) * np.sin(rho)
        if hdr["PC1_1"] >= 1.0:
            warnings.warn(f'{hdr["PC1_1"]=}, setting to  1.0.')
            hdr["PC1_1"] = 1.0
            hdr["PC2_2"] = 1.0
            hdr["PC1_2"] = 0.0
            hdr["PC2_1"] = 0.0
            hdr["CROTA"] = 0.0

        if 'CROTA' not in hdr:
            s = - np.sign(hdr["PC1_2"]) + (hdr["PC1_2"] == 0)
            hdr["CROTA"] = s * np.rad2deg(np.arccos(hdr["PC1_1"]))

        


    