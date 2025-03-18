import copy
from matplotlib import pyplot as plt
import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from matplotlib.gridspec import GridSpec
from astropy.visualization import ImageNormalize, LogStretch, PowerStretch, LinearStretch
import matplotlib.patches as patches
import cv2
import scipy
from astropy.io import fits
from ..utils.Util import AlignSpiceUtil, AlignEUIUtil, PlotFits, AlignCommonUtil
from astropy.time import Time
import os
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs.utils import WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS
from astropy.coordinates import SkyCoord
from matplotlib.backends.backend_pdf import PdfPages
import astropy.constants
from sunpy.coordinates import propagate_with_solar_surface
from matplotlib.colors import CenteredNorm


def interpol2d(image, x, y, order=1, fill=0, opencv=False, dst=None):
    """
    Interpolates in 2D image using either map_coordinates or opencv

    data: image to interpolate
    x, y: coordinates (in pixels) at which to interpolate the image
    order: if opencv is True:  0=nearest neighbor, 1=linear, 2=cubic
           if opencv is False: the order of the spline interpolation used by
                               map_coordinates (see scipy documentation)
    opencv: If True, uses opencv
            If False, uses scipy.ndimage.map_coordinates
            opencv can use only 32 bits floating point coordinates input
    fill: constant value usesd to fill in the edges
    dst: if present, ndarray in which to place the result
    """

    bad = np.logical_or(x == np.nan, y == np.nan)
    x = np.where(bad, -1, x)
    y = np.where(bad, -1, y)

    if dst is None:
        dst = np.empty(x.shape, dtype=image.dtype)

    if opencv:
        if order == 0:
            inter = cv2.INTER_NEAREST
        elif order == 1:
            inter = cv2.INTER_LINEAR
        elif order == 2:
            inter = cv2.INTER_CUBIC
        cv2.remap(image,
                  x.astype(np.float32),  # converts to float 32 for opencv
                  y.astype(np.float32),  # does nothing with default dtype
                  inter,  # interpolation method
                  dst,  # destination array
                  cv2.BORDER_CONSTANT,  # fills in with constant value
                  fill)  # constant value
    else:
        coords = np.stack((y.ravel(), x.ravel()), axis=0)

        scipy.ndimage.map_coordinates(image,  # input array
                                      coords,  # array of coordinates
                                      order=order,  # spline order
                                      mode='constant',  # fills in with constant value
                                      cval=fill,  # constant value
                                      output=dst.ravel(),
                                      prefilter=False)

    return dst


class PlotFunctions:
    @staticmethod
    def plot_correlation(corr, lag_crval1, lag_crval2, lag_crota=None, lag_cdelt1=None, lag_cdelt2=None,
                         path_save_figure=None, fig=None, ax=None, show=False, lag_dx_label='CRVAL1 [arcsec]'
                         , lag_dy_label='CRVAL2 [arcsec]',
                         shift: tuple=None,

                         unit_to_plot="arcsec",):
        """


        :param corr: (np.array) correlation Matrix obtained from co-alignment
        :param lag_crval1: (np.array): chosen lags for the CRVAL1 value on header. Must correspond to the correlation matrix
        :param lag_crval2: (np.array): chosen lags for the CRVAL2 value on header. Must correspond to the correlation matrix
        :param lag_crota: (np.array) (optional) chosen lags for the CROTA value on header. Must correspond to the correlation matrix
        :param lag_cdelt1: (np.array) (optional) chosen lags for the CDELT1 value on header. Must correspond to the correlation matrix
        :param lag_cdelt2: (np.array) (optional) chosen lags for the CDELT2 value on header. Must correspond to the correlation matrix
        :param path_save: (str) (optional) path to save the figure.
        :param fig: (matplotlib.figure.Figure) (optional) figure object where to plot the figure. If none, will create a new figure
        :param ax:  (matplotlib.axes.ax) (optional) ax where to plot the figure. If none, will create a new ax
        :param show: (bool) (optional) whether or not to show the figure.
        :param unit: (str) (optional) unit to use for the figure.
        :param shift: (tuple) (optional) shift array computed for AlignmentResults
        :param lag_dy_label: label for the dy axis
        :param lag_dx_label: label for the dx axis
        :param unit_to_plot: AlignmentResults param
        """
        max_index = np.unravel_index(np.nanargmax(corr), corr.shape)
        if unit_to_plot == "arcsec":
            unit = '\'\''
        elif unit_to_plot == "deg":
            unit = '°'
        else:
            raise NotImplementedError
        corr = corr[:, :, max_index[2], max_index[3], max_index[4]]

        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()

        lag_dx = u.Quantity(lag_crval1, "arcsec").to(unit_to_plot).value
        lag_dy = u.Quantity(lag_crval2, "arcsec").to(unit_to_plot).value
        dy = lag_dy[1] - lag_dy[0]
        dx = lag_dx[1] - lag_dx[0]
        if lag_cdelt1 is None:
            lag_cdelt1_ = np.array([0])
        else:
            lag_cdelt1_ = u.Quantity(lag_cdelt1, "arcsec").to(unit_to_plot).value
        if lag_cdelt2 is None:
            lag_cdelt2_ = np.array([0])
        else:
            lag_cdelt2_ = u.Quantity(lag_cdelt2, "arcsec").to(unit_to_plot).value
        if lag_crota is None:
            lag_crota_ = np.array([0])
        else:
            lag_crota_ = lag_crota
        if shift is None:
            shift = (
                u.Quantity(lag_dx[max_index[0]], "arcsec").to(unit_to_plot).value,
                u.Quantity(lag_dy[max_index[1]], "arcsec").to(unit_to_plot).value,
                u.Quantity(lag_cdelt1_[max_index[2]], "arcsec").to(unit_to_plot).value,
                u.Quantity(lag_cdelt2_[max_index[3]], "arcsec").to(unit_to_plot).value,
                lag_crota_[max_index[4]]
            )

        # norm = PowerNorm(gamma=2)
        isnan = np.isnan(corr)
        min = np.percentile(corr[~isnan], 30)
        # norm = ImageNormalize(stretch=LinearStretch(), vmin=min)
        norm = ImageNormalize(stretch=PowerStretch(a=3), vmin=min)

        im = ax.imshow(np.swapaxes(corr, axis1=0, axis2=1), origin='lower', interpolation="none",
                       norm=norm, cmap="plasma",
                       extent=(lag_dx[0] - 0.5 * dx, lag_dx[-1] + 0.5 * dx,
                               lag_dy[0] - 0.5 * dy, lag_dy[-1] + 0.5 * dy))
        rect = patches.Rectangle((lag_dx[max_index[0]] - 0.5 * dx, lag_dy[max_index[1]] - 0.5 * dy), dx, dy,
                                 edgecolor='r', linewidth=0.3, facecolor="none")
        ax.add_patch(rect)
        ax.axhline(y=shift[1], color='r', linestyle='--', linewidth=0.5)
        ax.axvline(x=shift[0], color='r', linestyle='--', linewidth=0.5)
        if (lag_crota is not None) & (lag_cdelt1 is None):
            textstr = '\n'.join((
                r'$dx=%.1f$ %s' % (shift[0], unit),
                r'$dy=%.1f$ %s' % (shift[1], unit),
                r'$drota=%.2f$ $^\circ$' % (shift[4]),
                r'max_cc = %.2f' % (np.nanmax(corr))
            ))
        elif (lag_crota is not None) & (lag_cdelt1 is not None):
            textstr = '\n'.join((
                r'$dx=%.1f$ %s' % (shift[0], unit),
                r'$dy=%.1f$ %s' % (shift[1], unit),
                r'$drota=%.2f$ $^\circ$' % (shift[4]),
                r'$cdelt1=%.2f$ $^\circ$' % (shift[2]),
                r'$cdelt2=%.2f$ $^\circ$' % (shift[3]),
                r'max_cc = %.2f' % (np.nanmax(corr))))

        else:
            textstr = '\n'.join((
                r'$\delta CRVAL1=%.2f$ %s' % (shift[0], unit),
                r'$\delta CRVAL2=%.2f$ %s' % (shift[1], unit),
                r'max_cc = %.2f' % (np.nanmax(corr))))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', bbox=props)
        if lag_dx_label is not None:
            ax.set_xlabel(lag_dx_label)
        if lag_dy_label is not None:
            ax.set_ylabel(lag_dy_label)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label="correlation")
        if show:
            fig.show()
        if path_save_figure is not None:
            fig.tight_layout()
            fig.savefig(path_save_figure)

    @staticmethod
    def plot_fov_rectangle(data, slc=None, path_save=None, show=True, plot_colorbar=True, norm=None, angle=0):
        fig = plt.figure()
        ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        PlotFunctions.plot_fov(data=data, show=False, fig=fig, ax=ax, norm=norm)
        rect = patches.Rectangle((slc[1].start, slc[0].start),
                                 slc[1].stop - slc[1].start, slc[0].stop - slc[0].start, linewidth=1,
                                 edgecolor='r', facecolor='none', angle=angle)
        ax.add_patch(rect)
        ax.axhline(y=(slc[1].start + slc[1].stop - 1) / 2, linestyle="--", linewidth=0.5, color="r")
        ax.axvline(x=(slc[0].start + slc[0].stop - 1) / 2, linestyle="--", linewidth=0.5, color="r")

        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)

    @staticmethod
    def plot_fov(data, slc=None, path_save=None, show=True, plot_colorbar=True, fig=None, ax=None, norm=None,
                 cmap="plasma", xlabel="X [px]", ylabel="Y [px]", aspect=1, return_im=False, extent=None,):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        if slc is not None:
            im = ax.imshow(data[slc[0], slc[1]], origin="lower", interpolation="none", norm=norm, aspect=aspect,
                           cmap=cmap, extent=extent)
        else:
            im = ax.imshow(data, cmap=cmap, origin="lower", interpolation="None", norm=norm, aspect=aspect, extent=extent)
        if plot_colorbar:
            fig.colorbar(im, label="DN/s")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)
        if return_im:
            return im
        

    @staticmethod
    def simple_plot_sunpy(m_main, path_save=None, show=False, ax=None, fig=None, norm=None,
                    show_xlabel=True, show_ylabel=True, plot_colorbar=True, cmap="plasma",  rsun = 1.004*astropy.constants.R_sun):
        from sunpy.map import Map

        rsun = rsun.to("m").value
        return_im = False
        if fig is None:
            fig = plt.figure()
            return_im = True
        if ax is None:
            ax = fig.add_subplot(projection=m_main)
        if norm is None:
            norm = PlotFits.get_range(m_main.data, stre=None)
        m_main.plot(axes=ax, norm=norm, cmap=cmap)

        # im = ax.imshow(data_main, origin="lower", interpolation="none", norm=norm,)

        if show_xlabel:
            ax.set_xlabel("Solar-X [arcsec]")
        if show_ylabel:
            ax.set_ylabel("Solar-Y [arcsec]")
        if plot_colorbar:
            if "bunit" in m_main.meta:
                fig.colorbar(ax=ax, mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap), label=m_main.meta["bunit"])
            else:
                fig.colorbar(ax=ax, mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap), )
        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)



    @staticmethod
    def simple_plot(hdr_main, data_main, path_save=None, show=False, ax=None, fig=None, norm=None,
                    show_xlabel=True, show_ylabel=True, plot_colorbar=True, cmap="plasma"):

        use_sunpy = False
        for mapping in [WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS]:
            if mapping[-1][0].__module__ == 'sunpy.coordinates.wcs_utils':
                use_sunpy = True

        if use_sunpy:
            w = WCS(hdr_main)
            idx_lon = np.where(np.array(w.wcs.ctype, dtype="str") == hdr_main["CTYPE1"])[0][0]
            idx_lat = np.where(np.array(w.wcs.ctype, dtype="str") == hdr_main["CTYPE2"])[0][0]
            x, y = np.meshgrid(np.arange(w.pixel_shape[idx_lon]),
                               np.arange(w.pixel_shape[idx_lat]), )  # t dépend de x,
            # should reproject on a new coordinate grid first : suppose slits at the same time :
            coords = w.pixel_to_world(x, y)
            if hdr_main["CTYPE1"] == "HPLN-TAN":
                longitude = AlignCommonUtil.ang2pipi(coords.Tx)
                latitude = AlignCommonUtil.ang2pipi(coords.Ty)
            else:
                longitude = coords.lon
                latitude = coords.lat
            longitude_grid, latitude_grid, dlon, dlat = PlotFits.build_regular_grid(longitude=longitude,
                                                                                    latitude=latitude)
            coords_grid = SkyCoord(longitude_grid, latitude_grid, frame=coords.frame)
            x, y = w.world_to_pixel(coords_grid)
        else:
            longitude, latitude, dsun = AlignEUIUtil.extract_EUI_coordinates(hdr_main, 
                                                                             lon_ctype=hdr_main["CTYPE1"], lat_ctype=hdr_main["CTYPE2"])
            longitude_grid, latitude_grid, dlon, dlat = PlotFits.build_regular_grid(longitude=longitude,
                                                                                    latitude=latitude)
            w = WCS(hdr_main)
            x, y = w.world_to_pixel(longitude_grid, latitude_grid)
        dlon = dlon.to("arcsec").value
        dlat = dlat.to("arcsec").value

        longitude_grid_arcsec = longitude_grid.to("arcsec").value
        latitude_grid_arcsec = latitude_grid.to("arcsec").value

        image_on_regular_grid = interpol2d(data_main, x=x, y=y, fill=-32762, order=1)
        image_on_regular_grid[image_on_regular_grid == -32762] = np.nan
        return_im = False
        if fig is None:
            fig = plt.figure()
            return_im = True
        if ax is None:
            ax = fig.add_subplot()
        if norm is None:
            norm = PlotFits.get_range(image_on_regular_grid, stre=None)
        im = ax.imshow(image_on_regular_grid, origin="lower", interpolation="none", norm=norm, cmap=cmap,
                       extent=(longitude_grid_arcsec[0, 0] - 0.5 * dlon, longitude_grid_arcsec[-1, -1] + 0.5 * dlon,
                               latitude_grid_arcsec[0, 0] - 0.5 * dlat, latitude_grid_arcsec[-1, -1] + 0.5 * dlat))
        # im = ax.imshow(data_main, origin="lower", interpolation="none", norm=norm,)

        if show_xlabel:
            ax.set_xlabel("Solar-X [arcsec]")
        if show_ylabel:
            ax.set_ylabel("Solar-Y [arcsec]")
        if plot_colorbar:
            if "BUNIT" in hdr_main:
                fig.colorbar(im, label=hdr_main["BUNIT"])
            else:
                fig.colorbar(im, )
        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)
        if return_im:
            return im

    # @staticmethod
    # def contour_plot_sunpy(hdr_main, data_main, hdr_contour, data_contour, path_save=None, show=True, levels=None,
    #                 ax=None, fig=None, norm=None, show_xlabel=True, show_ylabel=True, plot_colorbar=True,
    #                 header_coordinates_plot=None, cmap="plasma", rsun = 1.004*astropy.constants.R_sun):
    #     rsun = rsun.to("m").value
    #     if header_coordinates_plot is None:
    #         hdr_contour_ = copy.deepcopy(hdr_contour)
    #         hdr_contour_["RSUN_REF"] = rsun
    #         wcs_to_reproject = WCS(hdr_contour_)
    #     else:
    #         wcs_to_reproject = WCS(header_coordinates_plot)

    #     m_main = Map(data_main, hdr_main)
    #     m_contour = Map(data_contour, hdr_contour)

    #     m_main.meta["rsun_ref"] = rsun
    #     m_contour.meta["rsun_ref"] = rsun
    #     with propagate_with_solar_surface(): 
    #         m_main_rep = m_main.reproject_to(wcs_to_reproject)
    #         m_contour_rep = m_contour.reproject_to(wcs_to_reproject)
    #     return_im = True
    #     if fig is None:
    #         fig = plt.figure()
    #         return_im = False
    #     if ax is None:
    #         ax = fig.add_subplot(projection=m_main_rep)
    #     if norm is None:
    #         norm = ImageNormalize(stretch=LogStretch(5))
    #     m_main_rep.plot(axes=ax, norm=norm, cmap=cmap)

    #     if levels is None:
    #         max_small = np.nanmax(m_contour_rep.data)
    #         levels = [0.5 * max_small]
    #     m_contour_rep.draw_contours(axes=ax, levels=levels)
    #     if show_xlabel:
    #         ax.set_xlabel("Solar-X [arcsec]")
    #     if show_ylabel:
    #         ax.set_ylabel("Solar-Y [arcsec]")
    #     if plot_colorbar:
    #         divider = make_axes_locatable(ax)
    #         cax = divider.append_axes("right", size="5%", pad=0.05)
    #         if "BUNIT" in hdr_main:

    #             fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=hdr_main["BUNIT"])
    #         else:
    #             fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    #     if show:
    #         fig.show()
    #     if path_save is not None:
    #         fig.savefig(path_save)
    #     elif return_im:
    #         return im

    

    @staticmethod
    def contour_plot(hdr_main, data_main, hdr_contour, data_contour, path_save=None, show=True, levels=None,
                     ax=None, fig=None, norm=None, show_xlabel=True, show_ylabel=True, plot_colorbar=True,
                     header_coordinates_plot=None, cmap="plasma", return_grid=False, aspect=1):
        if header_coordinates_plot is None:
            longitude_main, latitude_main = AlignEUIUtil.extract_EUI_coordinates(hdr_contour, dsun=False, lon_ctype=hdr_contour["CTYPE1"], lat_ctype=hdr_contour["CTYPE2"])
        else:
            longitude_main, latitude_main = AlignEUIUtil.extract_EUI_coordinates(header_coordinates_plot, dsun=False, 
                                                                                  lon_ctype=header_coordinates_plot["CTYPE1"], lat_ctype=header_coordinates_plot["CTYPE2"])

        longitude_grid, latitude_grid, dlon, dlat = PlotFits.build_regular_grid(longitude=longitude_main,
                                                                                latitude=latitude_main)

        use_sunpy = False
        for mapping in [WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS]:
            if mapping[-1][0].__module__ == 'sunpy.coordinates.wcs_utils':
                use_sunpy = True

        w_xy_main = WCS(hdr_main)
        if use_sunpy:
            idx_lon = np.where(np.array(w_xy_main.wcs.ctype, dtype="str") == hdr_main["CTYPE1"])[0][0]
            idx_lat = np.where(np.array(w_xy_main.wcs.ctype, dtype="str") == hdr_main["CTYPE2"])[0][0]
            x, y = np.meshgrid(np.arange(w_xy_main.pixel_shape[idx_lon]),
                               np.arange(w_xy_main.pixel_shape[idx_lat]), )  # t dépend de x,
            # should reproject on a new coordinate grid first : suppose slits at the same time :
            coords_main = w_xy_main.pixel_to_world(x, y)
            coords_grid = SkyCoord(longitude_grid, latitude_grid, frame=coords_main)
            x_small, y_small = w_xy_main.world_to_pixel(coords_grid)

        else:
            x_small, y_small = w_xy_main.world_to_pixel(longitude_grid, latitude_grid)
        image_main_cut = interpol2d(np.array(data_main,
                                             dtype=np.float64), x=x_small, y=y_small,
                                    order=1, fill=-32768)
        image_main_cut[image_main_cut == -32768] = np.nan

        w_xy_contour = WCS(hdr_contour)
        if use_sunpy:
            x_contour, y_contour = w_xy_contour.world_to_pixel(coords_grid)

        else:
            x_contour, y_contour = w_xy_contour.world_to_pixel(longitude_grid, latitude_grid)
        image_contour_cut = interpol2d(np.array(data_contour, dtype=np.float64),
                                       x=x_contour, y=y_contour,
                                       order=1, fill=-32768)
        image_contour_cut[image_contour_cut == -32768] = np.nan
        if longitude_grid.unit == "deg":
            longitude_grid_arc =longitude_grid.to("arcsec").value
            latitude_grid_arc = latitude_grid.to("arcsec").value        
        else:
            longitude_grid_arc = AlignCommonUtil.ang2pipi(longitude_grid).to("arcsec").value
            latitude_grid_arc = AlignCommonUtil.ang2pipi(latitude_grid).to("arcsec").value
        dlon = dlon.to("arcsec").value
        dlat = dlat.to("arcsec").value
        return_im = True
        if fig is None:
            fig = plt.figure()
            return_im = False
        if ax is None:
            ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        im = ax.imshow(image_main_cut, origin="lower", interpolation="none", norm=norm, cmap=cmap, aspect=aspect,
                       extent=(longitude_grid_arc[0, 0] - 0.5 * dlon, longitude_grid_arc[-1, -1] + 0.5 * dlon,
                               latitude_grid_arc[0, 0] - 0.5 * dlat, latitude_grid_arc[-1, -1] + 0.5 * dlat))

        if levels is None:
            max_small = np.nanmax(image_contour_cut)
            levels = [0.5 * max_small]
        ax.contour(image_contour_cut, levels=levels, origin='lower', linewidths=0.5, colors='w',
                   extent=[longitude_grid_arc[0, 0] - 0.5 * dlon, longitude_grid_arc[-1, -1] + 0.5 * dlon,
                           latitude_grid_arc[0, 0] - 0.5 * dlat, latitude_grid_arc[-1, -1] + 0.5 * dlat])
        if show_xlabel:
            ax.set_xlabel("Solar-X [arcsec]")
        if show_ylabel:
            ax.set_ylabel("Solar-Y [arcsec]")
        if plot_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            if "BUNIT" in hdr_main:

                fig.colorbar(im, cax=cax, label=hdr_main["BUNIT"])
            else:
                fig.colorbar(im, cax=cax)
        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save)
        if return_im & return_grid:
            return im, longitude_grid, latitude_grid
        elif return_im:
            return im

    @staticmethod
    def compare_plot(hdr_main, data_main, hdr_contour_1, data_contour_1,
                     hdr_contour_2, data_contour_2, norm, norm_contour=None, path_save=None,
                     cmap1="plasma", cmap2="viridis", show=True,
                     levels=None, fig=None, gs=None, ax1=None, ax2=None, ax3=None, aspect=1, return_axes=False,
                     lmin=None, lmax=None):
        cm = 1 / 2.54  # centimeters in inches
        use_sunpy = False
        for mapping in [WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS]:
            if mapping[-1][0].__module__ == 'sunpy.coordinates.wcs_utils':
                use_sunpy = True

        if (norm.vmin is None) or (norm.vmax is None):
            raise ValueError("Must explicit vmin and vmax in norm, so that the cbar is the same for both figures.")
        if fig is None:
            fig = plt.figure(figsize=(12, 6))

        #

        # gs = GridSpec(1, 5, width_ratios=[1, 1, 0.2, 1, 0.2], wspace=0.5)
        gs = GridSpec(1, 5, width_ratios=[1, 1, 0.1, 1, 0.1], wspace=0.1)

        if ax1 is None:
            ax1 = fig.add_subplot(gs[0])
        if ax2 is None:
            ax2 = fig.add_subplot(gs[1])
        if ax3 is None:
            ax3 = fig.add_subplot(gs[3])

        # ax_cbar1 = fig.add_axes(
        #     [ax2.get_position().x1 + 0.013, ax2.get_position().y0 + 0.15 * ax2.get_position().height,
        #      0.01, ax2.get_position().height * 0.7])
        # ax_cbar2 = fig.add_axes(
        #     [ax3.get_position().x1 + 0.013, ax3.get_position().y0 + 0.15 * ax3.get_position().height,
        #      0.01, ax3.get_position().height * 0.7])

        im = PlotFunctions.contour_plot(hdr_main=hdr_main, data_main=data_main, plot_colorbar=False, aspect=aspect,
                                        hdr_contour=hdr_contour_1, data_contour=data_contour_1, cmap=cmap1,
                                        path_save=None, show=False, levels=levels, fig=fig, ax=ax1, norm=norm, )

        im, lon_grid, lat_grid = \
            PlotFunctions.contour_plot(hdr_main=hdr_main, data_main=data_main, show_ylabel=False,
                                       plot_colorbar=False, aspect=aspect,
                                       hdr_contour=hdr_contour_2, data_contour=data_contour_2,
                                       cmap=cmap1,
                                       path_save=None, show=False, levels=levels, fig=fig, ax=ax2,
                                       norm=norm,
                                       header_coordinates_plot=hdr_contour_1, return_grid=True)

        # b = np.logical_or(lat_ < lmin - 25, lat_ > lmax + 25)

        if norm_contour is None:
            # isnan = np.isnan(data_contour_2)
            # min = np.percentile(data_contour_2[~isnan], 5)
            # max = np.percentile(data_contour_2[~isnan], 98)
            # norm_contour = ImageNormalize(stretch=LinearStretch(), vmin=min, vmax=max)
            norm_contour = PlotFits.get_range(data_contour_2, imin=3, imax=97, stre=None)
        if lon_grid.unit == "deg":
            longitude_grid_arc =lon_grid.to("arcsec").value
            latitude_grid_arc = lat_grid.to("arcsec").value
        else:
            longitude_grid_arc = AlignCommonUtil.ang2pipi(lon_grid).to("arcsec").value
            latitude_grid_arc = AlignCommonUtil.ang2pipi(lat_grid).to("arcsec").value
        dlon = longitude_grid_arc[1, 1] - longitude_grid_arc[0, 0]
        dlat = latitude_grid_arc[1, 1] - latitude_grid_arc[0, 0]

        w_xy = WCS(hdr_contour_2)
        if use_sunpy:
            idx_lon = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
            idx_lat = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
            x, y = np.meshgrid(np.arange(w_xy.pixel_shape[idx_lon]),
                               np.arange(w_xy.pixel_shape[idx_lat]), )  # t dépend de x,
            # should reproject on a new coordinate grid first : suppose slits at the same time :
            coords_ = w_xy.pixel_to_world(x, y)
            coords = SkyCoord(lon_grid, lat_grid, frame=coords_.frame)
            x, y = w_xy.world_to_pixel(coords)

        else:

            x, y = w_xy.world_to_pixel(lon_grid, lat_grid)
        data_contour_2_interp = AlignCommonUtil.interpol2d(data_contour_2, x=x, y=y, order=1, fill=-32762)
        data_contour_2_interp = np.where(data_contour_2_interp == -32762, np.nan, data_contour_2_interp)
        im3 = ax3.imshow(data_contour_2_interp, origin="lower", interpolation="none", norm=norm_contour, cmap=cmap2,
                         aspect=aspect,
                         extent=[longitude_grid_arc[0, 0] - 0.5 * dlon, longitude_grid_arc[-1, -1] + 0.5 * dlon,
                                 latitude_grid_arc[0, 0] - 0.5 * dlat, latitude_grid_arc[-1, -1] + 0.5 * dlat])
        ax3.set_xlabel("Solar-X [arcsec]")
        ax3.set_ylabel("Solar-Y [arcsec]")

        ax_cbar1 = fig.add_axes(
            [ax2.get_position().x1 + 0.013, ax2.get_position().y0,
             0.01, ax2.get_position().height])
        ax_cbar2 = fig.add_axes(
            [ax3.get_position().x1 + 0.013, ax3.get_position().y0,
             0.01, ax3.get_position().height])

        ax3.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        if "BUNIT" in hdr_main:
            cbar = fig.colorbar(im, cax=ax_cbar1, label=hdr_main["BUNIT"], pad=-3,
                                fraction=0.001)  # , fraction=0.046, pad=0.04
        else:
            cbar = fig.colorbar(im, cax=ax_cbar1, label="unknown units", pad=-3, fraction=0.001)
        cbar.formatter.set_powerlimits((0, 0))

        if "BUNIT" in hdr_contour_2:
            cbar3 = fig.colorbar(im3, cax=ax_cbar2, label=hdr_contour_2["BUNIT"], pad=-0.5, fraction=0.001)

        else:
            cbar3 = fig.colorbar(im3, cax=ax_cbar2, label="unkown", pad=-0.5, fraction=0.001)
        cbar3.formatter.set_powerlimits((0, 0))

        ax1.set_title("(a) Before alignment")
        ax2.set_title("(b) After alignment")
        ax3.set_title("(c) Aligned image")
        if lmin is not None:
            ax1.set_ylim([lmin - 20, lmax + 20])
            ax2.set_ylim([lmin - 20, lmax + 20])
            ax3.set_ylim([lmin - 20, lmax + 20])

        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save, bbox_inches='tight')
        if return_axes:
            return fig, ax1, ax2, ax3, ax_cbar1, ax_cbar2

    @staticmethod
    def plot_co_alignment(reference_image_path: str, image_to_align_path: str,
                          reference_image_window: int | str, image_to_align_window: int | str,
                          corr: np.array = None,
                          path_save_figure: str = None,
                          lag_crval1: np.array = None, lag_crval2: np.array = None,
                          lag_crota: np.array = None, lag_cdelt1: np.array = None,
                          lag_cdelt2: np.array = None,
                          levels_percentile: list | None = None,
                          show: bool = False,
                          type_plot: str = "compare_plot",
                          wavelength_interval_to_sum: list[u.Quantity] | str = "all",
                          sub_fov_window: list[u.Quantity] | str = "all",
                          rsun: u.Quantity=1.004*astropy.constants.R_sun, 
                          small_fov_value_min: float = None,
                          small_fov_value_max: float = None,
                          shift_arcsec: list = None,
                          norm_type=None, imin=2, imax=97,
                          unit_to_plot="arcsec",
                          ) -> None:
        """
        plot and save figure comparing the reference image and the image to align before and after the pointing
        correction is applied.



        :param reference_image_path: path to the FITS file of the reference image. Must end with ".fits".
        :param image_to_align_path: path to the FITS file of the image which must be aligned. Must end with ".fits".
        :param corr: correlation array resulting from the Alignment class.
        :param reference_image_window: chosen window of the reference image HDULIST.
        :param image_to_align_window: Chosen window of the image to align HDLIST
        :param path_save_figure: path where the figure will be saved. Must end with ".pdf" (advised) or ".png".
        :param lag_crval1: shift array for CRVAL1  [arcsec].
        :param lag_crval2: shift array for CRVAL2  [arcsec].
        :param lag_crota: shift array for CROTA [degree].
        :param lag_cdelt1: shift array for cdelt1  [arcsec].
        :param lag_cdelt2: shift array for cdelt2  [arcsec].
        :param levels_percentile: percentiles of the contours to be plotted for the to align figure.
        :param show: True to plt.show() figure.
        :param type_plot: "compare_plot" (default) or "successive_plots" or "sunpy"
        :param sub_fov_window: for SPICE only. if "all", select the entire SPICE window. Else enter a list of the form
        [lon_min * u.arcsec, lon_max * u.arcsec, lat_min * u.arcsec, lat_max * u.arcsec].
        :param wavelength_interval_to_sum: has the form [wave_min * u.angstrom, wave_max * u.angstrom].
        for the given SPICE window, set the wavelength interval over which
        the sum is performed, to obtain image (X, Y) from the SPICE L2 data (X, Y, lambda).
        Default is "all" for the entire window.
        :param small_fov_value_min: add a minimal threshold on the absolute values of the to align image
        :param small_fov_value_max: add a maximal threshold on the absolute values of the to align image
        :param shift_arcsec: param for AlignmentResults class
        :param unit_to_plot: param for AlignmentResults class

        """
        if levels_percentile is None:
            levels_percentile = [85]

        # Check if sunpy is used, and if the helioprojective frame is loaded. If true, will change the output format of
        # wcs.pixel_to_world()
        use_sunpy = False
        for mapping in [WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS]:
            if mapping[-1][0].__module__ == 'sunpy.coordinates.wcs_utils':
                use_sunpy = True

        # Load and prepare all the data.
        if shift_arcsec is None:
            max_index = np.unravel_index(np.nanargmax(corr), corr.shape)
        with fits.open(reference_image_path) as hdul_reference:
            header_reference = hdul_reference[reference_image_window].header.copy()
            data_reference = hdul_reference[reference_image_window].data.copy()
            with fits.open(image_to_align_path) as hdul_to_align:
                header_to_align_original = hdul_to_align[image_to_align_window].header.copy()

                if "HRI_EUV" in header_to_align_original["TELESCOP"]:
                    # AlignEUIUtil.recenter_crpix_in_header(header_spice)
                    w_xy = WCS(header_to_align_original)
                    header_to_align = w_xy.to_header().copy()
                    data_to_align = np.array(hdul_to_align[image_to_align_window].data.copy(), dtype=float)

                elif "SPICE" in header_to_align_original["TELESCOP"]:
                    # AlignSpiceUtil.recenter_crpix_in_header_L2(header_spice)
                    w_to_align = WCS(header_to_align_original)
                    w_wave = w_to_align.sub(['spectral'])

                    ymin, ymax = AlignSpiceUtil.vertical_edges_limits(header_to_align_original)
                    w_xyt = w_to_align.dropaxis(2)

                    w_xyt.wcs.pc[2, 0] = 0
                    w_xy = w_xyt.dropaxis(2)
                    header_to_align = w_xy.to_header().copy()

                    data_to_align_tmp = np.array(hdul_to_align[image_to_align_window].data.copy(), dtype=float)
                    data_to_align_tmp[:, :, :ymin, :] = np.nan
                    data_to_align_tmp[:, :, ymax:, :] = np.nan

                    if wavelength_interval_to_sum is "all":
                        data_to_align = np.nansum(data_to_align_tmp[0, :, :, :], axis=0)
                    elif type(wavelength_interval_to_sum).__name__ == "list":
                        z = np.arange(data_to_align_tmp.shape[1])
                        wave = w_wave.pixel_to_world(z)
                        selection_wave = np.logical_and(wave >= wavelength_interval_to_sum[0],
                                                        wave <= wavelength_interval_to_sum[1])
                        data_to_align = np.nansum(data_to_align_tmp[0, selection_wave, :, :], axis=0)
                    else:
                        raise ValueError(
                            "wavelength_interval_to_sum must be a [wave_min * u.angstrom, wave_max * u.angstrom] "
                            "or 'all' str ")

                    idx_lon = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
                    idx_lat = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]

                    if sub_fov_window == "all":
                        pass
                    elif type(sub_fov_window).__name__ == "list":

                        x, y = np.meshgrid(np.arange(w_xy.pixel_shape[idx_lon]),
                                           np.arange(w_xy.pixel_shape[idx_lat]))

                        if use_sunpy:
                            coords_spice = w_xy.pixel_to_world(x, y)
                            lon_spice = coords_spice.Tx
                            lat_spice = coords_spice.Ty
                        else:
                            lon_spice, lat_spice = w_xy.pixel_to_world(x, y)

                        selection_subfov_lon = np.logical_and(lon_spice >= sub_fov_window[0],
                                                              lon_spice <= sub_fov_window[1], )
                        selection_subfov_lat = np.logical_and(lat_spice >= sub_fov_window[2],
                                                              lat_spice <= sub_fov_window[3], )

                        selection_subfov = np.logical_and(selection_subfov_lon, selection_subfov_lat)
                        data_to_align[~selection_subfov] = np.nan
                    else:
                        raise ValueError("sub_fov_window must be a [lon_min * u.arcsec, lon_max * u.arcsec,"
                                         " lat_min * u.arcsec, lat_max * u.arcsec] "
                                         "or 'all' str ")

                    data_to_align[:ymin, :] = np.nan
                    data_to_align[ymax:, :] = np.nan

                    # if spice_cut_from_center is not None:
                    #
                    #     xlen = spice_cut_from_center
                    #     xmid = data_to_align.shape[1] // 2
                    #     data_to_align[:, :(xmid - xlen // 2 - 1)] = np.nan
                    #     data_to_align[:, (xmid + xlen // 2):] = np.nan
                else:
                    warnings.warn("Instrument to align not recognised")
                    # Here we do the code for unrecognised imager
                    w_xy = WCS(header_to_align_original)
                    header_to_align = w_xy.to_header().copy()
                    data_to_align = np.array(hdul_to_align[image_to_align_window].data.copy(), dtype=float)

                condition_1 = np.zeros(data_to_align.shape, dtype='bool')
                condition_2 = np.zeros(data_to_align.shape, dtype='bool')

                if small_fov_value_min is not None:
                    condition_1 = np.array(np.abs(data_to_align) <= small_fov_value_min, dtype='bool')
                if small_fov_value_max is not None:
                    condition_2 = np.array(np.abs(data_to_align) >= small_fov_value_max, dtype='bool')
                set_to_nan = np.logical_or(condition_1,condition_2)
                data_to_align[set_to_nan] = np.nan
                
                header_to_align["NAXIS1"] = data_to_align.shape[1]
                header_to_align["NAXIS2"] = data_to_align.shape[0]
                data_to_align_ravel = data_to_align.flatten()
                not_nan = np.logical_not(np.isnan(data_to_align_ravel))
                levels = [np.percentile(data_to_align_ravel[not_nan], n) for n in levels_percentile]

                header_to_align_shifted = header_to_align.copy()
                parameter_alignment_values = None
                if shift_arcsec is None:
                    if lag_crval1 is None:
                        lag_crval1 = np.array([0])
                    if lag_crval2 is None:
                        lag_crval2 = np.array([0])
                    if lag_crota is None:
                        lag_crota = np.array([0])
                    if lag_cdelt1 is None:
                        lag_cdelt1 = np.array([0])
                    if lag_cdelt2 is None:
                        lag_cdelt2 = np.array([0])

                    parameter_alignment_values = {
                        "lag_crval1": lag_crval1[max_index[0]],
                        "lag_crval2": lag_crval2[max_index[1]],
                        "lag_crota": lag_crota[max_index[4]],
                        "lag_cdelt1": lag_cdelt1[max_index[2]],
                        "lag_cdelt2": lag_cdelt2[max_index[3]],
                    }
                else:
                    parameter_alignment_values = {
                        "lag_crval1": shift_arcsec[0],
                        "lag_crval2": shift_arcsec[1],
                        "lag_crota": shift_arcsec[4],
                        "lag_cdelt1": shift_arcsec[2],
                        "lag_cdelt2": shift_arcsec[3],
                    }

                AlignCommonUtil.correct_pointing_header(header=header_to_align_shifted,
                                                        **parameter_alignment_values)

                norm = PlotFits.get_range(data=data_reference, stre=norm_type, imin=imin, imax=imax)
                longitude, latitude = AlignEUIUtil.extract_EUI_coordinates(header_to_align.copy(), dsun=False,
                                                                        lon_ctype=header_to_align["CTYPE1"],  lat_ctype=header_to_align["CTYPE2"])
                longitude_grid, latitude_grid, dlon, dlat = PlotFits.build_regular_grid(longitude, latitude)
                dlon = dlon.to("arcsec").value
                dlat = dlat.to("arcsec").value

                lmin = None
                lmax = None
                norm_contour = PlotFits.get_range(data=data_to_align, stre=norm_type, imin=imin, imax=imax)

                if "SPICE" in header_to_align_original["TELESCOP"]:
                    lmin = AlignCommonUtil.ang2pipi(latitude).to("arcsec").value[ymin, 0]
                    lmax = AlignCommonUtil.ang2pipi(latitude).to("arcsec").value[ymax, 0]

                if type_plot == "compare_plot":
                    fig = plt.figure(figsize=(12, 6))
                    fig, ax1, ax2, ax3, ax_cbar1, ax_cbar2 = \
                        PlotFunctions.compare_plot(header_reference, data_reference, header_to_align, data_to_align,
                                                   header_to_align_shifted, data_to_align,
                                                   show=False, norm=norm, levels=levels, return_axes=True,
                                                   fig=fig, lmin=lmin, lmax=lmax,norm_contour=norm_contour,
                                                   cmap1="plasma", cmap2="viridis", path_save=None)
                    if "DETECTOR" in header_reference.keys():
                        detector = header_reference["DETECTOR"]
                    else :
                        detector = " UNKOWN"
                    if "WAVELNTH" in header_reference.keys():
                        wave = header_reference["WAVELNTH"]
                    else:
                        wave = "UNKNOWN"

                    ax1.set_title(f"{detector} {wave} & Small FOV (contour) NA ")
                    ax2.set_title(f"{detector} {wave} & Small FOV (contour) A ")
                    ax2.set_yticklabels([])
                    ax3.set_yticklabels([])

                    ax3.set_title("Small FOV (%s) aligned " % image_to_align_window)
                    date = Time(hdul_to_align[image_to_align_window].header["DATE-AVG"]).fits[:19]
                    date = date.replace(":", "_")
                    date = date.replace("-", "_")

                    date_str = header_to_align["DATE-OBS"][:19]
                    fig.suptitle(
                        f"Image to align  {date_str} aligned with {detector} {wave}. Aligned (A) ; Not Aligned (NA) ; ")
                    # fig.suptitle("Alignement of SPICE  using a synthetic raster of HRIEUV images")
                    if path_save_figure is not None:
                        fig.savefig(path_save_figure)
                    if show:
                        fig.show()

                elif type_plot == "successive_plot":
                    with PdfPages(path_save_figure) as pdf:

                        for data, header, title, norm_ in zip([data_reference, data_to_align, data_to_align],
                                                       [header_reference, header_to_align_shifted, header_to_align],
                                                       ["Reference image", "to align image shifted",
                                                        "to align not Shifted"], 
                                                        [norm, norm_contour, norm_contour]):
                            w_ = WCS(header)
                            if use_sunpy:
                                x, y = np.meshgrid(np.arange(1), np.arange(1))
                                coords_tmp = w_.pixel_to_world(x, y)
                                coords = SkyCoord(longitude_grid, latitude_grid, frame=coords_tmp.frame)
                                x, y = w_.world_to_pixel(coords)

                            else:
                                x, y = w_.world_to_pixel(longitude_grid, latitude_grid)

                            data_rep = AlignCommonUtil.interpol2d(image=data, x=x, y=y, fill=np.nan,
                                                                  order=3, )
                            fig = plt.figure(figsize=(6, 6))
                            ax = fig.add_subplot()
                            PlotFunctions.simple_plot(hdr_main=header_to_align, data_main=data_rep, fig=fig, ax=ax, norm=norm_)
                            ax.set_title(title)
                            pdf.savefig(fig)
    
                elif type_plot == "sunpy":
                    from sunpy.map import Map

                    rsun=rsun.to("m").value
                    with PdfPages(path_save_figure) as pdf:
                        header_to_align_ = copy.deepcopy(header_to_align)
                        header_to_align_["RSUN_REF"] = rsun
                        w_to_align = WCS(header_to_align_)
                        
                        
                        for data, header, title in zip([data_reference, data_to_align, data_to_align],
                                                       [header_reference, header_to_align_shifted, header_to_align],
                                                       ["Reference image", "to align image shifted",
                                                        "to align not Shifted"], ):
                            hdu = fits.PrimaryHDU(data=data, header=header)
                            hdu.verify('fix')
                            norm = PlotFits.get_range(data, stre=norm_type, imin=imin, imax=imax)
                            cmap = "viridis"                            
                            if "TELESCOP" in header:
                                if ("PHI" in header["TELESCOP"]) or ("HMI" in header["TELESCOP"]):
                                    isnan = np.isnan(data)
                                    p = np.percentile(np.abs(data[np.logical_not(isnan)]), 97)
                                    norm = CenteredNorm(0, halfrange=p)
                                    cmap = 'Greys'
                                # else:
                                #     norm = PlotFits.get_range(data, stre=None)
                                #     cmap = "viridis"
                            fig = plt.figure(figsize=(6, 6))
                            m = Map(hdu.data, hdu.header)
                            m.meta["rsun_ref"] = rsun
                            with propagate_with_solar_surface():
                                m_rep = m.reproject_to(w_to_align)
                            ax = fig.add_subplot(projection=m_rep)
                            PlotFunctions.simple_plot_sunpy(m_main=m_rep, fig=fig, ax=ax, norm=norm, cmap=cmap)
                            ax.set_title(title)
                            pdf.savefig(fig)
    
    #
    #
    # @staticmethod
    # def plot_co_alignment_old(large_fov_window, large_fov_path: str, corr: np.array,
    #                           small_fov_window, small_fov_path: str, levels_percentile=None,
    #                           lag_crval1=None, lag_crval2=None, lag_crota=None, lag_cdelt1=None, lag_cdelt2=None,
    #                           show=False, results_folder=None, cut_from_center=None, plot_all_figures=False
    #                           ):
    #     """
    #
    #     :param large_fov_window: path to the file to co-align. Must end with ".fits"
    #     :param large_fov_path: path to the reference file. Must end with ".fits"
    #     :param corr: correlation array resulting from the Alignment class
    #     :param small_fov_window: window of the fits file to align
    #     :param small_fov_path: window of the reference fits fils
    #     :param levels_percentile:
    #     :param lag_crval1: shift array for CRVAL1  [arcsec]
    #     :param lag_crval2: shift array for CRVAL2 [arcsce]
    #     :param lag_crota: shift array for CROTA [degree]
    #     :param lag_cdelt1: shift array for cdelt1  [arcsec]
    #     :param lag_cdelt2: shift array for cdelt2  [arcsec]
    #     :param show: if True, then show the figure
    #     :param results_folder: path where to save the figures
    #     :param cut_from_center: for spice only: cut the dumbells in the figure.
    #     :param plot_all_figures: if True, plot individual figures of the reference and to align images.
    #     """
    #     if levels_percentile is None:
    #         levels_percentile = [85]
    #     use_sunpy = False
    #     for mapping in [WCS_FRAME_MAPPINGS, FRAME_WCS_MAPPINGS]:
    #         if mapping[-1][0].__module__ == 'sunpy.coordinates.wcs_utils':
    #             use_sunpy = True
    #
    #     parameter_alignment = {
    #         "crval1": lag_crval1,
    #         "crval2": lag_crval2,
    #         "crota": lag_crota,
    #         "cdelt1": lag_cdelt1,
    #         "cdelt2": lag_cdelt2,
    #
    #     }
    #
    #     max_index = np.unravel_index(np.nanargmax(corr), corr.shape)
    #
    #     with fits.open(large_fov_path) as hdul_large:
    #         header_large = hdul_large[large_fov_window].header.copy()
    #         data_large = hdul_large[large_fov_window].data.copy()
    #         with fits.open(small_fov_path) as hdul_spice:
    #             header_spice_original = hdul_spice[small_fov_window].header.copy()
    #
    #             if "HRI_EUV" in header_spice_original["TELESCOP"]:
    #                 # AlignEUIUtil.recenter_crpix_in_header(header_spice)
    #                 w_xy = WCS(header_spice_original)
    #                 header_spice = w_xy.to_header().copy()
    #                 data_spice = np.array(hdul_spice[small_fov_window].data.copy(), dtype=np.float64)
    #             elif "SPICE" in header_spice_original["TELESCOP"]:
    #                 # AlignSpiceUtil.recenter_crpix_in_header_L2(header_spice)
    #                 w_spice = WCS(header_spice_original)
    #                 ymin, ymax = AlignSpiceUtil.vertical_edges_limits(header_spice_original)
    #                 w_xyt = w_spice.dropaxis(2)
    #                 w_xyt.wcs.pc[2, 0] = 0
    #                 w_xy = w_xyt.dropaxis(2)
    #                 header_spice = w_xy.to_header().copy()
    #
    #                 data_small = np.array(hdul_spice[small_fov_window].data.copy(), dtype=np.float64)
    #                 data_small[:, :, :ymin, :] = np.nan
    #                 data_small[:, :, ymax:, :] = np.nan
    #
    #                 data_spice = np.nansum(data_small[0, :, :, :], axis=0)
    #                 data_spice[:ymin, :] = np.nan
    #                 data_spice[ymax:, :] = np.nan
    #
    #                 if cut_from_center is not None:
    #
    #                     if cut_from_center is not None:
    #                         xlen = cut_from_center
    #                         xmid = data_spice.shape[1] // 2
    #                         data_spice[:, :(xmid - xlen // 2 - 1)] = np.nan
    #                         data_spice[:, (xmid + xlen // 2):] = np.nan
    #             else:
    #                 warnings.warn("Imager not recognised")
    #                 # Here we do the code for unrecognised imager
    #                 w_xy = WCS(header_spice_original)
    #                 header_spice = w_xy.to_header().copy()
    #                 data_spice = np.array(hdul_spice[small_fov_window].data.copy(), dtype=np.float64)
    #
    #                 # header_spice["CRPIX1"] = (data_spice.shape[1] + 1) / 2
    #                 # header_spice["CRPIX2"] = (data_spice.shape[0] + 1) / 2
    #
    #             # header_spice["SOLAR_B0"] = hdul_spice[small_fov_window].header["SOLAR_B0"]
    #             # header_spice["RSUN_REF"] = hdul_spice[small_fov_window].header["RSUN_REF"]
    #             # header_spice["DSUN_OBS"] = hdul_spice[small_fov_window].header["DSUN_OBS"]
    #             # data_spice = np.nansum(hdul_spice[raster_window].data.copy()[0, :, :, :], axis=0)
    #
    #             header_spice["NAXIS1"] = data_spice.shape[1]
    #             header_spice["NAXIS2"] = data_spice.shape[0]
    #             # AlignEUIUtil.recenter_crpix_in_header(header_spice)
    #             not_nan = np.isnan(data_spice)
    #             levels = [np.percentile(data_spice[~not_nan], n) for n in levels_percentile]
    #
    #             hdr_spice_shifted = header_spice.copy()
    #             if header_spice_original["PC1_1"] == 1.0:
    #                 for i, j in zip([1, 1, 2, 2], [1, 2, 1, 2]):
    #                     hdr_spice_shifted[f'PC{i}_{j}'] = header_spice_original[f'PC{i}_{j}']
    #
    #             hdr_spice_shifted["CRVAL1"] = hdr_spice_shifted["CRVAL1"] \
    #                                           + u.Quantity(parameter_alignment['crval1'][max_index[0]], "arcsec").to(
    #                 hdr_spice_shifted["CUNIT1"]).value
    #             hdr_spice_shifted["CRVAL2"] = hdr_spice_shifted["CRVAL2"] \
    #                                           + u.Quantity(parameter_alignment['crval2'][max_index[1]], "arcsec").to(
    #                 hdr_spice_shifted["CUNIT2"]).value
    #             change_pcij = False
    #
    #             if hdr_spice_shifted["PC1_1"] > 1.0:
    #                 warnings.warn(f'{hdr_spice_shifted["PC1_1"]=}, set it to 1.0')
    #                 hdr_spice_shifted["PC1_1"] = 1.0
    #                 hdr_spice_shifted["PC2_2"] = 1.0
    #                 hdr_spice_shifted["PC1_2"] = 0.0
    #                 hdr_spice_shifted["PC2_1"] = 0.0
    #                 hdr_spice_shifted["CROTA"] = 0.0
    #
    #             key_rota = None
    #             if "CROTA" in hdr_spice_shifted:
    #                 key_rota = "CROTA"
    #             elif "CROTA2" in hdr_spice_shifted:
    #                 key_rota = "CROTA2"
    #
    #             crota = np.rad2deg(np.arccos(copy.deepcopy(hdr_spice_shifted["PC1_1"])))
    #
    #             if parameter_alignment['crota'] is not None:
    #                 # hdr_spice_shifted["CROTA"] = hdul_spice[raster_window].header["CROTA"] +\
    #                 #                              parameter_alignement['crota'][max_index[4]]
    #                 if key_rota is None:
    #                     hdr_spice_shifted["CROTA"] = np.rad2deg(np.arccos(copy.deepcopy(hdr_spice_shifted["PC1_1"])))
    #
    #                     key_rota = "CROTA"
    #                 hdr_spice_shifted[key_rota] += parameter_alignment['crota'][max_index[4]]
    #                 crota += parameter_alignment['crota'][max_index[4]]
    #                 change_pcij = True
    #
    #             if parameter_alignment['cdelt1'] is not None:
    #                 cdelt1 = u.Quantity(hdr_spice_shifted["CDELT1"], hdr_spice_shifted["CUNIT1"]) + \
    #                          u.Quantity(parameter_alignment['cdelt1'][max_index[2]], "arcsec")
    #                 hdr_spice_shifted["CDELT1"] = cdelt1.to(hdr_spice_shifted["CUNIT1"]).value
    #                 change_pcij = True
    #
    #             if parameter_alignment['cdelt2'] is not None:
    #                 cdelt2 = u.Quantity(hdr_spice_shifted["CDELT2"], hdr_spice_shifted["CUNIT2"]) + \
    #                          u.Quantity(parameter_alignment['cdelt2'][max_index[3]], "arcsec")
    #                 hdr_spice_shifted["CDELT2"] = cdelt2.to(hdr_spice_shifted["CUNIT2"]).value
    #                 change_pcij = True
    #             if change_pcij:
    #                 s = - np.sign(hdr_spice_shifted["PC1_2"]) + (hdr_spice_shifted["PC1_2"] == 0.0)
    #                 theta = np.deg2rad(crota) * s
    #                 lam = hdr_spice_shifted["CDELT2"] / hdr_spice_shifted["CDELT1"]
    #                 hdr_spice_shifted["PC1_1"] = np.cos(theta)
    #                 hdr_spice_shifted["PC2_2"] = np.cos(theta)
    #                 hdr_spice_shifted["PC1_2"] = -lam * np.sin(theta)
    #                 hdr_spice_shifted["PC2_1"] = (1 / lam) * np.sin(theta)
    #             not_nan = np.isnan(data_large)
    #             # min = np.percentile(data_large[~not_nan], 3)
    #             # max = np.percentile(data_large[~not_nan], 99)
    #             # norm = ImageNormalize(stretch=LinearStretch(), vmin=np.max((min, 1)), vmax=max)
    #             norm = PlotFits.get_range(data=data_large, stre=None, imin=2, imax=97)
    #             longitude, latitude = AlignEUIUtil.extract_EUI_coordinates(header_spice.copy(), dsun=False)
    #             longitude_grid, latitude_grid, dlon, dlat = PlotFits.build_regular_grid(longitude, latitude)
    #             dlon = dlon.to("arcsec").value
    #             dlat = dlat.to("arcsec").value
    #
    #             data_fsi = hdul_large[large_fov_window].data
    #             header_fsi = hdul_large[large_fov_window].header
    #
    #             # header_spice["BUNIT"] = hdul_spice[small_fov_window].header["BUNIT"]
    #             # hdr_spice_shifted["BUNIT"] = hdul_spice[small_fov_window].header["BUNIT"]
    #             # header_spice["DATE-AVG"] = hdul_spice[small_fov_window].header["DATE-AVG"]
    #             # cm = 1 / 2.54  # centimeters in inches
    #             data_large_cp = copy.deepcopy(data_large)
    #             lmin = None
    #             lmax = None
    #             if "SPICE" in header_spice_original["TELESCOP"]:
    #                 lmin = AlignCommonUtil.ang2pipi(latitude).to("arcsec").value[ymin, 0]
    #                 lmax = AlignCommonUtil.ang2pipi(latitude).to("arcsec").value[ymax, 0]
    #
    #                 # data_large_cp[b] = np.nan
    #                 # data_large_cp[b] = np.nan
    #
    #             fig = plt.figure(figsize=(12, 6))
    #             fig, ax1, ax2, ax3, ax_cbar1, ax_cbar2 = \
    #                 PlotFunctions.compare_plot(header_large, data_large_cp, header_spice, data_spice, hdr_spice_shifted,
    #                                            data_spice, show=False, norm=norm, levels=levels, return_axes=True,
    #                                            fig=fig, lmin=lmin, lmax=lmax,
    #                                            cmap1="plasma", cmap2="viridis", path_save=None)
    #             detector = header_large["DETECTOR"]
    #             wave = header_large["WAVELNTH"]
    #
    #             ax1.set_title(f"{detector} {wave} & Small FOV (contour) NA ")
    #             ax2.set_title(f"{detector} {wave} & Small FOV (contour) A ")
    #             ax2.set_yticklabels([])
    #             ax3.set_yticklabels([])
    #
    #             ax3.set_title("Small FOV (%s) aligned " % small_fov_window)
    #             date = Time(hdul_spice[small_fov_window].header["DATE-AVG"]).fits[:19]
    #             date = date.replace(":", "_")
    #             date = date.replace("-", "_")
    #
    #             date_str = header_spice["DATE-OBS"][:19]
    #             fig.suptitle(f"Small FOV {date_str} aligned with {detector} {wave}. Aligned (A) ; Not Aligned (NA) ; ")
    #             # fig.suptitle("Alignement of SPICE  using a synthetic raster of HRIEUV images")
    #             if results_folder is not None:
    #                 fig.savefig('%s/compare_alignment.pdf' % (results_folder))
    #             if show:
    #                 fig.show()
    #             if plot_all_figures:
    #                 # fig.suptitle(f"Alignement SPICE {date}- HRI 174")
    #                 w_fsi = WCS(header_fsi)
    #                 w_spice = WCS(header_spice)
    #                 w_spice_shift = WCS(hdr_spice_shifted)
    #                 if use_sunpy:
    #                     idx_lon = np.where(np.array(w_fsi.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
    #                     idx_lat = np.where(np.array(w_fsi.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
    #                     x, y = np.meshgrid(np.arange(w_fsi.pixel_shape[idx_lon]),
    #                                        np.arange(w_fsi.pixel_shape[idx_lat]), )  # t dépend de x,
    #
    #                     # should reproject on a new coordinate grid first : suppose slits at the same time :
    #                     coords_fsi = w_fsi.pixel_to_world(x, y)
    #
    #                     idx_lon = np.where(np.array(w_spice.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
    #                     idx_lat = np.where(np.array(w_spice.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
    #                     x, y = np.meshgrid(np.arange(w_spice.pixel_shape[idx_lon]),
    #                                        np.arange(w_spice.pixel_shape[idx_lat]), )
    #                     coords_spice = w_spice.pixel_to_world(x, y)
    #
    #                     # breakpoint()
    #                     coords_grid = SkyCoord(longitude_grid, latitude_grid,
    #                                            frame="helioprojective", observer=coords_fsi.observer)
    #                     x_fsi, y_fsi = w_fsi.world_to_pixel(coords_grid)
    #                     coords_grid = SkyCoord(longitude_grid, latitude_grid,
    #                                            frame="helioprojective", observer=coords_spice.observer)
    #
    #                     x_spice, y_spice = w_spice.world_to_pixel(coords_grid)
    #                     x_spice_shift, y_spice_shift = w_spice_shift.world_to_pixel(coords_grid)
    #
    #                 else:
    #
    #                     x_fsi, y_fsi = w_fsi.world_to_pixel(longitude_grid, latitude_grid)
    #                     x_spice, y_spice = w_spice.world_to_pixel(longitude_grid, latitude_grid)
    #                     x_spice_shift, y_spice_shift = w_spice_shift.world_to_pixel(longitude_grid, latitude_grid)
    #
    #                 data_fsi_interp = AlignCommonUtil.interpol2d(data_fsi, x=x_fsi, y=y_fsi, fill=-32762, order=1)
    #                 data_spice_interp = AlignCommonUtil.interpol2d(data_spice, x=x_spice, y=y_spice, fill=-32762,
    #                                                                order=1)
    #                 data_spice_interp_shift = AlignCommonUtil.interpol2d(data_spice, x=x_spice_shift, y=y_spice_shift,
    #                                                                      fill=-32762,
    #                                                                      order=1)
    #
    #                 data_fsi_interp = np.where(data_fsi_interp == -32762, np.nan, data_fsi_interp)
    #                 data_spice_interp = np.where(data_spice_interp == -32762, np.nan, data_spice_interp)
    #                 data_spice_interp_shift = np.where(data_spice_interp_shift == -32762, np.nan,
    #                                                    data_spice_interp_shift)
    #
    #                 longitude_grid_arc = AlignCommonUtil.ang2pipi(longitude_grid.to("arcsec")).value
    #                 latitude_grid_arc = AlignCommonUtil.ang2pipi(latitude_grid.to("arcsec")).value
    #                 dlon = longitude_grid_arc[0, 1] - longitude_grid_arc[0, 0]
    #                 dlat = latitude_grid_arc[1, 0] - latitude_grid_arc[0, 0]
    #
    #                 isnan = np.isnan(data_spice_interp)
    #                 min = np.percentile(data_spice_interp[~isnan], 3)
    #                 max = np.percentile(data_spice_interp[~isnan], 99)
    #                 norm_spice = ImageNormalize(stretch=LinearStretch(), vmin=min, vmax=max)
    #
    #                 isnan = np.isnan(data_fsi_interp)
    #                 min = np.percentile(data_fsi_interp[~isnan], 3)
    #                 max = np.percentile(data_fsi_interp[~isnan], 99)
    #                 norm_fsi = ImageNormalize(stretch=LinearStretch(), vmin=min, vmax=max)
    #                 fig = plt.figure(figsize=(5, 5))
    #                 ax = fig.add_subplot()
    #                 im = ax.imshow(data_fsi_interp, origin='lower', interpolation='none', cmap='plasma', norm=norm_fsi,
    #                                extent=[longitude_grid_arc[0, 0] - 0.5 * dlon,
    #                                        longitude_grid_arc[-1, -1] + 0.5 * dlon,
    #                                        latitude_grid_arc[0, 0] - 0.5 * dlat,
    #                                        latitude_grid_arc[-1, -1] + 0.5 * dlat])
    #                 ax.set_title(f'{detector} {wave} \n {header_large["DATE-AVG"][:19]}')
    #                 ax.set_xlabel("Solar-X [arcsec]")
    #                 ax.set_ylabel("Solar-Y [arcsec]")
    #                 cbar = fig.colorbar(im, label=header_fsi["BUNIT"])
    #                 if results_folder is not None:
    #                     fig.savefig(os.path.join(results_folder, f"Synthetic_raster_on_grid_{date}.pdf"))
    #                 if show:
    #                     fig.show()
    #                 fig = plt.figure(figsize=(5, 5))
    #                 ax = fig.add_subplot()
    #                 im = ax.imshow(data_spice_interp, origin='lower', interpolation='none', cmap='viridis',
    #                                norm=norm_spice,
    #                                extent=[longitude_grid_arc[0, 0] - 0.5 * dlon,
    #                                        longitude_grid_arc[-1, -1] + 0.5 * dlon,
    #                                        latitude_grid_arc[0, 0] - 0.5 * dlat,
    #                                        latitude_grid_arc[-1, -1] + 0.5 * dlat])
    #
    #                 ax.set_xlabel("Solar-X [arcsec]")
    #                 ax.set_ylabel("Solar-Y [arcsec]")
    #                 ax.set_title("small FOV not aligned (%s) \n %s" % (small_fov_window, header_spice["DATE-OBS"][:19]))
    #                 cbar = fig.colorbar(im, label=header_spice["BUNIT"])
    #                 if results_folder is not None:
    #                     fig.savefig(os.path.join(results_folder, f"small_fov_before_alignment_on_grid_{date}.pdf"))
    #                 if show:
    #                     fig.show()
    #                 fig = plt.figure(figsize=(5, 5))
    #                 ax = fig.add_subplot()
    #                 im = ax.imshow(data_spice_interp_shift, origin='lower', interpolation='none', cmap='viridis',
    #                                norm=norm_spice,
    #                                extent=[longitude_grid_arc[0, 0] - 0.5 * dlon,
    #                                        longitude_grid_arc[-1, -1] + 0.5 * dlon,
    #                                        latitude_grid_arc[0, 0] - 0.5 * dlat,
    #                                        latitude_grid_arc[-1, -1] + 0.5 * dlat])
    #                 ax.set_title("Small FOV aligned (%s) \n %s" % (small_fov_window, header_spice["DATE-OBS"][:19]))
    #                 ax.set_xlabel("Solar-X [arcsec]")
    #                 ax.set_ylabel("Solar-Y [arcsec]")
    #                 cbar = fig.colorbar(im, label=header_spice["BUNIT"], )
    #                 if results_folder is not None:
    #                     fig.savefig(os.path.join(results_folder, f"small_fov_after_alignment_on_grid_{date}.pdf"))
    #                 if show:
    #                     fig.show()
    #                 # fig.savefig(os.path.join(results_folder, "SPICE_after_alignment_on_grid.png"), dpi=150)
    #
    #         hdul_spice.close()
    #     hdul_large.close()
