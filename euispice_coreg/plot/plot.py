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
    def plot_correlation(corr, lag_crval1, lag_crval2, lag_drot=None, lag_cdelta1=None, lag_cdelta2=None,
                         path_save=None, fig=None, ax=None, show=False,
                         lag_dx_label=None, lag_dy_label=None, unit='\'\'', type_plot="xy", ):

        max_index = np.unravel_index(np.nanargmax(corr), corr.shape)

        if type_plot == "xy":
            lag_dx_label = 'CRVAL1 [arcsec]'
            lag_dy_label = 'CRVAL2 [arcsec]'
            corr = corr[:, :, max_index[2], max_index[3], max_index[4]]
        else:
            raise NotImplementedError

        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()
        dy = lag_crval2[1] - lag_crval2[0]
        dx = lag_crval1[1] - lag_crval1[0]
        lag_dx = lag_crval1
        lag_dy = lag_crval2
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
        ax.axhline(y=lag_dy[max_index[1]], color='r', linestyle='--', linewidth=0.5)
        ax.axvline(x=lag_dx[max_index[0]], color='r', linestyle='--', linewidth=0.5)
        if (lag_drot is not None) & (lag_cdelta1 is None):
            textstr = '\n'.join((
                r'$dx=%.1f$ %s' % (lag_dx[max_index[0]], unit),
                r'$dy=%.1f$ %s' % (lag_dy[max_index[1]], unit),
                r'$drota=%.2f$ $^\circ$' % (lag_drot[max_index[4]]),
                r'max_cc = %.2f' % (np.nanmax(corr))
            ))
        elif (lag_drot is not None) & (lag_cdelta1 is not None):
            textstr = '\n'.join((
                r'$dx=%.1f$ %s' % (lag_dx[max_index[0]], unit),
                r'$dy=%.1f$ %s' % (lag_dy[max_index[1]], unit),
                r'$drota=%.2f$ $^\circ$' % (lag_drot[max_index[4]]),
                r'$cdelt1=%.2f$ $^\circ$' % (lag_cdelta1[max_index[2]]),
                r'$cdelt2=%.2f$ $^\circ$' % (lag_cdelta2[max_index[3]]),
                r'max_cc = %.2f' % (np.nanmax(corr))))

        else:
            textstr = '\n'.join((
                r'$\delta CRVAL1=%.2f$ %s' % (lag_dx[max_index[0]], unit),
                r'$\delta CRVAL2=%.2f$ %s' % (lag_dy[max_index[1]], unit),
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
        cbar = fig.colorbar(im, cax=cax, label="correlation" )
        if show:
            fig.show()
        if path_save is not None:
            fig.tight_layout()
            fig.savefig(path_save)

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
                 cmap="plasma", xlabel="X [px]", ylabel="Y [px]", aspect=1, return_im=False):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        if slc is not None:
            im = ax.imshow(data[slc[0], slc[1]], origin="lower", interpolation="none", norm=norm, aspect=aspect,
                           cmap=cmap)
        else:
            im = ax.imshow(data, cmap=cmap, origin="lower", interpolation="None", norm=norm, aspect=aspect)
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
    def simple_plot(hdr_main, data_main, path_save=None, show=False, ax=None, fig=None, norm=None,
                    show_xlabel=True, show_ylabel=True, plot_colorbar=True, cmap="plasma"):

        longitude, latitude, dsun = AlignEUIUtil.extract_EUI_coordinates(hdr_main)
        longitude_grid, latitude_grid = PlotFunctions._build_regular_grid(longitude=longitude, latitude=latitude)
        w = WCS(hdr_main)
        x, y = w.world_to_pixel(longitude_grid, latitude_grid)
        image_on_regular_grid = interpol2d(data_main, x=x, y=y, fill=-32762, order=1)
        image_on_regular_grid[image_on_regular_grid == -32762] = np.nan
        return_im = False
        if fig is None:
            fig = plt.figure()
            return_im = True
        if ax is None:
            ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        im = ax.imshow(image_on_regular_grid, origin="lower", interpolation="none", norm=norm, cmap=cmap,
                       extent=[longitude_grid[0, 0].to(u.arcsec).value, longitude_grid[-1, -1].to(u.arcsec).value,
                               latitude_grid[0, 0].to(u.arcsec).value, latitude_grid[-1, -1].to(u.arcsec).value])
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

    @staticmethod
    def contour_plot(hdr_main, data_main, hdr_contour, data_contour, path_save=None, show=True, levels=None,
                     ax=None, fig=None, norm=None, show_xlabel=True, show_ylabel=True, plot_colorbar=True,
                     header_coordinates_plot=None, cmap="plasma", return_grid=False, aspect=1):
        if header_coordinates_plot is None:
            longitude_main, latitude_main = AlignEUIUtil.extract_EUI_coordinates(hdr_contour, dsun=False)
        else:
            longitude_main, latitude_main = AlignEUIUtil.extract_EUI_coordinates(header_coordinates_plot, dsun=False)

        longitude_grid, latitude_grid = PlotFunctions._build_regular_grid(longitude=longitude_main,
                                                                          latitude=latitude_main)

        w_xy_main = WCS(hdr_main)
        x_small, y_small = w_xy_main.world_to_pixel(longitude_grid, latitude_grid)
        image_main_cut = interpol2d(np.array(data_main,
                                             dtype=np.float64), x=x_small, y=y_small,
                                    order=1, fill=-32768)
        image_main_cut[image_main_cut == -32768] = np.nan

        w_xy_contour = WCS(hdr_contour)
        x_contour, y_contour = w_xy_contour.world_to_pixel(longitude_grid, latitude_grid)
        image_contour_cut = interpol2d(np.array(data_contour, dtype=np.float64),
                                       x=x_contour, y=y_contour,
                                       order=1, fill=-32768)
        image_contour_cut[image_contour_cut == -32768] = np.nan
        longitude_grid_arc = AlignCommonUtil.ang2pipi(longitude_grid).to("arcsec").value
        latitude_grid_arc = AlignCommonUtil.ang2pipi(latitude_grid).to("arcsec").value
        dlon = longitude_grid_arc[1, 1] - longitude_grid_arc[0, 0]
        dlat = latitude_grid_arc[1, 1] - latitude_grid_arc[0, 0]
        return_im = True
        if fig is None:
            fig = plt.figure()
            return_im = False
        if ax is None:
            ax = fig.add_subplot()
        if norm is None:
            norm = ImageNormalize(stretch=LogStretch(5))
        im = ax.imshow(image_main_cut, origin="lower", interpolation="none", norm=norm, cmap=cmap, aspect=aspect,
                       extent=[longitude_grid_arc[0, 0] - 0.5 * dlon, longitude_grid_arc[-1, -1] + 0.5 * dlon,
                               latitude_grid_arc[0, 0] - 0.5 * dlat, latitude_grid_arc[-1, -1] + 0.5 * dlat])

        if levels is None:
            max_small = np.nanmax(image_contour_cut)
            levels = [0.5 * max_small]
        print(f'{levels=}')
        ax.contour(image_contour_cut, levels=levels, origin='lower', linewidths=0.5, colors='w', aspect=aspect,
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

                fig.colorbar(im,cax=cax, label=hdr_main["BUNIT"])
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
                     levels=None, fig=None, gs=None, ax1=None, ax2=None, ax3=None, aspect=1, return_axes=False):

        if (norm.vmin is None) or (norm.vmax is None):
            raise ValueError("Must explicit vmin and vmax in norm, so that the cbar is the same for both figures.")
        if fig is None:
            fig = plt.figure(figsize=(10, 4))


        cm = 1/2.54  # centimeters in inches
        #
        fig1 = plt.figure(figsize=(17*cm, 10*cm))
        gs1 = GridSpec(1, 2, wspace=0.5)

        axs1 = [fig.add_subplot(gs1[n]) for n in range(2)]

        gs = GridSpec(1, 5, width_ratios=[1, 1, 0.2, 1, 0.2], wspace=0.5)
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
                                        path_save=None, show=False, levels=levels, fig=fig, ax=ax1, norm=norm)

        im, lon_grid, lat_grid = \
            PlotFunctions.contour_plot(hdr_main=hdr_main, data_main=data_main, show_ylabel=False,
                                       plot_colorbar=False, aspect=aspect,
                                       hdr_contour=hdr_contour_2, data_contour=data_contour_2,
                                       cmap=cmap1,
                                       path_save=None, show=False, levels=levels, fig=fig, ax=ax2,
                                       norm=norm,
                                       header_coordinates_plot=hdr_contour_1, return_grid=True)


        if norm_contour is None:
            # isnan = np.isnan(data_contour_2)
            # min = np.percentile(data_contour_2[~isnan], 5)
            # max = np.percentile(data_contour_2[~isnan], 98)
            # norm_contour = ImageNormalize(stretch=LinearStretch(), vmin=min, vmax=max)
            norm_contour = PlotFits.get_range(data_contour_2, imin=3, imax=97, stre=None)
        longitude_grid_arc = AlignCommonUtil.ang2pipi(lon_grid).to("arcsec").value
        latitude_grid_arc = AlignCommonUtil.ang2pipi(lat_grid).to("arcsec").value
        dlon = longitude_grid_arc[1, 1] - longitude_grid_arc[0, 0]
        dlat = latitude_grid_arc[1, 1] - latitude_grid_arc[0, 0]

        w_xy = WCS(hdr_contour_2)
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

        if show:
            fig.show()
        if path_save is not None:
            fig.savefig(path_save, bbox_inches='tight')
        if return_axes:
            return fig, ax1, ax2, ax3, ax_cbar1, ax_cbar2

    @staticmethod
    def _build_regular_grid(longitude, latitude):
        dlon = np.abs((longitude[1, 1] - longitude[0, 0]).to(u.deg).value)
        dlat = np.abs((latitude[1, 1] - latitude[0, 0]).to(u.deg).value)
        longitude_grid, latitude_grid = np.meshgrid(
            np.arange(np.min(AlignCommonUtil.ang2pipi(longitude).to(u.deg).value),
                      np.max(AlignCommonUtil.ang2pipi(longitude).to(u.deg).value), dlon),
            np.arange(np.min(AlignCommonUtil.ang2pipi(latitude).to(u.deg).value),
                      np.max(AlignCommonUtil.ang2pipi(latitude).to(u.deg).value), dlat))

        longitude_grid = longitude_grid * u.deg
        latitude_grid = latitude_grid * u.deg

        return longitude_grid, latitude_grid

    @staticmethod
    def plot_co_alignment(large_fov_window, large_fov_path, corr,
                          small_fov_window, small_fov_path, levels_percentile=[85],
                          lag_crval1=None, lag_crval2=None, lag_crota=None, lag_cdelta1=None, lag_cdelta2=None,
                          show=False, results_folder=None, cut_from_center=None):

        parameter_alignment = {
            "crval1": lag_crval1,
            "crval2": lag_crval2,
            "crota": lag_crota,
            "cdelt1": lag_cdelta1,
            "cdelt2": lag_cdelta2,

        }

        max_index = np.unravel_index(np.nanargmax(corr), corr.shape)

        with fits.open(large_fov_path) as hdul_large:
            header_large = hdul_large[large_fov_window].header.copy()
            data_large = hdul_large[large_fov_window].data.copy()
            with fits.open(small_fov_path) as hdul_spice:
                header_spice_original = hdul_spice[small_fov_window].header.copy()

                if "HRI_EUV" in  header_spice_original["TELESCOP"]:
                    # AlignEUIUtil.recenter_crpix_in_header(header_spice)
                    w_xy = WCS(header_spice_original)
                    header_spice = w_xy.to_header().copy()
                    data_spice = np.array(hdul_spice[small_fov_window].data.copy(), dtype=np.float64)
                elif  "SPICE" in header_spice_original["TELESCOP"]:
                    # AlignSpiceUtil.recenter_crpix_in_header_L2(header_spice)
                    w_spice = WCS(header_spice_original)
                    ymin, ymax = AlignSpiceUtil.vertical_edges_limits(header_spice_original)
                    w_xyt = w_spice.dropaxis(2)
                    w_xyt.wcs.pc[2, 0] = 0
                    w_xy = w_xyt.dropaxis(2)
                    header_spice = w_xy.to_header().copy()

                    data_small = np.array(hdul_spice[small_fov_window].data.copy(), dtype=np.float64)
                    data_small[:, :, :ymin, :] = np.nan
                    data_small[:, :, ymax:, :] = np.nan

                    data_spice = np.nansum(data_small[0, :, :, :], axis=0)
                    data_spice[:ymin, :] = np.nan
                    data_spice[ymax:, :] = np.nan

                    if cut_from_center is not None:

                        if cut_from_center is not None:
                            xlen = cut_from_center
                            xmid = data_spice.shape[1] // 2
                            data_spice[:, :(xmid - xlen // 2 - 1)] = np.nan
                            data_spice[:, (xmid + xlen // 2):] = np.nan

                    # header_spice["CRPIX1"] = (data_spice.shape[1] + 1) / 2
                    # header_spice["CRPIX2"] = (data_spice.shape[0] + 1) / 2




                header_spice["SOLAR_B0"] = hdul_spice[small_fov_window].header["SOLAR_B0"]
                header_spice["RSUN_REF"] = hdul_spice[small_fov_window].header["RSUN_REF"]
                header_spice["DSUN_OBS"] = hdul_spice[small_fov_window].header["DSUN_OBS"]
                # data_spice = np.nansum(hdul_spice[raster_window].data.copy()[0, :, :, :], axis=0)

                header_spice["NAXIS1"] = data_spice.shape[1]
                header_spice["NAXIS2"] = data_spice.shape[0]
                # AlignEUIUtil.recenter_crpix_in_header(header_spice)
                not_nan = np.isnan(data_spice)
                levels = [np.percentile(data_spice[~not_nan], n) for n in levels_percentile]

                hdr_spice_shifted = header_spice.copy()
                if header_spice_original["PC1_1"] == 1.0:
                    for i, j in zip([1, 1, 2, 2], [1, 2, 1, 2]):
                        hdr_spice_shifted[f'PC{i}_{j}'] = header_spice_original[f'PC{i}_{j}']

                hdr_spice_shifted["CRVAL1"] = hdr_spice_shifted["CRVAL1"] \
                                              + u.Quantity(parameter_alignment['crval1'][max_index[0]], "arcsec").to(
                    hdr_spice_shifted["CUNIT1"]).value
                hdr_spice_shifted["CRVAL2"] = hdr_spice_shifted["CRVAL2"] \
                                              + u.Quantity(parameter_alignment['crval2'][max_index[1]], "arcsec").to(
                    hdr_spice_shifted["CUNIT2"]).value
                change_pcij = False

                if hdr_spice_shifted["PC1_1"] > 1.0:
                    warnings.warn(f'{hdr_spice_shifted["PC1_1"]=}, set it to 1.0')
                    hdr_spice_shifted["PC1_1"] = 1.0
                    hdr_spice_shifted["PC2_2"] = 1.0
                    hdr_spice_shifted["PC1_2"] = 0.0
                    hdr_spice_shifted["PC2_1"] = 0.0
                    hdr_spice_shifted["CROTA"] = 0.0

                key_rota = None
                if "CROTA" in hdr_spice_shifted:
                    key_rota = "CROTA"
                elif "CROTA2" in hdr_spice_shifted:
                    key_rota = "CROTA2"

                crota = np.rad2deg(np.arccos(copy.deepcopy(hdr_spice_shifted["PC1_1"])))



                if parameter_alignment['crota'] is not None:
                    # hdr_spice_shifted["CROTA"] = hdul_spice[raster_window].header["CROTA"] +\
                    #                              parameter_alignement['crota'][max_index[4]]
                    if key_rota is None:
                        hdr_spice_shifted["CROTA"] = np.rad2deg(np.arccos(copy.deepcopy(hdr_spice_shifted["PC1_1"])))

                        key_rota = "CROTA"
                    hdr_spice_shifted[key_rota] += parameter_alignment['crota'][max_index[4]]
                    crota += parameter_alignment['crota'][max_index[4]]
                    change_pcij = True

                if parameter_alignment['cdelt1'] is not None:
                    cdelt1 = u.Quantity(hdr_spice_shifted["CDELT1"], hdr_spice_shifted["CUNIT1"]) + \
                                                  u.Quantity(parameter_alignment['cdelt1'][max_index[2]], "arcsec")
                    hdr_spice_shifted["CDELT1"] = cdelt1.to(hdr_spice_shifted["CUNIT1"]).value
                    change_pcij = True

                if parameter_alignment['cdelt2'] is not None:
                    cdelt2 = u.Quantity(hdr_spice_shifted["CDELT2"], hdr_spice_shifted["CUNIT2"]) + \
                             u.Quantity(parameter_alignment['cdelt2'][max_index[3]], "arcsec")
                    hdr_spice_shifted["CDELT2"] = cdelt2.to(hdr_spice_shifted["CUNIT2"]).value
                    change_pcij = True
                if change_pcij:
                    s = - np.sign(hdr_spice_shifted["PC1_2"]) + (hdr_spice_shifted["PC1_2"] == 0.0)
                    theta = np.deg2rad(crota) * s
                    lam = hdr_spice_shifted["CDELT2"] / hdr_spice_shifted["CDELT1"]
                    hdr_spice_shifted["PC1_1"] = np.cos(theta)
                    hdr_spice_shifted["PC2_2"] = np.cos(theta)
                    hdr_spice_shifted["PC1_2"] = -lam * np.sin(theta)
                    hdr_spice_shifted["PC2_1"] = (1 / lam) * np.sin(theta)
                not_nan = np.isnan(data_large)
                min = np.percentile(data_large[~not_nan], 3)
                max = np.percentile(data_large[~not_nan], 99)
                norm = ImageNormalize(stretch=LinearStretch(), vmin=np.max((min, 1)), vmax=max)

                longitude, latitude = AlignEUIUtil.extract_EUI_coordinates(header_spice.copy(), dsun=False)
                longitude_grid, latitude_grid, dlon, dlat = PlotFits.build_regular_grid(longitude, latitude)
                dlon = dlon.to("arcsec").value
                dlat = dlat.to("arcsec").value

                data_fsi = hdul_large[large_fov_window].data
                header_fsi = hdul_large[large_fov_window].header

                header_spice["BUNIT"] = hdul_spice[small_fov_window].header["BUNIT"]
                hdr_spice_shifted["BUNIT"] = hdul_spice[small_fov_window].header["BUNIT"]
                header_spice["DATE-AVG"] = hdul_spice[small_fov_window].header["DATE-AVG"]

                fig = plt.figure(figsize=(10, 4))
                fig, ax1, ax2, ax3, ax_cbar1, ax_cbar2 = \
                    PlotFunctions.compare_plot(header_large, data_large, header_spice, data_spice, hdr_spice_shifted,
                                               data_spice, show=False, norm=norm, levels=levels, return_axes=True,
                                               fig=fig,
                                               cmap1="plasma", cmap2="viridis", path_save=None)
                detector = header_large["DETECTOR"]
                wave = header_large["WAVELNTH"]

                ax1.set_title(f"{detector} {wave} & Small FOV (contour) not aligned ")
                ax2.set_title(f"{detector} {wave} & Small FOV (contour) aligned ")
                ax3.set_title("Small FOV (%s) aligned " % small_fov_window)
                date = Time(hdul_spice[small_fov_window].header["DATE-AVG"]).fits[:19]
                date = date.replace(":", "_")
                date = date.replace("-", "_")

                date_str = header_spice["DATE-OBS"][:19]
                fig.suptitle(f"Small FOV {date_str} aligned with {detector} {wave}")
                # fig.suptitle("Alignement of SPICE  using a synthetic raster of HRIEUV images")
                if results_folder is not None:
                    fig.savefig('%s/compare_alignment.pdf' % (results_folder))
                if show:
                    fig.show()

                # fig.suptitle(f"Alignement SPICE {date}- HRI 174")
                w_fsi = WCS(header_fsi)
                w_spice = WCS(header_spice)
                w_spice_shift = WCS(hdr_spice_shifted)
                x_fsi, y_fsi = w_fsi.world_to_pixel(longitude_grid, latitude_grid)
                x_spice, y_spice = w_spice.world_to_pixel(longitude_grid, latitude_grid)
                x_spice_shift, y_spice_shift = w_spice_shift.world_to_pixel(longitude_grid, latitude_grid)

                data_fsi_interp = AlignCommonUtil.interpol2d(data_fsi, x=x_fsi, y=y_fsi, fill=-32762, order=1)
                data_spice_interp = AlignCommonUtil.interpol2d(data_spice, x=x_spice, y=y_spice, fill=-32762, order=1)
                data_spice_interp_shift = AlignCommonUtil.interpol2d(data_spice, x=x_spice_shift, y=y_spice_shift,
                                                                fill=-32762,
                                                                order=1)

                data_fsi_interp = np.where(data_fsi_interp == -32762, np.nan, data_fsi_interp)
                data_spice_interp = np.where(data_spice_interp == -32762, np.nan, data_spice_interp)
                data_spice_interp_shift = np.where(data_spice_interp_shift == -32762, np.nan, data_spice_interp_shift)

                longitude_grid_arc = AlignCommonUtil.ang2pipi(longitude_grid.to("arcsec")).value
                latitude_grid_arc = AlignCommonUtil.ang2pipi(latitude_grid.to("arcsec")).value
                dlon = longitude_grid_arc[0, 1] - longitude_grid_arc[0, 0]
                dlat = latitude_grid_arc[1, 0] - latitude_grid_arc[0, 0]

                isnan = np.isnan(data_spice_interp)
                min = np.percentile(data_spice_interp[~isnan], 3)
                max = np.percentile(data_spice_interp[~isnan], 99)
                norm_spice = ImageNormalize(stretch=LinearStretch(), vmin=min, vmax=max)

                isnan = np.isnan(data_fsi_interp)
                min = np.percentile(data_fsi_interp[~isnan], 3)
                max = np.percentile(data_fsi_interp[~isnan], 99)
                norm_fsi = ImageNormalize(stretch=LinearStretch(), vmin=min, vmax=max)
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot()
                im = ax.imshow(data_fsi_interp, origin='lower', interpolation='none', cmap='plasma', norm=norm_fsi,
                               extent=[longitude_grid_arc[0, 0] - 0.5 * dlon,
                                       longitude_grid_arc[-1, -1] + 0.5 * dlon,
                                       latitude_grid_arc[0, 0] - 0.5 * dlat,
                                       latitude_grid_arc[-1, -1] + 0.5 * dlat])
                ax.set_title(f"{detector} {wave} of %s %s files \n %s" % (header_large["DATE-AVG"][:19],
                                                                        detector, wave))
                ax.set_xlabel("Solar-X [arcsec]")
                ax.set_ylabel("Solar-Y [arcsec]")
                cbar = fig.colorbar(im, label=header_fsi["BUNIT"])
                if results_folder is not None:
                    fig.savefig(os.path.join(results_folder, f"Synthetic_raster_on_grid_{date}.pdf"))
                if show:
                    fig.show()
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot()
                im = ax.imshow(data_spice_interp, origin='lower', interpolation='none', cmap='viridis', norm=norm_spice,
                               extent=[longitude_grid_arc[0, 0] - 0.5 * dlon,
                                       longitude_grid_arc[-1, -1] + 0.5 * dlon,
                                       latitude_grid_arc[0, 0] - 0.5 * dlat,
                                       latitude_grid_arc[-1, -1] + 0.5 * dlat])

                ax.set_xlabel("Solar-X [arcsec]")
                ax.set_ylabel("Solar-Y [arcsec]")
                ax.set_title("small FOV not aligned (%s) \n %s" % (small_fov_window, header_spice["DATE-OBS"][:19]))
                cbar = fig.colorbar(im, label=header_spice["BUNIT"])
                if results_folder is not None:
                    fig.savefig(os.path.join(results_folder, f"small_fov_before_alignment_on_grid_{date}.pdf"))
                if show:
                    fig.show()
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot()
                im = ax.imshow(data_spice_interp_shift, origin='lower', interpolation='none', cmap='viridis',
                               norm=norm_spice,
                               extent=[longitude_grid_arc[0, 0] - 0.5 * dlon,
                                       longitude_grid_arc[-1, -1] + 0.5 * dlon,
                                       latitude_grid_arc[0, 0] - 0.5 * dlat,
                                       latitude_grid_arc[-1, -1] + 0.5 * dlat])
                ax.set_title("Small FOV aligned (%s) \n %s" % (small_fov_window, header_spice["DATE-OBS"][:19]))
                ax.set_xlabel("Solar-X [arcsec]")
                ax.set_ylabel("Solar-Y [arcsec]")
                cbar = fig.colorbar(im, label=header_spice["BUNIT"], )
                if results_folder is not None:
                    fig.savefig(os.path.join(results_folder, f"small_fov_after_alignment_on_grid_{date}.pdf"))
                if show:
                    fig.show()
                # fig.savefig(os.path.join(results_folder, "SPICE_after_alignment_on_grid.png"), dpi=150)

                hdul_spice.close()
            hdul_large.close()


# class Util:
#     @staticmethod
#     def ang2pipi(ang):
#         """ put angle between ]-180, +180] deg """
#         pi = u.Quantity(180, 'deg')
#         return - ((- ang + pi) % (2 * pi) - pi)
#
#     @staticmethod
#     def fits_create_submap(data, header, longitude_limits_arcsec: list, latitude_limits_arcsec: list):
#         longitude, latitude, dsun_obs = Util.extract_EUI_coordinates(hdr=header)
#         mask_longitude = np.array((longitude.to(u.arcsec) > longitude_limits_arcsec[0]) &
#                                   (longitude.to(u.arcsec) < longitude_limits_arcsec[1]), dtype="bool")
#
#         mask_latitude = np.array((latitude.to(u.arcsec) > latitude_limits_arcsec[0]) &
#                                  (latitude.to(u.arcsec) < latitude_limits_arcsec[1]), dtype="bool")
#
#     @staticmethod
#     def extract_EUI_coordinates(hdr):
#         w = WCS(hdr)
#         idx_lon = np.where(np.array(w.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
#         idx_lat = np.where(np.array(w.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
#         x, y = np.meshgrid(np.arange(w.pixel_shape[idx_lon]),
#                            np.arange(w.pixel_shape[idx_lat]), )
#         # should reproject on a new coordinate grid first : suppose slits at the same time :
#         longitude, latitude = w.pixel_to_world(x, y)
#         if "DSUN_OBS" in w.to_header():
#             dsun_obs_large = w.to_header()["DSUN_OBS"]
#         else:
#             dsun_obs_large = None
#         return Util.ang2pipi(longitude), Util.ang2pipi(latitude), dsun_obs_large
#
#     @staticmethod
#     def _extract_spice_coordinates(hdr):
#         w_small = WCS(hdr)
#         w2 = w_small.deepcopy()
#
#         w2.wcs.pc[3, 0] = 0
#         w2.wcs.pc[3, 1] = 0
#
#         w_xyt = w2.dropaxis(2)
#         w_xy = w_xyt.dropaxis(2)
#         idx_lon = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLN-TAN")[0][0]
#         idx_lat = np.where(np.array(w_xy.wcs.ctype, dtype="str") == "HPLT-TAN")[0][0]
#         x_small, y_small = np.meshgrid(np.arange(w_xy.pixel_shape[idx_lon]),
#                                        np.arange(w_xy.pixel_shape[idx_lat]),
#                                        indexing='ij')  # t dépend de x,
#         # should reproject on a new coordinate grid first : suppose slits at the same time :
#         longitude_small, latitude_small = w_xy.pixel_to_world(x_small, y_small)
#         return longitude_small, latitude_small
