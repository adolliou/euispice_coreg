import numpy as np
from ..plot.plot import PlotFunctions, PlotFits
from ..utils.Util import AlignCommonUtil
import warnings
from scipy.optimize import curve_fit
from astropy.io import fits
import astropy.units as u
import astropy


# from https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x, y = xy
    x0 = float(xo)
    y0 = float(yo)

    g = offset + amplitude * np.exp(
        -((((x - x0) ** 2) / (2 * sigma_x ** 2)) + (((y - y0) ** 2) / (2 * sigma_y ** 2)))
    )
    return g.ravel()


class AlignmentResults:

    def __init__(
            self,
            corr: np.array,
            lag_crval1: np.array,
            lag_crval2: np.array,
            lag_cdelt1: np.array,
            lag_cdelt2: np.array,
            lag_crota: np.array,
            unit_lag: str,
            image_to_align_path: str = None,
            image_to_align_window=None,
            reference_image_path: str = None,
            reference_image_window: int = None,
    ):
        """
        Initialize a AlignmentResults class, that simplyfies the treatment of the correlation results
        resulting from the Alignment class
        Args:
            corr (np.array): correlation matrix
            lag_crval1 (np.array): lag values in CRVAL1 header
            lag_crval2 (np.array): lag values in CRVAL2 header
            lag_cdelt1 (np.array): lag values in CDELT1 header
            lag_cdelt2 (np.array): lag values in CDELT2 header
            lag_crota (np.array): lag values in CROTA header, in degrees
            unit_lag (str): unit of the spatial shifts (e.g. "arcsec")
            image_to_align_path (str, optional): Path to the image to align FITS file. Defaults to None.
            image_to_align_window (_type_, optional): window to use for the to align FITS Hdulist. Defaults to None.
            reference_image_path (str, optional): Path to the reference image FITS file with known pointing. Defaults to None.
            reference_image_window (int, optional):  window to use for the reference FITS Hdulist. Defaults to None.
        """

        if lag_crval1 is None:
            lag_crval1 = np.array([0])
        if lag_crval2 is None:
            lag_crval2 = np.array([0])
        if lag_cdelt1 is None:
            lag_cdelt1 = np.array([0])
        if lag_cdelt2 is None:
            lag_cdelt2 = np.array([0])
        if lag_crota is None:
            lag_crota = np.array([0])

        self.max_index = np.unravel_index(np.nanargmax(corr), corr.shape)
        self.corr = corr
        self.parameters_alignment = {
            "lag_crval1": u.Quantity(lag_crval1, unit_lag),
            "lag_crval2": u.Quantity(lag_crval2, unit_lag),
            "lag_cdelt1": u.Quantity(lag_cdelt1, unit_lag),
            "lag_cdelt2": u.Quantity(lag_cdelt2, unit_lag),
            "lag_crota": u.Quantity(lag_crota, "deg"),
        }
        self.parameters_alignment_arcsec = {
            "lag_crval1": u.Quantity(lag_crval1, unit_lag).to("arcsec").value,
            "lag_crval2": u.Quantity(lag_crval2, unit_lag).to("arcsec").value,
            "lag_cdelt1": u.Quantity(lag_cdelt1, unit_lag).to("arcsec").value,
            "lag_cdelt2": u.Quantity(lag_cdelt2, unit_lag).to("arcsec").value,
            "lag_crota": u.Quantity(lag_crota, "deg").to("deg").value,
        }
        self.image_to_align_path = image_to_align_path
        self.image_to_align_window = image_to_align_window
        self.reference_image_path = reference_image_path
        self.reference_image_window = reference_image_window
        self.unit_lag = unit_lag
        self.shift_pixels = None
        self.shift_arcsec = None

        self._compute_shift()

    def plot_correlation(
            self, path_save_figure: str = None, show=False, fig=None, ax=None
    ):
        """
        Plot the correlation matrix. Calls the PlotFunctions.plot_correlation function from util.py

        Args:
            path_save_figure (str, optional): path where to save the figure. It is adviced that it ends with ".pdf". Defaults to None.
            show (bool, optional): set True if you want to interactively show the plots. Defaults to False.
            fig (_type_, optional): set a plt.figure object. Defaults to None.
            ax (_type_, optional): set a Axes object. Defaults to None.

        Returns:
            _type_: returns the results from PlotFunctions.plot_correlation.
        """
        return PlotFunctions.plot_correlation(
            corr=self.corr,
            show=show,
            path_save_figure=path_save_figure,
            fig=fig,
            ax=ax,
            shift=self.shift_arcsec,
            unit_to_plot=self.unit_lag,
            lag_dx_label=f"CRVAL1 [{self.unit_lag}]",
            lag_dy_label=f"CRVAL2 [{self.unit_lag}]",
            **self.parameters_alignment_arcsec,
        )

    def plot_co_alignment(
            self, path_save_figure: str = None, show=False, lonlims=None, latlims=None, **kwargs
    ):

        return PlotFunctions.plot_co_alignment(
            reference_image_path=self.reference_image_path,
            reference_image_window=self.reference_image_window,
            image_to_align_path=self.image_to_align_path,
            image_to_align_window=self.image_to_align_window,
            path_save_figure=path_save_figure,
            shift_arcsec=self.shift_arcsec,
            show=show,
            unit_to_plot=self.unit_lag,
            lonlims=lonlims, 
            latlims=latlims,
            **kwargs,
        )

    def write_corrected_fits(
            self,
            window_list_to_apply_shift: list,
            path_to_l3_output: str,
            path_to_l2_input: str = None,
    ):
        """
        Save the FITS with the corrected metadata in their headers. The hdu.data arrays are not changed, only the header values.
        It is adviced to call this method instead of manually changing the header values.

        Args:
            window_list_to_apply_shift (list): windows of the HDULIST where correction to the header values will be applied
            path_to_l3_output (str): path where the new corrected FITS will be saved.
            path_to_l2_input (str, optional): path where the initial L2 uncorrected FITS file is. If None, Then take the
            image_to_align_path attribute of the class.
            . Defaults to None.
        """
        if path_to_l2_input is None:
            if self.image_to_align_path is None:
                raise ValueError("Please provide a path_to_l2_input parameter")
            path_to_l2_input = self.image_to_align_path
        AlignCommonUtil.write_corrected_fits(
            corr=self.corr,
            path_to_l2_input=path_to_l2_input,
            path_to_l3_output=path_to_l3_output,
            window_list_to_apply_shift=window_list_to_apply_shift,
            shift_arcsec=self.shift_arcsec,
        )
        # has_corrected_window = 0


        # with fits.open(path_to_l2_input) as hdul:
        #     hdul_out = fits.HDUList()
        #     for ii in range(len(hdul)):
        #         hdu = hdul[ii]
        #         if "EXTNAME" in hdu.header:
        #             extname = hdu.header["EXTNAME"]
        #         else:
        #             extname = "nothing98695"
        #         if (extname in window_list_to_apply_shift) or (ii in window_list_to_apply_shift) or \
        #                 ((ii - len(hdul)) in window_list_to_apply_shift):
        #             header = hdu.header.copy()
        #             data = hdu.data.copy()
        #             AlignCommonUtil.correct_pointing_header(
        #                 header,
        #                 lag_crval1=self.shift_arcsec[0],
        #                 lag_crval2=self.shift_arcsec[1],
        #                 lag_cdelt1=self.shift_arcsec[2],
        #                 lag_cdelt2=self.shift_arcsec[3],
        #                 lag_crota=self.shift_arcsec[4],
        #             )
        #             if isinstance(hdu, astropy.io.fits.hdu.compressed.compressed.CompImageHDU):
        #                 hdu_out = fits.CompImageHDU(data=data, header=header)
        #             elif isinstance(hdu, astropy.io.fits.hdu.image.ImageHDU):
        #                 hdu_out = fits.ImageHDU(data=data, header=header)
        #             elif isinstance(hdu, astropy.io.fits.hdu.image.PrimaryHDU):
        #                 hdu_out = fits.PrimaryHDU(data=data, header=header)
        #             hdu_out.verify("silentfix")
        #             has_corrected_window += 1

        #         else:
        #             hdu_out = hdu
        #         hdul_out.append(hdu_out)

        #     hdul_out.writeto(path_to_l3_output, overwrite=True)
        #     if has_corrected_window == 0:
        #         raise ValueError("has not corrected any window.")


    def savefig(self, filename: str):
        raise NotImplementedError

    def saveyaml(self, filename: str, window: str, 
                 path_to_l2_input: str = None):
        # yaml with the corrected header values
        raise NotImplementedError

        if path_to_l2_input is None:
            if self.image_to_align_path is None:
                raise ValueError("Please provide a path_to_l2_input parameter")
            path_to_l2_input = self.image_to_align_path
        with fits.open(path_to_l2_input) as hdul:
            hdu = hdul[window]
            header_origin = hdu.header.copy()
            AlignCommonUtil.correct_pointing_header(
                header_origin,
                lag_crval1=self.shift_arcsec[0],
                lag_crval2=self.shift_arcsec[1],
                lag_cdelt1=self.shift_arcsec[2],
                lag_cdelt2=self.shift_arcsec[3],
                lag_crota=self.shift_arcsec[4],
            )

    def return_corrected_header(self, window: str, 
                 path_to_l2_input: str = None):
        """Return the header of a given window of the FITS file with the corrected pointing informations

        Args:
            window (str): window over which the header will be returned
            path_to_l2_input (str, optional): Initial FITS file path name. If None, Then take the
            image_to_align_path attribute of the class., .
        Returns:
            _type_: corrected header
        """        
        # yaml with the corrected header values
        if path_to_l2_input is None:
            if self.image_to_align_path is None:
                raise ValueError("Please provide a path_to_l2_input parameter")
            path_to_l2_input = self.image_to_align_path
        with fits.open(path_to_l2_input) as hdul:
            hdu = hdul[window]
            header_origin = hdu.header.copy()
            AlignCommonUtil.correct_pointing_header(
                header_origin,
                lag_crval1=self.shift_arcsec[0],
                lag_crval2=self.shift_arcsec[1],
                lag_cdelt1=self.shift_arcsec[2],
                lag_cdelt2=self.shift_arcsec[3],
                lag_crota=self.shift_arcsec[4],
            )
        return header_origin



    def _compute_shift(self, method="fitting_gaussian"):

        # return the shift values after 3d polynomial computation.
        corr2d = self.corr[
                 :, :, self.max_index[2], self.max_index[3], self.max_index[4]
                 ]
        p = [(self.max_index[0], self.max_index[1])]
        px = [self.max_index[0]]
        py = [self.max_index[1]]
        # Find points around the maximum
        lenx = corr2d.shape[0]
        leny = corr2d.shape[1]
        for ii, jj in zip(
                [
                    1,
                    1,
                    -1,
                    -1,
                    2,
                    2,
                    -2,
                    -2,
                    1,
                    1,
                    -1,
                    -1,
                    2,
                    2,
                    -2,
                    -2,
                ],
                [
                    1,
                    -1,
                    1,
                    -1,
                    2,
                    -2,
                    2,
                    -2,
                    2,
                    -2,
                    2,
                    -2,
                    1,
                    -1,
                    1,
                    -1,
                ],
        ):
            x = self.max_index[0] + ii
            y = self.max_index[1] + jj
            if (x != -1) and (x < lenx) and (y != -1) and (y < leny):
                p.append((x, y))
                px.append(x)
                py.append(y)
        if method == "fitting_gaussian":
            if len(px) < 4:
                warnings.warn(
                    " Cannot compute shift with Gaussian fitting: not enough points"
                )
                self.shift_pixels = (
                    self.max_index[0],
                    self.max_index[1],
                    self.max_index[2],
                    self.max_index[3],
                    self.max_index[4],
                )
                self.shift_arcsec = (
                    self.parameters_alignment["lag_crval1"][self.max_index[0]].to("arcsec").value,
                    self.parameters_alignment["lag_crval2"][self.max_index[1]].to("arcsec").value,
                    self.parameters_alignment["lag_cdelt1"][self.max_index[2]].to("arcsec").value,
                    self.parameters_alignment["lag_cdelt2"][self.max_index[3]].to("arcsec").value,
                    self.parameters_alignment["lag_crota"][self.max_index[4]].to("deg").value,
                )
                return None

            A = (np.float64(px), np.float64(py))
            B = np.float64(corr2d[px, py].ravel())

            p0 = (
                np.float64(corr2d[self.max_index[0], self.max_index[1]][0]),
                np.float64(self.max_index[0]),
                np.float64(self.max_index[1]),
                np.float64(1),
                np.float64(1),
                np.float64(0.9),
            )
            bounds = (
                [
                    np.float64(0),
                    np.float64(self.max_index[0] - 5),
                    np.float64(self.max_index[1] - 5),
                    np.float64(0),
                    np.float64(0),
                    np.float64(-10),
                ],
                [
                    np.float64(10),
                    np.float64(self.max_index[0] + 5),
                    np.float64(self.max_index[1] + 5),
                    np.float64(1000),
                    np.float64(1000),
                    np.float64(10),
                ],
            )
            try:
                popt, pcov = curve_fit(
                    f=twoD_Gaussian, xdata=A, ydata=B, p0=p0, bounds=bounds
                )
                lag_x = self.parameters_alignment["lag_crval1"].to("arcsec").value
                lag_y = self.parameters_alignment["lag_crval2"].to("arcsec").value

                shift_x_arcsec = np.interp(
                    popt[1],
                    np.arange(len(lag_x)),
                    lag_x,
                )
                shift_y_arcsec = np.interp(
                    popt[2],
                    np.arange(len(lag_y)),
                    lag_y,
                )
                self.shift_pixels = (
                    popt[1],
                    popt[2],
                    self.max_index[2],
                    self.max_index[3],
                    self.max_index[4],
                )
                self.shift_arcsec = (
                    shift_x_arcsec,
                    shift_y_arcsec,
                    self.parameters_alignment["lag_cdelt1"][self.max_index[2]].to("arcsec").value,
                    self.parameters_alignment["lag_cdelt2"][self.max_index[3]].to("arcsec").value,
                    self.parameters_alignment["lag_crota"][self.max_index[4]].to("deg").value,
                )

                return True
            except ValueError:
                warnings.warn(
                    "Gaussian fitting failed, setting shift params as the pixel of the maximal correlation"
                )
                self.shift_pixels = (
                    self.max_index[0],
                    self.max_index[1],
                    self.max_index[2],
                    self.max_index[3],
                    self.max_index[4],
                )
                self.shift_arcsec = (
                    self.parameters_alignment["lag_crval1"][self.max_index[0]].to("arcsec").value,
                    self.parameters_alignment["lag_crval2"][self.max_index[1]].to("arcsec").value,
                    self.parameters_alignment["lag_cdelt1"][self.max_index[2]].to("arcsec").value,
                    self.parameters_alignment["lag_cdelt2"][self.max_index[3]].to("arcsec").value,
                    self.parameters_alignment["lag_crota"][self.max_index[4]].to("deg").value,
                )
                return None
        elif method == "poly2d":
            A = np.array([1, px, py, px ** 2, py ** 2, px * py]).T
            B = corr2d[px, py]
            coeff, r, rank, s = np.linalg.lstsq(A, B)
            # But then I need to compute the maximum from the polynom points.
            # I dont know how to do that
            raise NotImplementedError

    def __repr__(self):
        print(
            f"\n Shift : \n x = {self.shift_arcsec[0]} '' \n y = {self.shift_arcsec[1]} '' \n dx = {self.shift_arcsec[2]} '"
            f"' \n dy = {self.shift_arcsec[3]} '' \n dcrot = {self.shift_arcsec[4]} deg"
        )

    def __str__(self):
        return (
            f"\n Shift : \n x = {self.shift_arcsec[0]} '' \n y = {self.shift_arcsec[1]} '' \n dx = {self.shift_arcsec[2]} '' "
            f"\n dy = {self.shift_arcsec[3]} '' \n dcrot = {self.shift_arcsec[4]} deg"
        )
