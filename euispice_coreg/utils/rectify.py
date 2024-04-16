#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:22:07 2020

@author: fauchere
"""

import numpy as np
import astropy.io.fits
import astropy.io.ascii
import astropy.constants
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib.collections as mcol
import configparser
import os.path
import scipy.ndimage
import scipy.interpolate
import cv2


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


def polyfit2d(x, y, f, deg, maxdegree=False):
    """
    x, y: coordinates of the data points
    f: data values at (x, y)
    deg: degree (on one axis, see maxdegree) of the fitted
         polynomial
    maxdegree: if True, degree represents the maximum degree of the
               fitting polynomial of all dimensions combined, rather
               than the maximum degree of the polynomial in a single
               variable.
    Adapted from https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
    """
    from numpy.polynomial import polynomial
    vander = polynomial.polyvander2d(x, y, [int(deg), int(deg)])
    vander = vander.reshape((-1, vander.shape[-1]))
    if maxdegree is True:
        # the summ of indices gives the combined degree for each coefficient,
        # which is then compared to maxdegree if not None
        dy, dx = np.indices((deg + 1, deg + 1))
        vander[:, (dx.reshape(-1) + dy.reshape(-1)) > deg] = 0
    c, _, _, _ = np.linalg.lstsq(vander,
                                 f.reshape((vander.shape[0],)),
                                 rcond=-1)
    return c.reshape((deg + 1, deg + 1))


def rotationmatrix(angle, axis):
    """
    Returns a rotation matrix about the specified axis (z=0, y=1, x=2) for the
    specififed angle (in radians).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)

    if axis == 0:  # Rz
        matrix = np.array([[cos, sin, 0],
                           [-sin, cos, 0],
                           [0, 0, 1]])
    elif axis == 1:  # Ry
        matrix = np.array([[cos, 0, sin],
                           [0, 1, 0],
                           [-sin, 0, cos]])
    elif axis == 2:  # Rx
        matrix = np.array([[1, 0, 0],
                           [0, cos, -sin],
                           [0, sin, cos]])

    return matrix


def gridpattern(nx=3072, ny=3072, s=16, t=3):
    """
    Creates an image containing a regular binary (0 1) grid.
    Used only for test purposes.

    nx, ny: size in pixels of the output image
    s: pitch of the grid in pixels
    t: thickness of the grid in pixels
    """
    image = np.zeros((nx, ny))
    for i in range(t):
        image[i::s, :] = 1
        image[:, i::s] = 1
    return image


class BaseTransform():
    """
    Adapted from astropy.visualization.BaseTransform
    """

    def __add__(self, other):
        return CompositeTransform(self, other)


class CompositeTransform(BaseTransform):
    """
    A combination of two transforms.
    Adapted from astropy.visualization.CompositeTransform

    Parameters
    ----------
    transform_1 : :class:`rectify.BaseTransform`
        The first transform to apply.
    transform_2 : :class:`rectify..BaseTransform`
        The second transform to apply.
    """

    def __init__(self, transform_1, transform_2):
        super().__init__()
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __call__(self, x=None, y=None):
        x, y = self.transform_1(x=x, y=y)
        return self.transform_2(x=x, y=y)


class Transform(BaseTransform):

    def __init__(self, direction='forward'):
        """
        direction: 'forward' (default) or 'inverse'. defines the direction of
                    the transform.
        """
        self.direction = direction

    def forward(self, x=None, y=None):
        raise NotImplementedError

    def inverse(self, x=None, y=None):
        raise NotImplementedError

    def __call__(self, x=None, y=None):

        if self.direction == 'forward':
            return self.forward(x=x, y=y)
        elif self.direction == 'inverse':
            return self.inverse(x=x, y=y)
        else:
            raise ValueError('Transfrom direction must be forward or inverse')


class LinearTransform(Transform):
    """
    Transformation of the type y = Ax where A is the transformation matrix.
    LinearTransforms must initialize _fmatrix and _imatrix attributes that
    contain respectively the forward and inverse transform matrices.
    """

    def transform(self, matrix, x=None, y=None):
        z = np.ones_like(x)
        xyz = np.stack((x.ravel(), y.ravel(), z.ravel()))
        nx, ny, _ = np.matmul(matrix, xyz)
        return nx.reshape(x.shape), ny.reshape(x.shape)

    def forward(self, x=None, y=None):
        return self.transform(self._fmatrix, x=x, y=y)

    def inverse(self, x=None, y=None):
        return self.transform(self._imatrix, x=x, y=y)


class PolarTransform(Transform):

    def __init__(self, *args, direction='forward', conformal=False, degrees=True):

        super().__init__(direction=direction)
        identity = lambda x: x
        self._convert = np.radians if degrees else identity
        self._iconvert = np.degrees if degrees else identity
        self.xc = args[0]
        self.yc = args[1]
        if direction == 'forward':
            if len(args) == 2:
                self.e = 1
                self.psi = 0
            elif len(args) == 4:
                self.e = args[2]
                self.psi = self._convert(args[3])
            else:
                raise ValueError('Invalid number of arguments')
        else:
            self.thetalims = args[2]
            self.rlims = args[3]
            if len(args) == 4:
                self.psi = 0
                self.e = 1
            elif len(args) == 6:
                self.psi = self._convert(args[4])
                self.e = args[5]
            else:
                raise ValueError('Invalid number of arguments')

    def forward(self, x=None, y=None):

        if x is None:
            theta = 0
        else:
            theta = self._convert(x) - self.psi
        if y is None:
            y = 1
        nx = y * np.cos(theta)
        ny = y * np.sin(theta)
        ny *= self.e
        if self.psi != 0:
            dum = nx * np.cos(self.psi) - ny * np.sin(self.psi)
            ny = nx * np.sin(self.psi) + ny * np.cos(self.psi)
            nx = dum
        return nx + self.xc, ny + self.yc


class EuclidianTransform(LinearTransform):

    def __init__(self,
                 dx, dy, theta, scale,
                 dtype=np.float64,
                 degrees=True, direction='forward'):
        super().__init__(direction=direction)
        identity = lambda x: x
        self._convert = np.radians if degrees else identity
        self.dx = dx
        self.dy = dy
        self.scale = scale
        self.theta = self._convert(theta)
        self._fmatrix = np.array([[np.cos(self.theta) * scale, -np.sin(self.theta) * scale, dx],
                                  [np.sin(self.theta) * scale, np.cos(self.theta) * scale, dy],
                                  [0, 0, 1]], dtype=dtype)
        self._imatrix = np.linalg.inv(self._fmatrix)


class HomographicTransform(LinearTransform):

    def __init__(self,
                 matrix,
                 dtype=np.float32,
                 direction='forward'):
        super().__init__(direction=direction)
        self._fmatrix = matrix.astype(dtype)
        self._imatrix = np.linalg.inv(self._fmatrix)


class DifferentialRotationTransform(Transform):

    def __init__(self,
                 delta_t,
                 rate_wave,
                 degrees=True,
                 direction='forward'):
        super().__init__(direction=direction)
        self._convert = np.radians if degrees else lambda x: x
        self.delta_t = delta_t
        self.carrington_rate = 14.18
        if rate_wave == '171':
            self.coeffs = (14.56, -2.65, 0.96)
        elif rate_wave == '195':
            self.coeffs = (14.50, -2.14, 0.66)
        elif rate_wave == '284':
            self.coeffs = (14.60, -0.71, -1.18)
        elif rate_wave == '304':
            self.coeffs = (14.51, -3.12, 0.34)
        else:
            self.coeffs = (self.carrington_rate, 0, 0)

    def forward(self, x=None, y=None):

        siny2 = np.sin(self._convert(y)) ** 2

        dx = self.delta_t * (self.coeffs[0] +
                             siny2 * (self.coeffs[1] + self.coeffs[2] * siny2) -
                             self.carrington_rate)
        return x - dx, y


class SphericalTransform(Transform):

    def __init__(self,
                 *args,
                 direction='forward',
                 zclip=0, degrees=True, c2limb=False):
        super().__init__(direction=direction)
        # if inputs are in degrees, convert to radians, else do nothing
        self._convert = np.radians if degrees else lambda x: x

        self.x = args[0]
        self.y = args[1]
        self.dist = args[2]
        self.lon = self._convert(args[3])
        self.lat = self._convert(args[4])
        self.roll = self._convert(args[5])
        self.cdelt1 = args[6]
        if len(args) == 7:
            self.cdelt2 = self.cdelt1
        elif len(args) == 8:
            self.cdelt2 = args[7]
        else:
            raise ValueError('Invalid number of arguments')
        self.zclip = zclip
        self.c2limb = c2limb

    def forward(self, x=None, y=None):

        lon = self._convert(x) - self.lon
        lat = self._convert(y)

        x = np.cos(lat) * np.sin(lon)
        y = np.sin(lat)
        z = np.cos(lat) * np.cos(lon)

        zz = z * np.cos(self.lat) + y * np.sin(self.lat)
        yy = y * np.cos(self.lat) - z * np.sin(self.lat)

        gd = zz >= self.zclip

        y = yy[gd] * np.cos(self.roll) - x[gd] * np.sin(self.roll)
        x = x[gd] * np.cos(self.roll) + yy[gd] * np.sin(self.roll)

        z = self.dist - zz[gd]

        nx = np.full_like(lon, np.nan)
        ny = np.full_like(lon, np.nan)

        nx[gd] = self.x + np.degrees(np.arctan(x / z)) * 3600 / self.cdelt1
        ny[gd] = self.y + np.degrees(np.arctan(y / z)) * 3600 / self.cdelt2

        if self.c2limb:
            mu = np.ones_like(lon)
            a = 0.1
            r = np.sqrt(x ** 2 + y ** 2)
            r[r > 1] = 1
            theta = np.arcsin(r)
            mu[gd] = -np.cos(theta) / a + np.sqrt(1 + 2 / a + (np.cos(theta) / a) ** 2)
            return nx, ny, mu
        else:
            return nx, ny


class CarringtonTransform(CompositeTransform):

    def __init__(self,
                 hdr,
                 radius_correction=1.0,
                 direction='forward',
                 reference_date=None,
                 rate_wave=None,
                 zclip=0,
                 c2limb=False):
        if 'CROTA' in hdr:
            roll = hdr['CROTA']
        elif 'CROTA2' in hdr:
            roll = hdr['CROTA2']
        else:
            raise ValueError('No roll value found in header')

        self.reference_date = reference_date

        cos = np.cos(np.radians(roll))
        sin = np.sin(np.radians(roll))

        dx = cos * hdr['CRVAL1'] + sin * hdr['CRVAL2']
        dy = -sin * hdr['CRVAL1'] + cos * hdr['CRVAL2']

        self.transform_2 = SphericalTransform(
            (hdr['CRPIX1'] - 1) - dx / hdr['CDELT1'],
            (hdr['CRPIX2'] - 1) - dy / hdr['CDELT2'],
            hdr['DSUN_OBS'] / (radius_correction * astropy.constants.R_sun.value),
            hdr['CRLN_OBS'],
            hdr['CRLT_OBS'],
            roll,
            hdr['CDELT1'],
            hdr['CDELT2'],
            direction=direction,
            zclip=zclip,
            c2limb=c2limb,
            degrees=True
        )
        if self.reference_date is None:
            self.reference_date = Time(hdr['DATE-OBS'])
        delta_t = (Time(hdr['DATE-OBS']) - Time(self.reference_date)).value
        self.transform_1 = DifferentialRotationTransform(
            delta_t,
            rate_wave,
            degrees=True
        )


class DistortionMatrix(Transform):
    """
    Class used to store distortion matrix data, as computed by Zemax

    The distortion matrix can be visualized using the plot method

    file: name of distortion poynomials file if rebuild is False,
          name of Zemax output file if rebuild is True
    rebuild: if True, read in Zemax data, performs polynomial fit and
             stores coefficients in specified file
    """

    class DistortionPolynomial:
        """
        Used to store and manipulate representations of distortion in the form
        of bivariate polynomials
        """

        def __init__(self,
                     file=None,
                     direction=None,
                     coefficients=None):
            """
            file: if present, then polynomial coefficients and scale (linear
                  approximation) are read from the [direction] section of the
                  file.
            direction: must be either 'field2pos' or 'pos2field'
            coefficients: if present, polynomial corefficients and scale are
                          iniitialized from a (scale, coeffs) 2-tuple. coeffs
                          must be a 2-tuple of [degree+1, degree+1] ndarrays.
            """
            if file is not None and coefficients is not None:
                raise ValueError

            if file is not None:
                if direction is None:
                    raise ValueError
                if os.path.isfile(file):
                    self.file = file
                    self.scale = None
                    self.coefficients = None
                    self.read(direction)
                else:
                    raise FileNotFoundError
            elif coefficients is not None:
                self.scale = coefficients[0]
                self.coefficients = coefficients[1]

        def __call__(self, x, y):
            """
            Evaluates distortion polynomial at points (x, y)
            Returns a tuple of output values.

            x, y: values at which to evaluate the distortion
            """

            def polyval(x, y, coefficients):
                """
                Evaluates bivariate polynomial at points (x, y)
                For some reason faster than numpy.polynomial.polynomial.polyval2d
                """
                degree = coefficients.shape[0] - 1
                poly = np.zeros_like(x)
                for j in range(degree, -1, -1):
                    dum = np.full_like(x, coefficients[degree, j])
                    for i in range(degree - 1, -1, -1):
                        dum *= x
                        dum += coefficients[i, j]
                    poly *= y
                    poly += dum
                return poly

            xpoly = polyval(x, y, self.coefficients[0])
            ypoly = polyval(x, y, self.coefficients[1])
            return xpoly, ypoly

        def read(self, direction):
            """
            Reads in distortion polynomial data file
            """

            def reform_poly(items, axis, scale):
                """
                Reformats the data read in the txt file
                """
                degree = np.int(items[axis + 'degree'])
                d = np.asarray(items['d' + axis + 'k'].split(),
                               dtype=np.float32).reshape((degree + 1, degree + 1))
                if axis == 'x':
                    d[1, 0] += scale  # poly encode the distortion: add scale
                elif axis == 'y':
                    d[0, 1] += scale  # poly encode the distortion: add scale
                else:
                    raise ValueError('Invalid axis')
                return d

            config = configparser.ConfigParser()
            config.read(self.file)
            items = dict(config.items(direction))
            self.scale = np.float32(items['scale'])
            self.coefficients = (reform_poly(items, 'x', self.scale),
                                 reform_poly(items, 'y', self.scale))

        #            self.coefficients = {}
        #            for sec in config.sections():
        #                self.coefficients[sec] = {'scale': scale,
        #                                          'xcoeffs': reform_poly(items, 'x', scale),
        #                                          'ycoeffs': reform_poly(items, 'y', scale)}

        def write(self):
            config = configparser.ConfigParser()
            config.write(self.file)

    class ZemaxData():
        """
        Used to store zemax distortion data

        xfield, yfield: field angles in degrees
        xchief, ychief: chief ray position ion the detector
        maxfield: maximum field angle considered in Zemax
        nsamples: number of sampling points on both axes
        """

        def __init__(self, file):

            if os.path.isfile(file):
                self.file = file
            else:
                raise FileNotFoundError

            self.xchief = None  # chief ray x on detector
            self.xfield = None  # corresponding x field angle
            self.ychief = None  # chief ray y on detector
            self.yfield = None  # corresponding y field angle
            self.maxfield = None  # maximum field angle considered
            self.nsamples = None  # number of samples in x an y
            self.step = None

            if file.endswith('.txt'):
                self.read_txt()
            elif file.endswith('.fits'):
                self.read_fits()
            else:
                raise ValueError('Invalid file extension')

        def read_fits(self):
            with astropy.io.fits.open(self.file) as hdu:
                self.maxfield = hdu[0].header['MAXFIELD']
                self.step = hdu[0].header['STEPSIZE']
                self.nsamples = hdu[1].header['NAXIS1']
                self.xchief = hdu[1].data
                self.ychief = hdu[2].data
                grid = np.linspace(-self.maxfield + self.step / 2, self.maxfield - self.step / 2, self.nsamples)
                self.xfield, self.yfield = np.meshgrid(grid, grid)

        def read_txt(self):
            """
            Reads in output the ASCII file written by the Zemax macro
            """

            try:
                columns = ['xchief', 'ychief', 'hx', 'hy']
                data = astropy.io.ascii.read(self.file,
                                             guess=False,
                                             comment=';',
                                             format='commented_header',
                                             names=columns)
            except:
                raise IOError()

            # maxfield: maximum field in degrees considered in Zemax
            # nsamples: number of data points on each axis
            # stored in the ASCII file in comment lines
            comments = data.meta['comments']
            self.maxfield = np.float((comments[-5]).split()[-1])
            self.nsamples = np.int(np.float((comments[-4]).split()[-1]))
            self.step = 2 * self.maxfield / self.nsamples

            shape = (self.nsamples, self.nsamples)

            self.xchief = np.asarray(data.columns.pop('xchief')).reshape(shape)

            self.ychief = np.asarray(data.columns.pop('ychief')).reshape(shape)

            self.xfield = np.asarray(data.columns.pop('hx')).reshape(shape)
            self.xfield *= self.maxfield

            self.yfield = np.asarray(data.columns.pop('hy')).reshape(shape)
            self.yfield *= self.maxfield

        def write_fits(self):
            pass
            # outfile = self.file.replace(".txt", ".fits")
            # primary_hdu = astropy.io.fits.PrimaryHDU()
            # primary_hdu.header['MAXFIELD'] = self.maxfield
            # primary_hdu.header['STEPSIZE'] = 2*self.maxfield/self.nsamples
            # primary_hdu.header['UNITS'] = 'degrees'
            # hdu_list = astropy.io.fits.HDUList([primary_hdu])
            # #On purpose inversion of xchief and ychief wrt Zemax conventions
            # hdu_list.append(astropy.io.fits.ImageHDU(self.ychief))
            # hdu_list.append(astropy.io.fits.ImageHDU(self.xchief))
            # hdu_list.writeto(outfile, overwrite=True)

        def fit(self, direction):
            """
            """

            if direction == 'pos2field':
                x = self.xchief
                y = self.ychief
                f1 = self.xfield
                f2 = self.yfield
                d1 = 5
                d2 = 3
                mx1 = True
                mx2 = False
            elif direction == 'field2pos':
                x = self.xfield
                y = self.yfield
                f1 = self.xchief
                f2 = self.ychief
                d1 = 5
                d2 = 3
                mx1 = True
                mx2 = False

            c1 = polyfit2d(x, y, f1, d1, maxdegree=mx1)
            c2 = polyfit2d(x, y, f2, d2, maxdegree=mx2)

            return (c1, c2)

        def write_polynomials(self, outfile):

            comment_str = "\
            # Stores the polynomial fit parameters of the plate\
            # plate scale distortion in FSI images.\
            #\
            # The distrotion field computed from the Zemax model of\
            # the instrument is fitted using the IDL sfit function.\
            # The degree of the polynomials used to fit the x and\
            # y axes can be different. The dimensions of the arrays\
            # dxk and dyk may vary depending on whether or not the\
            # /max_degree keyword was used for the fit. See the\
            # documentation of the sfit function for details.\
            #\
            # This file was automatically generated.\
            #\
            # scale   - the average plate scale\
            # dxk     - the polynomial fit parameters for the x axis\
            # dyk     - the polynomial fit parameters for the y axis\
            # xdegree - the degree of the polynomial fit for the x axis\
            # ydegree - the degree of the polynomial fit for the y axis\
            "
            config = configparser.ConfigParser()
            config.comment(comment_str)
            config['gen'] = {'phys_pix_size': '0.01',
                             'ref_x_pix': '1536',
                             'ref_y_pix': '1536'}
            config['field2pos'] = {'scale': self.field2pos.scale,
                                   'xdegree': '5',
                                   'dxk': self.field2pos.coefficients[0].flatten(),
                                   'ydegree': '3',
                                   'dyk': self.field2pos.coefficients[1].flatten()}
            config['pos2field'] = {'scale': self.pos2field.scale,
                                   'xdegree': '5',
                                   'dxk': self.pos2field.coefficients[0].flatten(),
                                   'ydegree': '3',
                                   'dyk': self.pos2field.coefficients[1].flatten()}
            with open(outfile, 'w') as configfile:
                config.write(configfile)

        def field2pos(self, x, y):
            nx = self.nsamples * (x / self.maxfield + 1) / 2
            ny = self.nsamples * (y / self.maxfield + 1) / 2
            x = interpol2d(self.xchief, nx, ny)
            y = interpol2d(self.ychief, nx, ny)
            return x, y

        def pos2field(self, x, y):
            xidx, yidx = np.indices(self.xchief.shape)
            points = np.stack((self.xchief.ravel(), self.ychief.ravel()), axis=1)
            nx = scipy.interpolate.griddata(points, xidx.ravel(), (x, y), method='nearest')
            ny = scipy.interpolate.griddata(points, yidx.ravel(), (x, y), method='nearest')
            x = interpol2d(self.xfield, nx, ny)
            y = interpol2d(self.yfield, nx, ny)
            return x, y

    def __init__(self, file, rebuild=False, flip=False, direction='forward'):
        """
        file (string): name of distortion poynomials file if rebuild is False,
                       name of Zemax input file if rebuild is True
        rebuild (boolean)
        """
        super().__init__(direction=direction)

        self.exact = file.endswith('.fits') is True

        if rebuild is False:  # Does not rebuid polynomials from Zemax data
            if os.path.isfile(file):
                self.file = file
                if self.exact:
                    self.phys_pix_size = 0.01
                    self.ref_x_pix = 1535.5
                    self.ref_y_pix = 1535.5
                else:
                    config = configparser.ConfigParser()
                    config.read(self.file)
                    items = dict(config.items('gen'))
                    self.phys_pix_size = np.float32(items['phys_pix_size'])
                    self.ref_x_pix = np.float32(items['ref_x_pix'])
                    self.ref_y_pix = np.float32(items['ref_y_pix'])
            else:
                raise FileNotFoundError
            # pos2field and field2pos are DistortionPolynomial objects initialized
            # from a configuration file
            if self.exact:
                data = self.ZemaxData(file)
                self.pos2field = data.pos2field
                self.field2pos = data.field2pos
            else:
                self.pos2field = self.DistortionPolynomial(file, 'pos2field')
                self.field2pos = self.DistortionPolynomial(file, 'field2pos')
        else:  # Rebuids polynomials from Zemax data
            self.phys_pix_size = np.float32(0.01)
            self.ref_x_pix = np.float32(1535.5)
            self.ref_y_pix = np.float32(1535.5)
            self.zemax_data = self.ZemaxData(file)
            coeffs = self.zemax_data.fit('pos2field')
            self.pos2field = self.DistortionPolynomial(coefficients=(0, coeffs))
            coeffs = self.zemax_data.fit('field2pos')
            self.field2pos = self.DistortionPolynomial(coefficients=(0, coeffs))

        self.flipped_images = flip

    def forward(self, x=None, y=None):
        """
        Computes the forward transform for given input coordinates.
        """
        if self.flipped_images:
            y, x = self.field2pos(-y, x)
            y = -y
        else:
            x, y = self.field2pos(x, y)
        x /= self.phys_pix_size
        x += self.ref_x_pix
        y /= self.phys_pix_size
        y += self.ref_y_pix

        return x, y

    def inverse(self, x=None, y=None):
        """
        Computes the inverse transform for given input coordinates.
        """

        x -= self.ref_x_pix
        x *= self.phys_pix_size
        y -= self.ref_y_pix
        y *= self.phys_pix_size
        if self.flipped_images:
            y, x = self.pos2field(-y, x)
            y = -y
        else:
            x, y = self.pos2field(x, y)
        return x, y

    def plot(self, s=10):
        """
        Produces a distortion plot.

        Blue grid is the original (sky) regular grid, projected on the
            detector assuming a linear transformation from angles on the sky
            to position in mm, using the average plate scale
        Red grid is the true (distorted) output as projected on the dtector by
            the telescope
        The black vectors join the original grid points to the distorted ones

        s: scaling factor used to better visualize the distortion. The vectors
           are magnified by this factor.
        """
        scale = self.pos2field.scale
        x = np.linspace(-3072 / 2 + 1, 3072 / 2, 48) * self.phys_pix_size
        y = np.linspace(-3072 / 2 + 1, 3072 / 2, 48) * self.phys_pix_size
        ox, oy = np.meshgrid(x, y)

        # pos2field and field2pos are DistortionPolynomial objects initialized
        # from a configuration file
        # The evaluate method computes the output for the given input
        nx, ny = self.field2pos(ox * scale, oy * scale)
        dx = nx - ox
        dy = ny - oy

        fig, ax = plt.subplots()
        ax.quiver(x, y, s * dx, s * dy, angles='xy', scale_units='xy', scale=1, linewidth=0.25)

        lines = np.stack((ox, oy), axis=2)
        lc = mcol.LineCollection(lines, colors=(0, 0, 1, 1), linewidth=0.25)
        ax.add_collection(lc)
        lines = np.stack((ox.T, oy.T), axis=2)
        lc = mcol.LineCollection(lines, colors=(0, 0, 1, 1), linewidth=0.25)
        ax.add_collection(lc)

        lines = np.stack((ox + s * dx, oy + s * dy), axis=2)
        lc = mcol.LineCollection(lines, colors=(1, 0, 0, 1), linewidth=0.25)
        ax.add_collection(lc)
        lines = np.stack(((ox + s * dx).T, (oy + s * dy).T), axis=2)
        lc = mcol.LineCollection(lines, colors=(1, 0, 0, 1), linewidth=0.25)
        ax.add_collection(lc)

        ax.set_xlabel('Position on detector (mm)')
        ax.set_ylabel('Position on detector (mm)')
        ax.set_aspect('equal')

        return fig


class Rectifier():
    """
    Rectifier class to be initialized with a Transform object instance.
    Provides an interpolator method to resample images on a regular grid

    order: if using map_coordinates (opencv set to False), this is the order
           of the spline interpolation, must be in the range 0-5.
           if opencv is True, order=0, 1, 2, corresponds to nearest neighbor,
           linear and cubic spline interpolation respectively.
    opencv: if False (default), uses scipy map_coordinates. If True, uses
            opencv remap function (faster)

    dtype: np.float32 (default) or np.float64. 32 bit computations are faster
           and should be sufficient in most cases. If opencv is True,
           computations are made in 32 bits in any case.

    fill: constant value used to fill in missing data
    """

    def __init__(self, transform):
        """
        """

        self.transform = transform
        self.coordinates = None
        self.shape = None
        self.xlims = None
        self.ylims = None

    def __call__(self,
                 image, shape, xlims, ylims,
                 order=1, dst=None, opencv=False, dtype=np.float32, fill=0):
        """
        image: ndarray containing the image to rectify
        shape: shape of the regular grid on which to interpolate
        xlims: x limits of the regular grid on which to interpolate
        ylims: y limits of the regular grid on which to interpolate
        """

        if shape != self.shape or xlims != self.xlims or ylims != self.ylims:
            self.coordinates = np.meshgrid(np.linspace(xlims[0], xlims[1], shape[0], dtype=dtype),
                                           np.linspace(ylims[0], ylims[1], shape[1], dtype=dtype))
            self.shape, self.xlims, self.ylims = shape, xlims, ylims

        x, y = self.coordinates
        dum = self.transform(x=x, y=y)
        if len(dum) == 2:
            nx, ny = dum
            mu = 1
        else:
            nx, ny, mu = dum

        return interpol2d(image, nx, ny, dst=dst, order=order, opencv=opencv, fill=fill) / mu


