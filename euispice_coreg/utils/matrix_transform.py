import numpy as np


class MatrixTransform:
    @staticmethod
    def displacement_matrix(ndim=2, dx=0, dy=0):
        if ndim == 2:
            return np.array([[1, 0, dx],
                             [0, 1, dy],
                             [0, 0, 1]])
        else:
            raise NotImplementedError
        # xx, yy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))

    @staticmethod
    def rotation_matrix(ndim=2, theta=0, units='radian'):
        if ndim == 2:
            identity = lambda x: x
            if units == 'radian':
                theta = identity(theta)
            elif units == 'degree':
                theta = np.radians(theta)
            return np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0],
                             [0, 0, 1]])
        else:
            raise NotImplementedError

    @staticmethod
    def linear_transform(*args, matrix, ):
        if len(args) == 2:
            assert matrix.ndim == 2
            assert args[0].shape == args[1].shape
            xx = args[0]
            yy = args[1]
            zz = np.ones(args[0].shape)
            xyz = np.stack((xx.ravel(), yy.ravel(), zz.ravel()))
            nx, ny, _ = np.matmul(matrix, xyz)
            return nx.reshape(xx.shape), ny.reshape(yy.shape)
        else:
            raise NotImplementedError

    @staticmethod
    def to_polar_coordinates(*args, direction='forward'):
        if len(args) == 2:
            xx = args[0]
            yy = args[1]

            assert xx.shape == yy.shape

        elif len(args) == 4:
            xx = args[0]
            yy = args[1]
            xc = args[2]
            yc = args[3]
            assert xx.shape == yy.shape

        else:
            raise NotImplementedError
        if direction == 'forward':
            if len(args) == 2:
                xc = xx[round(xx.shape[0] / 2), round(xx.shape[1] / 2)]
                yc = yy[round(xx.shape[0] / 2), round(xx.shape[1] / 2)]
            nr = np.sqrt(np.power(xx - xc, 2) + np.power(yy - yc, 2))
            ntheta = np.arctan2(yy - yc, xx - xc)
            ntheta[np.isnan(ntheta)] = 0
            return nr, ntheta
        elif direction == 'backward':
            if len(args) == 2:
                xc = 0
                yc = 0
            # here xx = r and yy = theta
            nx = np.multiply(xx, np.cos(yy)) + xc
            ny = np.multiply(xx, np.sin(yy)) + yc
        return nx, ny

    @staticmethod
    def polar_transform(*args, theta=0, units='radian'):
        identity = lambda x: x
        if units == 'radian':
            theta = identity(theta)
        elif units == 'degree':
            theta = np.radians(theta)

        if len(args) == 2:

            xx = args[0]
            yy = args[1]
            assert xx.shape == yy.shape
            xc = xx[round(xx.shape[0] / 2), round(xx.shape[1] / 2)]
            yc = yy[round(xx.shape[0] / 2), round(xx.shape[1] / 2)]

        elif len(args) == 4:
            xx = args[0]
            yy = args[1]
            xc = args[2]
            yc = args[3]
            assert xx.shape == yy.shape


        else:
            raise NotImplementedError
        nr, ntheta = MatrixTransform.to_polar_coordinates(xx, yy, xc, yc, direction='forward')
        ntheta = ntheta + theta
        nx, ny = MatrixTransform.to_polar_coordinates(nr, ntheta, xc, yc, direction='backward')
        return nx, ny
