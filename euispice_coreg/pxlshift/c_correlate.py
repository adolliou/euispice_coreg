# From  @wtbarnes
# wtbarnes

import numpy as np
from tqdm import tqdm
from numba import jit


@jit(nopython=True, inline='always', cache=True)
def c_correlate3D(s_1, s_2, lags):
    """
    Numpy implementation of c_correlate.pro IDL routine
    """
    # ensure signals are of equal length

    n_s = s_1.shape[2]
    # center both signals
    s_1_center = s_1 - np.repeat(s_1.mean(axis=(2))[:, :, np.newaxis], s_1.shape[2], axis=2)
    s_2_center = s_2 - np.repeat(s_2.mean(axis=(2))[:, :, np.newaxis], s_2.shape[2], axis=2)

    # allocate space for correlation
    correlation = np.zeros((s_1_center.shape[0], s_1_center.shape[1], lags.shape[0]))
    # iterate over lags
    for i, l in range(len(lags)):
        l = lags[i]
        if l >= 0:
            tmp = np.multiply(s_1_center[:, :, :(n_s - l)], s_2_center[:, :, l:])
        else:
            tmp = np.multiply(s_1_center[:, :, -l:], s_2_center[:, :, :(n_s + l)])
        correlation[:, :, i] = tmp.sum(axis=2)

    # Divide by standard deviation of both
    correlation = np.divide(correlation, np.repeat(
        np.sqrt(np.multiply((np.power(s_1_center, 2)).sum(axis=(2)), np.power(s_2_center, 2).sum(axis=(2))))[:, :,
        np.newaxis], correlation.shape[2], axis=2))

    return correlation


@jit(nopython=True, inline='always', cache=True)
def c_correlate(s_1, s_2, lags):
    """
    Numpy implementation of c_correlate.pro IDL routine
    """
    # ensure signals are of equal length
    n_s = s_1.shape[0]
    # center both signals
    s_1_center = s_1 - s_1.mean()
    s_2_center = s_2 - s_2.mean()
    # allocate space for correlation
    correlation = np.zeros(len(lags), dtype="float32")
    # iterate over lags
    for i in range(len(lags)):
        l = lags[i]
        if l >= 0:
            tmp = s_1_center[:(n_s - l)] * s_2_center[l:]
        else:
            tmp = s_1_center[-l:] * s_2_center[:(n_s + l)]
        correlation[i] = tmp.sum()
    # Divide by standard deviation of both
    correlation = correlation/np.sqrt((s_1_center ** 2).sum() * (s_2_center ** 2).sum())

    return correlation
