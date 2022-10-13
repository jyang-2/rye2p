import numpy as np
from pathlib import Path
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def norm_traces(sp):
    """ Normalizes fluorescence responses to (0, 1)"""
    spn = zscore(sp, axis=1)
    spn = np.minimum(8, spn)
    spn = np.maximum(-4, spn) + 4
    spn /= 12
    return spn


def running_average(sp, nbin=50):
    """ Performs a running average filter over the first dimension of X. Faster than gaussian filtering.

    Args:
        sp (np.ndarray): fluorescence responses, cells x time
        nbin (int): bin size

    Returns:
        sp_avg (np.ndarray): downsampled fluorescence responses, cells x (time/bins)
    """
    sp_avg = np.cumsum(sp, axis=0)
    sp_avg = sp_avg[nbin:, :] - sp_avg[:-nbin, :]
    return sp_avg


def plot_rastermap(S, nbin=50, figure_kwargs={}, imshow_kwargs={}):

    S_filt = running_average(S, nbin=nbin)
    S_filt = zscore(S_filt, axis=1)

    figure_kwargs0 = dict(figsize=(11, 8.5), tight_layout=True)
    figure_kwargs0.update(figure_kwargs)

    imshow_kwargs0 = dict(vmin=0, vmax=1, aspect='auto', cmap='gray_r', origin='lower')
    imshow_kwargs0.update(imshow_kwargs)

    fig, ax = plt.subplots(**figure_kwargs0)
    img = ax.imshow(S_filt, **imshow_kwargs0)
    fig.colorbar(img)

    plt.xlabel('time points')
    plt.ylabel('sorted neurons')
    return fig, ax


def draw_stim_lines(ax, x, labels, colors):
    """Draws line on axes to indicate stimulus times."""
    axvline_kwargs = dict(linewidth=0.5, linestyle='--')

    for x0, label, clr in zip(x, labels, colors):
        ax.axvline(x0, label=label, color=clr, **axvline_kwargs)

        ax.text(x0, ax.get_ylim()[1]+20, label,
                fontsize=8, rotation=45,
                verticalalignment='bottom',
                horizontalalignment='left')

    return ax

