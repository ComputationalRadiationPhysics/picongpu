import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import colorConverter
from warnings import warn


def get_different_colors(n, cmap="tab20"):
    """
    Parameters
    ----------
    n: int
        the number of distinct colors
    cmap: str
        name of the matplotlib colormap to use when the
        prop cycle does not give enough colors.
    Returns
    -------
    a list of length n with unique color codes where each
    code is a list of length 3.
    """

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if n <= len(colors):
        return colors[:n]
    else:
        # This allows arbitrary number of colors but colors are less well
        # distinguishable
        try:
            cm = plt.get_cmap(cmap, n)
        except ValueError:
            fallback_cmap = "tab20"
            warn("Colormap {0} not known. Using {1} instead!".format(
                cmap, fallback_cmap))
            cm = plt.get_cmap(fallback_cmap, n)

        return cm(np.linspace(0, 1, n))


def get_different_colormaps(n):

    myCmaps = [None] * n
    colors = get_different_colors(n)

    # generate the colors for colormaps
    for i, c in enumerate(colors):
        myColor1 = colorConverter.to_rgba('white', alpha=0.5)
        myColor2 = colorConverter.to_rgba(c, alpha=0.5)
        myCmap = mpl.colors.LinearSegmentedColormap.from_list(
            'my_cmap_' + str(i), [myColor1, myColor2], 1200)
        # create the _lut array, with rgba values
        myCmap._init()
        myCmaps[i] = myCmap

    return myCmaps
