import numpy as np
from scipy.spatial import distance


def calc_pts_diameter2(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set). Faster but requires more memory than
    calc_pts_diameter.

    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    dists = distance.cdist(pts, pts, 'euclidean')
    diameter = np.max(dists)
    return diameter
