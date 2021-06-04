import numpy as np
from math import sqrt


class Node:
    def __init__(self, idx, location, left, right, axis):
        self.idx = idx
        self.location = location
        self.left = left
        self.right = right
        self.axis = axis

    def __repr__(self):
        return "{0} - L {1} R {2}".format(self.location, self.left, self.right)

    def traverse_closer(self, position, dist2, stack=[], dists=[]):
        diff = [pos - self.location[i] for i, pos in enumerate(position)]
        if self.left and self.right:
            dist1 = diff[self.axis]

            if dist1 < 0:
                self.left.traverse_closer(position, dist2, stack, dists)
            else:
                self.right.traverse_closer(position, dist2, stack, dists)

        d2 = sqrt(sum(i**2 for i in diff))
        if d2 < dist2:
            stack.append(self)
            dists.append(d2)


class PointKDtree:
    """uses a k-d tree for storing points."""
    def __init__(self, pts):
        # FIXME: make sure it works with even number of points
        self._dimensions = len(pts[0])  # assume all points have same dimensions
        indexed = [(i, pt) for i, pt in enumerate(pts)]

        self.root = self._build_kd_tree(indexed)

    @staticmethod
    def _split_axis(indexed_pts):
        """Returns axis with the largest span, used as the splitting axis for building the k-d tree"""
        pts = [pt[1] for pt in indexed_pts]
        bound_min = np.min(pts, 0)
        bound_max = np.max(pts, 0)

        diff = bound_max - bound_min
        dmax = max(diff)

        return next(i for i, d in enumerate(diff) if d == dmax)

    def _build_kd_tree(self, pts):
        """Each pt is a (index, location) tuple"""
        size = len(pts)
        if size == 0:
            return

        axis = self._split_axis(pts)
        sort_pts = sorted(pts, key=lambda pt: pt[1][axis])

        median = size >> 1
        return Node(
            idx=sort_pts[median][0],
            location=sort_pts[median][1],
            left=self._build_kd_tree(sort_pts[:median]),
            right=self._build_kd_tree(sort_pts[median + 1:]),
            axis=axis
        )

    def get_points(self, position, radius):
        """Returns all points to the given position within the given radius.
        Calls the given pointFound function for each point found.

        :param position:
        :param radius:
        :return:
        """
        stack = []
        dists = []
        self.root.traverse_closer(position, radius, stack, dists)

        for i, node in enumerate(stack):
            yield node, dists[i]


if __name__ == "__main__":
    # test
    import random
    from random import randrange

    dimensions = 3
    size = 50
    points = [[random.uniform(0, 1) for d in range(dimensions)] for i in range(size)]

    p_cloud = PointKDtree(points)
    pt_idx = randrange(0, size)
    radius = 4.0

    print("neighbours of {0} ({1}) in a radius of {2}".format(pt_idx, points[pt_idx], radius))
    p_in_rad = p_cloud.get_points(points[pt_idx], radius)

    for pin, dist in p_in_rad:
        print(pin.idx, pin.location, dist)
