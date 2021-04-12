
from operator import itemgetter

import numpy as np
from math import sqrt


class Node:
    def __init__(self, location, left, right, axis):
        self.location = location
        self.left = left
        self.right = right
        self.axis = axis

    def __repr__(self):
        return "{0} - L {1} R {2}".format(self.location, self.left, self.right)

    def traverse_closer(self, position, dist2, stack=[]):
        diff = [pos - self.location[i] for i, pos in enumerate(position)]
        if self.left and self.right:
            dist1 = diff[self.axis]

            if dist1 < 0:
                self.left.traverse_closer(position, dist2, stack)
            else:
                self.right.traverse_closer(position, dist2, stack)

        d2 = sqrt(sum(i**2 for i in diff))
        if d2 < dist2:
            stack.append(self)


class PointCloud:
    """uses a k-d tree for storing points."""
    def __init__(self, pts):
        # FIXME: make sure it works with even number of points
        self._dimensions = len(pts[0])  # assume all points have same dimensions
        self.root = self._build_kd_tree(pts)

    @staticmethod
    def _split_axis(pts):
        """Returns axis with the largest span, used as the splitting axis for building the k-d tree"""
        bound_min = np.min(pts, 0)
        bound_max = np.max(pts, 0)

        diff = bound_max - bound_min
        dmax = max(diff)

        return next(i for i, d in enumerate(diff) if d == dmax)

    def _build_kd_tree(self, pts):
        size = len(pts)
        if size == 0:
            return

        axis = self._split_axis(pts)
        sort_pts = sorted(pts, key=itemgetter(axis))

        median = size >> 1
        return Node(
            location=sort_pts[median],
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
        self.root.traverse_closer(position, radius, stack)

        for node in stack:
            yield node.location


if __name__ == "__main__":
    # test
    import random

    dimensions = 3
    size = 50
    #points = [[random.uniform(0, 1) for d in range(dimensions)] for i in range(size)]
    points = [(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3), (1, 1)]

    p_cloud = PointCloud(points)
    # print(p_cloud.root)
    #
    # position = (8, 2)
    # print("look up", position, "dist", 1)
    #
    # stack = []
    # p_cloud.root.traverse_closer(position, 4, stack)
    #
    # for p in stack:
    #     print(p.location)

    # for p in closer:
    #     print(p.location)
    #p_cloud.root.left.traverse_closer(position, 1)

    p_in_rad = p_cloud.get_points((8, 5), 4)
    for pin in p_in_rad:
        print(pin)
