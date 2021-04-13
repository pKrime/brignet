
import copy
import logging
import numpy as np
from math import sqrt

from . import point_utils

from importlib import reload
reload(point_utils)


class QueueEntry:
    def __init__(self, idx, weight):
        self.idx = idx
        self.weight = weight

    def __repr__(self):
        return f'{self.idx, self.weight}'


class MeshSampler:
    def __init__(self, triangles, vertices, v_normals, triangle_areas):
        self._has_vertex_normals = False
        self._has_vertex_colors = False

        self.triangles = triangles
        self.vertices = vertices
        self.v_normals = v_normals
        self.areas = triangle_areas

    @property
    def has_vertex_normals(self):
        return bool(self.v_normals)

    @property
    def has_vertex_colors(self):
        return self._has_vertex_colors

    @property
    def surface_area(self):
        return sum(self.areas)

    def compute_triangle_normals(self, normalized=True):
        # TODO
        raise NotImplementedError

    def sample_points_uniformlyImpl(self, number_of_points):
        triangle_areas = copy.deepcopy(self.areas)
        surface_area = self.surface_area

        triangle_areas[0] /= surface_area
        for i, area in enumerate(triangle_areas[1:]):
            triangle_areas[i + 1] = area / surface_area + triangle_areas[i]

        points = []
        for i, triangle in enumerate(self.triangles):
            n = round(triangle_areas[i] * number_of_points)
            while i < n:
                r1 = np.random.uniform()
                r2 = np.random.uniform()

                a = 1 - sqrt(r1)
                b = sqrt(r1) * (1 - r2)
                c = sqrt(r1) * r2

                points.append(a * self.vertices[triangle[0]] +
                              b * self.vertices[triangle[1]] +
                              c * self.vertices[triangle[2]])

                i += 1

        return points

    def sample_points_poissondisk(self, number_of_points, init_factor=5):
        logger = logging.getLogger("SamplePointsPoissonDisk")
        if number_of_points < 1:
            logger.error("zero or negative number of points")
            return

        if not self.triangles.any:
            logger.error("input mesh has no triangles")
            return

        if init_factor < 1:
            logger.error("please provide either a point cloud or an init_factor greater than 0")

        pcl = self.sample_points_uniformlyImpl(init_factor * number_of_points)

        # Set-up sample elimination
        alpha = 8    # constant defined in paper
        beta = 0.5   # constant defined in paper
        gamma = 1.5  # constant defined in paper

        pcl_size = len(pcl)
        ratio = number_of_points / pcl_size
        r_max = 2 * sqrt((self.surface_area / number_of_points) / (2 * sqrt(3.0)))
        r_min = r_max * beta * (1 - pow(ratio, gamma))

        deleted = [False] * pcl_size
        kdtree = point_utils.PointKDtree(pcl)

        def weight_fcn(d2):
            d = sqrt(d2)
            if d < r_min:
                d = r_min

            return pow(1 - d / r_max, alpha)

        def compute_point_weight(pidx0):
            nbs = kdtree.get_points(pcl[pidx0], r_max)
            weight = 0

            for neighbour, dist2 in nbs:
                # only count weights if not the same point if not deleted
                if neighbour.idx == pidx0:
                    continue
                if deleted[neighbour.idx]:
                    continue

                weight += weight_fcn(dist2)

            return weight

        # init weights and priority queue
        queue = []

        for idx in range(pcl_size):
            weight = compute_point_weight(idx)
            queue.append(QueueEntry(idx, weight))

        priority = copy.copy(queue)
        current_number_of_points = pcl_size
        while current_number_of_points > number_of_points:
            priority.sort(key=lambda q: q.weight)

            last = priority.pop()
            weight, pidx = last.weight, last.idx
            deleted[pidx] = True
            current_number_of_points -= 1

            # update weights
            nbs = kdtree.get_points(pcl[pidx], r_max)

            for nb, dist in nbs:
                queue[nb.idx].weight = compute_point_weight(nb.idx)

        for i, point in enumerate(pcl):
            if deleted[i]:
                continue

            yield point
