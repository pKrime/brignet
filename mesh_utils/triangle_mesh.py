
import copy
import logging
import numpy as np

from collections import namedtuple
from queue import PriorityQueue
from math import sqrt

from . import point_cloud

from importlib import reload
reload(point_cloud)


class BrigMesh:
    def __init__(self, triangles, vertices, v_normals, triangle_areas):
        #self._mesh = mesh
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

    # def triangle_areas(self):
    #     # TODO area of each triangle
    #     pass

    def compute_triangle_normals(self, normalized=True):
        # TODO
        raise NotImplementedError

    def sample_points_uniformlyImpl(self, number_of_points, use_triang_normal=False, seed=None):
        triangle_areas = copy.deepcopy(self.areas)
        surface_area = self.surface_area

        triangle_areas[0] /= surface_area
        for i, area in enumerate(triangle_areas[1:]):
            triangle_areas[i + 1] = area / surface_area + triangle_areas[i]

        # mt = np.random.MT19937(seed)
        # dist = np.random.uniform()

        # pcd = PointCloud()
        # pcd.points_.resize(number_of_points)

        # if self.has_vertex_colors:
        #     pcd.normals_.resize(number_of_points)
        #
        # if use_triang_normal and self.has_triangle_normals:
        #     self.compute_triangle_normals(True)
        #
        # if self.has_vertex_normals:
        #     pcd.colors_.resize(number_of_points)

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

                # if use_triang_normal:
                #     pcd.normals_[i] = self.triangle_normals_[i]
                # elif self.has_vertex_normals:
                #     pcd.normals_[i] = a * self.vertex_normals_[triangle[0]] + \
                #                       b * self.vertex_normals_[triangle[1]] + \
                #                       c * self.vertex_normals_[triangle[2]]
                # if self.has_vertex_colors:
                #     pcd.colors_[i] = a * self.colors_[triangle[0]] + \
                #                      b * self.colors_[triangle[1]] + \
                #                      c * self.colors_[triangle[2]]

                i += 1

        return points

    def sample_points_poissondisk(self, number_of_points, init_factor=5, given_sample=None, use_triang_normal=False, seed=None):
        logger = logging.getLogger("SamplePointsPoissonDisk")
        if number_of_points < 1:
            logger.error("zero or negative number of points")
            return

        if not self.triangles.any:
            logger.error("input mesh has no triangles")
            return

        if not given_sample and init_factor < 1:
            logger.error("please provide either a point cloud or an init_factor greater than 0")
            return

        if given_sample and given_sample.size < number_of_points:
            logger.error("either pass pcl_init with #points > number_of_points, or init_factor > 1")
            return

        if not given_sample:
            pcl = self.sample_points_uniformlyImpl(init_factor * number_of_points, use_triang_normal=use_triang_normal, seed=seed)
        else:
            raise NotImplementedError
            # pcl = PointCloud()
            # pcl.points_ = point_cloud.points_
            # pcl.normals_ = point_cloud.normals_
            # pcl.colors_ = point_cloud.colors_

        # Set-up sample elimination
        alpha = 8    # constant defined in paper
        beta = 0.5   # constant defined in paper
        gamma = 1.5  # constant defined in paper

        pcl_size = len(pcl)
        ratio = number_of_points / pcl_size
        r_max = 2 * sqrt((self.surface_area / number_of_points) / (2 * sqrt(3.0)))
        r_min = r_max * beta * (1 - pow(ratio, gamma))

        weights = [0] * pcl_size
        deleted = [False] * pcl_size
        kdtree = point_cloud.PointCloud(pcl)

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

        class QueueEntry:
            def __init__(self, idx, weight):
                self.idx = idx
                self.weight = weight

            def __repr__(self):
                return f'{self.idx, self.weight}'

        for idx in range(pcl_size):
            weight = compute_point_weight(idx)
            queue.append(QueueEntry(idx, weight))

        priority = sorted(queue, key=lambda q: q.weight, reverse=True)
        current_number_of_points = pcl_size
        while current_number_of_points > number_of_points:
            last = priority.pop()
            weight, pidx = last.weight, last.idx
            deleted[pidx] = True
            current_number_of_points -= 1

            # update weights
            nbs = kdtree.get_points(pcl[pidx], r_max)

            for nb, dist in nbs:
                queue[nb.idx].weight = compute_point_weight(nb.idx)

            priority.sort(key=lambda q: q.weight, reverse=True)

        # update pcl
        # has_vert_normal = pcl.HasNormals()
        # has_vert_color = pcl.HasColors()

        for i, point in enumerate(pcl):
            if deleted[i]:
                continue

            yield point
