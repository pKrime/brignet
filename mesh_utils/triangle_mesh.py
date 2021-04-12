
import logging
import numpy as np

from collections import namedtuple
from queue import PriorityQueue
from math import sqrt

from . import point_cloud

class BrigMesh:
    def __init__(self, mesh):
        self._mesh = mesh
        self._has_vertex_normals = False
        self._has_vertex_colors = False

        self.triangles = []

    @property
    def has_vertex_normals(self):
        return self._has_vertex_normals

    @property
    def has_vertex_colors(self):
        return self._has_vertex_colors

    def triangle_areas(self):
        # TODO area of each triangle
        pass

    def surface_area(self):
        # TODO
        # sum of all triangle areas
        pass

    def compute_triangle_normals(self, normalized=True):
        # TODO
        raise NotImplementedError

    def sample_points_uniformlyImpl(self, number_of_points, triangle_areas, surface_area, use_triang_normal, seed=None):
        triangle_areas[0] /= surface_area
        for i, area in enumerate(triangle_areas[1:]):
            triangle_areas[i + 1] = area / surface_area + triangle_areas[i]

        # mt = np.random.MT19937(seed)
        # dist = np.random.uniform()

        pcd = PointCloud()
        pcd.points_.resize(number_of_points)

        if self.has_vertex_colors:
            pcd.normals_.resize(number_of_points)

        if use_triang_normal and self.has_triangle_normals:
            self.comput_triangle_normals(True)

        if self.has_vertex_normals:
            pcd.colors_.resize(number_of_points)

        for i, triangle in enumerate(self.triangles):
            n = round(triangle_areas[i] * number_of_points)
            while i < n:
                r1 = np.random.uniform()
                r2 = np.random.uniform()

                a = 1- sqrt(r1)
                b = sqrt(r1) * (1 - r2)
                c = sqrt(r1) * r2

                pcd.points_[i] = a * self.vertices_[triangle[0]] + \
                                 b * self.vertices_[triangle[1]] + \
                                 c * self.vertices_[triangle[2]]

                if use_triang_normal:
                    pcd.normals_[i] = self.triangle_normals_[i]
                elif self.has_vertex_normals:
                    pcd.normals_[i] = a * self.vertex_normals_[triangle[0]] + \
                                      b * self.vertex_normals_[triangle[1]] + \
                                      c * self.vertex_normals_[triangle[2]]
                if self.has_vertex_colors:
                    pcd.colors_[i] = a * self.colors_[triangle[0]] + \
                                     b * self.colors_[triangle[1]] + \
                                     c * self.colors_[triangle[2]]

                i += 1

        return pcd

    def sample_points_poissondisk(self, number_of_points, init_factor=5, point_cloud=None, use_triang_normal=False, seed=None):
        logger = logging.getLogger("SamplePointsPoissonDisk")
        if number_of_points < 1:
            logger.error("zero or negative number of points")
            return

        if not self.triangles:
            logger.error("input mesh has no triangles")
            return

        if not point_cloud and init_factor < 1:
            logger.error("please provide either a point cloud or an init_factor greater than 0")
            return

        if point_cloud and point_cloud.size < number_of_points:
            logger.error("either pass pcl_init with #points > number_of_points, or init_factor > 1")
            return

        if not point_cloud:
            pcl = self.sample_points_uniformlyImpl(init_factor * number_of_points, use_triang_normal=use_triang_normal, seed=seed)
        else:
            pcl = PointCloud()
            pcl.points_ = point_cloud.points_
            pcl.normals_ = point_cloud.normals_
            pcl.colors_ = point_cloud.colors_

        # Set-up sample elimination
        alpha = 8    # constant defined in paper
        beta = 0.5   # constant defined in paper
        gamma = 1.5  # constant defined in paper
        ratio = number_of_points / pcl.points_.size()
        r_max = 2 * sqrt( (self.surface_area / number_of_points) / (2 * sqrt(3.0)))
        r_min = r_max * beta * (1 - pow(ratio, gamma))

        weights = [0] * pcl.points_.size()
        deleted = [False] * pcl.points_.size()
        kdtree = KDTreeFlann(pcl)

        def weight_fcn(d2):
            d = sqrt(d2)
            if d < r_min:
                d = r_min

            return pow(1 - d / r_max, alpha)

        def compute_point_weight(pidx0):
            nbs = []  # indices of neighbours
            dists2 = []
            kdtree.SearchRadius(pcl.points_[pidx0], r_max, nbs, dists2)
            weight = 0

            for i, pidx1 in enumerate(nbs):
                # only count weights if not the same point if not deleted
                if pidx1 == pidx0:
                    continue
                if deleted[pidx1]:
                    continue
                weight += weight_fcn(dists2[i])

            weights[pidx0] = weight


        # init weights and priority queue
        QueueEntry = namedtuple('QueueEntry', 'idx weight')
        # order points per weight, lower weights last so they can be popped
        priority = PriorityQueue()
        for idx in pcl.points_:
            priority.put((idx, compute_point_weight(idx)))

        current_number_of_points = len(pcl.points_)
        while current_number_of_points > number_of_points:

            pidx, weight = priority.get()
            deleted[pidx] = True
            current_number_of_points -= 1

            # update weights
            nbs = []
            dists2 = []
            kdtree.SearchRadius(pcl.points_[pidx], r_max, nbs, dists2)

            for nb in nbs:
                compute_point_weight(nb)
                priority.put(nb, compute_point_weight(nb))

        # update pcl
        has_vert_normal = pcl.HasNormals()
        has_vert_color = pcl.HasColors()
        next_free = 0

        for i, point in pcl.points_:
            if deleted[i]:
                continue

            pcl.points_[next_free] = point
            if has_vert_normal:
                pcl.normals_[next_free] = pcl.normals_[i]
            if has_vert_color:
                pcl.colors_[next_free] = pcl.colors_[i]

        pcl.resize(next_free)
        return pcl
