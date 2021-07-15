import copy
import logging
from math import sqrt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
import time

from mathutils.kdtree import KDTree


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

        self._surface_geodesic = None

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

        point_idx = 0
        points = []
        normals = []
        for i, triangle in enumerate(self.triangles):
            n = round(triangle_areas[i] * number_of_points)
            while point_idx < n:
                r1 = np.random.uniform()
                r2 = np.random.uniform()

                a = 1 - sqrt(r1)
                b = sqrt(r1) * (1 - r2)
                c = sqrt(r1) * r2

                points.append(a * self.vertices[triangle[0]] +
                              b * self.vertices[triangle[1]] +
                              c * self.vertices[triangle[2]])

                normals.append(a * self.v_normals[triangle[0]] +
                               b * self.v_normals[triangle[1]] +
                               c * self.v_normals[triangle[2]])

                point_idx += 1

        return points, normals

    def sample_points_poissondisk(self, number_of_points, init_factor=5, approximate=False):
        logger = logging.getLogger("SamplePointsPoissonDisk")
        if number_of_points < 1:
            logger.error("zero or negative number of points")
            return

        if not self.triangles.any:
            logger.error("input mesh has no triangles")
            return

        if init_factor < 1:
            logger.error("please provide either a point cloud or an init_factor greater than 0")

        all_points, normals = self.sample_points_uniformlyImpl(init_factor * number_of_points)

        # Set-up sample elimination
        alpha = 8    # constant defined in paper
        beta = 0.5   # constant defined in paper
        gamma = 1.5  # constant defined in paper

        pcl_size = len(all_points)
        ratio = number_of_points / pcl_size
        r_max = 2 * sqrt((self.surface_area / number_of_points) / (2 * sqrt(3.0)))
        r_min = r_max * beta * (1 - pow(ratio, gamma))

        deleted = [False] * pcl_size

        kd = KDTree(len(all_points))
        for i, v in enumerate(all_points):
            kd.insert(v, i)

        kd.balance()

        def weight_fcn(d):
            if d < r_min:
                d = r_min

            return pow(1 - d / r_max, alpha)

        def weight_fcn_squared(d2):
            d = sqrt(d2)
            return weight_fcn(d)

        def compute_point_weight(pidx0):
            nbs = kd.find_range(all_points[pidx0], r_max)
            weight = 0

            for neighbour, nb_idx, nb_dist in nbs:
                # only count weights if not the same point if not deleted
                if nb_idx == pidx0:
                    continue
                if deleted[nb_idx]:
                    continue

                weight += weight_fcn(nb_dist)

            return weight

        # init weights and priority queue
        queue = []

        for idx in range(pcl_size):
            weight = compute_point_weight(idx)
            queue.append(QueueEntry(idx, weight))

        priority = copy.copy(queue)
        current_number_of_points = pcl_size

        if approximate:
            first_slice = number_of_points + number_of_points * int(init_factor/2)
            step = init_factor * 2
            while current_number_of_points > first_slice:
                priority.sort(key=lambda q: q.weight)
                for p in priority[-step:]:
                    deleted[p.idx] = True
                for p in priority[-step:]:
                    nbs = kd.find_range(all_points[p.idx], r_max)
                    for nb, nb_idx, nb_dist in nbs:
                        queue[nb_idx].weight = compute_point_weight(nb_idx)

                priority = priority[:-step]
                current_number_of_points -= step

        while current_number_of_points > number_of_points:
            priority.sort(key=lambda q: q.weight)

            last = priority.pop()
            weight, pidx = last.weight, last.idx
            deleted[pidx] = True
            current_number_of_points -= 1

            # update weights
            nbs = kd.find_range(all_points[pidx], r_max)

            for nb, nb_idx, nb_dist in nbs:
                queue[nb_idx].weight = compute_point_weight(nb_idx)

        for i, point in enumerate(all_points):
            if deleted[i]:
                continue

            yield point, normals[i]

    def calc_geodesic(self, samples=2000):
        if self._surface_geodesic is not None:
            return self._surface_geodesic

        # RigNet uses 4000 samples, not sure this script can handle that.

        sampled = [(pt, normal) for pt, normal in self.sample_points_poissondisk(samples)]
        sample_points, sample_normals = list(zip(*sampled))

        pts = np.asarray(sample_points)
        pts_normal = np.asarray(sample_normals)

        time1 = time.time()
        
        verts_dist = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
        verts_nn = np.argsort(verts_dist, axis=1)

        N = len(pts)
        conn_matrix = lil_matrix((N, N), dtype=np.float32)
        for p in range(N):
            nn_p = verts_nn[p, 1:6]
            norm_nn_p = np.linalg.norm(pts_normal[nn_p], axis=1)
            norm_p = np.linalg.norm(pts_normal[p])
            cos_similar = np.dot(pts_normal[nn_p], pts_normal[p]) / (norm_nn_p * norm_p + 1e-10)
            nn_p = nn_p[cos_similar > -0.5]
            conn_matrix[p, nn_p] = verts_dist[p, nn_p]
        dist = dijkstra(conn_matrix, directed=False, indices=range(N),
                        return_predecessors=False, unweighted=False)

        # replace inf distance with euclidean distance + 8
        # 6.12 is the maximal geodesic distance without considering inf, I add 8 to be safer.
        inf_pos = np.argwhere(np.isinf(dist))
        if len(inf_pos) > 0:
            euc_distance = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
            dist[inf_pos[:, 0], inf_pos[:, 1]] = 8.0 + euc_distance[inf_pos[:, 0], inf_pos[:, 1]]

        verts = self.vertices
        vert_pts_distance = np.sqrt(np.sum((verts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
        vert_pts_nn = np.argmin(vert_pts_distance, axis=0)
        surface_geodesic = dist[vert_pts_nn, :][:, vert_pts_nn]
        time2 = time.time()
        logger = logging.getLogger("Geodesic")
        logger.debug('surface geodesic calculation: {} seconds'.format((time2 - time1)))

        self._surface_geodesic = surface_geodesic
        return surface_geodesic
