import numpy as np
from scipy.signal import convolve
from random import random

import bpy
import bmesh
from mathutils import Matrix
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def normalize_obj(mesh_v):
    dims = [max(mesh_v[:, 0]) - min(mesh_v[:, 0]),
            max(mesh_v[:, 1]) - min(mesh_v[:, 1]),
            max(mesh_v[:, 2]) - min(mesh_v[:, 2])]
    scale = 1.0 / max(dims)
    pivot = np.array([(min(mesh_v[:, 0]) + max(mesh_v[:, 0])) / 2, min(mesh_v[:, 1]),
                      (min(mesh_v[:, 2]) + max(mesh_v[:, 2])) / 2])
    mesh_v[:, 0] -= pivot[0]
    mesh_v[:, 1] -= pivot[1]
    mesh_v[:, 2] -= pivot[2]
    mesh_v *= scale
    return mesh_v, pivot, scale


def obj_simple_export(filepath, vertices, polygons):
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in vertices:
            f.write("v {0:.4f} {1:.4f} {2:.4f}\n".format(*v))

        for p in polygons:
            f.write("f " + " ".join(str(i + 1) for i in p) + "\n")


def get_geo_edges(surface_geodesic, remesh_obj_v):
    edge_index = []
    surface_geodesic += 1.0 * np.eye(len(surface_geodesic))  # remove self-loop edge here
    for i in range(len(remesh_obj_v)):
        geodesic_ball_samples = np.argwhere(surface_geodesic[i, :] <= 0.06).squeeze(1)
        if len(geodesic_ball_samples) > 10:
            geodesic_ball_samples = np.random.choice(geodesic_ball_samples, 10, replace=False)
        edge_index.append(np.concatenate((np.repeat(i, len(geodesic_ball_samples))[:, np.newaxis],
                                          geodesic_ball_samples[:, np.newaxis]), axis=1))
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def get_tpl_edges(remesh_obj_v, remesh_obj_f):
    edge_index = []
    for v in range(len(remesh_obj_v)):
        face_ids = np.argwhere(remesh_obj_f == v)[:, 0]
        neighbor_ids = []
        for face_id in face_ids:
            for v_id in range(3):
                if remesh_obj_f[face_id, v_id] != v:
                    neighbor_ids.append(remesh_obj_f[face_id, v_id])
        neighbor_ids = list(set(neighbor_ids))
        neighbor_ids = [np.array([v, n])[np.newaxis, :] for n in neighbor_ids]
        neighbor_ids = np.concatenate(neighbor_ids, axis=0)
        edge_index.append(neighbor_ids)
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


class Voxels:
    """ Holds a binvox model.
    data is either a three-dimensional numpy boolean array (dense representation)
    or a two-dimensional numpy float array (coordinate representation).

    dims, translate and scale are the model metadata.

    dims are the voxel dimensions, e.g. [32, 32, 32] for a 32x32x32 model.

    scale and translate relate the voxels to the original model coordinates.

    To translate voxel coordinates i, j, k to original coordinates x, y, z:

    x_n = (i+.5)/dims[0]
    y_n = (j+.5)/dims[1]
    z_n = (k+.5)/dims[2]
    x = scale*x_n + translate[0]
    y = scale*y_n + translate[1]
    z = scale*z_n + translate[2]

    """

    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)


class NormalizedMeshData:
    def __init__(self, mesh_obj):
        # triangulate first
        bm = bmesh.new()
        bm.from_object(mesh_obj, bpy.context.evaluated_depsgraph_get())

        # apply modifiers
        mesh_obj.data.clear_geometry()
        for mod in reversed(mesh_obj.modifiers):
            mesh_obj.modifiers.remove(mod)

        bm.to_mesh(mesh_obj.data)
        bpy.context.evaluated_depsgraph_get()
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')

        # rotate -90 deg on X axis
        mat = Matrix(((1.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 1.0, 0.0),
                      (0.0, -1.0, 0.0, 0.0),
                      (0.0, 0.0, 0.0, 1.0)))

        bmesh.ops.transform(bm, matrix=mat, verts=bm.verts[:])
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        mesh_v = np.asarray([list(v.co) for v in bm.verts])
        self.mesh_f = np.asarray([[v.index for v in f.verts] for f in bm.faces])

        self.mesh_vn = np.asarray([list(v.normal) for v in bm.verts])
        self.tri_areas = [t.calc_area() for t in bm.faces]

        bm.free()

        self.mesh_v, self.translation_normalize, self.scale_normalize = normalize_obj(mesh_v)
        self._bvh_tree = BVHTree.FromPolygons(self.mesh_v.tolist(), self.mesh_f.tolist(), all_triangles=True)

    @property
    def bound_min(self):
        return -0.5, 0.0, min(self.mesh_v[:, 2])

    @property
    def bound_max(self):
        return 0.5, 1.0, max(self.mesh_v[:, 2])

    @property
    def bvh_tree(self):
        return self._bvh_tree

    def is_inside_volume(self, vector, samples=2):
        direction = Vector((random(), random(), random())).normalized()
        hits = self._count_hits(vector, direction)

        if hits == 0:
            return False
        if hits == 1:
            return True

        hits_modulo = hits % 2
        for i in range(samples):
            direction = Vector((random(), random(), random())).normalized()
            check_modulo = self._count_hits(vector, direction) % 2
            if hits_modulo == check_modulo:
                return hits_modulo == 1

            hits_modulo = check_modulo

        return hits_modulo == 1

    def _count_hits(self, start, direction):
        hits = 0
        offset = direction * 0.0001
        bvh_tree = self.bvh_tree

        location = bvh_tree.ray_cast(start, direction)[0]

        while location is not None:
            hits += 1
            location = bvh_tree.ray_cast(location + offset, direction)[0]

        return hits

    def _on_surface(self, point, radius):
        return self.bvh_tree.find_nearest(point, radius)

    def _remove_isolated_voxels(self, voxels):
        # convolution against kernel of 'Trues' with a 'False' center
        kernel = np.ones((3, 3, 3), 'int')
        kernel[1, 1, 1] = 0

        # expand to allow convolution on boundaries
        res_x, res_y, res_z = voxels.shape
        blank = np.zeros((res_x, 1, res_z), 'bool')
        expanded = np.hstack((blank, voxels, blank))
        blank = np.zeros((1, res_y + 2, res_z), 'bool')
        expanded = np.vstack((blank, expanded, blank))
        blank = np.zeros((res_x + 2, res_y + 2, 1), 'bool')
        expanded = np.dstack((blank, expanded, blank))

        it = np.nditer(voxels, flags=['multi_index'])
        for vox in it:
            if not vox:
                continue
            x, y, z = it.multi_index

            vox_slice = expanded[x:x+3, y:y+3, z:z+3].astype('int')
            if not convolve(vox_slice, kernel, 'valid').all():
                voxels[x, y, z] = False
            # TODO: if voxels[x, y, z] is True, we might jump the surroundings

    def voxels(self, resolution=88, remove_isolated=True):
        voxels = np.zeros([resolution, resolution, resolution], dtype=bool)
        res_x, res_y, res_z = voxels.shape  # redundant, but might get useful if we change uniform res in the future

        vox_size = 1.0 / resolution
        vox_radius = vox_size / 2.0

        bound_min = self.bound_min
        min_x, min_y, min_z = bound_min

        z_co = min_z + vox_radius
        for z in range(res_z):
            y_co = min_y + vox_radius
            for y in range(res_y):
                x_co = min_x + vox_radius
                for x in range(res_x):
                    voxels[x, y, z] = self.is_inside_volume(Vector((x_co, y_co, z_co)))

                    x_co += vox_size
                y_co += vox_size
            z_co += vox_size

        if remove_isolated:
            self._remove_isolated_voxels(voxels)

        return Voxels(voxels, voxels.shape, bound_min, 1.0, 'xyz')
