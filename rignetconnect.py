import os

import numpy as np
import itertools as it

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from .RigNet.utils.rig_parser import Info
from .RigNet.utils.tree_utils import TreeNode
from .RigNet.utils.cluster_utils import meanshift_cluster, nms_meanshift
from .RigNet.utils.mst_utils import increase_cost_for_outside_bone, primMST_symmetry, loadSkel_recur, inside_check, flip
from .RigNet.utils.mst_utils import sample_on_bone

from .RigNet.models.GCN import JOINTNET_MASKNET_MEANSHIFT as JOINTNET
from .RigNet.models.ROOT_GCN import ROOTNET
from .RigNet.models.PairCls_GCN import PairCls as BONENET
from .RigNet.models.SKINNING import SKINNET

import bpy
from mathutils import Matrix

from .ob_utils import sampling as mesh_sampling
from .ob_utils.geometry import get_tpl_edges
from .ob_utils.geometry import get_geo_edges
from .ob_utils.geometry import NormalizedMeshData

from .ob_utils.objects import ArmatureGenerator


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MeshStorage:
    """Store Mesh Data and samples"""
    _instance = None  # stores singleton instance

    _mesh_data = None
    _mesh_sampler = None
    _surf_geodesic = None
    _voxels = None

    def __init__(self, samples=2000):
        self._samples = samples

    def set_mesh_data(self, mesh_obj):
        self._mesh_data = NormalizedMeshData(mesh_obj)

    @property
    def mesh_data(self):
        assert self._mesh_data is not None
        return self._mesh_data

    @property
    def surface_geodesic(self):
        if self._surf_geodesic is None:
            assert self._mesh_data is not None
            self._surf_geodesic = self.mesh_sampler.calc_geodesic(samples=self._samples)
        return self._surf_geodesic

    @property
    def voxels(self):
        if self._voxels is None:
            assert self._mesh_data is not None
            self._voxels = self._mesh_data.voxels()

        return self._voxels

    @property
    def mesh_sampler(self):
        if self._mesh_sampler is None:
            assert self._mesh_data is not None
            self._mesh_sampler = mesh_sampling.MeshSampler(self._mesh_data.mesh_f, self._mesh_data.mesh_v,
                                                           self._mesh_data.mesh_vn, self._mesh_data.tri_areas)
        return self._mesh_sampler


def getInitId(data, model):
    """
    predict root joint ID via rootnet
    :param data:
    :param model:
    :return:
    """
    with torch.no_grad():
        root_prob, _ = model(data, shuffle=False)
        root_prob = torch.sigmoid(root_prob).data.cpu().numpy()
    root_id = np.argmax(root_prob)
    return root_id


def create_single_data(mesh_storage: MeshStorage):
    """
    create input data for the network. The data is wrapped by Data structure in pytorch-geometric library
    """

    mesh_data = mesh_storage.mesh_data

    # vertices
    v = np.concatenate((mesh_data.mesh_v, mesh_data.mesh_vn), axis=1)
    v = torch.from_numpy(v).float()
    # topology edges
    print("     gathering topological edges.")
    tpl_e = get_tpl_edges(mesh_data.mesh_v, mesh_data.mesh_f).T
    tpl_e = torch.from_numpy(tpl_e).long()
    tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.size(0))
    # surface geodesic distance matrix
    print("     calculating surface geodesic matrix.")

    surface_geodesic = mesh_storage.surface_geodesic
    # geodesic edges
    print("     gathering geodesic edges.")
    geo_e = get_geo_edges(surface_geodesic, mesh_data.mesh_v).T
    geo_e = torch.from_numpy(geo_e).long()
    geo_e, _ = add_self_loops(geo_e, num_nodes=v.size(0))
    # batch
    batch = torch.zeros(len(v), dtype=torch.long)

    geo_data = Data(x=v[:, 3:6], pos=v[:, 0:3], tpl_edge_index=tpl_e, geo_edge_index=geo_e, batch=batch)
    return geo_data


def add_joints_data(input_data, vox, joint_pred_net, threshold, bandwidth=None, mesh_filename=None):
    """
    Predict joints
    :param input_data: wrapped input data
    :param vox: voxelized mesh
    :param joint_pred_net: network for predicting joints
    :param threshold: density threshold to filter out shifted points
    :param bandwidth: bandwidth for meanshift clustering
    :param mesh_filename: mesh filename for visualization
    :return: wrapped data with predicted joints, pair-wise bone representation added.
    """
    data_displacement, _, attn_pred, bandwidth_pred = joint_pred_net(input_data)
    y_pred = data_displacement + input_data.pos
    y_pred_np = y_pred.data.cpu().numpy()
    attn_pred_np = attn_pred.data.cpu().numpy()
    y_pred_np, index_inside = inside_check(y_pred_np, vox)
    attn_pred_np = attn_pred_np[index_inside, :]
    y_pred_np = y_pred_np[attn_pred_np.squeeze() > 1e-3]
    attn_pred_np = attn_pred_np[attn_pred_np.squeeze() > 1e-3]

    # symmetrize points by reflecting
    y_pred_np_reflect = y_pred_np * np.array([[-1, 1, 1]])
    y_pred_np = np.concatenate((y_pred_np, y_pred_np_reflect), axis=0)
    attn_pred_np = np.tile(attn_pred_np, (2, 1))

    if not bandwidth:
        bandwidth = bandwidth_pred.item()
    y_pred_np = meanshift_cluster(y_pred_np, bandwidth, attn_pred_np, max_iter=40)

    Y_dist = np.sum(((y_pred_np[np.newaxis, ...] - y_pred_np[:, np.newaxis, :]) ** 2), axis=2)
    density = np.maximum(bandwidth ** 2 - Y_dist, np.zeros(Y_dist.shape))
    density = np.sum(density, axis=0)
    density_sum = np.sum(density)
    y_pred_np = y_pred_np[density / density_sum > threshold]
    density = density[density / density_sum > threshold]

    pred_joints = nms_meanshift(y_pred_np, density, bandwidth)
    pred_joints, _ = flip(pred_joints)

    # prepare and add new data members
    pairs = list(it.combinations(range(pred_joints.shape[0]), 2))
    pair_attr = []
    for pr in pairs:
        dist = np.linalg.norm(pred_joints[pr[0]] - pred_joints[pr[1]])
        bone_samples = sample_on_bone(pred_joints[pr[0]], pred_joints[pr[1]])
        bone_samples_inside, _ = inside_check(bone_samples, vox)
        outside_proportion = len(bone_samples_inside) / (len(bone_samples) + 1e-10)
        attr = np.array([dist, outside_proportion, 1])
        pair_attr.append(attr)
    pairs = np.array(pairs)
    pair_attr = np.array(pair_attr)
    pairs = torch.from_numpy(pairs).float()
    pair_attr = torch.from_numpy(pair_attr).float()
    pred_joints = torch.from_numpy(pred_joints).float()
    joints_batch = torch.zeros(len(pred_joints), dtype=torch.long)
    pairs_batch = torch.zeros(len(pairs), dtype=torch.long)

    input_data.joints = pred_joints
    input_data.pairs = pairs
    input_data.pair_attr = pair_attr
    input_data.joints_batch = joints_batch
    input_data.pairs_batch = pairs_batch
    return input_data


def predict_skeleton(input_data, vox, root_pred_net, bone_pred_net):
    """
    Predict skeleton structure based on joints
    :param input_data: wrapped data
    :param vox: voxelized mesh
    :param root_pred_net: network to predict root
    :param bone_pred_net: network to predict pairwise connectivity cost
    :return: predicted skeleton structure
    """
    root_id = getInitId(input_data, root_pred_net)
    pred_joints = input_data.joints.data.cpu().numpy()

    with torch.no_grad():
        connect_prob, _ = bone_pred_net(input_data, permute_joints=False)
        connect_prob = torch.sigmoid(connect_prob)
    pair_idx = input_data.pairs.long().data.cpu().numpy()
    prob_matrix = np.zeros((len(input_data.joints), len(input_data.joints)))
    prob_matrix[pair_idx[:, 0], pair_idx[:, 1]] = connect_prob.data.cpu().numpy().squeeze()
    prob_matrix = prob_matrix + prob_matrix.transpose()
    cost_matrix = -np.log(prob_matrix + 1e-10)
    cost_matrix = increase_cost_for_outside_bone(cost_matrix, pred_joints, vox)

    pred_skel = Info()
    parent, key, _ = primMST_symmetry(cost_matrix, root_id, pred_joints)
    for i in range(len(parent)):
        if parent[i] == -1:
            pred_skel.root = TreeNode('root', tuple(pred_joints[i]))
            break
    loadSkel_recur(pred_skel.root, i, None, pred_joints, parent)
    pred_skel.joint_pos = pred_skel.get_joint_dict()

    return pred_skel


def pts2line(pts, lines):
    '''
    Calculate points-to-bone distance. Point to line segment distance refer to
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    :param pts: N*3
    :param lines: N*6, where [N,0:3] is the starting position and [N, 3:6] is the ending position
    :return: origins are the neatest projected position of the point on the line.
             ends are the points themselves.
             dist is the distance in between, which is the distance from points to lines.
             Origins and ends will be used for generate rays.
    '''
    l2 = np.sum((lines[:, 3:6] - lines[:, 0:3]) ** 2, axis=1)
    origins = np.zeros((len(pts) * len(lines), 3))
    ends = np.zeros((len(pts) * len(lines), 3))
    dist = np.zeros((len(pts) * len(lines)))
    for l in range(len(lines)):
        if np.abs(l2[l]) < 1e-8:  # for zero-length edges
            origins[l * len(pts):(l + 1) * len(pts)] = lines[l][0:3]
        else:  # for other edges
            t = np.sum((pts - lines[l][0:3][np.newaxis, :]) * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :], axis=1) / \
                l2[l]
            t = np.clip(t, 0, 1)
            t_pos = lines[l][0:3][np.newaxis, :] + t[:, np.newaxis] * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :]
            origins[l * len(pts):(l + 1) * len(pts)] = t_pos
        ends[l * len(pts):(l + 1) * len(pts)] = pts
        dist[l * len(pts):(l + 1) * len(pts)] = np.linalg.norm(
            origins[l * len(pts):(l + 1) * len(pts)] - ends[l * len(pts):(l + 1) * len(pts)], axis=1)
    return origins, ends, dist


def calc_pts2bone_visible_mat(bvhtree, origins, ends):
    '''
    Check whether the surface point is visible by the internal bone.
    Visible is defined as no occlusion on the path between.
    :param mesh:
    :param surface_pts: points on the surface (n*3)
    :param origins: origins of rays
    :param ends: ends of the rays, together with origins, we can decide the direction of the ray.
    :return: binary visibility matrix (n*m), where 1 indicate the n-th surface point is visible to the m-th ray
    '''
    ray_dirs = ends - origins

    min_hit_distance = []
    for ray_dir, origin in zip(ray_dirs, origins):
        # FIXME: perhaps we should sample more distances
        location, normal, index, distance = bvhtree.ray_cast(origin, ray_dir + 1e-15)
        if location:
            min_hit_distance.append(np.linalg.norm(np.array(location) - origin))
        else:
            min_hit_distance.append(np.linalg.norm(ray_dir))

    min_hit_distance = np.array(min_hit_distance)
    distance = np.linalg.norm(ray_dirs, axis=1)
    vis_mat = (np.abs(min_hit_distance - distance) < 1e-4)
    return vis_mat


def calc_geodesic_matrix(bones, mesh_v, surface_geodesic, bvh_tree, use_sampling=False):
    """
    calculate volumetric geodesic distance from vertices to each bones
    :param bones: B*6 numpy array where each row stores the starting and ending joint position of a bone
    :param mesh_v: V*3 mesh vertices
    :param surface_geodesic: geodesic distance matrix of all vertices
    :return: an approaximate volumetric geodesic distance matrix V*B, were (v,b) is the distance from vertex v to bone b
    """

    if use_sampling:
        # TODO: perhaps not required with blender's bvh tree
        # will have to decimate the mesh otherwise
        # also, this should rather be done outside the function
        subsamples = mesh_v
    else:
        subsamples = mesh_v

    origins, ends, pts_bone_dist = pts2line(subsamples, bones)
    pts_bone_visibility = calc_pts2bone_visible_mat(bvh_tree, origins, ends)
    pts_bone_visibility = pts_bone_visibility.reshape(len(bones), len(subsamples)).transpose()
    pts_bone_dist = pts_bone_dist.reshape(len(bones), len(subsamples)).transpose()
    # remove visible points which are too far
    for b in range(pts_bone_visibility.shape[1]):
        visible_pts = np.argwhere(pts_bone_visibility[:, b] == 1).squeeze(1)
        if len(visible_pts) == 0:
            continue
        threshold_b = np.percentile(pts_bone_dist[visible_pts, b], 15)
        pts_bone_visibility[pts_bone_dist[:, b] > 1.3 * threshold_b, b] = False

    visible_matrix = np.zeros(pts_bone_visibility.shape)
    visible_matrix[np.where(pts_bone_visibility == 1)] = pts_bone_dist[np.where(pts_bone_visibility == 1)]
    for c in range(visible_matrix.shape[1]):
        unvisible_pts = np.argwhere(pts_bone_visibility[:, c] == 0).squeeze(1)
        visible_pts = np.argwhere(pts_bone_visibility[:, c] == 1).squeeze(1)
        if len(visible_pts) == 0:
            visible_matrix[:, c] = pts_bone_dist[:, c]
            continue
        for r in unvisible_pts:
            dist1 = np.min(surface_geodesic[r, visible_pts])
            nn_visible = visible_pts[np.argmin(surface_geodesic[r, visible_pts])]
            if np.isinf(dist1):
                visible_matrix[r, c] = 8.0 + pts_bone_dist[r, c]
            else:
                visible_matrix[r, c] = dist1 + visible_matrix[nn_visible, c]
    if use_sampling:
        nn_dist = np.sum((mesh_v[:, np.newaxis, :] - subsamples[np.newaxis, ...]) ** 2, axis=2)
        nn_ind = np.argmin(nn_dist, axis=1)
        visible_matrix = visible_matrix[nn_ind, :]
    return visible_matrix


def add_duplicate_joints(skel):
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            if len(p_node.children) > 1:
                new_children = []
                for dup_id in range(len(p_node.children)):
                    p_node_new = TreeNode(p_node.name + '_dup_{:d}'.format(dup_id), p_node.pos)
                    p_node_new.overlap=True
                    p_node_new.parent = p_node
                    p_node_new.children = [p_node.children[dup_id]]
                    # for user interaction, we move overlapping joints a bit to its children
                    p_node_new.pos = np.array(p_node_new.pos) + 0.03 * np.linalg.norm(np.array(p_node.children[dup_id].pos) - np.array(p_node_new.pos))
                    p_node_new.pos = (p_node_new.pos[0], p_node_new.pos[1], p_node_new.pos[2])
                    p_node.children[dup_id].parent = p_node_new
                    new_children.append(p_node_new)
                p_node.children = new_children
            p_node.overlap = False
            next_level += p_node.children
        this_level = next_level
    return skel


def mapping_bone_index(bones_old, bones_new):
    bone_map = {}
    for i in range(len(bones_old)):
        bone_old = bones_old[i][np.newaxis, :]
        dist = np.linalg.norm(bones_new - bone_old, axis=1)
        ni = np.argmin(dist)
        bone_map[i] = ni
    return bone_map


def get_bones(skel):
    """
    extract bones from skeleton struction
    :param skel: input skeleton
    :return: bones are B*6 array where each row consists starting and ending points of a bone
             bone_name are a list of B elements, where each element consists starting and ending joint name
             leaf_bones indicate if this bone is a virtual "leaf" bone.
             We add virtual "leaf" bones to the leaf joints since they always have skinning weights as well
    """
    bones = []
    bone_name = []
    leaf_bones = []
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            p_pos = np.array(p_node.pos)
            next_level += p_node.children
            for c_node in p_node.children:
                c_pos = np.array(c_node.pos)
                bones.append(np.concatenate((p_pos, c_pos))[np.newaxis, :])
                bone_name.append([p_node.name, c_node.name])
                leaf_bones.append(False)
                if len(c_node.children) == 0:
                    bones.append(np.concatenate((c_pos, c_pos))[np.newaxis, :])
                    bone_name.append([c_node.name, c_node.name+'_leaf'])
                    leaf_bones.append(True)
        this_level = next_level
    bones = np.concatenate(bones, axis=0)
    return bones, bone_name, leaf_bones



def assemble_skel_skin(skel, attachment):
    bones_old, bone_names_old, _ = get_bones(skel)
    skel_new = add_duplicate_joints(skel)
    bones_new, bone_names_new, _ = get_bones(skel_new)
    bone_map = mapping_bone_index(bones_old, bones_new)
    skel_new.joint_pos = skel_new.get_joint_dict()
    skel_new.joint_skin = []

    for v in range(len(attachment)):
        vi_skin = [str(v)]
        skw = attachment[v]
        skw = skw / (np.sum(skw) + 1e-10)
        for i in range(len(skw)):
            if i == len(bones_old):
                break
            if skw[i] > 1e-5:
                bind_joint_name = bone_names_new[bone_map[i]][0]
                bind_weight = skw[i]
                vi_skin.append(bind_joint_name)
                vi_skin.append(str(bind_weight))
        skel_new.joint_skin.append(vi_skin)
    return skel_new


def post_filter(skin_weights, topology_edge, num_ring=1):
    skin_weights_new = np.zeros_like(skin_weights)
    for v in range(len(skin_weights)):
        adj_verts_multi_ring = []
        current_seeds = [v]
        for r in range(num_ring):
            adj_verts = []
            for seed in current_seeds:
                adj_edges = topology_edge[:, np.argwhere(topology_edge == seed)[:, 1]]
                adj_verts_seed = list(set(adj_edges.flatten().tolist()))
                adj_verts_seed.remove(seed)
                adj_verts += adj_verts_seed
            adj_verts_multi_ring += adj_verts
            current_seeds = adj_verts
        adj_verts_multi_ring = list(set(adj_verts_multi_ring))
        if v in adj_verts_multi_ring:
            adj_verts_multi_ring.remove(v)
        skin_weights_neighbor = [skin_weights[int(i), :][np.newaxis, :] for i in adj_verts_multi_ring]
        skin_weights_neighbor = np.concatenate(skin_weights_neighbor, axis=0)
        #max_bone_id = np.argmax(skin_weights[v, :])
        #if np.sum(skin_weights_neighbor[:, max_bone_id]) < 0.17 * len(skin_weights_neighbor):
        #    skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)
        #else:
        #    skin_weights_new[v, :] = skin_weights[v, :]
        skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)

    #skin_weights_new[skin_weights_new.sum(axis=1) == 0, :] = skin_weights[skin_weights_new.sum(axis=1) == 0, :]
    return skin_weights_new


def predict_skinning(input_data, pred_skel, skin_pred_net, surface_geodesic, bvh_tree):
    """
    predict skinning
    :param input_data: wrapped input data
    :param pred_skel: predicted skeleton
    :param skin_pred_net: network to predict skinning weights
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: predicted rig with skinning weights information
    """
    global DEVICE, output_folder
    num_nearest_bone = 5
    bones, bone_names, bone_isleaf = get_bones(pred_skel)
    mesh_v = input_data.pos.data.cpu().numpy()
    print("     calculating volumetric geodesic distance from vertices to bone. This step takes some time...")

    geo_dist = calc_geodesic_matrix(bones, mesh_v, surface_geodesic, bvh_tree)
    input_samples = []  # joint_pos (x, y, z), (bone_id, 1/D)*5
    loss_mask = []
    skin_nn = []
    for v_id in range(len(mesh_v)):
        geo_dist_v = geo_dist[v_id]
        bone_id_near_to_far = np.argsort(geo_dist_v)
        this_sample = []
        this_nn = []
        this_mask = []
        for i in range(num_nearest_bone):
            if i >= len(bones):
                this_sample += bones[bone_id_near_to_far[0]].tolist()
                this_sample.append(1.0 / (geo_dist_v[bone_id_near_to_far[0]] + 1e-10))
                this_sample.append(bone_isleaf[bone_id_near_to_far[0]])
                this_nn.append(0)
                this_mask.append(0)
            else:
                skel_bone_id = bone_id_near_to_far[i]
                this_sample += bones[skel_bone_id].tolist()
                this_sample.append(1.0 / (geo_dist_v[skel_bone_id] + 1e-10))
                this_sample.append(bone_isleaf[skel_bone_id])
                this_nn.append(skel_bone_id)
                this_mask.append(1)
        input_samples.append(np.array(this_sample)[np.newaxis, :])
        skin_nn.append(np.array(this_nn)[np.newaxis, :])
        loss_mask.append(np.array(this_mask)[np.newaxis, :])

    skin_input = np.concatenate(input_samples, axis=0)
    loss_mask = np.concatenate(loss_mask, axis=0)
    skin_nn = np.concatenate(skin_nn, axis=0)
    skin_input = torch.from_numpy(skin_input).float()
    input_data.skin_input = skin_input
    input_data.to(DEVICE)

    skin_pred = skin_pred_net(input_data)
    skin_pred = torch.softmax(skin_pred, dim=1)
    skin_pred = skin_pred.data.cpu().numpy()
    skin_pred = skin_pred * loss_mask

    skin_nn = skin_nn[:, 0:num_nearest_bone]
    skin_pred_full = np.zeros((len(skin_pred), len(bone_names)))
    for v in range(len(skin_pred)):
        for nn_id in range(len(skin_nn[v, :])):
            skin_pred_full[v, skin_nn[v, nn_id]] = skin_pred[v, nn_id]
    print("     filtering skinning prediction")
    tpl_e = input_data.tpl_edge_index.data.cpu().numpy()
    skin_pred_full = post_filter(skin_pred_full, tpl_e, num_ring=1)
    skin_pred_full[skin_pred_full < np.max(skin_pred_full, axis=1, keepdims=True) * 0.35] = 0.0
    skin_pred_full = skin_pred_full / (skin_pred_full.sum(axis=1, keepdims=True) + 1e-10)
    skel_res = assemble_skel_skin(pred_skel, skin_pred_full)
    return skel_res


class Networks:
    def __init__(self, model_dir="", load_networks=True, load_skinning=True):
        self.joint_net = None
        self.root_net = None
        self.bone_net = None
        self.skin_net = None

        self._load_skinning = load_skinning
        self.model_dir = model_dir if model_dir else bpy.context.preferences.addons[__package__].preferences.model_path

        if load_networks:
            self.load_networks()

    def load_networks(self):
        print("loading all networks...")
        joint_net = JOINTNET()
        joint_net.to(DEVICE)
        joint_net.eval()
        joint_net_checkpoint = torch.load(os.path.join(self.model_dir, 'gcn_meanshift/model_best.pth.tar'))
        joint_net.load_state_dict(joint_net_checkpoint['state_dict'])
        self.joint_net = joint_net
        print("     joint prediction network loaded.")

        root_net = ROOTNET()
        root_net.to(DEVICE)
        root_net.eval()
        root_net_checkpoint = torch.load(os.path.join(self.model_dir, 'rootnet/model_best.pth.tar'))
        root_net.load_state_dict(root_net_checkpoint['state_dict'])
        self.root_net = root_net
        print("     root prediction network loaded.")

        bone_net = BONENET()
        bone_net.to(DEVICE)
        bone_net.eval()
        bone_net_checkpoint = torch.load(os.path.join(self.model_dir, 'bonenet/model_best.pth.tar'))
        bone_net.load_state_dict(bone_net_checkpoint['state_dict'])
        self.bone_net = bone_net
        print("     connection prediction network loaded.")

        if self._load_skinning:
            skin_net = SKINNET(nearest_bone=5, use_Dg=True, use_Lf=True)
            skin_net_checkpoint = torch.load(os.path.join(self.model_dir, 'skinnet/model_best.pth.tar'))
            skin_net.load_state_dict(skin_net_checkpoint['state_dict'])
            skin_net.to(DEVICE)
            skin_net.eval()
            self.skin_net = skin_net
            print("     skinning prediction network loaded.")


def init_data(mesh_obj, samples=2000):
    mesh_storage = MeshStorage(samples)
    mesh_storage.set_mesh_data(mesh_obj)

    predict_data = create_single_data(mesh_storage)
    predict_data.to(DEVICE)

    return predict_data, mesh_storage


def predict_joint(predict_data, joint_network, mesh_storage: MeshStorage, bandwidth, threshold):
    print("predicting joints")
    predict_data = add_joints_data(predict_data, mesh_storage.voxels, joint_network, threshold, bandwidth=bandwidth)
    predict_data.to(DEVICE)
    return predict_data


def predict_hierarchy(predict_data, networks: Networks, mesh_storage: MeshStorage):
    print("predicting connectivity")
    predicted_skeleton = predict_skeleton(predict_data, mesh_storage.voxels, networks.root_net, networks.bone_net)
    return predicted_skeleton


def predict_weights(predict_data, predicted_skeleton, skin_network, mesh_storage: MeshStorage):
    print("predicting skinning")
    mesh_data = mesh_storage.mesh_data
    bvh_tree = mesh_data.bvh_tree
    predicted_rig = predict_skinning(predict_data, predicted_skeleton, skin_network, mesh_storage.surface_geodesic, bvh_tree)

    # here we reverse the normalization to the original scale and position
    predicted_rig.normalize(mesh_data.scale_normalize, -mesh_data.translation_normalize)
    return predicted_rig


def create_armature(mesh_obj, predicted_rig):
    mesh_obj.vertex_groups.clear()

    for obj in bpy.data.objects:
        obj.select_set(False)

    mat = Matrix(((1.0, 0.0, 0.0, 0.0),
                  (0.0, 0.0, -1.0, 0.0),
                  (0.0, 1.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0, 1.0)))
    new_arm = ArmatureGenerator(predicted_rig, mesh_obj).generate(matrix=mat)
    torch.cuda.empty_cache()

    return new_arm


def clear():
    torch.cuda.empty_cache()
