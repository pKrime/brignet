import bpy
from bpy.props import BoolProperty
from bpy.props import FloatProperty

from math import floor
from mathutils import Vector

from . import bone_utils
from . import bone_mapping


class LimbChain:
    def __init__(self, chain_root, object, direction_change_stop=False):
        self.object = object
        self.length = chain_root.length
        self.bones = [chain_root]
        self.direction_change_stop = direction_change_stop

        self.get_children()

    @property
    def root(self):
        return self.bones[0]

    @property
    def end(self):
        return self.bones[-1]

    @property
    def mid(self):
        mid_idx = floor(len(self.bones) / 2)
        # TODO: compare length to mid bone with (self.length - self.end.length) / 2

        mid_bone = self.bones[mid_idx]
        return mid_bone

    def get_children(self):
        try:
            child = next(c for c in self.root.children if self.object.data.bones[c.name].use_connect)
        except (IndexError, StopIteration):
            return

        self.bones.append(child)
        self.length += child.length
        while child:
            try:
                child = next(c for c in child.children if self.object.data.bones[c.name].use_connect)
            except (IndexError, StopIteration):
                break
            else:
                self.bones.append(child)
                self.length += child.length

                if self.direction_change_stop and child.parent.vector.normalized().dot(child.vector.normalized()) < 0.6:
                    break


class MergeBones(bpy.types.Operator):
    """Merge two deformation bones and their vertex weights"""

    bl_idname = "object.brignet_merge_bones"
    bl_label = "Merge Deform Bones"
    bl_description = "Merge bones and their assigned vertex weights"
    bl_options = {'REGISTER', 'UNDO'}

    mirror: BoolProperty(name="Mirror", description="merge bones from the other side too", default=True)
    remove_merged: BoolProperty(name="Remove Merged", description="Remove merged groups", default=True)
    merge_tails: BoolProperty(name="Merge Tails",
                              description="Move the resulting bone tail where the merged bone tail was", default=True)

    _armature = None

    @classmethod
    def poll(cls, context):
        if context.mode != 'POSE':
            return False

        # In case of multiple bones, we should choose which tail we should keep (perhaps most distant?)
        # for now we limit the merge to just two bones
        return len(context.selected_pose_bones) < 3

    def merge_bones(self, ebone, target_bone):
        """Merge selected bones and their vertex weights"""
        if self.merge_tails:
            target_bone.tail = ebone.tail

        for child in ebone.children:
            if child.head != target_bone.tail:
                child.use_connect = False
            child.parent = target_bone

        self._armature.data.edit_bones.remove(ebone)

    def execute(self, context):
        self._armature = context.active_object

        bone_names = [b.name for b in context.selected_pose_bones if b != context.active_pose_bone]
        target_name = context.active_pose_bone.name

        other_target_name = ''

        if self.mirror:
            side, other_side = side_from_bone_name(target_name)

            if side and other_side:
                other_target_name = other_side_name(target_name, side, other_side)

        for ob in bone_utils.iterate_rigged_obs(self._armature):
            for name in bone_names:
                bone_utils.merge_vertex_groups(ob, target_name, name, remove_merged=self.remove_merged)

                if other_target_name:
                    other_name = other_side_name(name, side, other_side)
                    bone_utils.merge_vertex_groups(ob, other_target_name, other_name, remove_merged=self.remove_merged)

        bpy.ops.object.mode_set(mode='EDIT')
        target_bone = self._armature.data.edit_bones[target_name]
        other_target_bone = None
        if other_target_name:
            try:
                other_target_bone = self._armature.data.edit_bones[other_target_name]
            except KeyError:
                pass

        for name in bone_names:
            ebone = self._armature.data.edit_bones[name]
            self.merge_bones(ebone, target_bone)

            if other_target_bone:
                try:
                    ebone = self._armature.data.edit_bones[name[:-2] + other_side]
                except KeyError:
                    pass
                else:
                    self.merge_bones(ebone, other_target_bone)

        bpy.ops.object.mode_set(mode='POSE')

        return {'FINISHED'}


class SpineFix(bpy.types.Operator):
    """Rename deformation bones as generated via rigify. Rigify should be enabled"""

    bl_idname = "object.brignet_spinefix"
    bl_label = "Fix Spine"
    bl_description = "Extend collapsed spine joints"
    bl_options = {'REGISTER', 'UNDO'}

    factor: FloatProperty(name='Factor', min=0.0, max=1.0, default=1.0)
    fwd_roll: FloatProperty(name='Roll', min=0.0, max=1.0, default=1.0)
    _central_tolerance = 0.01  # max dislocation for joint to be considered central

    @classmethod
    def poll(cls, context):
        if context.object.type != 'ARMATURE':
            return False
        if 'root' not in context.object.data.bones:
            return False

        return True

    def get_central_child(self, ebone):
        try:
            child = next(c for c in ebone.children if abs(c.tail.x) < self._central_tolerance)
        except StopIteration:
            return

        return child

    def execute(self, context):
        armature = context.active_object.data
        bpy.ops.object.mode_set(mode='EDIT')

        root_bone = armature.edit_bones['root']
        hip_bones = [c for c in root_bone.children if abs(c.tail.x) > self._central_tolerance]

        if not hip_bones:
            return {'FINISHED'}

        diff = root_bone.head.z - hip_bones[0].tail.z
        new_z = hip_bones[0].tail.z - diff/2

        new_head = Vector((0.0, root_bone.head.y, new_z))
        root_bone.head = self.factor * new_head + (1 - self.factor) * root_bone.head

        fwd = Vector((0.0, -1.0, 0.0))
        root_bone.roll = self.fwd_roll * bone_utils.ebone_roll_to_vector(root_bone, fwd) + (1 - self.fwd_roll) * root_bone.roll

        for bone in hip_bones:
            bone.use_connect = False
            bone.head.z = root_bone.head.z

        child = self.get_central_child(root_bone)
        while child:
            child.use_connect = True
            if child.length < 0.02:
                new_head = Vector((0.0, child.head.y, child.head.z - child.parent.length/2))
                child.head = self.factor * new_head + (1 - self.factor) * child.head
                child.tail.x = self.factor * 0.0 + (1 - self.factor) * child.tail.x

            child.roll = self.fwd_roll * bone_utils.ebone_roll_to_vector(child, fwd) + (1 - self.fwd_roll) * child.roll
            child = self.get_central_child(child)

        bpy.ops.object.mode_set(mode='POSE')
        return {'FINISHED'}


def side_from_bone_name(bone_name):
    if bone_name.endswith(('.R', '.L')):
        side = bone_name[-2:]
    elif bone_name[:-1].endswith(tuple(f'.{sd}.00' for sd in ('L', 'R'))):
        side = bone_name[-6:-5]
    else:
        return "", ""

    return side, '.L' if side == '.R' else '.R'


def other_side_name(bone_name, side, other_side):
    bone_name = bone_name.replace(f'{side}.00', f'{other_side}.00')
    if bone_name.endswith(side):
        bone_name = bone_name[:-2] + other_side

    return bone_name


class NamiFy(bpy.types.Operator):
    """Rename deformation bones as generated via rigify. Rigify should be enabled"""

    bl_idname = "object.brignet_namify"
    bl_label = "Namify"
    bl_description = "Rename deformation bones as generated via rigify"
    bl_options = {'REGISTER', 'UNDO'}

    rename_mirrored: BoolProperty(name='Rename mirrored bones', default=True,
                                  description='Rename mirrored bones if found')

    @classmethod
    def poll(cls, context):
        return context.active_object.type == 'ARMATURE'

    def rename_def_bones(self, armature):
        for bone in armature.pose.bones:
            if bone.name.startswith('DEF-'):
                continue
            try:
                rigify_type = bone.rigify_type
            except AttributeError:
                self.report('ERROR', "Rigify attribute not found, please make sure Rigify is enabled")
                return

            if not rigify_type:
                continue

            side, other_side = side_from_bone_name(bone.name)
            rename_mirrored = self.rename_mirrored and side in ('.L', '.R')

            chain = LimbChain(bone, armature)
            rigify_parameters = bone.rigify_parameters

            if rigify_type == 'limbs.super_limb':
                if rigify_parameters.limb_type == 'arm':
                    root_name, mid_name, end_name = 'DEF-upper_arm', 'DEF-forearm', 'DEF-hand'
                    parent_name = 'DEF-shoulder'
                elif rigify_parameters.limb_type == 'leg':
                    chain = LimbChain(bone, armature, direction_change_stop=True)
                    root_name, mid_name, end_name = 'DEF-thigh', 'DEF-shin', 'DEF-foot'
                    parent_name = 'DEF-pelvis'

                if rename_mirrored:
                    other_parent_name = other_side_name(bone.parent.name, side, other_side)
                    other_parent = armature.pose.bones[other_parent_name]
                    other_parent.name = parent_name + other_side
                bone.parent.name = parent_name + side

                basename = root_name
                for cbone in chain.bones:
                    if cbone == chain.mid:
                        basename = mid_name
                    elif cbone == chain.end:
                        basename = end_name
                    if cbone.name.startswith(basename):
                        # already named
                        continue

                    if rename_mirrored:
                        other_name = other_side_name(cbone.name, side, other_side)
                        try:
                            other_bone = armature.pose.bones[other_name]
                        except KeyError:
                            pass
                        else:
                            other_bone.name = basename + other_side

                    cbone.name = basename + side

                try:
                    cbone = cbone.children[0]
                except IndexError:
                    pass
                else:
                    basename = 'DEF-toe'
                    if rename_mirrored:
                        other_name = other_side_name(cbone.name, side, other_side)
                        try:
                            other_bone = armature.pose.bones[other_name]
                        except KeyError:
                            pass
                        else:
                            other_bone.name = basename + other_side

                    cbone.name = basename + side

            elif rigify_type in ('spines.basic_spine', 'spines.super_spine'):
                basename = 'DEF-spine'
                for cbone in chain.bones:
                    cbone.name = basename

                if rename_mirrored:
                    other_name = bone.name[:-2] + other_side
                    try:
                        other_bone = armature.pose.bones[other_name]
                    except KeyError:
                        pass
                    else:
                        other_bone.name = basename + other_side

                bone.name = basename + side

    def execute(self, context):
        armature = context.active_object
        self.rename_def_bones(armature)

        # trigger update as vgroup indices have changed
        for obj in bone_utils.iterate_rigged_obs(armature):
            obj.update_tag(refresh={'DATA'})

        context.view_layer.update()
        return {'FINISHED'}


class ExtractMetarig(bpy.types.Operator):
    """Create Metarig from current object"""
    bl_idname = "object.brignet_extract_metarig"
    bl_label = "Extract Metarig"
    bl_description = "Create Metarig from current object"
    bl_options = {'REGISTER', 'UNDO'}

    remove_missing: BoolProperty(name='Remove Unmatched Bones',
                                 default=True,
                                 description='Rigify will generate to the active object')

    assign_metarig: BoolProperty(name='Assign metarig',
                                 default=True,
                                 description='Rigify will generate to the active object')

    roll_knee_to_foot: BoolProperty(name='Roll knees to foot',
                                    default=True,
                                    description='Align knee roll with foot direction')

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        if context.mode != 'POSE':
            return False
        if context.object.type != 'ARMATURE':
            return False

        return True

    def adjust_toes(self, armature):
        """Align toe joint with foot"""
        for side in '.R', '.L':
            foot_bone = armature.edit_bones[f'foot{side}']
            vector = foot_bone.vector.normalized()

            toe_bone = armature.edit_bones[f'toe{side}']
            dist = (toe_bone.tail - foot_bone.head).length

            vector *= dist
            new_loc = vector + foot_bone.head
            new_loc.z = toe_bone.tail.z

            toe_bone.tail = new_loc

    def adjust_knees(self, armature):
        """Straighten knee joints"""
        for side in '.R', '.L':
            thigh_bone = armature.edit_bones[f'thigh{side}']
            foot_bone = armature.edit_bones[f'foot{side}']

            leg_direction = (foot_bone.tail - thigh_bone.head).normalized()
            leg_direction *= thigh_bone.length
            thigh_bone.tail = thigh_bone.head + leg_direction

            if self.roll_knee_to_foot:
                foot_direction = -foot_bone.vector.normalized()
                thigh_bone.roll = bone_utils.ebone_roll_to_vector(thigh_bone, foot_direction)
                shin_bone = armature.edit_bones[f'shin{side}']
                shin_bone.roll = bone_utils.ebone_roll_to_vector(shin_bone, foot_direction)

                up_axis = Vector((0.0, 0.0, 1.0))
                foot_bone.roll = bone_utils.ebone_roll_to_vector(foot_bone, -up_axis)
                if foot_bone.children:
                    foot_bone.children[0].roll = bone_utils.ebone_roll_to_vector(foot_bone, up_axis)

    def adjust_elbows(self, armature):
        """Straighten knee joints"""
        for side in '.R', '.L':
            arm_bone = armature.edit_bones[f'upper_arm{side}']
            forearm_bone = armature.edit_bones[f'forearm{side}']
            hand_bone = armature.edit_bones[f'hand{side}']

            arm_mid = forearm_bone.tail/2 - arm_bone.head/2
            bow_dir = arm_bone.tail - arm_mid
            bow_dir = bow_dir.cross(Vector((-1.0, 0.0, 0.0)))
            bow_dir.normalize()

            arm_bone.roll = bone_utils.ebone_roll_to_vector(arm_bone, bow_dir)
            forearm_bone.roll = bone_utils.ebone_roll_to_vector(forearm_bone, bow_dir)
            hand_bone.roll = bone_utils.ebone_roll_to_vector(hand_bone, bow_dir)

    def execute(self, context):
        src_object = context.object
        src_armature = context.object.data

        try:
            metarig = next(ob for ob in bpy.data.objects if ob.type == 'ARMATURE' and ob.data.rigify_target_rig == src_object)
            met_armature = metarig.data
            create_metarig = False
        except StopIteration:
            create_metarig = True
            met_armature = bpy.data.armatures.new('metarig')
            metarig = bpy.data.objects.new("metarig", met_armature)

            context.collection.objects.link(metarig)

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')

        metarig.select_set(True)
        bpy.context.view_layer.objects.active = metarig
        bpy.ops.object.mode_set(mode='EDIT')

        if create_metarig:
            from rigify.metarigs.Basic import basic_human
            basic_human.create(metarig)

        met_skeleton = bone_mapping.RigifyMeta()

        def match_meta_bone(met_bone_group, src_bone_group, bone_attr):
            met_bone = met_armature.edit_bones[getattr(met_bone_group, bone_attr)]
            src_bone = src_armature.bones.get(getattr(src_bone_group, bone_attr), None)

            if not src_bone:
                print(getattr(src_bone_group, bone_attr, None), "not found in", src_armature)
                if self.remove_missing:
                    met_armature.edit_bones.remove(met_bone)
                return

            met_bone.head = src_bone.head_local
            met_bone.tail = src_bone.tail_local

        src_skeleton = bone_mapping.RigifySkeleton()
        for bone_attr in ['hips', 'spine', 'spine1', 'spine2', 'neck', 'head']:
            match_meta_bone(met_skeleton.spine, src_skeleton.spine, bone_attr)

        if self.remove_missing and 'DEF-spine.005' not in src_armature.edit_bones:
            # TODO: should rather check all DEF-bones at once
            met_armature.edit_bones.remove(met_armature.edit_bones['spine.005'])

        for bone_attr in ['shoulder', 'arm', 'forearm', 'hand']:
            match_meta_bone(met_skeleton.right_arm, src_skeleton.right_arm, bone_attr)
            match_meta_bone(met_skeleton.left_arm, src_skeleton.left_arm, bone_attr)

        for bone_attr in ['upleg', 'leg', 'foot', 'toe']:
            match_meta_bone(met_skeleton.right_leg, src_skeleton.right_leg, bone_attr)
            match_meta_bone(met_skeleton.left_leg, src_skeleton.left_leg, bone_attr)

        self.adjust_toes(met_armature)
        self.adjust_knees(met_armature)
        self.adjust_elbows(met_armature)

        # find foot vertices
        foot_verts = {}
        foot_ob = None
        # pick object with most foot verts
        for ob in bone_utils.iterate_rigged_obs(src_object):
            if src_skeleton.left_leg.foot not in ob.vertex_groups:
                continue
            grouped_verts = [gv for gv, _ in bone_utils.get_group_verts_weight(ob, src_skeleton.left_leg.foot, threshold=0.8)]
            if len(grouped_verts) > len(foot_verts):
                foot_verts = grouped_verts
                foot_ob = ob

        if foot_verts:
            # find rear verts (heel)
            mat = ob.matrix_world

            rearest_y = max([(mat @ foot_ob.data.vertices[v].co)[1] for v in foot_verts])
            leftmost_x = max([(mat @ foot_ob.data.vertices[v].co)[0] for v in foot_verts])  # FIXME: we should counter rotate verts for more accuracy
            rightmost_x = min([(mat @ foot_ob.data.vertices[v].co)[0] for v in foot_verts])

            inv = src_object.matrix_world.inverted()
            for side in "L", "R":
                heel_bone = met_armature.edit_bones['heel.02.' + side]

                heel_bone.head.y = rearest_y
                heel_bone.tail.y = rearest_y

                if heel_bone.head.x > 0:
                    heel_head = leftmost_x
                    heel_tail = rightmost_x
                else:
                    heel_head = rightmost_x * -1
                    heel_tail = leftmost_x * -1
                heel_bone.head.x = heel_head
                heel_bone.tail.x = heel_tail

                heel_bone.head = inv @ heel_bone.head
                heel_bone.tail = inv @ heel_bone.tail

            for side in "L", "R":
                spine_bone = met_armature.edit_bones['spine']
                pelvis_bone = met_armature.edit_bones['pelvis.' + side]
                thigh_bone = met_armature.edit_bones['thigh.' + side]
                pelvis_bone.head = spine_bone.head
                pelvis_bone.tail.x = thigh_bone.tail.x
                pelvis_bone.tail.y = spine_bone.tail.y
                pelvis_bone.tail.z = spine_bone.tail.z

                spine_bone = met_armature.edit_bones['spine.003']
                breast_bone = met_armature.edit_bones['breast.' + side]
                breast_bone.head.x = pelvis_bone.tail.x
                breast_bone.head.y = spine_bone.head.y
                breast_bone.head.z = spine_bone.head.z
                #
                breast_bone.tail.x = breast_bone.head.x
                breast_bone.tail.z = breast_bone.head.z

        bpy.ops.object.mode_set(mode='POSE')
        if self.assign_metarig:
            met_armature.rigify_target_rig = src_object

        metarig.parent = src_object.parent
        metarig.matrix_local = src_object.matrix_local

        return {'FINISHED'}
