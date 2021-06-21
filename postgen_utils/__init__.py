import bpy
from bpy.props import BoolProperty
from bpy.props import FloatProperty

from math import floor
from mathutils import Vector

from . import bone_utils
from . import bone_mapping


class LimbChain:
    def __init__(self, chain_root, object):
        self.object = object
        self.length = chain_root.length
        self.bones = [chain_root]

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
            child = self.root.children[0]
        except IndexError:
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

        return child


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

    def side_from_name(self, bone):
        if bone.name.endswith(('.R', '.L')):
            side = bone.name[-2:]
        else:
            side = ""

        if side and self.rename_mirrored:
            other_side = '.R' if side == '.L' else '.L'
        else:
            other_side = ""

        return side, other_side

    def rename_def_bones(self, armature):
        for bone in armature.pose.bones:
            try:
                rigify_type = bone.rigify_type
            except AttributeError:
                self.report('ERROR', "Rigify attribute not found, please make sure Rigify is enabled")
                return

            if not rigify_type:
                continue

            side, other_side = self.side_from_name(bone)
            rename_mirrored = self.rename_mirrored and bool(side)

            chain = LimbChain(bone, armature)
            rigify_parameters = bone.rigify_parameters

            if rigify_type == 'limbs.super_limb':
                if rigify_parameters.limb_type == 'arm':
                    root_name, mid_name, end_name = 'DEF-upper_arm', 'DEF-forearm', 'DEF-hand'
                elif rigify_parameters.limb_type == 'leg':
                    root_name, mid_name, end_name = 'DEF-thigh', 'DEF-shin', 'DEF-foot'

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
                        other_name = cbone.name[:-2] + other_side
                        try:
                            other_bone = armature.pose.bones[other_name]
                        except KeyError:
                            pass
                        else:
                            other_bone.name = basename + other_side

                    cbone.name = basename + side
            elif rigify_type == 'spines.basic_spine':
                basename = 'DEF-spine'
                for cbone in chain.bones:
                    cbone.name = basename
            elif rigify_type == 'basic.super_copy':
                try:
                    child = bone.children[0]
                except KeyError:
                    continue
                if child.rigify_type == 'limbs.super_limb':
                    if child.rigify_parameters.limb_type == 'arm':
                        basename = 'DEF-shoulder'
                    elif child.rigify_parameters.limb_type == 'leg':
                        basename = 'DEF-pelvis'
                    else:
                        continue

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
        return {'FINISHED'}


class ExtractMetarig(bpy.types.Operator):
    """Create Metarig from current object"""
    bl_idname = "object.brignet_extract_metarig"
    bl_label = "Extract Metarig"
    bl_description = "Create Metarig from current object"
    bl_options = {'REGISTER', 'UNDO'}


    offset_knee: FloatProperty(name='Offset Knee',
                               default=0.0)

    offset_elbow: FloatProperty(name='Offset Elbow',
                                default=0.0)

    assign_metarig: BoolProperty(name='Assign metarig',
                                 default=True,
                                 description='Rigify will generate to the active object')

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        if context.mode != 'POSE':
            return False
        if context.object.type != 'ARMATURE':
            return False

        return True

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
                print(bone_attr, "not found in", src_armature)
                return

            met_bone.head = src_bone.head_local
            met_bone.tail = src_bone.tail_local

            if met_bone.parent and met_bone.use_connect:
                bone_dir = met_bone.vector.normalized()
                parent_dir = met_bone.parent.vector.normalized()

                if bone_dir.dot(parent_dir) < -0.6:
                    print(met_bone.name, "non aligned")
                    # TODO

            # TODO: set roll
            # met_bone.roll = 0.0
            #
            # src_z_axis = Vector((0.0, 0.0, 1.0)) @ src_bone.matrix_local.to_3x3()
            # inv_rot = met_bone.matrix.to_3x3().inverted()
            # trg_z_axis = src_z_axis @ inv_rot
            # dot_z = (met_bone.z_axis @ met_bone.matrix.inverted()).dot(trg_z_axis)
            # met_bone.roll = dot_z * pi

        src_skeleton = bone_mapping.RigifySkeleton()
        for bone_attr in ['hips', 'spine', 'spine1', 'spine2', 'neck', 'head']:
            match_meta_bone(met_skeleton.spine, src_skeleton.spine, bone_attr)

        for bone_attr in ['shoulder', 'arm', 'forearm', 'hand']:
            match_meta_bone(met_skeleton.right_arm, src_skeleton.right_arm, bone_attr)
            match_meta_bone(met_skeleton.left_arm, src_skeleton.left_arm, bone_attr)

        for bone_attr in ['upleg', 'leg', 'foot', 'toe']:
            match_meta_bone(met_skeleton.right_leg, src_skeleton.right_leg, bone_attr)
            match_meta_bone(met_skeleton.left_leg, src_skeleton.left_leg, bone_attr)

        right_leg = met_armature.edit_bones[met_skeleton.right_leg.leg]
        left_leg = met_armature.edit_bones[met_skeleton.left_leg.leg]

        offset = Vector((0.0, self.offset_knee, 0.0))
        for bone in right_leg, left_leg:
            bone.head += offset

        right_knee = met_armature.edit_bones[met_skeleton.right_arm.forearm]
        left_knee = met_armature.edit_bones[met_skeleton.left_arm.forearm]
        offset = Vector((0.0, self.offset_elbow, 0.0))

        for bone in right_knee, left_knee:
            bone.head += offset

            # TODO: roll
            # for met_bone_name, src_bone_name in zip(met_bone_names, src_bone_names):
            #     met_bone = met_armature.edit_bones[met_bone_name]
            #     try:
            #         src_bone = src_armature.bones[src_bone_name]
            #     except KeyError:
            #         print("source bone not found", src_bone_name)
            #         continue

                # met_bone.head = src_bone.head_local
                # try:
                #     met_bone.tail = src_bone.children[0].head_local
                # except IndexError:
                #     align_to_closer_axis(src_bone, met_bone)
                #
                # met_bone.roll = 0.0
                #
                # src_z_axis = Vector((0.0, 0.0, 1.0)) @ src_bone.matrix_local.to_3x3()
                # inv_rot = met_bone.matrix.to_3x3().inverted()
                # trg_z_axis = src_z_axis @ inv_rot
                # dot_z = (met_bone.z_axis @ met_bone.matrix.inverted()).dot(trg_z_axis)
                # met_bone.roll = dot_z * pi


        met_armature.edit_bones['spine.003'].tail = met_armature.edit_bones['spine.004'].head
        met_armature.edit_bones['spine.005'].head = (met_armature.edit_bones['spine.004'].head + met_armature.edit_bones['spine.006'].head) / 2

        # find foot vertices
        foot_verts = {}
        foot_ob = None
        # pick object with most foot verts
        for ob in bone_utils.iterate_rigged_obs(src_object):
            if src_skeleton.left_leg.foot not in ob.vertex_groups:
                continue
            grouped_verts = bone_utils.get_group_verts(ob, src_skeleton.left_leg.foot, threshold=0.8)
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
                breast_bone.tail.y = breast_bone.head.y + 0.25
                breast_bone.tail.z = breast_bone.head.z

        bpy.ops.object.mode_set(mode='POSE')
        if self.assign_metarig:
            met_armature.rigify_target_rig = src_object

        metarig.parent = src_object.parent
        metarig.matrix_local = src_object.matrix_local

        return {'FINISHED'}