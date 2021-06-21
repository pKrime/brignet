import bpy
from bpy.props import BoolProperty
from math import floor


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
