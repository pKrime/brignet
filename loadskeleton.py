import os
import bpy

from .ob_utils.objects import ArmatureGenerator


class LoadSkeletonPanel(bpy.types.Panel):
    """Access LoadSkeleton operator"""
    bl_label = "Load Skeleton"
    bl_idname = "RIGNET_PT_skeleton"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'RigNet'

    def draw(self, context):
        wm = context.window_manager
        layout = self.layout

        row = layout.row()
        row.prop(wm, 'brignet_obj_path')
        row = layout.row()
        row.prop(wm, 'brignet_skel_path')
        row = layout.row()
        row.operator("object.brignet_load")


class LoadRignetSkeleton(bpy.types.Operator):
    """Load characters generated using RigNet from the command line"""
    bl_idname = "object.brignet_load"
    bl_label = "Load rignet character"

    @classmethod
    def poll(cls, context):
        return True  # TODO: object mode

    def execute(self, context):
        wm = context.window_manager
        if not os.path.isfile(wm.brignet_skel_path):
            return {'CANCELLED'}

        from utils.rig_parser import Info
        skel_info = Info(filename=wm.brignet_skel_path)

        if os.path.isfile(wm.brignet_obj_path):
            bpy.ops.import_scene.obj(filepath=wm.brignet_obj_path, use_edges=True, use_smooth_groups=True,
                                     use_groups_as_vgroups=False, use_image_search=True, split_mode='OFF',
                                     axis_forward='-Z', axis_up='Y')

            mesh_obj = context.selected_objects[0]
        else:
            mesh_obj = None

        ArmatureGenerator(skel_info, mesh_obj).generate()
        return {'FINISHED'}
