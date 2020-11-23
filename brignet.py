import bpy
from bpy.props import IntProperty, BoolProperty, FloatProperty, PointerProperty, StringProperty
from .import rigutils


from importlib import reload
reload(rigutils)


class BrigNetPredict(bpy.types.Operator):
    """Predict joint position of chosen mesh using a trained model"""
    bl_idname = "object.brignet_predict"
    bl_label = "Predict joints and skinning"

    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        if not wm.brignet_targetmesh:
            return False

        return wm.brignet_targetmesh.type == 'MESH'

    def execute(self, context):
        wm = context.window_manager
        rigutils.remove_modifiers(wm.brignet_targetmesh, type_list=('ARMATURE',))

        try:
            from . import rignetconnect
            reload(rignetconnect)
        except ModuleNotFoundError:
            self.report({'WARNING'}, "Some modules not found, please check bRigNet preferences")
            return {'CANCELLED'}

        bandwidth = (1 - wm.brignet_density) / 10
        threshold = wm.brignet_threshold/1000

        rignetconnect.predict_rig(wm.brignet_targetmesh, bandwidth, threshold,
                                  wm.brignet_downsample_skin,
                                  wm.brignet_downsample_decimate,
                                  wm.brignet_downsample_sampling)

        if wm.brignet_highrescollection:
            rigutils.copy_weights(wm.brignet_highrescollection.objects, wm.brignet_targetmesh)

        return {'FINISHED'}


class BrignetPanel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Layout Demo"
    bl_idname = "RIGNET_PT_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'bRigNet'

    def draw(self, context):
        layout = self.layout

        wm = context.window_manager

        row = layout.row()
        row.prop(wm, 'brignet_downsample_skin', text='Downsample Skinning')

        if wm.brignet_downsample_skin:
            row = layout.row()
            col = row.column()
            col.prop(wm, 'brignet_downsample_decimate', text='Decimation')
            col = row.column()
            col.prop(wm, 'brignet_downsample_sampling', text='Sampling')

        row = layout.row()
        row.prop(wm, 'brignet_targetmesh', text='Target')

        row = layout.row()
        row.prop(wm, 'brignet_highrescollection', text='HighRes')

        row = layout.row()
        row.operator("object.brignet_predict")

        row = layout.row()
        row.prop(wm, 'brignet_density', text='Density')

        row = layout.row()
        row.prop(wm, 'brignet_threshold', text='Treshold')


def register_properties():
    bpy.types.WindowManager.brignet_downsample_skin = BoolProperty(name="downsample_skinning", default=True)
    bpy.types.WindowManager.brignet_downsample_decimate = IntProperty(name="downsample_decimate", default=3000)
    bpy.types.WindowManager.brignet_downsample_sampling = IntProperty(name="downsample_sampling", default=1500)

    bpy.types.WindowManager.brignet_targetmesh = PointerProperty(type=bpy.types.Object,
                                                                 name="bRigNet Target Object",
                                                                 description="Mesh to use for skin prediction. Keep below 5000 triangles",
                                                                 poll=lambda self, obj: obj.type == 'MESH' and obj.data is not self)

    bpy.types.WindowManager.brignet_highrescollection = PointerProperty(type=bpy.types.Collection,
                                                                        name="bRigNet HighRes Objects",
                                                                        description="Meshes to use for final skinning")

    bpy.types.WindowManager.brignet_density = FloatProperty(name="density", default=0.571, min=0.1, max=1.0,
                                                            description="Bone Density")

    bpy.types.WindowManager.brignet_threshold = FloatProperty(name="threshold", default=0.75e-2,
                                                              description='Minimum skin weight',
                                                              min=0.01,
                                                              max=1.0)

    bpy.types.WindowManager.brignet_obj_path = StringProperty(name='Mesh obj',
                                                              description='Path to Mesh file',
                                                              subtype='FILE_PATH')

    bpy.types.WindowManager.brignet_skel_path = StringProperty(name='Skeleton txt',
                                                               description='Path to Skeleton File',
                                                               subtype='FILE_PATH')

    bpy.utils.register_class(BrigNetPredict)  # FIXME: not a property


def unregister_properties():
    bpy.utils.unregister_class(BrigNetPredict)

    del bpy.types.WindowManager.brignet_downsample_skin
    del bpy.types.WindowManager.brignet_downsample_decimate
    del bpy.types.WindowManager.brignet_downsample_sampling
    del bpy.types.WindowManager.brignet_targetmesh
    del bpy.types.WindowManager.brignet_highrescollection
    del bpy.types.WindowManager.brignet_density
    del bpy.types.WindowManager.brignet_threshold
    del bpy.types.WindowManager.brignet_obj_path
    del bpy.types.WindowManager.brignet_skel_path
