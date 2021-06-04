import bpy
from bpy.props import IntProperty, BoolProperty, FloatProperty, PointerProperty, StringProperty

from .ob_utils import objects


class BrignetRemesh(bpy.types.Operator):
    """Create remeshed model from highres objects"""
    bl_idname = "object.brignet_remesh"
    bl_label = "Create Remesh model from Collection"

    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        if not wm.brignet_highrescollection:
            return False

        return True

    def execute(self, context):
        wm = context.window_manager
        new_ob = objects.mesh_from_collection(wm.brignet_highrescollection, name='brignet_remesh')

        remesh = new_ob.modifiers.new(name='remesh', type='REMESH')
        remesh.voxel_size = 0.01

        decimate = new_ob.modifiers.new(name='decimate', type='DECIMATE')
        decimate.use_collapse_triangulate = True

        context.evaluated_depsgraph_get()
        decimate.ratio = 1800 / decimate.face_count

        wm.brignet_targetmesh = new_ob
        wm.brignet_highrescollection.hide_viewport = True
        return {'FINISHED'}


class BrignetCollection(bpy.types.Operator):
    """Create collection from selected objects"""
    bl_idname = 'collection.brignet_collection'
    bl_label = 'Create collection from selected objects'

    @classmethod
    def poll(cls, context):
        if not context.selected_objects:
            return False
        if not next((ob for ob in context.selected_objects if ob.type == 'MESH'), None):
            return False
        return True

    def execute(self, context):
        default_collection = bpy.data.collections['Collection']
        collection = bpy.data.collections.new("BrignetGeometry")
        for ob in context.selected_objects:
            if ob.type != 'MESH':
                continue
            collection.objects.link(ob)
            try:
                default_collection.objects.unlink(ob)
            except RuntimeError:
                pass

        bpy.context.scene.collection.children.link(collection)
        context.window_manager.brignet_highrescollection = collection

        return {'FINISHED'}


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
        objects.remove_modifiers(wm.brignet_targetmesh, type_list=('ARMATURE',))

        bandwidth = (1 - wm.brignet_density) / 10
        threshold = wm.brignet_threshold/1000

        from . import rignetconnect
        rignetconnect.predict_rig(wm.brignet_targetmesh, bandwidth, threshold,
                                  wm.brignet_downsample_skin,
                                  wm.brignet_downsample_decimate,
                                  wm.brignet_downsample_sampling)

        if wm.brignet_highrescollection:
            wm.brignet_highrescollection.hide_viewport = False
            objects.copy_weights(wm.brignet_highrescollection.objects, wm.brignet_targetmesh)

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
        col = row.column()
        col.prop(wm, 'brignet_highrescollection', text='HighRes')
        col = row.column()
        col.operator('collection.brignet_collection', text='<-')

        row = layout.row()
        col = row.column()
        col.prop(wm, 'brignet_targetmesh', text='Target')
        col = row.column()
        col.operator('object.brignet_remesh', text='<-')

        if wm.brignet_targetmesh:
            remesh_mod = next((mod for mod in wm.brignet_targetmesh.modifiers if mod.type == 'REMESH'), None)
            decimate_mod = next((mod for mod in wm.brignet_targetmesh.modifiers if mod.type == 'DECIMATE'), None)
            if remesh_mod:
                row = layout.row()
                row.prop(remesh_mod, 'voxel_size')
            if decimate_mod:
                row = layout.row()
                row.prop(decimate_mod, 'ratio')
                row = layout.row()
                row.label(text='face count: {0}'.format(decimate_mod.face_count))

        row = layout.row()
        row.operator('object.brignet_predict')

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
                                                              min=0.01e-2,
                                                              max=1.0)

    bpy.types.WindowManager.brignet_obj_path = StringProperty(name='Mesh obj',
                                                              description='Path to Mesh file',
                                                              subtype='FILE_PATH')

    bpy.types.WindowManager.brignet_skel_path = StringProperty(name='Skeleton txt',
                                                               description='Path to Skeleton File',
                                                               subtype='FILE_PATH')


def unregister_properties():
    del bpy.types.WindowManager.brignet_downsample_skin
    del bpy.types.WindowManager.brignet_downsample_decimate
    del bpy.types.WindowManager.brignet_downsample_sampling
    del bpy.types.WindowManager.brignet_targetmesh
    del bpy.types.WindowManager.brignet_highrescollection
    del bpy.types.WindowManager.brignet_density
    del bpy.types.WindowManager.brignet_threshold
    del bpy.types.WindowManager.brignet_obj_path
    del bpy.types.WindowManager.brignet_skel_path
