from enum import Enum

import blf
import bpy
from bpy.props import IntProperty, BoolProperty, FloatProperty, PointerProperty, StringProperty

from .ob_utils import objects
from .preferences import BrignetPrefs

try:
    from . import rignetconnect
except ModuleNotFoundError:
    pass


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
        if wm.brignet_targetmesh:
            # remove previous mesh
            bpy.data.objects.remove(wm.brignet_targetmesh, do_unlink=True)
        new_ob = objects.mesh_from_collection(wm.brignet_highrescollection, name='brignet_remesh')

        remesh = new_ob.modifiers.new(name='remesh', type='REMESH')
        remesh.voxel_size = 0.01

        decimate = new_ob.modifiers.new(name='decimate', type='DECIMATE')
        decimate.use_collapse_triangulate = True

        context.evaluated_depsgraph_get()
        decimate.ratio = 1800 / decimate.face_count

        new_ob.hide_render = True
        wm.brignet_targetmesh = new_ob

        collection_name = wm.brignet_highrescollection.name
        view_layer = bpy.context.view_layer.layer_collection.children.get(collection_name)
        view_layer.hide_viewport = True
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
        collection = bpy.data.collections.new("BrignetGeometry")
        for ob in context.selected_objects:
            if ob.type != 'MESH':
                continue
            collection.objects.link(ob)

        bpy.context.scene.collection.children.link(collection)
        context.window_manager.brignet_highrescollection = collection

        return {'FINISHED'}


def draw_callback_px(self, context):
    font_id = 0

    # draw some text
    blf.color(font_id, 0.9, 0.9, 0.9, 0.8)
    blf.position(font_id, 15, 30, 0)
    blf.size(font_id, 20, 72)
    blf.draw(font_id, self.current_step.name.replace('_', ' '))


class PredictSteps(Enum):
    NotStarted = 0
    Loading_Networks = 1
    Creating_Data = 2
    Predicting_Joints = 3
    Predicting_Hierarchy = 4
    Predicting_Weights = 5
    Creating_Armature = 6
    Finished = 7


class BrigNetPredict(bpy.types.Operator):
    """Predict joint position of chosen mesh using a trained model"""
    bl_idname = "object.brignet_predict"
    bl_label = "Predict joints and skinning"

    bandwidth: FloatProperty()
    threshold: FloatProperty()
    current_step: PredictSteps

    _timer = None
    _handle = None

    _networks = None
    _mesh_storage = None
    _pred_data = None
    _pred_skeleton = None
    _pred_rig = None

    @classmethod
    def poll(cls, context):
        modules_found = bpy.context.preferences.addons[__package__].preferences.modules_found
        if not modules_found:
            # TODO: we should rather gray out the whole panel and display a warning
            return False

        wm = context.window_manager
        if not wm.brignet_targetmesh:
            return False

        return wm.brignet_targetmesh.type == 'MESH'

    def clean_up(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        context.window_manager.event_timer_remove(self._timer)
        rignetconnect.clear()

    def modal(self, context, event):
        """Go through the prediction steps and show feedback"""
        context.area.tag_redraw()
        if event.type == 'ESC':
            self.clean_up(context)
            return {'CANCELLED'}

        wm = context.window_manager
        if self.current_step == PredictSteps.Loading_Networks:
            self._networks = rignetconnect.Networks()
        elif self.current_step == PredictSteps.Creating_Data:
            self._pred_data, self._mesh_storage = rignetconnect.init_data(wm.brignet_targetmesh)
        elif self.current_step == PredictSteps.Predicting_Joints:
            self._pred_data = rignetconnect.predict_joint(self._pred_data, self._networks.joint_net, self._mesh_storage,
                                                          self.bandwidth, self.threshold)
        elif self.current_step == PredictSteps.Predicting_Hierarchy:
            self._pred_skeleton = rignetconnect.predict_hierarchy(self._pred_data, self._networks, self._mesh_storage)
        elif self.current_step == PredictSteps.Predicting_Weights:
            self._pred_rig = rignetconnect.predict_weights(self._pred_data, self._pred_skeleton,
                                                        self._networks.skin_net, self._mesh_storage)
        elif self.current_step == PredictSteps.Creating_Armature:
            rignetconnect.create_armature(wm.brignet_targetmesh, self._pred_rig)
        elif self.current_step == PredictSteps.Finished:
            self.clean_up(context)

            if wm.brignet_highrescollection:
                wm.brignet_highrescollection.hide_viewport = False
                objects.copy_weights(wm.brignet_highrescollection.objects, wm.brignet_targetmesh)

            return {'FINISHED'}

        # Advance current state
        try:
            self.current_step = PredictSteps(self.current_step.value + 1)
        except ValueError:
            self.clean_up(context)
            return {'FINISHED'}

        return {'INTERFACE'}

    def invoke(self, context, event):
        global rignetconnect
        from . import rignetconnect

        wm = context.window_manager

        self.bandwidth = (1 - wm.brignet_density) / 10
        self.threshold = wm.brignet_threshold/1000
        self.current_step = PredictSteps(0)

        args = (self, context)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_PIXEL')
        # timer event makes sure that the modal script is executed even without user interaction
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}


class BrignetPanel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "bRigNet Meshes"
    bl_idname = "RIGNET_PT_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'bRigNet'

    def draw(self, context):
        layout = self.layout

        wm = context.window_manager

        row = layout.row()
        row.label(text="Character Collection:")

        split = layout.split(factor=0.8, align=False)
        col = split.column()
        col.prop(wm, 'brignet_highrescollection', text='')
        col = split.column()
        col.operator(BrignetCollection.bl_idname, text='<-')

        row = layout.row()
        row.label(text="Simple Mesh:")

        split = layout.split(factor=0.8, align=False)
        col = split.column()
        col.prop(wm, 'brignet_targetmesh', text='')
        col = split.column()
        col.operator(BrignetRemesh.bl_idname, text='<-')

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
    del bpy.types.WindowManager.brignet_targetmesh
    del bpy.types.WindowManager.brignet_highrescollection
    del bpy.types.WindowManager.brignet_density
    del bpy.types.WindowManager.brignet_threshold
    del bpy.types.WindowManager.brignet_obj_path
    del bpy.types.WindowManager.brignet_skel_path
