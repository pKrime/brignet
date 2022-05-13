from enum import Enum

import bpy
from bpy.props import BoolProperty, FloatProperty, PointerProperty, StringProperty

from .ob_utils import objects
from .postgen_utils.bone_utils import NameFix

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

        for ob in bpy.data.collections[collection_name].all_objects:
            ob.hide_set(True)

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


class PredictSteps(Enum):
    NotStarted = 0
    Loading_Networks = 1
    Creating_Data = 2
    Predicting_Joints = 3
    Predicting_Hierarchy = 4
    Predicting_Weights = 5
    Creating_Armature = 6
    Post_Generation = 7
    Finished = 8

    @staticmethod
    def last():
        return PredictSteps.Finished

    @property
    def icon(self):
        if self.value == self.Loading_Networks.value:
            return 'NETWORK_DRIVE'
        if self.value == self.Creating_Data.value:
            return 'OUTLINER_DATA_POINTCLOUD'
        if self.value == self.Predicting_Joints.value:
            return 'BONE_DATA'
        if self.value == self.Predicting_Hierarchy.value:
            return 'ARMATURE_DATA'
        if self.value == self.Predicting_Weights.value:
            return 'OUTLINER_OB_ARMATURE'
        if self.value == self.Creating_Armature.value:
            return 'SCENE_DATA'
        if self.value == self.Finished.value:
            return 'CHECKMARK'
        return 'NONE'

    @property
    def nice_name(self):
        nice_name = self.name.replace('_', ' ')

        if self.value == self.Creating_Data.value:
            nice_name += " (takes a while...)"

        return nice_name


class BrigNetPredict(bpy.types.Operator):
    """Predict joint position of chosen mesh using a trained model"""
    bl_idname = "object.brignet_predict"
    bl_label = "Predict Rig"

    bandwidth: FloatProperty()
    threshold: FloatProperty()
    current_step: PredictSteps.NotStarted

    _timer = None
    _networks = None
    _mesh_storage = None
    _pred_data = None
    _pred_skeleton = None
    _pred_rig = None
    _armature = None

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
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.brignet_current_progress = 0.0
        rignetconnect.clear()

    def modal(self, context, event):
        """Go through the prediction steps and show feedback"""
        context.area.tag_redraw()
        if event.type == 'ESC':
            self.clean_up(context)
            return {'CANCELLED'}

        wm = context.window_manager
        wm.brignet_targetmesh.hide_set(False)  # hidden target mesh might cause crashes
        if self.current_step == PredictSteps.Loading_Networks:
            self._networks = rignetconnect.Networks(load_skinning=wm.brignet_predict_weights)
        elif self.current_step == PredictSteps.Creating_Data:
            self._pred_data, self._mesh_storage = rignetconnect.init_data(wm.brignet_targetmesh, wm.brignet_samples)
        elif self.current_step == PredictSteps.Predicting_Joints:
            self._pred_data = rignetconnect.predict_joint(self._pred_data, self._networks.joint_net, self._mesh_storage,
                                                          self.bandwidth, self.threshold)
        elif self.current_step == PredictSteps.Predicting_Hierarchy:
            self._pred_skeleton = rignetconnect.predict_hierarchy(self._pred_data, self._networks, self._mesh_storage)
        elif self.current_step == PredictSteps.Predicting_Weights:
            if wm.brignet_predict_weights:
                self._pred_rig = rignetconnect.predict_weights(self._pred_data, self._pred_skeleton,
                                                               self._networks.skin_net, self._mesh_storage)
            else:
                self._pred_rig = self._pred_skeleton
                mesh_data = self._mesh_storage.mesh_data
                self._pred_rig.normalize(mesh_data.scale_normalize, -mesh_data.translation_normalize)
        elif self.current_step == PredictSteps.Creating_Armature:
            self._armature = rignetconnect.create_armature(wm.brignet_targetmesh, self._pred_rig)
        elif self.current_step == PredictSteps.Post_Generation and self._armature:
            if wm.brignet_mirror_names:
                renamer = NameFix(self._armature)
                renamer.name_left_right()
        elif self.current_step == PredictSteps.Finished:
            self.clean_up(context)

            if wm.brignet_highrescollection:
                wm.brignet_highrescollection.hide_viewport = False
                objects.copy_weights(wm.brignet_highrescollection.objects, wm.brignet_targetmesh)

                for ob in wm.brignet_highrescollection.all_objects:
                    ob.hide_set(False)
                wm.brignet_targetmesh.hide_set(True)

            return {'FINISHED'}

        # Advance current state
        try:
            self.current_step = PredictSteps(self.current_step.value + 1)
            wm.brignet_current_progress = self.current_step.value
        except ValueError:
            self.clean_up(context)
            return {'FINISHED'}

        return {'INTERFACE'}

    def invoke(self, context, event):
        # we import the rignet module here to prevent import errors before the dependencies are installed
        global rignetconnect
        from . import rignetconnect

        wm = context.window_manager

        self.bandwidth = (1 - wm.brignet_density) / 10
        self.threshold = wm.brignet_threshold/1000
        self.current_step = PredictSteps(0)

        # timer event makes sure that the modal script is executed even without user interaction
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}


class BrignetPanel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Neural Rigging"
    bl_idname = "RIGNET_PT_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'RigNet'

    def draw(self, context):
        layout = self.layout

        wm = context.window_manager

        row = layout.row()
        row.label(text="Character Collection:")

        split = layout.split(factor=0.8, align=False)
        col = split.column()
        col.prop(wm, 'brignet_highrescollection', text='')
        col = split.column()
        col.operator(BrignetCollection.bl_idname, text='', icon='RESTRICT_SELECT_OFF')

        row = layout.row()
        row.label(text="Simple Mesh:")

        split = layout.split(factor=0.8, align=False)
        col = split.column()
        col.prop(wm, 'brignet_targetmesh', text='')
        col = split.column()
        col.operator(BrignetRemesh.bl_idname, text='', icon='RESTRICT_SELECT_OFF')

        if wm.brignet_targetmesh:
            remesh_mod = next((mod for mod in wm.brignet_targetmesh.modifiers if mod.type == 'REMESH'), None)
            decimate_mod = next((mod for mod in wm.brignet_targetmesh.modifiers if mod.type == 'DECIMATE'), None)
            if remesh_mod:
                row = layout.row()
                row.prop(remesh_mod, 'voxel_size', slider=True)
            if decimate_mod:
                row = layout.row()
                row.prop(decimate_mod, 'ratio', slider=True)
                row = layout.row()
                row.label(text='Face count: {0}'.format(decimate_mod.face_count))

                max_face_count = 5000
                if decimate_mod.face_count > max_face_count:
                    row = layout.row()
                    row.label(text=f'Face count too high (exceeds {max_face_count})', icon='ERROR')
                min_face_count = 1000
                if decimate_mod.face_count < min_face_count:
                    row = layout.row()
                    row.label(text=f'Face count too low (less than {min_face_count})', icon='ERROR')

        if wm.brignet_current_progress > 0.1:
            layout.separator()
            row = layout.row()
            current_step = PredictSteps(int(wm.brignet_current_progress))
            row.label(text=current_step.nice_name, icon=current_step.icon)
            row = layout.row()
            row.prop(wm, 'brignet_current_progress', slider=True)
        else:
            row = layout.row()
            row.operator('object.brignet_predict')

            row = layout.row()
            row.prop(wm, 'brignet_density', text='Density')

            row = layout.row()
            row.prop(wm, 'brignet_threshold', text='Treshold')

            row = layout.row()
            row.prop(wm, 'brignet_samples', text='Samples')

            row = layout.row()
            row.prop(wm, 'brignet_predict_weights')

            row = layout.row()
            row.prop(wm, 'brignet_mirror_names')


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

    bpy.types.WindowManager.brignet_samples = FloatProperty(name="samples", default=2000,
                                                            description='Poisson Disks Samples',
                                                            min=100,
                                                            max=5000)

    bpy.types.WindowManager.brignet_obj_path = StringProperty(name='Mesh obj',
                                                              description='Path to Mesh file',
                                                              subtype='FILE_PATH')

    bpy.types.WindowManager.brignet_skel_path = StringProperty(name='Skeleton txt',
                                                               description='Path to Skeleton File',
                                                               subtype='FILE_PATH')

    bpy.types.WindowManager.brignet_current_progress = FloatProperty(name="Progress", default=0.0,
                                                                     description='Progress of ongoing rig',
                                                                     min=0.0, max=PredictSteps.last().value,
                                                                     options={'HIDDEN', 'SKIP_SAVE'}
                                                                     )

    bpy.types.WindowManager.brignet_predict_weights = BoolProperty(name='Predict Weights', default=True,
                                                                   description='Predict Bone weights')

    bpy.types.WindowManager.brignet_mirror_names = BoolProperty(name='Mirror Bone Names', default=True,
                                                                description='Apply .L/.R names to symmetric bones')


def unregister_properties():
    del bpy.types.WindowManager.brignet_targetmesh
    del bpy.types.WindowManager.brignet_highrescollection
    del bpy.types.WindowManager.brignet_density
    del bpy.types.WindowManager.brignet_threshold
    del bpy.types.WindowManager.brignet_samples
    del bpy.types.WindowManager.brignet_obj_path
    del bpy.types.WindowManager.brignet_skel_path
    del bpy.types.WindowManager.brignet_current_progress
    del bpy.types.WindowManager.brignet_predict_weights
    del bpy.types.WindowManager.brignet_mirror_names
