import bpy
import bmesh


def mesh_from_collection(collection, name=None):
    name = name if name else collection.name + '_join'
    objects = collection.objects[:]

    bm = bmesh.new()
    first_object = objects.pop()
    bm.from_object(first_object, bpy.context.evaluated_depsgraph_get())
    bm.transform(first_object.matrix_world)

    for ob in objects:
        bm.verts.ensure_lookup_table()
        last_idx = len(bm.verts)

        other_bm = bmesh.new()
        other_bm.from_object(ob, bpy.context.evaluated_depsgraph_get())
        other_bm.transform(ob.matrix_world)
        other_bm.verts.ensure_lookup_table()
        other_bm.edges.ensure_lookup_table()
        other_bm.faces.ensure_lookup_table()

        for vert in other_bm.verts:
            bm.verts.new(vert.co)
        bm.verts.ensure_lookup_table()

        for edge in other_bm.edges:
            bm.edges.new([bm.verts[vert.index + last_idx] for vert in edge.verts])

        for face in other_bm.faces:
            bm.faces.new([bm.verts[vert.index + last_idx] for vert in face.verts])

    new_mesh = bpy.data.meshes.new(name)
    bm.to_mesh(new_mesh)
    bm.free()
    new_ob = bpy.data.objects.new(name, new_mesh)

    bpy.context.scene.collection.objects.link(new_ob)
    return new_ob


def get_armature_modifier(ob):
    return next((mod for mod in ob.modifiers if mod.type == 'ARMATURE'), None)


def copy_weights(ob_list, ob_source, apply_modifier=True):
    src_mod = get_armature_modifier(ob_source)
    src_mod.show_viewport = False
    src_mod.show_render = False
    ob_source.hide_set(True)

    for ob in ob_list:
        remove_modifiers(ob)

        transf = ob.modifiers.new('weight_transf', 'DATA_TRANSFER')
        if not transf:
            continue

        transf.object = ob_source
        transf.use_vert_data = True
        transf.data_types_verts = {'VGROUP_WEIGHTS'}
        transf.vert_mapping = 'POLY_NEAREST'

        arm = ob.modifiers.new('Armature', 'ARMATURE')
        arm.object = src_mod.object
        arm.show_in_editmode = True
        arm.show_on_cage = True

        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.datalayout_transfer(modifier=transf.name)

        if apply_modifier:
            bpy.ops.object.modifier_apply(modifier=transf.name)


def remove_modifiers(ob, type_list=('DATA_TRANSFER', 'ARMATURE')):
    for mod in reversed(ob.modifiers):
        if mod.type in type_list:
            ob.modifiers.remove(mod)


class ArmatureGenerator(object):
    def __init__(self, info, mesh=None):
        self._info = info
        self._mesh = mesh

    def generate(self, matrix=None):
        basename = self._mesh.name if self._mesh else ""
        arm_data = bpy.data.armatures.new(basename + "_armature")
        arm_obj = bpy.data.objects.new('brignet_rig', arm_data)

        bpy.context.collection.objects.link(arm_obj)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='EDIT')

        this_level = [self._info.root]
        hier_level = 1
        while this_level:
            next_level = []
            for p_node in this_level:
                pos = p_node.pos
                parent = p_node.parent.name if p_node.parent is not None else None

                e_bone = arm_data.edit_bones.new(p_node.name)
                if self._mesh and e_bone.name not in self._mesh.vertex_groups:
                    self._mesh.vertex_groups.new(name=e_bone.name)

                e_bone.head.x, e_bone.head.z, e_bone.head.y = pos[0], pos[2], pos[1]

                if parent:
                    e_bone.parent = arm_data.edit_bones[parent]
                    if e_bone.parent.tail == e_bone.head:
                        e_bone.use_connect = True

                if len(p_node.children) == 1:
                    pos = p_node.children[0].pos
                    e_bone.tail.x, e_bone.tail.z, e_bone.tail.y = pos[0], pos[2], pos[1]
                elif len(p_node.children) > 1:
                    x_offset = [abs(c_node.pos[0] - pos[0]) for c_node in p_node.children]

                    idx = x_offset.index(min(x_offset))
                    pos = p_node.children[idx].pos
                    e_bone.tail.x, e_bone.tail.z, e_bone.tail.y = pos[0], pos[2], pos[1]

                elif e_bone.parent:
                    offset = e_bone.head - e_bone.parent.head
                    e_bone.tail = e_bone.head + offset / 2
                else:
                    e_bone.tail.x, e_bone.tail.z, e_bone.tail.y = pos[0], pos[2], pos[1]
                    e_bone.tail.y += .1

                for c_node in p_node.children:
                    next_level.append(c_node)

            this_level = next_level
            hier_level += 1

        if matrix:
            arm_data.transform(matrix)

        bpy.ops.object.mode_set(mode='POSE')

        if self._mesh:
            for v_skin in self._info.joint_skin:
                v_idx = int(v_skin.pop(0))

                for i in range(0, len(v_skin), 2):
                    self._mesh.vertex_groups[v_skin[i]].add([v_idx], float(v_skin[i + 1]), 'REPLACE')

            arm_obj.matrix_world = self._mesh.matrix_world
            mod = self._mesh.modifiers.new('rignet', 'ARMATURE')
            mod.object = arm_obj

        return arm_obj
