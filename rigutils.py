import bpy


def get_armature_modifier(ob):
    # TODO
    return ob.modifiers[0]


def copy_weights(ob_list, ob_source):
    src_mod = get_armature_modifier(ob_source)
    src_mod.show_viewport = False
    src_mod.show_render = False
    ob_source.hide_viewport = True
    ob_source.hide_render = True

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

        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.datalayout_transfer(modifier="weight_transf")


def remove_modifiers(ob, type_list=('DATA_TRANSFER', 'ARMATURE')):
    for mod in reversed(ob.modifiers):
        if mod.type in type_list:
            ob.modifiers.remove(mod)
