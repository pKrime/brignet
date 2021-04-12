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




if __name__ == '__main__':
    mesh_from_collection(bpy.data.collections['test_collection'])
