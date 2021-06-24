from ..postgen_utils import NamiFy
from ..postgen_utils import ExtractMetarig
from ..postgen_utils import SpineFix
from ..postgen_utils import MergeBones


def menu_header(layout):
    row = layout.row()
    row.separator()

    row = layout.row()
    row.label(text="Neural Rig Utils")


def pose_context_options(self, context):
    layout = self.layout
    menu_header(layout)

    row = layout.row()
    row.operator(ExtractMetarig.bl_idname)

    row = layout.row()
    row.operator(NamiFy.bl_idname)

    row = layout.row()
    row.operator(SpineFix.bl_idname)

    row = layout.row()
    row.operator(MergeBones.bl_idname)
