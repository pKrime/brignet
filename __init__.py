# ====================== BEGIN GPL LICENSE BLOCK ======================
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation, version 3.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ======================= END GPL LICENSE BLOCK ========================


bl_info = {
    "name": "Neural Rigging (RigNet)",
    "version": (0, 1, 0),
    "author": "Paolo Acampora",
    "blender": (2, 90, 0),
    "description": "Armature and Weights prediction using RigNet",
    "location": "3D Viewport",
    "doc_url": "https://github.com/pKrime/brignet",
    "category": "Rigging",
}

import bpy

from . import brignet, preferences, loadskeleton
from . import postgen_utils
from .ui import menus

from importlib import reload
try:
    reload(brignet)
    reload(preferences)
    reload(loadskeleton)
    reload(postgen_utils)
except NameError:
    pass

from .brignet import BrignetPanel, BrigNetPredict, BrignetRemesh, BrignetCollection
from .preferences import BrignetPrefs, BrignetEnvironment
from .loadskeleton import LoadRignetSkeleton, LoadSkeletonPanel
from .postgen_utils import NamiFy, ExtractMetarig, SpineFix, MergeBones


# REGISTER #

def register():
    brignet.register_properties()
    bpy.utils.register_class(BrignetEnvironment)
    bpy.utils.register_class(BrignetPrefs)
    bpy.utils.register_class(BrignetCollection)
    bpy.utils.register_class(BrignetRemesh)
    bpy.utils.register_class(BrigNetPredict)

    bpy.utils.register_class(NamiFy)
    bpy.utils.register_class(ExtractMetarig)
    bpy.utils.register_class(SpineFix)
    bpy.utils.register_class(MergeBones)

    bpy.utils.register_class(BrignetPanel)
    bpy.utils.register_class(LoadRignetSkeleton)
    bpy.utils.register_class(LoadSkeletonPanel)

    BrignetPrefs.check_cuda()
    if not BrignetPrefs.add_module_paths():
        print("Modules path not found, please set in bRigNet preferences")
    BrignetPrefs.check_modules()

    bpy.types.VIEW3D_MT_pose_context_menu.append(menus.pose_context_options)


def unregister():
    try:
        from . import rignetconnect
        rignetconnect.clear()
    except ModuleNotFoundError:
        # if we have failed to load rignetconnect, we have no device to clear
        pass

    bpy.types.VIEW3D_MT_pose_context_menu.remove(menus.pose_context_options)
    BrignetPrefs.reset_module_paths()

    bpy.utils.unregister_class(BrignetPanel)
    bpy.utils.unregister_class(BrignetPrefs)
    bpy.utils.unregister_class(BrignetEnvironment)
    bpy.utils.unregister_class(BrignetCollection)
    bpy.utils.unregister_class(BrignetRemesh)
    bpy.utils.unregister_class(BrigNetPredict)
    bpy.utils.unregister_class(NamiFy)
    bpy.utils.unregister_class(ExtractMetarig)
    bpy.utils.unregister_class(SpineFix)
    bpy.utils.unregister_class(MergeBones)
    bpy.utils.unregister_class(LoadSkeletonPanel)
    bpy.utils.unregister_class(LoadRignetSkeleton)
    brignet.unregister_properties()
