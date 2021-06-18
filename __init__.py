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
    "name": "bRigNet",
    "version": (0, 0, 1),
    "author": "Paolo Acampora",
    "blender": (2, 90, 0),
    "description": "Armature and Weights prediction using RigNet",
    "location": "Armature properties",
    "doc_url": "https://github.com/pKrime/brignet",
    "category": "Rigging",
}

import bpy

from . import brignet, preferences, loadskeleton

from importlib import reload
try:
    reload(brignet)
    reload(preferences)
    reload(loadskeleton)
except NameError:
    pass

from .brignet import BrignetPanel, BrigNetPredict, BrignetRemesh, BrignetCollection
from .preferences import BrignetPrefs, BrignetEnvironment, BrignetCheckpoints
from .loadskeleton import LoadRignetSkeleton, LoadSkeletonPanel


# REGISTER #

def register():
    brignet.register_properties()
    bpy.utils.register_class(BrignetEnvironment)
    bpy.utils.register_class(BrignetCheckpoints)
    bpy.utils.register_class(BrignetPrefs)
    bpy.utils.register_class(BrignetCollection)
    bpy.utils.register_class(BrignetRemesh)
    bpy.utils.register_class(BrigNetPredict)
    bpy.utils.register_class(BrignetPanel)
    bpy.utils.register_class(LoadRignetSkeleton)
    bpy.utils.register_class(LoadSkeletonPanel)

    BrignetPrefs.check_cuda()
    if not BrignetPrefs.add_module_paths():
        print("Modules path not found, please set in bRigNet preferences")
    BrignetPrefs.check_modules()


def unregister():
    try:
        from . import rignetconnect
        rignetconnect.clear()
    except ModuleNotFoundError:
        # if we have failed to load rignetconnect, we have no device to clear
        pass

    BrignetPrefs.reset_module_paths()

    bpy.utils.unregister_class(BrignetPanel)
    bpy.utils.unregister_class(BrignetPrefs)
    bpy.utils.unregister_class(BrignetCheckpoints)
    bpy.utils.unregister_class(BrignetEnvironment)
    bpy.utils.unregister_class(BrignetCollection)
    bpy.utils.unregister_class(BrignetRemesh)
    bpy.utils.unregister_class(BrigNetPredict)
    bpy.utils.unregister_class(LoadSkeletonPanel)
    bpy.utils.unregister_class(LoadRignetSkeleton)
    brignet.unregister_properties()
