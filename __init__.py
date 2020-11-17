# ====================== BEGIN GPL LICENSE BLOCK ======================
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
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

# <pep8 compliant>

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
from . import brignet
from .brignet import BrignetPanel
from .preferences import BrignetPrefs


from importlib import reload
try:
    reload(brignet)
    reload(preferences)
except NameError:
    pass


# REGISTER #

def register():
    from importlib import reload
    try:
        reload(brignet)
        reload(preferences)
    except NameError:
        pass

    brignet.register_properties()
    bpy.utils.register_class(BrignetPrefs)
    bpy.utils.register_class(BrignetPanel)

    if not preferences.append_rignet():
        print("RigNet not found, please set in bRigNet preferences")
    if not preferences.append_modules():
        print("Modules path not found, please set in bRigNet preferences")


def unregister():
    try:
        from . import rignetconnect
        rignetconnect.clear()
    except ModuleNotFoundError:
        # if we have failed to load rignetconnect, we have no device to clear
        pass

    bpy.utils.unregister_class(BrignetPanel)
    bpy.utils.unregister_class(BrignetPrefs)
    brignet.unregister_properties()
