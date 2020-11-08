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
from . import rignetconnect
from .brignet import BrignetPanel


from importlib import reload
try:
    reload(brignet)
except NameError:
    pass


# REGISTER #

def register():
    from importlib import reload
    try:
        reload(brignet)
        reload(rignetconnect)
    except NameError:
        pass

    rignetconnect.load_networks()
    brignet.register_properties()
    bpy.utils.register_class(BrignetPanel)


def unregister():
    rignetconnect.clear()
    bpy.utils.unregister_class(BrignetPanel)
    brignet.unregister_properties()
