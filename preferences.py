import os
from pathlib import Path
import sys

import bpy

from .setup_utils import cuda_utils
from .setup_utils import venv_utils
from importlib.util import find_spec


class BrignetEnvironment(bpy.types.Operator):
    """Create virtual environment with required modules"""
    bl_idname = "wm.brignet_environment"
    bl_label = "Create Remesh model from Collection"

    @classmethod
    def poll(cls, context):
        env_path = bpy.context.preferences.addons[__package__].preferences.modules_path
        if not env_path:
            return False

        return len(BrignetPrefs.missing_modules) > 0

    def execute(self, context):
        env_path = bpy.context.preferences.addons[__package__].preferences.modules_path
        venv_utils.setup_environment(env_path)
        BrignetPrefs.add_module_paths()
        return {'FINISHED'}


class BrignetPrefs(bpy.types.AddonPreferences):
    bl_idname = __package__

    _cuda_info = None
    _added_paths = []
    missing_modules = []

    @staticmethod
    def check_cuda():
        BrignetPrefs._cuda_info = cuda_utils.CudaDetect()

    @staticmethod
    def add_module_paths():
        BrignetPrefs.reset_module_paths()
        env_path = bpy.context.preferences.addons[__package__].preferences.modules_path

        if not os.path.isdir(env_path):
            return False
            
        if sys.platform.startswith("linux"):
            lib_path = os.path.join(env_path, 'lib')
            sitepackages = os.path.join(lib_path, 'python3.7', 'site-packages')
        else:
            lib_path = os.path.join(env_path, 'Lib')
            sitepackages = os.path.join(lib_path, 'site-packages')

        if not os.path.isdir(sitepackages):
            # not a python path, but the user might be still typing
            return False

        platformpath = os.path.join(sitepackages, sys.platform)
        platformlibs = os.path.join(platformpath, 'lib')

        mod_paths = [lib_path, sitepackages, platformpath, platformlibs]
        if sys.platform.startswith("win"):
            mod_paths.append(os.path.join(env_path, 'DLLs'))
            mod_paths.append(os.path.join(sitepackages, 'Pythonwin'))

        for mod_path in mod_paths:
            if not os.path.isdir(mod_path):
                print(f'{mod_path} not a directory, skipping')
                continue
            if mod_path not in sys.path:
                print(f'adding {mod_path}')
                sys.path.append(mod_path)
                BrignetPrefs._added_paths.append(mod_path)

        BrignetPrefs.check_modules()
        return True

    @staticmethod
    def reset_module_paths():
        # FIXME: even if we do this, additional modules are still available
        for mod_path in BrignetPrefs._added_paths:
            print(f"removing module path: {mod_path}")
            sys.path.remove(mod_path)
        BrignetPrefs._added_paths.clear()

    def update_modules(self, context):
        self.add_module_paths()

    modules_path: bpy.props.StringProperty(
        name='RigNet environment path',
        description='Path to additional modules (torch, torch_geometric...)',
        subtype='DIR_PATH',
        update=update_modules,
        default=os.path.join(os.path.join(os.path.dirname(__file__)), '_additional_modules')
    )

    model_path: bpy.props.StringProperty(
        name='Model path',
        description='Path to RigNet code',
        subtype='DIR_PATH',
        default=os.path.join(os.path.join(os.path.dirname(__file__)), 'RigNet', 'checkpoints')
    )

    modules_found: bpy.props.BoolProperty(
        name='Required Modules',
        description="Whether required modules have been found or not"
    )

    @staticmethod
    def check_modules():
        BrignetPrefs.missing_modules.clear()
        for mod_name in ('torch', 'torch_geometric', 'torch_cluster', 'torch_sparse', 'torch_scatter', 'scipy'):
            if not find_spec(mod_name):
                BrignetPrefs.missing_modules.append(mod_name)

        preferences = bpy.context.preferences.addons[__package__].preferences
        preferences.modules_found = len(BrignetPrefs.missing_modules) == 0

    def draw(self, context):
        layout = self.layout
        column = layout.column()

        info = BrignetPrefs._cuda_info
        if info:
            py_ver = sys.version_info
            row = column.row()
            row.label(text=f"Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
            if info.result == cuda_utils.CudaResult.SUCCESS:
                row = column.row()
                row.label(text=f"Cuda Version: {info.major}.{info.minor}.{info.micro}")
            elif info.result == cuda_utils.CudaResult.NOT_FOUND:
                row = column.row()
                row.label(text="CUDA Toolkit not found", icon='ERROR')

                if info.has_cuda_hardware:
                    row = column.row()
                    split = row.split(factor=0.1, align=False)
                    split.column()
                    col = split.column()
                    col.label(text="CUDA hardware is present. Please make sure that CUDA Toolkit is installed")

                    op = col.operator(
                        'wm.url_open',
                        text='nVidia Downloads',
                        icon='URL'
                    )
                    op.url = 'https://developer.nvidia.com/downloads'

        if self.missing_modules:
            row = column.row()
            row.label(text=f"Modules not found: {','.join(self.missing_modules)}", icon='ERROR')

        box = column.box()
        col = box.column()

        row = col.row()
        split = row.split(factor=0.8, align=False)
        sp_col = split.column()
        sp_col.prop(self, 'modules_path', text='Modules Path')

        if self.missing_modules:
            sp_col = split.column()
            sp_col.operator(BrignetEnvironment.bl_idname, text='Install')

        row = col.row()
        split = row.split(factor=0.8, align=False)
        sp_col = split.column()
        sp_col.prop(self, 'model_path', text='Model Path')
        if not os.path.isdir(self.model_path) or 'bonenet' not in os.listdir(self.model_path):
            sp_col = split.column()
            op = sp_col.operator(
                'wm.url_open',
                text='Download'
            )
            op.url = "https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/EYKLCvYTWFJArehlo3-H2SgBABnY08B4k5Q14K7H1Hh0VA"

            row = col.row()

            if self.model_path:
                row.label(text="Please, unpack the content of 'checkpoints' to")
                row = col.row()
                row.label(text=f"    {self.model_path}")
            else:
                row.label(text="Please, unpack the content of 'checkpoints' to the 'Model Path' folder")

        row = layout.row()
        row.label(text="End of bRigNet Preferences")
