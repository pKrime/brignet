import bpy

import os
import sys

from enum import Enum
import subprocess


class CudaResult(Enum):
    SUCCESS = 1
    NOT_FOUND = 2


class CudaDetect:
    """Checks Cuda version installed in the system"""
    def __init__(self):
        self.result = None
        self.major = 0
        self.minor = 0
        self.micro = 0
        self.has_cuda_hardware = False

        self.has_cuda_device()
        self.detect_cuda_ver()

    @staticmethod
    def get_cuda_path():
        try:
            return os.environ['CPATH']
        except KeyError:
            pass

        where_cmd = "where" if sys.platform.startswith('win') else "whereis"
        result = subprocess.check_output([where_cmd, 'nvcc']).decode('UTF-8')

        if '\n' in result:
            nvcc_path = result.split('\n', 1)[0]
        else:
            nvcc_path = result.split('\t', 1)[0]

        nvcc_dir, _ = os.path.split(nvcc_path)
        cuda_dir = os.path.dirname(nvcc_dir)

        return cuda_dir

    def has_cuda_device(self):
        """Checks for cuda hardware in cycles preferences"""
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences

        if bpy.app.version[0] > 2:
            # devices are iterated differently in blender 3.0/blender 2.9
            cprefs.refresh_devices()

            def get_dev():
                for dev in cprefs.devices:
                    yield dev
        else:
            def get_dev():
                for dev in cprefs.get_devices(bpy.context):
                    for dev_entry in dev:
                        yield dev_entry

        for device in get_dev():
            if device.type == 'CUDA':
                    self.has_cuda_hardware = True
                    return

    def detect_cuda_ver(self):
        """Try execute the cuda compiler with the --version flag"""
        try:
            nvcc_out = subprocess.check_output(["nvcc", "--version"])
        except FileNotFoundError:
            self.result = CudaResult.NOT_FOUND
            return

        nvcc_out = str(nvcc_out)
        ver = nvcc_out.rsplit(" V", 1)[-1]
        ver = ver.strip("'\\r\\n")
        ver_ends = next((i for i, c in enumerate(ver) if not (c.isdigit() or c == '.')), len(ver))
        ver = ver[:ver_ends]

        self.major, self.minor, self.micro = ver.split(".", 2)
        self.result = CudaResult.SUCCESS
