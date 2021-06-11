import bpy

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
        self.release = 0
        self.has_cuda_hardware = False

        self.has_cuda_device()
        self.detect_cuda_ver()

    def has_cuda_device(self):
        """Checks for cuda hardware in cycles preferences"""
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences

        for device in cprefs.get_devices(bpy.context):
            for dev_entry in device:
                if dev_entry.type == 'CUDA':
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

        self.major, self.minor, self.release = ver.split(".", 2)
        self.result = CudaResult.SUCCESS
