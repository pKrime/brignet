import os
from pathlib import Path
import requests
import sys
import subprocess
import shutil
import tarfile
import tempfile
import venv

from .cuda_utils import CudaDetect


class VenvAutoSetup:
    def __init__(self, environment_path):
        self.env_path = environment_path
        self._on_win = sys.platform.startswith("win")
        self.py_exe = ""

    def create_venv(self, with_pip=True):
        if os.path.isdir(self.env_path):
            if len(os.listdir(self.env_path)) > 0:
                msg = "Can't create Virtual Env in existing, non empty directory)"
                # TODO: Custom Exception
                raise Exception(msg)
        
        venv.create(self.env_path, with_pip=with_pip)

        self.py_exe = self._get_py_exe()

    def _get_py_exe(self):
        v_py = os.path.join(self.env_path, "Scripts", "python")
        v_py = os.path.normpath(v_py)

        if self._on_win:
            v_py += ".exe"

        return v_py

    def venv_activate_line(self):
        v_activate = os.path.join(self.env_path, "Scripts", "activate")
        v_activate = os.path.normpath(v_activate)

        if self._on_win:
            v_activate = 'call "' + v_activate
            v_activate += '.bat"'
        else:
            v_activate = "source '{0}'".format(v_activate)

        return v_activate

    def pip_install_lines(self):
        lines = [
            f'"{self.py_exe}" -m ensurepip',
            f'{self.py_exe}" -m pip install wheel'
        ]

        return lines

    def pip_install_script(self):
        ba_file = tempfile.NamedTemporaryFile(mode='w+b',
                                              prefix="vpip_install_",
                                              suffix='.bat' if self._on_win else None,
                                              delete=False)

        with ba_file as f:
            if self._on_win:
                f.write(b"@echo off\n")
            else:
                f.write(b"#!/bin/bash\n")
            f.write(bytes(self.venv_activate_line(), 'utf-8'))
            f.write(b"\n")

            for line in self.pip_install_lines():
                f.write(bytes(line, 'utf-8'))
                f.write(b"\n")

        return ba_file.name

    def torch_install_script(self, torch_version="1.11.0", cuda_version="113", torch_url=""):
        if not torch_url:
            torch_url = "https://download.pytorch.org/whl"

        ba_file = tempfile.NamedTemporaryFile(mode='w+b',
                                              prefix="torch_install_",
                                              suffix='.bat' if self._on_win else None,
                                              delete=False)

        torch_line = f'"{self.py_exe}" -m pip install torch=={torch_version}+cu{cuda_version} --extra-index-url {torch_url}/cu{cuda_version}'

        with ba_file as f:
            if self._on_win:
                f.write(b"@echo off\n")
            else:
                f.write(b"#!/bin/bash\n")
            f.write(bytes(self.venv_activate_line(), 'utf-8'))
            f.write(b"\n")

            f.write(bytes(torch_line, 'utf-8'))
            f.write(b"\n")

        return ba_file.name

    def pkg_install_script(self, package_name, env_vars=dict(), additional_parameter=""):
        ba_file = tempfile.NamedTemporaryFile(mode='w+b',
                                              prefix=f"{package_name}_install_",
                                              suffix='.bat' if self._on_win else None,
                                              delete=False)

        pkg_line = f'"{self.py_exe}" -m pip install {package_name}'
        if additional_parameter:
            pkg_line += f" {additional_parameter}"

        with ba_file as f:
            if self._on_win:
                f.write(b"@echo off\n")
                for k, v in env_vars.items():
                    f.write(bytes(f'set "{k}={v}"\n', 'utf-8'))
            else:
                f.write(b"#!/bin/bash\n")
                for k, v in env_vars.items():
                    f.write(bytes(f'{k}="{v}"\n', 'utf-8'))

            f.write(bytes(self.venv_activate_line(), 'utf-8'))
            f.write(b"\n")

            f.write(bytes(pkg_line, 'utf-8'))
            f.write(b"\n")

        return ba_file.name

    def pkg_download_script(self, download_dir, packages=('torch-sparse', 'torch-cluster')):
        ba_file = tempfile.NamedTemporaryFile(mode='w+b',
                                              prefix="sparse_install_",
                                              suffix='.bat' if self._on_win else None,
                                              delete=False)

        pkg_lines = [f'python -m pip download --no-deps {pkg_name} -d {download_dir}\n' for pkg_name in packages]

        with ba_file as f:
            if self._on_win:
                f.write(b"@echo off\n")
            else:
                f.write(b"#!/bin/bash\n")
            f.write(bytes(self.venv_activate_line(), 'utf-8'))
            f.write(b"\n")

            for line in pkg_lines:
                f.write(bytes(line, 'utf-8'))
            f.write(b"\n")

        return ba_file.name


def fix_source_absolute_paths(package_name, download_dir):
    pkg_archive = ""
    for fn in os.listdir(download_dir):
        if fn.startswith(f'{package_name}-') and fn.endswith('.tar.gz'):
            pkg_archive = fn
            break

    if not pkg_archive:
        raise FileNotFoundError(f'{package_name} archive not found')

    pkg_namever = pkg_archive
    pkg_namever = os.path.splitext(pkg_namever)[0]
    pkg_namever = os.path.splitext(pkg_namever)[0]

    pkg_dir = os.path.join(download_dir, pkg_namever)
    ar_file = tarfile.open(os.path.join(download_dir, pkg_archive))
    ar_file.extractall(pkg_dir)
    ar_file.close()

    src_info = os.path.join(pkg_dir, pkg_namever, f'{package_name}.egg-info', 'SOURCES.txt')
    src_old = os.path.join(pkg_dir, pkg_namever, f'{package_name}.egg-info', 'SOURCES_orig.txt')

    src_info = os.path.normpath(src_info)
    src_old = os.path.normpath(src_old)

    shutil.move(src_info, src_old)

    with open(src_old) as old, open(src_info, 'w') as new:
        lines = old.readlines()
        new.writelines([line for line in lines if not line.startswith('/')])

    dist_dir = os.path.join(download_dir, 'fix')
    Path(dist_dir).mkdir(0o755, exist_ok=True)
    fix_tar = os.path.join(dist_dir, f'{pkg_namever}.tar')
    with tarfile.open(fix_tar, "w") as tar:
        tar.add(os.path.join(pkg_dir, pkg_namever), arcname=pkg_namever)

    shutil.rmtree(pkg_dir)
    return os.path.normpath(fix_tar)


def download_python_headers(download_dir):
    v_info = sys.version_info
    v_str = f"{v_info.major}.{v_info.minor}.{v_info.micro}"
    py_name = f"Python-{v_str}"
    f_name = f"{py_name}.tgz"

    py_dir = os.path.join(download_dir, "_python")
    Path(py_dir).mkdir(0o755, exist_ok=True)
    f_path = os.path.join(py_dir, f_name)

    if os.path.isfile(f_path):
        print(f"Using cached {f_path}")
    else:
        src_url = f"https://www.python.org/ftp/python/{v_str}/{f_name}"
        if not os.path.isfile(f_path):
            r = requests.get(src_url, allow_redirects=True)
            open(f_path, 'wb').write(r.content)

    include_dir = f'{py_name}/Include'
    ar_file = tarfile.open(f_path)
    for file_name in ar_file.getnames():
        if file_name.startswith(include_dir):
            ar_file.extract(file_name, os.path.join(download_dir, py_dir))

    ar_file.close()
    headers_dir = os.path.join(py_dir, include_dir)
    return os.path.normpath(headers_dir)


def install_headers(env_path, download_dir):
    extracted_headers_dir = download_python_headers(download_dir)
    py_include_dir = os.path.join(env_path, 'Include')

    for item in os.listdir(extracted_headers_dir):
        src_path = os.path.join(extracted_headers_dir, item)
        dst_path = os.path.join(py_include_dir, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            continue

        shutil.copy(src_path, dst_path)


def setup_environment(environment_path, with_pip=True, torch_version="1.11.0", cuda_version='113'):
    ve_setup = VenvAutoSetup(environment_path)
    ve_setup.create_venv(with_pip=with_pip)

    if with_pip:
        # TODO: install wheels
        pass
    else:
        # install pip via script
        print("installing pip")
        pip_install_script = ve_setup.pip_install_script()
        subprocess.check_call(pip_install_script)

    # install pytorch
    print("installing torch")
    torch_install_script = ve_setup.torch_install_script(torch_version=torch_version, cuda_version=cuda_version)
    subprocess.check_call(torch_install_script)

    # install torch-geometric
    if cuda_version in ('101', '102', '111'):
        # wheels are provided for these versions
        find_link = f"-f https://pytorch-geometric.com/whl/torch-{torch_version}+cu{cuda_version}.html"
        for pkg in ("torch-scatter", "torch-sparse", "torch-cluster", "torch-geometric"):
            print(f"Installing {pkg}")
            pkg_inst_script = ve_setup.pkg_install_script(pkg, additional_parameter=find_link)
            subprocess.check_call(pkg_inst_script)
    elif cuda_version == '113':
        find_link = f"-f https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_version}.html"
        for pkg in ("torch-scatter", "torch-sparse", "torch-cluster", "torch-geometric"):
            print(f"Installing {pkg}")
            pkg_inst_script = ve_setup.pkg_install_script(pkg, additional_parameter=find_link)
            subprocess.check_call(pkg_inst_script)
    else:
        # we gotta build'em wheels
        platform = sys.platform
        if platform.startswith('linux'):
            cuda_lib_path = os.path.join(cuda_detect.get_cuda_path(), "lib64")
            os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
        elif platform == 'darwin':
            cuda_lib_path = os.path.join(cuda_detect.get_cuda_path(), "lib")
            os.environ['DYLD_LIBRARY_PATH '] = f"{cuda_lib_path}:{os.environ['DYLD_LIBRARY_PATH']}"
        else:
            raise NotImplementedError(f"Auto-Build not supported on {platform}")

        # install torch-scatter
        scatter_inst_script = ve_setup.pkg_install_script('torch-scatter')
        subprocess.check_call(scatter_inst_script)

        # install torch-sparse
        cuda_include_path = os.path.join(cuda_detect.get_cuda_path(), "include")
        if platform.startswith('win'):
            # TODO: needs python3.lib
            # TODO: check if cl.exe is available
            # we need to patch the absolute paths in Source to fix build error
            download_dir = os.path.join(ve_setup.env_path, '_download')

            # we need python headers
            install_headers(environment_path, download_dir)

            Path(download_dir).mkdir(0o755, exist_ok=True)
            pkg_dw_script = ve_setup.pkg_download_script(download_dir)
            subprocess.check_call(pkg_dw_script)

            fixed_sparse = fix_source_absolute_paths('torch_sparse', download_dir)
            pkg_install_script = ve_setup.pkg_install_script(fixed_sparse, env_vars=dict(CPATH=cuda_include_path))
        else:
            pkg_install_script = ve_setup.pkg_install_script('torch-sparse')

        subprocess.check_call(pkg_install_script)

        # install torch-cluster
        if platform.startswith('win'):
            # we need to patch the absolute paths in Source to fix build error
            fixed_cluster = fix_source_absolute_paths('torch_cluster', download_dir)
            pkg_install_script = ve_setup.pkg_install_script(fixed_cluster, env_vars=dict(CPATH=cuda_include_path))
        else:
            pkg_install_script = ve_setup.pkg_install_script('torch_cluster')
        subprocess.check_call(pkg_install_script)

        # install geometric
        pkg_install_script = ve_setup.pkg_install_script('torch_geometric')
        subprocess.check_call(pkg_install_script)
