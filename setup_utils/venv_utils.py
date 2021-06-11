import os
import sys
import subprocess
import tempfile
import venv


def create_venv(vpath):
    if not os.path.isdir(vpath):
        venv.create(vpath)


def torch_install_lines(vpath, torch_version="1.8.1", cuda_version="102", torch_url='https://download.pytorch.org/whl/torch_stable.html'):
    v_pip = os.path.join(vpath, "Scripts", "pip3")
    if sys.platform.startswith("win"):
        v_pip += ".exe"
    v_pip = os.path.normpath(v_pip)

    lines = f"{v_pip} install torch=={torch_version}+cu{cuda_version} -f {torch_url}\n"
    extension_base = v_pip + " install torch-{0}"

    if torch_version.startswith("1.8.") or torch_version.startswith("1.7."):
        torch_version = torch_version[:-1] + "0"

    extension_template = f" -f https://pytorch-geometric.com/whl/torch-{torch_version}+cu{cuda_version}.html\n"
    extension_template = extension_base + extension_template

    lines += extension_template.format("scatter")
    lines += extension_template.format("sparse")
    lines += extension_template.format("cluster")
    lines += extension_template.format("spline-conv")
    lines += extension_base.format("geometric")
    return lines


def create_vpip_batch_file(vpath):
    on_win = sys.platform.startswith("win")
    ba_file = tempfile.NamedTemporaryFile(mode='w+b',
                                          prefix="vpip_install_",
                                          suffix='.bat' if on_win else None,
                                          delete=False)

    v_activate = os.path.join(vpath, "Scripts", "activate")
    v_py = os.path.join(vpath, "Scripts", "python")

    v_activate = os.path.normpath(v_activate)
    v_py = os.path.normpath(v_py)

    if on_win:
        v_activate = "call " + v_activate
        v_activate += ".bat"
        v_py += ".exe"

    with ba_file as f:
        if on_win:
            f.write(b"@echo off\n")
        f.write(bytes(v_activate, 'utf-8'))
        f.write(b"\n")
        f.write(bytes(f"{v_py} -m ensurepip", 'utf-8'))
        f.write(b"\n")
        f.write(bytes(torch_install_lines(vpath), 'utf-8'))

    return ba_file.name
