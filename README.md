# bRigNet
---------
Neural Rigging for [blender](https://www.blender.org/ "Blender Home Page") using [RigNet](https://zhan-xu.github.io/rig-net/ "RigNet Home Page")

Blender is the open source 3D application from the Blender Foundation. RigNet has been presented in the following papers

``` 
  @InProceedings{AnimSkelVolNet,
    title={Predicting Animation Skeletons for 3D Articulated Models via Volumetric Nets},
    author={Zhan Xu and Yang Zhou and Evangelos Kalogerakis and Karan Singh},
    booktitle={2019 International Conference on 3D Vision (3DV)},
    year={2019}
  }
```

```
  @article{RigNet,
    title={RigNet: Neural Rigging for Articulated Characters},
    author={Zhan Xu and Yang Zhou and Evangelos Kalogerakis and Chris Landreth and Karan Singh},
    journal={ACM Trans. on Graphics},
    year={2020},
    volume={39}
  }
```

**Warning:** this addon is experimental, the install procedure is clumsy and the code could be better.
Everything about it could change, hopefully in good.


## Setup

bRigNet requires SciPy, PyTorch and torch-geometric, along with torch-scatter and torch-sparse.

Environment managers, like conda or virtualenv can be used to ease the install.
To take advantage of GPU hardware, PyTorch requires the CUDA toolkit, which can be found at the
[manifacturer website](https://developer.nvidia.com)

### Install the required modules using *conda*

Anaconda is a data science platform from Anaconda Inc., it can be downloaded from the
[company website](https://www.anaconda.com/).

A lightweight version called [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is available.
Both versions include the package manager 'conda' used in the following steps.

- Open a Miniconda or Anaconda prompt
- Create a Conda Environment and activate it

```
conda create -n brignet python=3.7
conda activate brignet_deps
```

- Install PyTorch. If CUDA is installed, the CUDA version can be queried in a command prompt. For example

```
nvcc --version
```
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:32:27_Pacific_Daylight_Time_2019
Cuda compilation tools, release 10.2, V10.2.89
```

In this case pytorch can be installed in the command prompt via

```
conda install pytorch==1.7.1 cudatoolkit=10.2 -c pytorch
```

More complete information on the PyTorch command line can be found at the [PyTorch website](https://pytorch.org/)
The install command on non-cuda devices is

```
conda install pytorch==1.7.1 cpuonly -c pytorch
```

- Install torch utilities. The syntax follows the pattern

```
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-[version]+cu[cuda-version].html
```

Pre-built packages for 1.7.1 and 1.8.1 are 1.7.0 and 1.8.0 anyway. Example:
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-geometric
```

Alternatively, pip can try and build the libraries. Even if part of torch-sparse fails without a proper environment,
the rilevant modules are usually built

```
pip install torch-scatter
pip install torch-sparse
pip install torch-geometric
```

The directory of each environment can be obtained via

```
conda info --envs
```

The environment directory can be set in the "Additional Modules" setting of the bRigNet preferences

### Install the required modules using virtualenv

Alternatively, virtualenv can be used to create a Python environment with the required packages.
First, python 3.7 must be installed on the system. It can be found at https://www.python.org/downloads/

Make sure that **Add Python 3.7 to PATH** is checked in the setup options.    

Usually, python comes with its package manager installed (pip). Please, refer
to the [pip documentation](https://pypi.org/project/pip/) if pip is not present in your system.

Next step is to install virtualenv. Open a command prompt and reach a folder where python packages will be kept
please execute.

```
pip install virtualenv
```

then create the virtual environment and activate it

```
virtualenv brignet_deps
cd brignet_deps
Scripts\activate
```

now we can install the torch library. At present, version 1.8.1 is provided. Cuda 10.2 is used in this example:

```
pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-geometric
```

the virtual environment directory can be set as the "Additional modules path" in the brignet preferences

 
 

### Using pip with blender's Python interpreter

You can install pip following [this guide](http://www.codeplastic.com/2019/03/12/how-to-install-python-modules-in-blender/ "pip in blender"),
then

```
[BLENDER_DIR]\[BLENDER_VER]\python\bin\python -m pip install --upgrade Pillow
```

## Installation

Download bRigNet as a .zip file and install it from the blender addons window,
or copy the code to the blender scripts path

## Usage 
Enable *bRigNet* in the blender addons, the preferences will show up.
Set the Modules path properties to the RigNet environment from the previous step

RigNet requires a trained model. They have made theirs available at [this address](https://umass.box.com/s/l7dxfayrubf5qzxcyg7can715xnislwm)
The checkpoint folder can be copied to the RigNet subfolder.
A different location can be set in the addon preferences.

#### Rig Generation

the **bRigNet** tab will show up in the Viewport tools. Select a character mesh as target.
Please make sure it doesn't exceed the 5K triangles. You can use the *Decimator* modifier
to reduce the polycount on a copy of the mesh, and select a *Collection* of high res model
on which to transfer the final weights  

#### Load generated rigs

Rigs generated using RigNet from the command line can be loaded via the **Load Skeleton** panel.
Please select the *.obj and *.txt file and press the button **Load Rignet character**

## License

bRigNet is released under the GNU General Public License version 3. RigNet is licensed under the General Public License Version 3 (GPLv3), or under a Commercial License.
