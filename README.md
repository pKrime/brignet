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
Follow [RigNet instruction](https://github.com/zhan-xu/RigNet "RigNet repository")
and create a Python environment with all required packages

For compatibility reasons, [Pillow](https://python-pillow.org/) must be installed in blender.
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
