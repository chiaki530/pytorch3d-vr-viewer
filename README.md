# PyTorch3D VR Viewer（Experimental）

## Installation
Install [SteamVR](https://www.steamvr.com/zh-cn/) which is needed by OpenVR applications.

Create a conda environment
```
conda create -n pytorch3d_vr_test python=3.9
conda activate pytorch3d_vr_test
```

Install pyopenvr
```
pip install openvr
```
Install PyTorch3D， refer to 
```
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c fvcore -c iopath -c conda-forge fvcore iopath plotly
pip install matplotlib opencv-python open3d
```
Install pyGLFW, pyOpenGL
```
pip install glfw
pip install PyOpenGL PyOpenGL_accelerate
```
Install numpysocket for Client-Server mode
```
pip install numpysocket
```

## Test
- [X] Cube Actor
- [X] Mesh Actor
- [X] Point Actor
- [ ] Volume Actor
- [ ] Implicit Actor
- [ ] Remote Actor