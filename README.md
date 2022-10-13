# PyTorch3D VR Viewer
This is an experimental repository aims to develop a customizable VR neural rendering viewer written in Python for evaluating and developing neural rendering methods for VR. 

## Pipeline
We adopts the popular [Pytorch3D]() as our backbone which can be used to implement major differentiable neural rendering methods (e.g. NeRF). For VR connection, we uses the [pyopenvr](https://github.com/cmbruns/pyopenvr), a python bindings for Valve's OpenVR virtual reality SDK to make the whole pipeline all in Python. While Pytorch3D may not be optimized for the real-time performance, it can be replaced with any latest neural renderers with Python interface.

## Installation
Install [SteamVR](https://www.steamvr.com/zh-cn/) and [Oculus App](https://www.meta.com/ch/en/quest/setup/) which is needed by OpenVR applications.

Create a conda environment.
```
conda create -n pytorch3d_vr python=3.9
conda activate pytorch3d_vr
```

Install pyopenvr, pyglfw, and pyopengl.
```
pip install openvr glfw PyOpenGL PyOpenGL_accelerate
```
Install PyTorch3D. More details can see [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
```
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c fvcore -c iopath -c conda-forge fvcore iopath plotly
```
Install other dependencies.
```
pip install matplotlib opencv-python open3d numpysocket
```

## Usage
As running OpenVR and Pytorch3D on the same GPU is likely to reach the GPU memory limit can signicantly slow down the rendering speed, we provide two modes here.
- *All-in-One mode*: Running OpenVR and Pytorch3D renderer on the same machine (single GPU). This is suitable when the Pytorch3D GPU usage is small (e.g. point/mesh renderers) or the GPU memory is big enough.
- *Client-Server mode*: Running OpenVR on one machine and Pytorch3D renderer on another machine (two GPUs). This is suitable when the GPU memory is small or the powerful GPU is avaiable at the remote cluster. We transfer data using socket. 

For *All-in-one* mode, we tested it on a single GeForce RTX 3060 Laptop (6GB). For *Client-Server* mode, we tested it with server running on GeForce RTX 3060 Laptop (6GB) and client running on GeForce RTX 2060 Laptop (6GB). 

### All-in-One mode
Firstly, open the Oculus Quest 2 and connect to the PC with Oculus-Link or Oculus-Air-Link. Then simply run
```
python run_glfw.py [--actor ACTOR]
```
We currently supports default Pytorch3D renderers including mesh, point, volume and implicit renderers.

### Client-Server Mode
Similarly, make sure the Oculus Quest 2 is connected to the client machine with Oculus-Link or Oculus-Air-Lnik. Additinally, find out the IP address of the server machine (the one running actor) and make sure it is accessible from the client.

On the server machine, start a actor in a remote mode by running on a given port. For example, to run a mesh actor we can use
```
python actor/pytorch3d_mesh_actor.py [--port PORT]
```
On the client machine, run
```
python run_glfw.py --actor remote [--host_ip HOST_IP] [--port PORT]
```
where HOST_IP and PORT is the server's IP address and port.

## Todo
- [X] Add default Pytorch3D renderers
- [X] Add support for client-server mode
- [ ] Solve flashing problem for low FPS cases
- [ ] Optimize for speed
